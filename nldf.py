"""N-LDF: Normalized Local Density Factor oversampling.

This module is the sampler class that produced every result in the manuscript,
transcribed verbatim from the cell titled "Cell 3 (FINAL FIXED)" of the
experiment notebook, together with ``build_sampler``, which fixes the parameter
values used in the benchmark.

Provenance. The notebook defines ``NLDFv2`` twice. The earlier cell ("REVISED
COMPLETE") applies a synthetic cap of 20 * n_min, a 3 * N_target attempt cap, a
deterministic interpolation coefficient, and a two-stage acceptance test. The
later cell, reproduced here, applies no cap, allows 5 * N_target attempts,
draws the interpolation coefficient from a Beta distribution, uses OAS
covariance shrinkage, projects one-hot blocks before acceptance, and applies a
three-stage acceptance test. The archived ``results/fold_level_results.csv``
carries the diagnostic columns (``n_attempted``, ``n_rejected_density``, ...)
that only this later class writes, and its accepted counts exceed 20 * n_min on
21 datasets, so this is the class that ran. ``reproduce_paper_tables.py --check``
verifies both facts against the archive.

Usage:
    from nldf import build_sampler
    sampler = build_sampler("N-LDF", y_train, projector, seed)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    sampler.last_diagnostics_   # per-call log, same keys as the archive
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import LinAlgError, cholesky
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42


def effective_k(y_train: np.ndarray, requested: int = 5) -> int:
    """Neighbor count shared by every sampler: 5, or n_min - 1 when smaller."""
    n_minority = int((y_train == 1).sum())
    if n_minority < 2:
        raise ValueError("At least two training minority observations are required")
    return min(requested, n_minority - 1)


class NLDFv2:
    """
    N-LDF 2.0: Normalized Local Density Factor - FINAL FIXED VERSION

    Semua perbaikan kritis telah diterapkan:
    - Single prior correction (λ controls class-size normalization)
    - Reproducible random numbers
    - Correct bandwidth in acceptance filter
    - Correct acceptance_threshold semantics
    - Projection before acceptance
    - OAS covariance for all dimensions
    - Duplicate check includes synthetic samples
    """

    def __init__(
        self,
        # Core parameters
        k_neighbors: int = 5,
        k_density: int = 7,

        # Boundary parameters
        q0: float = 0.20,
        beta: float = 0.075,
        q_min: float = 0.10,
        q_max: float = 0.45,

        # Prior correction (SINGLE formulation)
        lambda_prior: float = 0.7,  # 0=raw counts, 1=full class-size normalization

        # Soft weighting
        tau_boundary: float = 0.5,  # FIXED: lebih masuk akal (0.1 terlalu sempit)
        tau_noise: float = 0.05,
        delta_noise: float = 0.02,

        # Generation
        lambda_boundary: float = 1.5,
        lambda_safe: float = 1.0,

        # Jitter
        jitter_base: float = 0.01,
        use_covariance: bool = True,

        # Acceptance
        acceptance_threshold: float = -0.10,  # FIXED: default negatif
        min_distance: float = 0.001,

        # Sampling ratio
        sampling_ratio: float = 0.75,

        # Other
        epsilon: float = 1e-8,
        projector: Callable[[np.ndarray], np.ndarray] | None = None,
        random_state: int = RANDOM_STATE,

        # Inner CV (non-recursive)
        use_inner_cv: bool = False,
        ratio_grid: list[float] = None,
    ):
        self.k_neighbors = k_neighbors
        self.k_density = k_density
        self.q0 = q0
        self.beta = beta
        self.q_min = q_min
        self.q_max = q_max
        self.lambda_prior = lambda_prior
        self.tau_boundary = tau_boundary
        self.tau_noise = tau_noise
        self.delta_noise = delta_noise
        self.lambda_boundary = lambda_boundary
        self.lambda_safe = lambda_safe
        self.jitter_base = jitter_base
        self.use_covariance = use_covariance
        self.acceptance_threshold = acceptance_threshold
        self.min_distance = min_distance
        self.sampling_ratio = sampling_ratio
        self.epsilon = epsilon
        self.projector = projector
        self.random_state = random_state
        self.use_inner_cv = use_inner_cv
        self.ratio_grid = ratio_grid or [0.25, 0.50, 0.75, 1.00]
        self.last_diagnostics_: dict[str, float | int] = {}

    # ============================================================
    # IMPROVEMENT 2: Adaptive kernel with local bandwidth
    # ============================================================
    def _compute_bandwidth(self, X: np.ndarray) -> np.ndarray:
        """Compute local bandwidth using median k-nearest neighbor distance."""
        n_neighbors = min(self.k_density + 1, len(X))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        distances, _ = nn.kneighbors(X)
        h = np.median(distances[:, 1:], axis=1)
        return np.maximum(h, self.epsilon)

    def _kernel_weight(self, distances: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Kernel with local bandwidth. K(d) = 1 / (1 + (d/h)^2)."""
        h_safe = np.maximum(h, self.epsilon)
        scaled = distances / h_safe[:, np.newaxis]
        return 1.0 / (1.0 + scaled**2)

    def _kernel_weight_single(self, distances: np.ndarray, h: float) -> np.ndarray:
        """Kernel for a single sample with scalar bandwidth."""
        h_safe = max(h, self.epsilon)
        scaled = distances / h_safe
        return 1.0 / (1.0 + scaled**2)

    # ============================================================
    # IMPROVEMENT 1: Class-prior-corrected eta (SINGLE correction)
    # ============================================================
    def _eta_corrected(self, X_min: np.ndarray, X_all: np.ndarray,
                       y_all: np.ndarray) -> np.ndarray:
        """
        Class-prior-corrected local density factor.

        FORMULATION (SINGLE correction):
        A_i = S_min / (n_min ** lambda_prior)
        B_i = S_maj / (n_maj ** lambda_prior)

        lambda_prior=0: raw local support (no correction)
        lambda_prior=1: full class-size normalization
        0 < lambda_prior < 1: partial prior correction
        """
        n_min = int(np.sum(y_all == 1))
        n_maj = int(np.sum(y_all == 0))

        if n_min == 0 or n_maj == 0:
            return np.zeros(len(X_min))

        # Compute bandwidths for minority samples
        h = self._compute_bandwidth(X_all)
        h_min = h[y_all == 1]

        # All pairwise distances from minority to all samples
        n_neighbors = min(self.k_density + 1, len(X_all))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_all)
        distances, indices = nn.kneighbors(X_min)

        etas = np.empty(len(X_min), dtype=float)
        minority_global_indices = np.flatnonzero(y_all == 1)

        for row, global_idx in enumerate(minority_global_indices):
            # Exclude self
            mask = indices[row] != global_idx
            neighbor_idx = indices[row][mask][:self.k_density]
            neighbor_dist = distances[row][mask][:self.k_density]

            # Kernel weights
            weights = self._kernel_weight(
                neighbor_dist.reshape(1, -1),
                np.array([h_min[row]])
            )[0]

            # Separate minority and majority neighbors
            min_mask = y_all[neighbor_idx] == 1
            maj_mask = y_all[neighbor_idx] == 0

            # FIXED: Single prior correction
            S_min = np.sum(weights[min_mask])
            S_maj = np.sum(weights[maj_mask])

            A = S_min / max(n_min ** self.lambda_prior, self.epsilon)
            B = S_maj / max(n_maj ** self.lambda_prior, self.epsilon)

            denom = A + B + self.epsilon
            etas[row] = (A - B) / denom

        return etas

    # ============================================================
    # IMPROVEMENT 3: Adaptive quantile boundary
    # ============================================================
    def _compute_delta(self, eta: np.ndarray, IR: float) -> float:
        """Boundary that WIDENS with IR, not narrows."""
        log_ir = np.log10(max(IR, 2.0))
        q = np.clip(self.q0 + self.beta * log_ir, self.q_min, self.q_max)

        abs_eta = np.abs(eta)
        delta = np.quantile(abs_eta, q) if len(abs_eta) > 0 else 0.1

        return max(delta, self.epsilon)

    # ============================================================
    # IMPROVEMENT 4: Soft seed weighting (ACTUALLY USES delta)
    # ============================================================
    def _compute_weights(self, eta: np.ndarray, delta: float) -> np.ndarray:
        """
        Soft weighting with delta properly integrated.

        z_i = eta_i / delta
        u_i = exp(-z_i^2 / (2 * tau_b^2))
        """
        # Normalize eta by delta
        z = eta / (delta + self.epsilon)

        # Boundary proximity (Gaussian, centered at eta=0)
        u = np.exp(-z**2 / (2 * self.tau_boundary**2))

        # Noise suppression (sigmoid)
        r = 1.0 / (1.0 + np.exp(-(eta + self.delta_noise) / self.tau_noise))

        # Safe weight (positive eta gets weight)
        safe_weight = np.maximum(eta, 0)

        # Combined weight
        w = self.lambda_boundary * u * r + self.lambda_safe * safe_weight

        # Normalize
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(len(eta)) / len(eta)

        return w

    # ============================================================
    # IMPROVEMENT 5: Safe-directed LOCAL neighbor selection
    # ============================================================
    def _select_neighbor(self, seed_idx: int, eta_min: np.ndarray,
                         X_min: np.ndarray, rng: np.random.Generator) -> int:
        """Prefer safer neighbors from LOCAL k-nearest neighbors."""
        n_samples = len(X_min)
        if n_samples <= 1:
            return 0

        k_eff = min(self.k_neighbors, n_samples - 1)
        if k_eff <= 0:
            return 0

        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
        distances, indices = nn.kneighbors(X_min[seed_idx].reshape(1, -1))

        neighbor_indices = indices[0][1:k_eff + 1]
        neighbor_distances = distances[0][1:k_eff + 1]

        if len(neighbor_indices) == 0:
            return seed_idx

        h = np.median(neighbor_distances) if len(neighbor_distances) > 0 else 1.0
        h = max(h, self.epsilon)

        safe_weights = np.maximum(eta_min[neighbor_indices], 0)
        dist_weights = np.exp(-neighbor_distances**2 / h**2)

        probs = dist_weights * safe_weights

        if np.sum(probs) == 0:
            return int(rng.choice(neighbor_indices))

        probs = probs / np.sum(probs)
        return int(rng.choice(neighbor_indices, p=probs))

    # ============================================================
    # IMPROVEMENT 5: Alpha dengan RNG dan safety difference
    # ============================================================
    def _compute_alpha(self, eta_seed: float, eta_target: float,
                       rng: np.random.Generator) -> float:
        """
        Adaptive alpha based on safety difference.
        Uses provided rng for reproducibility.
        """
        safety_diff = eta_target - eta_seed
        mu_alpha = np.clip(0.5 + 0.5 * safety_diff, 0.2, 0.8)
        kappa = 5.0
        alpha = rng.beta(kappa * mu_alpha, kappa * (1 - mu_alpha))
        return np.clip(alpha, 0.1, 0.9)

    # ============================================================
    # IMPROVEMENT 6: Local covariance jitter with OAS
    # ============================================================
    def _compute_local_covariance(self, X: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Compute local covariance with OAS shrinkage for stability."""
        n_features = X.shape[1]
        n_samples = X.shape[0]

        if n_samples < k + 1:
            return None, np.ones(n_samples) * self.jitter_base

        n_neighbors = min(k + 1, n_samples)
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, indices = nn.kneighbors(X)

        covariances = np.zeros((n_samples, n_features, n_features))
        scales = np.ones(n_samples) * self.jitter_base

        for i in range(n_samples):
            neighbors = X[indices[i][1:]]

            # FIXED: OAS if at least 2 neighbors
            if len(neighbors) >= 2:
                try:
                    oas = OAS(assume_centered=False)
                    cov = oas.fit(neighbors).covariance_
                    cov += np.eye(n_features) * self.jitter_base
                    covariances[i] = cov
                    scales[i] = self.jitter_base * np.sqrt(np.mean(np.diag(cov)))
                except Exception:
                    covariances[i] = np.eye(n_features) * self.jitter_base**2
                    scales[i] = self.jitter_base
            else:
                covariances[i] = np.eye(n_features) * self.jitter_base**2
                scales[i] = self.jitter_base

        return covariances, scales

    def _generate_sample(self, seed: np.ndarray, target: np.ndarray,
                        alpha: float, seed_idx: int,
                        local_cov: np.ndarray | None,
                        scale: float, eta_seed: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        """
        Generate with local covariance jitter.
        Returns: (sample, covariance_failed)
        """
        displacement = target - seed

        if np.all(np.abs(displacement) < self.epsilon):
            displacement = rng.normal(0, 0.01, size=seed.shape)

        sample = seed + alpha * displacement

        risk_scale = np.clip((eta_seed + 1) / 2, 0.1, 1.0)
        jitter_scale = self.jitter_base * risk_scale * 0.5
        covariance_failed = False

        if self.use_covariance and local_cov is not None:
            try:
                L = cholesky(local_cov, lower=True)
                z = rng.normal(0, 1, size=seed.shape)
                jitter = L @ z * jitter_scale
                sample += jitter
            except LinAlgError:
                covariance_failed = True
                jitter = rng.normal(0, jitter_scale, size=seed.shape)
                sample += jitter
        else:
            jitter = rng.normal(0, jitter_scale, size=seed.shape)
            sample += jitter * np.std(displacement)

        return sample, covariance_failed

    # ============================================================
    # IMPROVEMENT 7: Acceptance filtering dengan FULL AUDIT
    # ============================================================
    def _compute_eta_for_sample(self, sample: np.ndarray, X_all: np.ndarray,
                                y_all: np.ndarray) -> float:
        """Compute eta for a single candidate sample."""
        n_min = int(np.sum(y_all == 1))
        n_maj = int(np.sum(y_all == 0))

        if n_min == 0 or n_maj == 0:
            return 0.0

        n_neighbors = min(self.k_density + 1, len(X_all))
        nn_all = NearestNeighbors(n_neighbors=n_neighbors).fit(X_all)
        distances, indices = nn_all.kneighbors(sample.reshape(1, -1))

        # FIXED: Bandwidth from candidate itself
        candidate_dist = distances[0]
        h_new = max(np.median(candidate_dist), self.epsilon)
        weights = self._kernel_weight_single(candidate_dist, h_new)

        min_mask = y_all[indices[0]] == 1
        maj_mask = y_all[indices[0]] == 0

        S_min = np.sum(weights[min_mask])
        S_maj = np.sum(weights[maj_mask])

        A = S_min / max(n_min ** self.lambda_prior, self.epsilon)
        B = S_maj / max(n_maj ** self.lambda_prior, self.epsilon)

        denom = A + B + self.epsilon
        return (A - B) / denom

    def _accept_sample(self, sample: np.ndarray, X_existing: np.ndarray,
                      y_existing: np.ndarray, X_min: np.ndarray,
                      eta_min: np.ndarray, delta: float,
                      X_all: np.ndarray, y_all: np.ndarray,
                      generated: list) -> tuple[bool, str]:
        """
        Acceptance filtering with detailed diagnostic reasons.
        Includes duplicate check against synthetic samples.
        """
        # Check 1: Distance to existing samples (including generated)
        if len(X_existing) > 0:
            min_dist = np.min(cdist(sample.reshape(1, -1), X_existing))
            if min_dist < self.min_distance:
                return False, "too_close"

        # FIXED: Check against generated samples too
        if generated:
            dist_generated = np.min(cdist(sample.reshape(1, -1), np.asarray(generated)))
            if dist_generated < self.min_distance:
                return False, "too_close_generated"

        # Check 2: Eta-based acceptance (FIXED semantics)
        if len(X_min) > 1 and len(X_all) > 1:
            eta_new = self._compute_eta_for_sample(sample, X_all, y_all)
            # FIXED: Correct threshold semantics
            if eta_new < self.acceptance_threshold:
                return False, "low_density"

        # Check 3: Density ratio (safety net)
        if len(X_min) > 1 and len(X_existing[y_existing == 0]) > 1:
            nn_min = NearestNeighbors(n_neighbors=min(3, len(X_min))).fit(X_min)
            dist_to_min, _ = nn_min.kneighbors(sample.reshape(1, -1))

            X_maj = X_existing[y_existing == 0]
            nn_maj = NearestNeighbors(n_neighbors=min(3, len(X_maj))).fit(X_maj)
            dist_to_maj, _ = nn_maj.kneighbors(sample.reshape(1, -1))

            ratio = np.mean(dist_to_min) / (np.mean(dist_to_maj) + self.epsilon)

            if ratio > 1.5:
                return False, "maj_dominated"

        return True, "accepted"

    # ============================================================
    # IMPROVEMENT 8: Inner CV for sampling ratio (NON-RECURSIVE)
    # ============================================================
    def _fit_resample_fixed_ratio(self, X: np.ndarray, y: np.ndarray,
                                   ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """Internal method for fixed-ratio resampling without inner CV recursion."""
        original_ratio = self.sampling_ratio
        self.sampling_ratio = ratio

        use_inner_cv_original = self.use_inner_cv
        self.use_inner_cv = False

        try:
            X_res, y_res = self.fit_resample(X, y)
        finally:
            self.sampling_ratio = original_ratio
            self.use_inner_cv = use_inner_cv_original

        return X_res, y_res

    def _select_ratio_inner_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        """Select sampling ratio via inner cross-validation (NON-RECURSIVE)."""
        if not self.use_inner_cv or len(self.ratio_grid) <= 1:
            return self.sampling_ratio

        n_min = int(np.sum(y == 1))
        if n_min < 10:
            return self.sampling_ratio

        best_ratio = self.sampling_ratio
        best_score = -np.inf

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)

        for ratio in self.ratio_grid:
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                try:
                    X_res, y_res = self._fit_resample_fixed_ratio(X_train, y_train, ratio)

                    from sklearn.neighbors import KNeighborsClassifier
                    clf = KNeighborsClassifier(n_neighbors=3)
                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X_val)

                    f1 = f1_score(y_val, y_pred, zero_division=0)
                    recall = recall_score(y_val, y_pred, zero_division=0)

                    if recall < 0.3:
                        f1 *= 0.5

                    scores.append(f1)
                except Exception:
                    scores.append(0)

            avg_score = np.mean(scores) if scores else 0

            if avg_score > best_score:
                best_score = avg_score
                best_ratio = ratio

        return best_ratio

    # ============================================================
    # Main fit_resample dengan semua improvements
    # ============================================================
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        N-LDF 2.0: Complete resampling with all mathematical improvements.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Binary labels required (0=majority, 1=minority)")

        X_min = X[y == 1]
        X_maj = X[y == 0]

        if len(X_min) < 2:
            self.last_diagnostics_ = {"error": "Too few minority samples"}
            return X.copy(), y.copy()

        if len(X_maj) == 0:
            self.last_diagnostics_ = {"error": "No majority samples"}
            return X.copy(), y.copy()

        rng = np.random.default_rng(self.random_state)

        if self.use_inner_cv:
            self.sampling_ratio = self._select_ratio_inner_cv(X, y)

        # FIXED: Single prior correction
        eta_min = self._eta_corrected(X_min, X, y)

        IR = len(X_maj) / len(X_min)
        delta = self._compute_delta(eta_min, IR)

        weights = self._compute_weights(eta_min, delta)

        # NO 20x cap
        target_minority = int(len(X_maj) * self.sampling_ratio)
        n_generate = max(0, target_minority - len(X_min))

        if n_generate <= 0:
            self.last_diagnostics_ = {
                "n_generated": 0,
                "n_accepted": 0,
                "sampling_ratio": self.sampling_ratio,
                "delta": delta,
                "IR": IR,
                "target_minority": target_minority,
                "n_attempted": 0,
                "n_rejected_distance": 0,
                "n_rejected_density": 0,
                "n_rejected_maj": 0,
                "n_covariance_failures": 0,
                "eta_min": float(np.min(eta_min)),
                "eta_max": float(np.max(eta_min)),
                "eta_mean": float(np.mean(eta_min)),
                "q_boundary": float(np.clip(self.q0 + self.beta * np.log10(max(IR, 2)),
                                            self.q_min, self.q_max)),
            }
            return X.copy(), y.copy()

        local_covs = None
        scales = None
        if self.use_covariance:
            try:
                local_covs, scales = self._compute_local_covariance(X_min, k=min(5, len(X_min)-1))
            except Exception:
                local_covs = None
                scales = np.ones(len(X_min)) * self.jitter_base

        generated = []
        n_accepted = 0
        n_attempted = 0
        n_rejected_distance = 0
        n_rejected_density = 0
        n_rejected_maj = 0
        n_covariance_failures = 0

        max_attempts = n_generate * 5

        for attempt in range(max_attempts):
            if n_accepted >= n_generate:
                break

            n_attempted += 1

            seed_idx = int(rng.choice(len(X_min), p=weights))
            seed = X_min[seed_idx]
            eta_seed = eta_min[seed_idx]

            neighbor_idx = self._select_neighbor(seed_idx, eta_min, X_min, rng)

            if neighbor_idx == seed_idx:
                continue

            target = X_min[neighbor_idx]
            eta_target = eta_min[neighbor_idx]

            # FIXED: Alpha dengan RNG
            alpha = self._compute_alpha(eta_seed, eta_target, rng)

            local_cov = None
            scale = self.jitter_base
            if local_covs is not None:
                try:
                    local_cov = local_covs[seed_idx]
                    scale = scales[seed_idx] if scales is not None else self.jitter_base
                except Exception:
                    n_covariance_failures += 1
                    local_cov = None
                    scale = self.jitter_base

            sample, cov_failed = self._generate_sample(
                seed, target, alpha, seed_idx,
                local_cov, scale, eta_seed, rng
            )
            if cov_failed:
                n_covariance_failures += 1

            # FIXED: Project BEFORE acceptance
            if self.projector is not None:
                try:
                    sample = self.projector(sample.reshape(1, -1))[0]
                except Exception:
                    pass

            accepted, reason = self._accept_sample(
                sample, X, y, X_min, eta_min, delta, X, y, generated
            )

            if accepted:
                generated.append(sample)
                n_accepted += 1
            else:
                if reason in ["too_close", "too_close_generated"]:
                    n_rejected_distance += 1
                elif reason == "low_density":
                    n_rejected_density += 1
                elif reason == "maj_dominated":
                    n_rejected_maj += 1

        if len(generated) > 0:
            X_resampled = np.vstack([X, np.array(generated)])
            y_resampled = np.concatenate([y, np.ones(len(generated), dtype=int)])
        else:
            X_resampled = X.copy()
            y_resampled = y.copy()

        # Diagnostics with FULL AUDIT
        self.last_diagnostics_ = {
            "IR": IR,
            "delta": delta,
            "n_generated": len(generated),
            "n_accepted": n_accepted,
            "n_attempted": n_attempted,
            "n_rejected_distance": n_rejected_distance,
            "n_rejected_density": n_rejected_density,
            "n_rejected_maj": n_rejected_maj,
            "n_covariance_failures": n_covariance_failures,
            "sampling_ratio": self.sampling_ratio,
            "target_minority": target_minority,
            "eta_min": float(np.min(eta_min)),
            "eta_max": float(np.max(eta_min)),
            "eta_mean": float(np.mean(eta_min)),
            "weights_max": float(np.max(weights)),
            "lambda_prior": self.lambda_prior,
            "q_boundary": float(np.clip(self.q0 + self.beta * np.log10(max(IR, 2)),
                                        self.q_min, self.q_max)),
            "acceptance_rate": n_accepted / max(n_attempted, 1),
            "achieved_ratio": (len(X_min) + n_accepted) / len(X_maj),
            "target_shortfall": n_generate - n_accepted,
            "rejection_rate": 1 - n_accepted / max(n_attempted, 1),
        }

        return X_resampled, y_resampled


def build_sampler(
    method: str,
    y_train: np.ndarray,
    projector: Callable[[np.ndarray], np.ndarray],
    seed: int,
):
    """Return the sampler used in the benchmark for ``method``.

    The N-LDF branch fixes every parameter to the value used in the run; the
    class defaults above are *not* the benchmark values (tau_boundary and
    acceptance_threshold in particular). The comparison samplers are the
    imbalanced-learn implementations at their library defaults with
    k_neighbors = effective_k(y_train); they are constructed in the notebook.
    """
    if method == "Baseline":
        return None
    if method != "N-LDF":
        raise KeyError(
            f"{method}: comparison samplers are built in the notebook "
            "(imbalanced-learn defaults with k_neighbors=effective_k(y_train))"
        )

    k = effective_k(y_train)
    return NLDFv2(
        k_neighbors=k,
        k_density=7,
        # Boundary parameters (widens with IR)
        q0=0.20,
        beta=0.075,
        q_min=0.10,
        q_max=0.45,
        # Prior correction
        lambda_prior=0.7,
        # Soft weighting
        tau_boundary=0.1,
        tau_noise=0.05,
        delta_noise=0.02,
        lambda_boundary=1.5,
        lambda_safe=1.0,
        # Generation
        jitter_base=0.01,
        use_covariance=True,
        # Acceptance
        acceptance_threshold=0.0,
        min_distance=0.001,
        # Sampling ratio (not always 1:1)
        sampling_ratio=0.75,
        # Other
        projector=projector,
        random_state=seed,
        use_inner_cv=False,
    )
