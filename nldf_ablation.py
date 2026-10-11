"""Controlled variants of the N-LDF sampler for the ablation and matched-amount
experiments requested at review.

Nothing in this module changes ``nldf.py`` (its SHA-256 is quoted in the
manuscript). Every variant is built from the archived benchmark configuration,
``nldf.build_sampler``, with exactly one thing changed:

Matched amount
    N-LDF-r1.0        sampling_ratio 1.0 (parity) instead of 0.75

Replacing the support statistic (everything else unchanged)
    ABL-lp0           lambda_prior = 0  (raw affinity sums, no normalization)
    ABL-lp1           lambda_prior = 1  (full class-size normalization)
    ABL-localprop     eta replaced by the count-based local minority
                      proportion contrast (n_min - n_maj) / k over the same
                      k_density neighbours: no kernel, no bandwidth, no
                      normalization
    ABL-uniform       uniform seed weights, distance-only partner selection,
                      interpolation centred at 0.5 (the statistic is not used
                      in generation; the acceptance tests are unchanged)

Removing a component
    ABL-nojitter      covariance perturbation removed (jitter_base = 0,
                      use_covariance = False)
    ABL-nodist        duplicate / minimum-distance test removed
    ABL-nosupport     support test (eta >= 0) removed
    ABL-noratio       class-distance-ratio test (R <= 1.5) removed

Parameter sensitivity (one value moved from the archived setting)
    SENS-lp0.5, SENS-lp0.9        lambda_prior 0.7 -> 0.5, 0.9
    SENS-kd5, SENS-kd11           k_density (bandwidth neighbourhood) 7 -> 5, 11
    SENS-acc-0.1, SENS-acc+0.1    acceptance_threshold 0.0 -> -0.1, +0.1
    SENS-rb2.0                    class-distance-ratio bound 1.5 -> 2.0

``NLDFAblation`` with its default flags reproduces ``NLDFv2`` exactly
(``selftest()`` asserts this bit for bit on synthetic data), so a variant
differs from the archived sampler only in the one flag it sets.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from nldf import NLDFv2, build_sampler as _build_archived, effective_k

MATCHED = ["N-LDF-r1.0"]
ABLATION_STATISTIC = ["ABL-lp0", "ABL-lp1", "ABL-localprop", "ABL-uniform"]
ABLATION_COMPONENTS = ["ABL-nojitter", "ABL-nodist", "ABL-nosupport", "ABL-noratio"]
SENSITIVITY = ["SENS-lp0.5", "SENS-lp0.9", "SENS-kd5", "SENS-kd11", "SENS-acc-0.1", "SENS-acc+0.1", "SENS-rb2.0"]
SENSITIVITY_PARAMS = {   # name -> (parameter, archived value, value used)
    "SENS-lp0.5": ("lambda_prior", 0.7, 0.5), "SENS-lp0.9": ("lambda_prior", 0.7, 0.9),
    "SENS-kd5": ("k_density", 7, 5), "SENS-kd11": ("k_density", 7, 11),
    "SENS-acc-0.1": ("acceptance_threshold", 0.0, -0.1), "SENS-acc+0.1": ("acceptance_threshold", 0.0, 0.1),
    "SENS-rb2.0": ("ratio_bound", 1.5, 2.0),
}
VARIANTS = MATCHED + ABLATION_STATISTIC + ABLATION_COMPONENTS + SENSITIVITY


def benchmark_kwargs(y_train: np.ndarray, projector: Callable, seed: int) -> dict:
    """The archived configuration, read from nldf.build_sampler (never retyped)."""
    s = _build_archived("N-LDF", y_train, projector, seed)
    skip = {"last_diagnostics_"}
    return {k: v for k, v in vars(s).items() if k not in skip}


class NLDFAblation(NLDFv2):
    """NLDFv2 with switches for the ablation. Default switches == NLDFv2."""

    def __init__(self, *args, statistic: str = "nldf", uniform_weights: bool = False,
                 use_distance_test: bool = True, use_support_test: bool = True,
                 use_ratio_test: bool = True, ratio_bound: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        if statistic not in ("nldf", "local_proportion"):
            raise ValueError(statistic)
        self.statistic = statistic
        self.uniform_weights = uniform_weights
        self.use_distance_test = use_distance_test
        self.use_support_test = use_support_test
        self.use_ratio_test = use_ratio_test
        self.ratio_bound = ratio_bound

    # ---------------------------------------------------------------- statistic
    def _eta_corrected(self, X_min, X_all, y_all):
        if self.statistic == "nldf":
            return super()._eta_corrected(X_min, X_all, y_all)
        # count-based local minority proportion contrast over the same
        # k_density neighbours (seed excluded): (n_min - n_maj) / k in [-1, 1]
        n_min = int(np.sum(y_all == 1))
        n_maj = int(np.sum(y_all == 0))
        if n_min == 0 or n_maj == 0:
            return np.zeros(len(X_min))
        n_neighbors = min(self.k_density + 1, len(X_all))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_all)
        _, indices = nn.kneighbors(X_min)
        etas = np.empty(len(X_min), dtype=float)
        for row, global_idx in enumerate(np.flatnonzero(y_all == 1)):
            mask = indices[row] != global_idx
            neighbor_idx = indices[row][mask][:self.k_density]
            k = max(len(neighbor_idx), 1)
            etas[row] = (np.sum(y_all[neighbor_idx] == 1) - np.sum(y_all[neighbor_idx] == 0)) / k
        return etas

    def _compute_eta_for_sample(self, sample, X_all, y_all):
        if self.statistic == "nldf":
            return super()._compute_eta_for_sample(sample, X_all, y_all)
        n_min = int(np.sum(y_all == 1))
        n_maj = int(np.sum(y_all == 0))
        if n_min == 0 or n_maj == 0:
            return 0.0
        n_neighbors = min(self.k_density + 1, len(X_all))
        nn_all = NearestNeighbors(n_neighbors=n_neighbors).fit(X_all)
        _, indices = nn_all.kneighbors(sample.reshape(1, -1))
        labels = y_all[indices[0]]
        return float((np.sum(labels == 1) - np.sum(labels == 0)) / max(len(labels), 1))

    # ---------------------------------------------------------------- weighting
    def _compute_weights(self, eta, delta):
        if not self.uniform_weights:
            return super()._compute_weights(eta, delta)
        return np.ones(len(eta)) / len(eta)

    def _select_neighbor(self, seed_idx, eta_min, X_min, rng):
        if not self.uniform_weights:
            return super()._select_neighbor(seed_idx, eta_min, X_min, rng)
        # distance-only partner selection in the same local pool
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
        h = max(np.median(neighbor_distances), self.epsilon)
        probs = np.exp(-neighbor_distances**2 / h**2)
        if np.sum(probs) == 0:
            return int(rng.choice(neighbor_indices))
        return int(rng.choice(neighbor_indices, p=probs / np.sum(probs)))

    def _compute_alpha(self, eta_seed, eta_target, rng):
        if not self.uniform_weights:
            return super()._compute_alpha(eta_seed, eta_target, rng)
        # same Beta family, centre fixed at 0.5 (support difference ignored)
        return float(np.clip(rng.beta(2.5, 2.5), 0.1, 0.9))

    # ---------------------------------------------------------------- acceptance
    def _accept_sample(self, sample, X_existing, y_existing, X_min, eta_min, delta,
                       X_all, y_all, generated):
        """nldf.NLDFv2._accept_sample with each test switchable and the
        ratio bound a parameter. With all tests on and ratio_bound = 1.5 the
        decisions are identical to the archived class (see selftest)."""
        if self.use_distance_test:
            if len(X_existing) > 0:
                min_dist = np.min(cdist(sample.reshape(1, -1), X_existing))
                if min_dist < self.min_distance:
                    return False, "too_close"
            if generated:
                dist_generated = np.min(cdist(sample.reshape(1, -1), np.asarray(generated)))
                if dist_generated < self.min_distance:
                    return False, "too_close_generated"
        if self.use_support_test and len(X_min) > 1 and len(X_all) > 1:
            eta_new = self._compute_eta_for_sample(sample, X_all, y_all)
            if eta_new < self.acceptance_threshold:
                return False, "low_density"
        if self.use_ratio_test and len(X_min) > 1 and len(X_existing[y_existing == 0]) > 1:
            nn_min = NearestNeighbors(n_neighbors=min(3, len(X_min))).fit(X_min)
            dist_to_min, _ = nn_min.kneighbors(sample.reshape(1, -1))
            X_maj = X_existing[y_existing == 0]
            nn_maj = NearestNeighbors(n_neighbors=min(3, len(X_maj))).fit(X_maj)
            dist_to_maj, _ = nn_maj.kneighbors(sample.reshape(1, -1))
            ratio = np.mean(dist_to_min) / (np.mean(dist_to_maj) + self.epsilon)
            if ratio > self.ratio_bound:
                return False, "maj_dominated"
        return True, "accepted"


def build_variant(name: str, y_train: np.ndarray, projector: Callable, seed: int):
    """The sampler for one experimental strategy. Every variant starts from
    the archived configuration and changes exactly one thing."""
    kw = benchmark_kwargs(y_train, projector, seed)
    if name == "N-LDF":
        return _build_archived("N-LDF", y_train, projector, seed)
    if name == "N-LDF-r1.0":
        kw["sampling_ratio"] = 1.0
        return NLDFv2(**kw)
    if name == "ABL-lp0":
        kw["lambda_prior"] = 0.0
        return NLDFv2(**kw)
    if name == "ABL-lp1":
        kw["lambda_prior"] = 1.0
        return NLDFv2(**kw)
    if name == "ABL-nojitter":
        kw["jitter_base"] = 0.0
        kw["use_covariance"] = False
        return NLDFv2(**kw)
    if name == "ABL-localprop":
        return NLDFAblation(statistic="local_proportion", **kw)
    if name == "ABL-uniform":
        return NLDFAblation(uniform_weights=True, **kw)
    if name == "ABL-nodist":
        return NLDFAblation(use_distance_test=False, **kw)
    if name == "ABL-nosupport":
        return NLDFAblation(use_support_test=False, **kw)
    if name == "ABL-noratio":
        return NLDFAblation(use_ratio_test=False, **kw)
    if name in SENSITIVITY_PARAMS:
        param, archived, value = SENSITIVITY_PARAMS[name]
        if param == "ratio_bound":
            return NLDFAblation(ratio_bound=value, **kw)
        assert kw[param] == archived, f"{param}: archived value is {kw[param]}, expected {archived}"
        kw[param] = value
        return NLDFv2(**kw)
    raise KeyError(f"unknown variant: {name}")


# ---------------------------------------------------------------- self-test
def _synthetic(seed: int, n: int = 240, n_min: int = 30, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    y = np.zeros(n, dtype=int)
    y[:n_min] = 1
    X[:n_min] += 1.2
    return X, y


def selftest(verbose: bool = True) -> None:
    """(1) NLDFAblation with default flags == NLDFv2 bit for bit;
    (2) every variant runs and differs from the archived sampler where it must."""
    proj = lambda X: X  # noqa: E731
    for seed in (1, 2, 3):
        X, y = _synthetic(seed)
        kw = benchmark_kwargs(y, proj, 42)
        ref = NLDFv2(**kw).fit_resample(X, y)
        abl = NLDFAblation(**kw).fit_resample(X, y)
        assert ref[0].shape == abl[0].shape and np.array_equal(ref[0], abl[0]) and np.array_equal(ref[1], abl[1]), seed
        base = build_variant("N-LDF", y, proj, 42).fit_resample(X, y)
        assert np.array_equal(base[0], ref[0]), "build_variant('N-LDF') must equal the archived sampler"
    X, y = _synthetic(7)
    ref_X, _ = build_variant("N-LDF", y, proj, 42).fit_resample(X, y)
    n_maj = int((y == 0).sum())
    for name in VARIANTS:
        s = build_variant(name, y, proj, 42)
        Xr, yr = s.fit_resample(X, y)
        assert yr.sum() >= y.sum() and Xr.shape[1] == X.shape[1], name
        d = s.last_diagnostics_
        assert {"n_attempted", "n_rejected_density", "n_rejected_maj", "n_rejected_distance"} <= set(d), name
        if name == "N-LDF-r1.0":
            assert d["target_minority"] == int(n_maj * 1.0), d
        if name == "ABL-nosupport":
            assert d["n_rejected_density"] == 0
        if name == "ABL-noratio":
            assert d["n_rejected_maj"] == 0
        if name == "ABL-nodist":
            assert d["n_rejected_distance"] == 0
        if name in SENSITIVITY_PARAMS:
            param, archived, value = SENSITIVITY_PARAMS[name]
            got = getattr(s, param)
            assert got == value and archived != value, (name, got)
            # every other archived parameter is untouched
            kw0 = benchmark_kwargs(y, proj, 42)
            for k0, v0 in kw0.items():
                if k0 != param and not k0.startswith("_") and k0 not in ("last_diagnostics_",):
                    assert getattr(s, k0) == v0 or (isinstance(v0, float) and np.isclose(getattr(s, k0), v0)), (name, k0)
        if name == "ABL-nojitter":
            # without perturbation every synthetic point lies on a segment
            # between two training minority points: check co-linearity
            syn = Xr[len(X):]
            Xm = X[y == 1]
            ok = 0
            for p in syn[:20]:
                dmat = cdist(p.reshape(1, -1), Xm)[0]
                i = int(np.argmin(dmat))
                # a point on segment (a, b) satisfies |p-a| + |p-b| = |a-b| for some b
                a = Xm[i]
                rest = np.delete(Xm, i, axis=0)
                gaps = np.linalg.norm(p - a) + np.linalg.norm(rest - p, axis=1) - np.linalg.norm(rest - a, axis=1)
                ok += bool(np.min(np.abs(gaps)) < 1e-9)
            assert ok >= 18, f"nojitter: {ok}/20 points on a segment"
        if verbose:
            print(f"  {name:<14} accepted={d['n_accepted']:>4} attempted={d['n_attempted']:>4} "
                  f"rej(dist/sup/ratio)={d['n_rejected_distance']}/{d['n_rejected_density']}/{d['n_rejected_maj']} "
                  f"target={d['target_minority']}")
    if verbose:
        print("nldf_ablation selftest: OK")


if __name__ == "__main__":
    selftest()
