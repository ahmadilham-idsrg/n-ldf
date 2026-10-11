"""Additional comparison strategies for the extension experiments.

Three oversamplers that neighbourhood-guided methods are usually compared
with, plus the class-weighting rule for the weighted baseline. All of them
target minority--majority parity, like the archived comparison methods, and
all expose the imbalanced-learn ``fit_resample`` interface with the
``last_diagnostics_`` dictionary the benchmark harness records
(``SamplerStatus``, ``SamplerFallback``, ``SamplerError``).

* ``SafeLevelSMOTE``: Safe-Level-SMOTE (Bunkhumpornpat, Sinapiromsaran and
  Lursinsap, PAKDD 2009), implemented here from the paper's pseudocode because
  no maintained package installs cleanly on current Python. It is the direct
  relative of N-LDF's support statistic: the safe level of an instance is the
  *count* of minority instances among its k nearest neighbours, and the ratio
  of the safe levels of the seed and its partner decides where on the segment
  the synthetic instance is placed.
* ``AuditedSVMSMOTE``: imbalanced-learn ``SVMSMOTE`` (boundary-guided), with the
  same SMOTE fallback and audit record as the archived ``AuditedADASYN``.
* ``AuditedKMeansSMOTE``: imbalanced-learn ``KMeansSMOTE`` (density-guided),
  likewise audited; it raises on small or diffuse minority classes, and the
  fallback is then recorded, not hidden.
* ``class_weighted(clf, y)``: the balanced class weights for the four
  classifier policies of the benchmark (``class_weight="balanced"`` for the
  scikit-learn estimators, ``scale_pos_weight = n_maj / n_min`` for XGBoost,
  which is the same minority-to-majority weight ratio).

``python nldf_comparators.py`` runs the self-test.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ["SafeLevelSMOTE", "AuditedSVMSMOTE", "AuditedKMeansSMOTE", "class_weighted",
           "COMPARATORS", "selftest"]

COMPARATORS = ["Baseline-cw", "SLS", "SVM-SMOTE", "KMeans-SMOTE"]


def _target_count(y: np.ndarray, sampling_strategy: float) -> int:
    n_min = int((y == 1).sum())
    n_maj = int((y == 0).sum())
    return max(0, int(np.floor(sampling_strategy * n_maj)) - n_min)


class SafeLevelSMOTE:
    """Safe-Level-SMOTE for a binary problem with the minority class labelled 1.

    For a seed p drawn at random from the minority class, a partner n is drawn
    at random from the k nearest neighbours of p among *all* training
    instances (as in the original algorithm, where n may belong to either
    class). With sl(.) the number of minority instances among the k nearest
    neighbours, the ratio r = sl(p) / sl(n) sets the interpolation gap of each
    attribute independently: gap = 0 when sl(n) = 0 and sl(p) > 0 (duplicate
    p); gap ~ U(0, 1) when r = 1; gap ~ U(0, 1/r) when r > 1 (stay near the
    safer p); gap ~ U(1 - r, 1) when r < 1 (move towards the safer n). A seed
    with sl(p) = 0 whose partner also has sl(n) = 0 is treated as noise and
    generates nothing. Generation stops when ``sampling_strategy`` times the
    majority count is reached, or after ``max_attempts`` draws.
    """

    def __init__(self, sampling_strategy: float = 1.0, k_neighbors: int = 5,
                 random_state: int | None = None, max_attempts_factor: int = 20):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.max_attempts_factor = max_attempts_factor
        self.last_diagnostics_: dict[str, object] = {}

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        n_target = _target_count(y, self.sampling_strategy)
        min_idx = np.flatnonzero(y == 1)
        self.last_diagnostics_ = {"SamplerStatus": "success", "SamplerFallback": "none", "SamplerError": "",
                                  "n_attempted": 0, "n_accepted": 0, "n_noise_skipped": 0}
        if n_target == 0 or len(min_idx) == 0:
            return X.copy(), y.copy()
        k = max(1, min(self.k_neighbors, len(X) - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        neigh = nn.kneighbors(X, return_distance=False)[:, 1:]          # drop self
        safe_level = (y[neigh] == 1).sum(axis=1)                        # sl(.) for every instance
        synth = []
        attempts, noise = 0, 0
        max_attempts = self.max_attempts_factor * n_target
        while len(synth) < n_target and attempts < max_attempts:
            attempts += 1
            p = int(rng.choice(min_idx))
            n = int(rng.choice(neigh[p]))
            sl_p, sl_n = int(safe_level[p]), int(safe_level[n])
            if sl_p == 0 and sl_n == 0:
                noise += 1
                continue
            d = X.shape[1]
            if sl_n == 0:                       # ratio infinite, p safe: duplicate p
                gap = np.zeros(d)
            else:
                r = sl_p / sl_n
                if r == 1.0:
                    gap = rng.uniform(0.0, 1.0, d)
                elif r > 1.0:
                    gap = rng.uniform(0.0, 1.0 / r, d)
                else:
                    gap = rng.uniform(1.0 - r, 1.0, d)
            synth.append(X[p] + gap * (X[n] - X[p]))
        self.last_diagnostics_.update({"n_attempted": attempts, "n_accepted": len(synth),
                                       "n_noise_skipped": noise})
        if not synth:
            return X.copy(), y.copy()
        X_new = np.vstack([X, np.asarray(synth)])
        y_new = np.concatenate([y, np.ones(len(synth), dtype=int)])
        return X_new, y_new


class _AuditedImblearn:
    """Common fallback-and-audit wrapper, identical in spirit to AuditedADASYN."""
    name = ""

    def __init__(self, k_neighbors: int, random_state: int, sampling_strategy: float = 1.0):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.last_diagnostics_: dict[str, object] = {}

    def _make(self):
        raise NotImplementedError

    def fit_resample(self, X, y):
        from imblearn.over_sampling import SMOTE
        self.last_diagnostics_ = {"SamplerStatus": "success", "SamplerFallback": "none", "SamplerError": ""}
        try:
            return self._make().fit_resample(X, y)
        except (RuntimeError, ValueError) as exc:
            X_res, y_res = SMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors,
                                 random_state=self.random_state).fit_resample(X, y)
            self.last_diagnostics_ = {"SamplerStatus": "fallback", "SamplerFallback": "SMOTE",
                                      "SamplerError": str(exc)[:200]}
            return X_res, y_res


class AuditedSVMSMOTE(_AuditedImblearn):
    name = "SVM-SMOTE"

    def _make(self):
        from imblearn.over_sampling import SVMSMOTE
        return SVMSMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors,
                        random_state=self.random_state)


class AuditedKMeansSMOTE(_AuditedImblearn):
    name = "KMeans-SMOTE"

    def _make(self):
        from imblearn.over_sampling import KMeansSMOTE
        return KMeansSMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors,
                           random_state=self.random_state)


def class_weighted(clf, y_train: np.ndarray):
    """Return ``clf`` reconfigured with balanced class weights (in place).

    scikit-learn estimators get ``class_weight="balanced"`` (weight of class c
    = n / (2 n_c), so the minority-to-majority weight ratio is n_maj / n_min);
    XGBoost gets ``scale_pos_weight = n_maj / n_min``, the same ratio.
    """
    y = np.asarray(y_train).astype(int)
    n_min = max(1, int((y == 1).sum()))
    n_maj = int((y == 0).sum())
    params = clf.get_params()
    if "scale_pos_weight" in params:
        clf.set_params(scale_pos_weight=n_maj / n_min)
    elif "class_weight" in params:
        clf.set_params(class_weight="balanced")
    else:
        raise TypeError(f"{type(clf).__name__} supports neither class_weight nor scale_pos_weight")
    return clf


class AuditedADASYNRatio:
    """ADASYN at a given minority/majority target with the archived fallback
    rule of AuditedADASYN (Cell 2 of the notebooks): SMOTE at the same target
    when ADASYN reports that the data are not suited or no samples could be
    generated; any other error propagates."""

    def __init__(self, n_neighbors: int, random_state: int, sampling_strategy: float = 1.0):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.last_diagnostics_: dict[str, object] = {}

    def fit_resample(self, X, y):
        from imblearn.over_sampling import ADASYN, SMOTE
        self.last_diagnostics_ = {"SamplerStatus": "success", "SamplerFallback": "none", "SamplerError": ""}
        try:
            return ADASYN(sampling_strategy=self.sampling_strategy, n_neighbors=self.n_neighbors,
                          random_state=self.random_state).fit_resample(X, y)
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            if "not suited" in message or "No samples" in message:
                X_res, y_res = SMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=self.n_neighbors,
                                     random_state=self.random_state).fit_resample(X, y)
                self.last_diagnostics_ = {"SamplerStatus": "fallback", "SamplerFallback": "SMOTE",
                                          "SamplerError": message[:200]}
                return X_res, y_res
            raise


# every oversampler at the N-LDF target r = 0.75 (package "matched_all"); the
# parity versions are the archived strategies and the comparators above
COMPARATORS_R075 = ["B-SMOTE-r0.75", "ADASYN-r0.75", "SMOTE-Tomek-r0.75",
                    "SLS-r0.75", "SVM-SMOTE-r0.75", "KMeans-SMOTE-r0.75"]


def build_comparator(method: str, k: int, seed: int):
    """Sampler for a comparator strategy; None for the weighted baseline."""
    if method == "Baseline-cw":
        return None
    if method in COMPARATORS_R075:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE
        r = 0.75
        if method == "B-SMOTE-r0.75":
            return BorderlineSMOTE(sampling_strategy=r, k_neighbors=k, random_state=seed)
        if method == "ADASYN-r0.75":
            return AuditedADASYNRatio(n_neighbors=k, random_state=seed, sampling_strategy=r)
        if method == "SMOTE-Tomek-r0.75":
            return SMOTETomek(smote=SMOTE(sampling_strategy=r, k_neighbors=k, random_state=seed),
                              random_state=seed)
        if method == "SLS-r0.75":
            return SafeLevelSMOTE(sampling_strategy=r, k_neighbors=k, random_state=seed)
        if method == "SVM-SMOTE-r0.75":
            return AuditedSVMSMOTE(k_neighbors=k, random_state=seed, sampling_strategy=r)
        if method == "KMeans-SMOTE-r0.75":
            return AuditedKMeansSMOTE(k_neighbors=k, random_state=seed, sampling_strategy=r)
    if method == "SLS":
        return SafeLevelSMOTE(sampling_strategy=1.0, k_neighbors=k, random_state=seed)
    if method == "SVM-SMOTE":
        return AuditedSVMSMOTE(k_neighbors=k, random_state=seed)
    if method == "KMeans-SMOTE":
        return AuditedKMeansSMOTE(k_neighbors=k, random_state=seed)
    raise KeyError(method)


# --------------------------------------------------------------------------
def selftest(verbose: bool = True) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(260, 4))
    y = np.zeros(260, dtype=int)
    y[:40] = 1
    X[:40] += 1.8
    n_min, n_maj = 40, 220

    # Safe-Level-SMOTE: reaches parity, synthetic points lie on seed-partner segments
    s = SafeLevelSMOTE(k_neighbors=5, random_state=1)
    Xs, ys = s.fit_resample(X, y)
    assert int(ys.sum()) == n_maj, ys.sum()
    assert len(Xs) == 2 * n_maj
    assert s.last_diagnostics_["n_accepted"] == n_maj - n_min
    new = Xs[len(X):]
    nn = NearestNeighbors(n_neighbors=6).fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    # every synthetic point is a per-attribute convex combination of some minority p and a neighbour n of p
    ok = 0
    for z in new[:60]:
        hit = False
        for p in np.flatnonzero(y == 1):
            for n in neigh[p]:
                lo, hi = np.minimum(X[p], X[n]), np.maximum(X[p], X[n])
                if np.all(z >= lo - 1e-9) and np.all(z <= hi + 1e-9):
                    hit = True
                    break
            if hit:
                break
        ok += hit
    assert ok == 60, ok
    # determinism
    Xs2, _ = SafeLevelSMOTE(k_neighbors=5, random_state=1).fit_resample(X, y)
    assert np.array_equal(Xs, Xs2)
    # a pure-noise minority (two isolated seeds whose majority partners have no
    # minority neighbour either, so sl(p) = sl(n) = 0) generates nothing
    Xn = np.vstack([rng.normal(size=(58, 3)), [[100.0] * 3], [[-100.0] * 3]])
    yn = np.zeros(60, dtype=int)
    yn[-2:] = 1
    sn = SafeLevelSMOTE(k_neighbors=5, random_state=0)
    Xo, yo = sn.fit_resample(Xn, yn)
    assert int(yo.sum()) == 2 and sn.last_diagnostics_["n_noise_skipped"] > 0, sn.last_diagnostics_
    # ...whereas a noise seed whose partner is "safer" (sl(n) > 0) is placed at
    # the partner, gap = 1, as the published algorithm prescribes
    Xq = np.vstack([rng.normal(size=(30, 2)), [[0.0, 0.0]], [[0.05, 0.0]]])
    yq = np.zeros(32, dtype=int)
    yq[-1] = 1                                   # one minority point next to a majority point
    sq = SafeLevelSMOTE(k_neighbors=1, random_state=0)
    Xo, yo = sq.fit_resample(Xq, yq)
    assert int(yo.sum()) == 31 and np.allclose(Xo[32:], Xq[30]), "gap-1 case"

    # audited wrappers: success on the easy problem, recorded fallback on a hard one
    for cls in (AuditedSVMSMOTE, AuditedKMeansSMOTE):
        w = cls(k_neighbors=5, random_state=1)
        Xw, yw = w.fit_resample(X, y)
        assert int(yw.sum()) == n_maj, (cls.__name__, yw.sum())
        assert w.last_diagnostics_["SamplerStatus"] in ("success", "fallback")
    hard_X = rng.normal(size=(40, 3))
    hard_y = np.zeros(40, dtype=int)
    hard_y[:3] = 1
    w = AuditedKMeansSMOTE(k_neighbors=2, random_state=1)
    Xw, yw = w.fit_resample(hard_X, hard_y)
    assert int(yw.sum()) == 37 and w.last_diagnostics_["SamplerStatus"] in ("success", "fallback")

    # class weighting
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    rf = class_weighted(RandomForestClassifier(n_estimators=5, random_state=0), y)
    assert rf.get_params()["class_weight"] == "balanced"
    svc = class_weighted(SVC(kernel="linear"), y)
    assert svc.get_params()["class_weight"] == "balanced"
    try:
        from xgboost import XGBClassifier
        xgb = class_weighted(XGBClassifier(n_estimators=5), y)
        assert abs(xgb.get_params()["scale_pos_weight"] - n_maj / n_min) < 1e-12
    except ImportError:
        pass
    for m in COMPARATORS:
        build_comparator(m, 5, 42)
    if verbose:
        print("nldf_comparators selftest: OK")


if __name__ == "__main__":
    selftest()
