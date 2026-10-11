"""Memoised NearestNeighbors for the N-LDF samplers (speed only).

nldf.py refits sklearn.neighbors.NearestNeighbors on the same reference set
for every candidate it evaluates (four fits per accepted instance). A fitted
index answers kneighbors() identically however many times it is queried, so
reusing the index built for identical data and parameters returns exactly the
same neighbours and distances. ``install()`` replaces the name
NearestNeighbors inside nldf and nldf_ablation by a factory that returns a
cached fitted index when (parameters, shape, dtype, bytes) of the training
array match a previous fit. nldf.py itself is not modified (its SHA-256 is
unchanged). Equality with the archived records is checked by
verify_reproduction.py on re-run N-LDF units.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
from sklearn.neighbors import NearestNeighbors as _SkNN

_CACHE: "OrderedDict[tuple, _SkNN]" = OrderedDict()
_MAX = 32
STATS = {"hits": 0, "misses": 0}


class CachedNearestNeighbors:
    def __init__(self, **params):
        self._params = params

    def fit(self, X, y=None):
        A = np.ascontiguousarray(X)
        key = (tuple(sorted(self._params.items())), A.shape, A.dtype.str,
               hashlib.blake2b(A.tobytes(), digest_size=16).digest())
        nn = _CACHE.get(key)
        if nn is None:
            STATS["misses"] += 1
            nn = _SkNN(**self._params).fit(X)
            _CACHE[key] = nn
            if len(_CACHE) > _MAX:
                _CACHE.popitem(last=False)
        else:
            STATS["hits"] += 1
            _CACHE.move_to_end(key)
        return nn


def install() -> None:
    import nldf
    import nldf_ablation
    nldf.NearestNeighbors = CachedNearestNeighbors
    nldf_ablation.NearestNeighbors = CachedNearestNeighbors
