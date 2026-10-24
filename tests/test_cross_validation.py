"""Validation croisée avec des implémentations indépendantes (pymcdm, pyDecision) sur des problèmes aléatoires.

Ces bibliothèques sont des dépendances de développement : les tests sont ignorés si elles sont absentes.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mcdakit.methods import topsis, weighted_sum
from mcdakit.weights import ahp

pm = pytest.importorskip("pymcdm.methods")
pn = pytest.importorskip("pymcdm.normalizations")
N = 150


def _problems(seed):
    rng = np.random.default_rng(seed)
    for it in range(N):
        n, m = int(rng.integers(3, 13)), int(rng.integers(2, 9))
        x = rng.uniform(1, 100, (n, m))
        if it % 3 == 0:
            x = np.round(x / 10) + 1  # nombreux ex aequo
        w = rng.dirichlet(np.ones(m))
        types = rng.choice([1, -1], m)
        cols = [f"C{j}" for j in range(m)]
        yield (
            rng,
            x,
            w,
            types,
            pd.DataFrame(x, columns=cols),
            pd.Series(w, index=cols),
            pd.Series(np.where(types > 0, "max", "min"), index=cols),
        )


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-12)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_topsis_wsm_match_pymcdm():
    for _, x, w, t, m, ws, d in _problems(1):
        _close(topsis(m, ws, d, normalization="vector").scores, pm.TOPSIS(pn.vector_normalization)(x, w, t))
        _close(topsis(m, ws, d, normalization="minmax").scores, pm.TOPSIS(pn.minmax_normalization)(x, w, t))
        _close(
            weighted_sum(m, ws, d, normalization="max").scores,
            pm.WSM(pn.linear_normalization)(x, w, t, validation=False),
        )


def _random_reciprocal(rng, k):
    a = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = rng.choice([1 / 9, 1 / 7, 1 / 5, 1 / 3, 1 / 2, 1, 2, 3, 5, 7, 9])
            a[i, j], a[j, i] = v, 1 / v
    return a


def test_ahp_matches_pymcdm_and_pydecision():
    from pymcdm.weights.subjective import AHP

    pyd = pytest.importorskip("pyDecision.algorithm")
    rng = np.random.default_rng(3)
    for _ in range(N):
        a = _random_reciprocal(rng, int(rng.integers(3, 15)))
        res = ahp(a)
        _close(res.weights, AHP(matrix=a)())
        w, cr = pyd.ahp_method(a, "me")
        _close(res.weights, w)
        assert res.cr == pytest.approx(cr, abs=1e-12)
        w, cr = pyd.ahp_method(a, "g")
        geo = ahp(a, method="geometric")
        _close(geo.weights, w)
        assert geo.cr == pytest.approx(cr, abs=1e-12)
