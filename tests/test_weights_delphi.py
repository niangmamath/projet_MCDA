import numpy as np
import pytest

from mcdakit import ahp, ahp_from_judgements, delphi, rank_order_centroid, resolve_weights

STRATEMA_W = [0.084257, 0.163348, 0.259502, 0.049097, 0.328454, 0.115342]


def test_ahp_stratema(stratema1):
    w, res = resolve_weights(stratema1)
    assert np.allclose(w.values, STRATEMA_W, atol=1e-6)
    assert res.lambda_max == pytest.approx(6.145258, abs=1e-6)
    assert res.cr == pytest.approx(0.02343, abs=1e-5)
    assert res.consistent


def test_ahp_judgements_equals_matrix(stratema1, stratema2):
    w1, _ = resolve_weights(stratema1)
    w2, _ = resolve_weights(stratema2)
    assert np.allclose(w1.values, w2.values)


def test_ahp_consistent_matrix_has_zero_cr():
    w = np.array([0.5, 0.3, 0.2])
    res = ahp(w[:, None] / w[None, :])
    assert np.allclose(res.weights.values, w)
    assert res.cr == pytest.approx(0, abs=1e-9)


def test_ahp_geometric_close_to_eigen(stratema1):
    m = stratema1.weights_spec["matrix"]
    assert np.allclose(ahp(m, method="geometric").weights.values, ahp(m).weights.values, atol=0.01)


def test_ahp_rejects_non_reciprocal():
    with pytest.raises(ValueError):
        ahp([[1, 2], [2, 1]])


def test_ahp_inconsistent_detected():
    res = ahp_from_judgements(["a", "b", "c"], {"a>b": 9, "b>c": 9, "a>c": "1/9"})
    assert not res.consistent
    assert len(res.worst_judgements()) == 3


def test_roc():
    w = rank_order_centroid(["x", "y", "z"])
    assert w.sum() == pytest.approx(1)
    assert w["x"] == pytest.approx(11 / 18)


def test_delphi_stratema_cells():
    res = delphi(["2 · 1 · 3 · 3 · 2", "2 · 2 · 2 · 3 · 2"])
    assert res.rounds[0].q1 == 2 and res.rounds[0].q3 == 3
    assert res.value == 2 and res.consensus and res.stable
    res = delphi([[3, 4, 3, 4, 2]])
    assert (res.final.median, res.final.q1, res.final.q3) == (3, 3, 4)


def test_delphi_no_consensus():
    assert not delphi([[1, 5, 1, 5, 3]]).consensus
