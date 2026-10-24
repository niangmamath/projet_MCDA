"""Propriétés mathématiques que toute implémentation correcte doit vérifier (tests aléatoires reproductibles)."""
import numpy as np
import pandas as pd
import pytest

from mcdakit import DecisionProblem, delphi, evaluate, monte_carlo, weight_thresholds
from mcdakit.methods import METHODS
from mcdakit.sensitivity import WeightScorer, set_weight
from mcdakit.weights import ahp

CONFIGS = [("wsm", {"normalization": "minmax"}), ("wsm", {"normalization": "max"}), ("wpm", {"normalization": "max"}),
           ("topsis", {"normalization": "vector"}), ("topsis", {"normalization": "minmax"})]


def _random(seed, n=None, m=None):
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(3, 10))
    m = m or int(rng.integers(2, 7))
    cols = [f"C{j}" for j in range(m)]
    x = pd.DataFrame(rng.uniform(1, 50, (n, m)), columns=cols, index=[f"A{i}" for i in range(n)])
    d = pd.Series(rng.choice(["max", "min"], m), index=cols)
    w = pd.Series(rng.dirichlet(np.ones(m)), index=cols)
    return rng, x, w, d


def _call(key, params, x, w, d):
    return METHODS[key](x, w, d, **params).scores


@pytest.mark.parametrize("key,params", CONFIGS)
def test_dominance(key, params):
    """Une alternative qui domine une autre (≥ partout, > quelque part) n'est jamais classée derrière elle."""
    for seed in range(60):
        rng, x, w, d = _random(seed)
        better = x.iloc[0].copy()
        for c in x.columns:
            better[c] = better[c] + rng.uniform(0, 5) if d[c] == "max" else better[c] * rng.uniform(0.5, 1)
        x.loc["DOM"] = better
        s = _call(key, params, x, w, d)
        assert s["DOM"] >= s["A0"] - 1e-12


@pytest.mark.parametrize("key,params", CONFIGS)
def test_permutation_of_alternatives(key, params):
    """L'ordre de saisie des alternatives ne change aucun score."""
    for seed in range(30):
        rng, x, w, d = _random(seed)
        s1 = _call(key, params, x, w, d)
        perm = x.sample(frac=1, random_state=seed)
        s2 = _call(key, params, perm, w, d)
        np.testing.assert_allclose(s1[perm.index].to_numpy(), s2.to_numpy(), atol=1e-12)


@pytest.mark.parametrize("key,params", CONFIGS)
def test_change_of_unit(key, params):
    """Changer d'unité (×k sur un critère) ne change pas le classement."""
    for seed in range(30):
        rng, x, w, d = _random(seed)
        s1 = _call(key, params, x, w, d)
        y = x.copy()
        y[y.columns[0]] *= 1000.0
        s2 = _call(key, params, y, w, d)
        np.testing.assert_allclose(s1.to_numpy(), s2.to_numpy(), atol=1e-9)


@pytest.mark.parametrize("key,params", CONFIGS)
def test_weights_scale_free(key, params):
    """Multiplier tous les poids par une constante ne change rien (ils sont renormalisés)."""
    for seed in range(20):
        _, x, w, d = _random(seed)
        np.testing.assert_allclose(_call(key, params, x, w, d).to_numpy(), _call(key, params, x, w * 7.3, d).to_numpy(), atol=1e-12)


@pytest.mark.parametrize("key,params", [c for c in CONFIGS if c[0] != "wpm"])
def test_single_criterion(key, params):
    """Tout le poids sur un critère : le classement suit ce critère."""
    for seed in range(30):
        rng, x, w, d = _random(seed)
        c = x.columns[0]
        w1 = pd.Series(1e-12, index=x.columns)
        w1[c] = 1.0
        s = _call(key, params, x, w1, d)
        expected = x[c] if d[c] == "max" else -x[c]
        assert list(s.sort_values(ascending=False).index) == list(expected.sort_values(ascending=False).index)


def test_bounds():
    for seed in range(50):
        _, x, w, d = _random(seed)
        ci = METHODS["topsis"](x, w, d).scores
        assert ((ci >= 0) & (ci <= 1)).all()
        r = METHODS["wsm"](x, w, d).details["Matrice normalisée"]
        assert (r.min().min() >= -1e-12) and (r.max().max() <= 1 + 1e-12)


def test_ahp_consistent_matrix_and_transpose():
    rng = np.random.default_rng(0)
    for _ in range(30):
        w = rng.dirichlet(np.ones(int(rng.integers(3, 10))))
        a = w[:, None] / w[None, :]
        res = ahp(a)
        np.testing.assert_allclose(res.weights.to_numpy(), w, atol=1e-10)
        assert res.cr == pytest.approx(0, abs=1e-9)
        inv = ahp(a.T).weights.to_numpy()          # jugements inversés -> poids ∝ 1/w
        np.testing.assert_allclose(inv, (1 / w) / (1 / w).sum(), atol=1e-10)


def test_delphi_statistics_match_numpy():
    rng = np.random.default_rng(1)
    for _ in range(50):
        votes = rng.integers(1, 6, int(rng.integers(3, 12))).tolist()
        r = delphi([votes]).final
        assert r.median == np.median(votes)
        assert r.q1 == np.percentile(votes, 25) and r.q3 == np.percentile(votes, 75)


def _portfolio_problem(seed):
    rng = np.random.default_rng(seed)
    items = [{"id": f"E{i}", "values": {"C1": float(rng.integers(1, 6)), "C2": float(rng.integers(1, 6)),
                                        "C3": float(rng.integers(1, 6)), "COUT": float(rng.integers(10, 40))}}
             for i in range(5)]
    return DecisionProblem.from_dict({
        "criteria": [{"id": c, "aggregation": "mean"} for c in ("C1", "C2", "C3")],
        "attributes": [{"id": "COUT", "aggregation": "sum"}],
        "alternatives": items, "weights": {"C1": rng.uniform(), "C2": rng.uniform(), "C3": rng.uniform()},
        "portfolio": {"size": 2}, "constraints": [{"column": "COUT", "op": "<=", "value": 55}]})


def test_portfolio_linearity_and_constraints():
    """Somme pondérée sans normalisation + moyenne : score(XY) = (score(X) + score(Y)) / 2 ; contraintes respectées."""
    for seed in range(20):
        p = _portfolio_problem(seed)
        cand = p.candidates()
        try:
            res = evaluate(p, "wsm", normalization="none")
        except ValueError:
            continue                              # aucune paire admissible
        single = p.copy()
        single.portfolio = None
        single.constraints = []
        base = evaluate(single, "wsm", normalization="none").scores
        for pair, s in res.scores.items():
            a, b = pair.split("+")
            assert s == pytest.approx((base[a] + base[b]) / 2)
            assert cand.at[pair, "COUT"] <= 55


def test_threshold_is_exact_switch_point():
    """Juste avant le seuil de basculement le gagnant est inchangé, juste après il change."""
    for seed in range(15):
        rng, x, w, d = _random(seed, n=5, m=4)
        data = {"criteria": [{"id": c, "direction": d[c]} for c in x.columns],
                "alternatives": [{"id": i, "values": x.loc[i].to_dict()} for i in x.index],
                "weights": w.to_dict()}
        p = DecisionProblem.from_dict(data)
        sc = WeightScorer(p, "topsis")
        base = sc.winner(sc.base_weights.to_numpy())
        for c, row in weight_thresholds(p, "topsis").iterrows():
            for col, sign in (("seuil haut", 1), ("seuil bas", -1)):
                t = row[col]
                if np.isnan(t):
                    continue
                before = set_weight(sc.base_weights, c, t - sign * 1e-6).to_numpy()
                after = set_weight(sc.base_weights, c, t + sign * 1e-6).to_numpy()
                assert sc.winner(before) == base and sc.winner(after) != base


def test_monte_carlo_degenerate_spread():
    """Dispersion quasi nulle : le gagnant de référence gagne dans ~100 % des tirages."""
    p = DecisionProblem.load("examples/stratema_methode2.json")
    mc = monte_carlo(p, "topsis", n=2000, spread=0.001)
    assert mc.first_place["B+C"] > 0.999
