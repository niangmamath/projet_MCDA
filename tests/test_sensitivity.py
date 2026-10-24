import numpy as np
import pytest

from mcdakit import (
    WeightScorer,
    data_scenarios,
    evaluate,
    full_analysis,
    monte_carlo,
    to_excel,
    to_html,
    to_markdown,
    value_threshold,
    weight_profiles,
    weight_sweep,
    weight_thresholds,
)


@pytest.mark.parametrize("method", ["wsm", "topsis"])
def test_vectorised_scorer_matches_methods(stratema2, method):
    sc = WeightScorer(stratema2, method)
    rng = np.random.default_rng(0)
    for w in rng.dirichlet(np.ones(6), size=5):
        ref = evaluate(stratema2, method, weights=dict(zip(sc.criteria, w, strict=True))).scores.to_numpy()
        assert np.allclose(sc.scores(w), ref, atol=1e-12)


def test_weight_thresholds_match_report(stratema2):
    t = weight_thresholds(stratema2, "topsis")
    assert t.at["C2", "seuil haut"] == pytest.approx(0.245, abs=1e-3)
    assert t.at["C3", "seuil bas"] == pytest.approx(0.171, abs=1e-3)
    assert t.at["C5", "seuil haut"] == pytest.approx(0.441, abs=1e-3)
    assert t.at["C6", "bascule vers (haut)"] == "C+D"
    assert t.at["C1", "seuil haut"] == pytest.approx(0.471, abs=1e-3)


def test_wsm_part1_analytic_threshold(stratema1):
    # Rapport partie 1 : Δ(t) = 0,007178 − 2t lorsqu'on transfère t de C3 vers C2
    sc = WeightScorer(stratema1, "wsm", {"normalization": "none"})
    w = sc.base_weights.copy()
    t = 0.007178 / 2
    for eps, win in ((-1e-4, "B+C"), (1e-4, "B+D")):
        w2 = w.copy()
        w2["C3"] -= t + eps
        w2["C2"] += t + eps
        assert sc.winner(w2.to_numpy()) == win


def test_sweep_and_profiles(stratema2):
    s = weight_sweep(stratema2, "topsis", "C2", steps=11)
    assert s.loc[0.0, "gagnant"] == "B+C" and s.loc[1.0, "gagnant"] == "B+D"
    prof = weight_profiles(stratema2, "topsis", {"Financier": {"C2": 2}, "Égaux": {"__equal__": True}})
    assert prof.at["Financier", "gagnant"] == "B+D"
    assert prof.at["Égaux", "gagnant"] == "C+D"


def test_monte_carlo_reproducible(stratema2):
    a = monte_carlo(stratema2, "topsis", n=2000, seed=1)
    b = monte_carlo(stratema2, "topsis", n=2000, seed=1)
    assert a.first_place.equals(b.first_place)
    assert a.first_place.sum() == pytest.approx(1)
    assert a.first_place.idxmax() == "B+C"


def test_data_scenarios_report_part2(stratema2):
    sc = data_scenarios(
        stratema2,
        "topsis",
        {
            "C5 de C : 3 → 2": [("C", "C5", 2)],
            "B + 10 JH": [("B", "C1", 42)],
        },
    )
    assert sc.at["C5 de C : 3 → 2", "gagnant"] == "B+D"
    assert sc.at["B + 10 JH", "gagnant"] == "B+C"
    assert "B+D" in sc.at["B + 10 JH", "exclues (contraintes)"]


def test_value_threshold(stratema2):
    r = value_threshold(stratema2, "topsis", "C", "C5", 1, 5)
    assert r["seuil"] == pytest.approx(2.4, abs=0.05)
    assert r["nouveau gagnant"] == "B+D"


def test_full_analysis_and_reports(stratema2, fournisseur):
    fa = full_analysis(stratema2, {"topsis": {}, "wsm": {}}, mc_runs=500)
    assert fa.winner == "B+C"
    assert fa.diagnosis.level in ("robuste", "modérément robuste", "fragile")
    md = to_markdown(fa)
    assert "TOPSIS" in md and "Diagnostic" in md
    assert to_html(fa).startswith("<!doctype html>")
    assert len(to_excel(fa)) > 1000
    fb = full_analysis(fournisseur, mc_runs=500)
    assert fb.winner == "F3" and "F2" not in fb.matrix.index  # F2 exclu par délai > 25
