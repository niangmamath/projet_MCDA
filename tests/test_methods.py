import numpy as np
import pandas as pd
import pytest

from mcdakit import evaluate, topsis, weighted_sum
from mcdakit.problem import DecisionProblem


def test_portfolio_generation(stratema1):
    cand = stratema1.candidates()
    assert list(cand.index) == ["A+B", "A+C", "A+D", "B+C", "B+D", "C+D"]
    assert cand.at["B+C", "C3"] == 4.5 and cand.at["B+D", "JH"] == 57
    assert stratema1.feasibility()["admissible"].all()


def test_wsm_reproduces_report_part1(stratema1):
    res = evaluate(stratema1, "wsm", normalization="none")
    expected = {"B+C": 3.8746, "B+D": 3.8675, "C+D": 3.6515, "A+B": 3.4638, "A+C": 3.2479, "A+D": 3.2407}
    for k, v in expected.items():
        assert res.scores[k] == pytest.approx(v, abs=1e-4)
    assert res.order == list(expected)


def test_overrides_delphi(stratema2):
    m = stratema2.decision_matrix()
    assert m.at["A+D", "C3"] == 2 and m.at["B+C", "C4"] == 2 and m.at["B+D", "C6"] == 2
    assert m.at["A+B", "C1"] == 62 and m.at["B+D", "C2"] == 81000


def test_topsis_reproduces_report_part2(stratema2):
    res = evaluate(stratema2, "topsis")
    expected = {"B+C": 0.7228, "B+D": 0.6559, "A+B": 0.5986, "C+D": 0.4898, "A+C": 0.4086, "A+D": 0.3175}
    for k, v in expected.items():
        assert res.scores[k] == pytest.approx(v, abs=1e-4)
    d = res.details["Distances"]
    assert d.at["B+C", "D⁺"] == pytest.approx(0.0371, abs=1e-4)
    assert d.at["B+C", "D⁻"] == pytest.approx(0.0968, abs=1e-4)


def test_topsis_minmax_switches(stratema2):
    res = evaluate(stratema2, "topsis", normalization="minmax")
    assert res.best == "B+D"
    assert res.scores["B+D"] == pytest.approx(0.7019, abs=1e-4)


def test_constraint_filters():
    p = DecisionProblem.from_dict(
        {
            "criteria": [{"id": "x"}],
            "attributes": [{"id": "cost"}],
            "alternatives": [
                {"id": "a", "values": {"x": 1, "cost": 5}},
                {"id": "b", "values": {"x": 2, "cost": 9}},
                {"id": "c", "values": {"x": 3, "cost": 4}},
            ],
            "portfolio": {"size": 2},
            "constraints": [{"column": "cost", "op": "<=", "value": 10}],
        }
    )
    assert list(p.decision_matrix().index) == ["a+c"]


def test_dominance_respected_by_all_methods():
    m = pd.DataFrame({"g1": [5, 3, 1], "g2": [10, 20, 30]}, index=["best", "mid", "worst"])
    d = pd.Series({"g1": "max", "g2": "min"})
    w = pd.Series({"g1": 0.5, "g2": 0.5})
    for f in (weighted_sum, topsis):
        assert f(m, w, d).order == ["best", "mid", "worst"]


def test_wsm_rejects_unoriented_min():
    m = pd.DataFrame({"g": [1.0, 2.0]}, index=["a", "b"])
    with pytest.raises(ValueError):
        weighted_sum(m, pd.Series({"g": 1.0}), pd.Series({"g": "min"}), normalization="vector")


def test_roundtrip_json_and_excel(tmp_path, stratema2):
    stratema2.save(tmp_path / "p.json")
    p = DecisionProblem.load(tmp_path / "p.json")
    assert evaluate(p, "topsis").scores.equals(evaluate(stratema2, "topsis").scores)
    from mcdakit.io_excel import write_excel

    write_excel(stratema2, tmp_path / "p.xlsx")
    px = DecisionProblem.load(tmp_path / "p.xlsx")
    px.overrides = stratema2.overrides
    assert np.allclose(evaluate(px, "topsis").scores, evaluate(stratema2, "topsis").scores)
