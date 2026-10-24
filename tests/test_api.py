import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)
EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def strat():
    return json.loads((EX / "stratema_methode2.json").read_text(encoding="utf-8"))


def test_meta_and_examples():
    assert "topsis" in client.get("/api/meta").json()["methods"]
    ids = [e["id"] for e in client.get("/api/examples").json()]
    assert ids == ["stratema_methode1", "stratema_methode2"]  # seuls les cas Stratema dans le menu
    assert client.get("/api/examples/stratema_methode1").json()["criteria"]
    assert client.get("/api/examples/choix_fournisseur").status_code == 404  # plus de cas hors Stratema


def test_validate(strat):
    r = client.post("/api/validate", json={"problem": strat}).json()
    assert r["ahp"]["cr"] == pytest.approx(0.02343, abs=1e-5)
    assert len(r["matrix"]["index"]) == 6


def test_analyze(strat):
    r = client.post("/api/analyze", json={"problem": strat, "methods": {"topsis": {}, "wsm": {}}}).json()
    assert r["results"]["topsis"]["best"] == "B+C"
    assert r["results"]["topsis"]["scores"]["B+C"] == pytest.approx(0.7228, abs=1e-4)
    assert r["comparison"]["same_winner"] is False


def test_sensitivity(strat):
    r = client.post(
        "/api/sensitivity",
        json={
            "problem": strat,
            "method": "topsis",
            "sweep_criterion": "C2",
            "mc_runs": 2000,
            "profiles": {"Financier": {"C2": 2}},
            "scenarios": {"C5 de C": [["C", "C5", 2]]},
        },
    ).json()
    assert r["winner"] == "B+C"
    assert r["diagnosis"]["level"] in ("robuste", "modérément robuste", "fragile")
    assert r["sweep"]["winner"][-1] == "B+D"
    assert "B+D" in json.dumps(r["scenarios"])


def test_value_threshold_and_delphi(strat):
    r = client.post(
        "/api/value-threshold",
        json={"problem": strat, "method": "topsis", "alternative": "C", "column": "C5", "low": 1, "high": 5},
    ).json()
    assert r["threshold"] == pytest.approx(2.4, abs=0.05)
    d = client.post("/api/delphi", json={"items": [{"rounds": ["2,1,3,3,2", "2,2,2,3,2"]}]}).json()
    assert d[0]["value"] == 2 and d[0]["consensus"]


def test_reports_and_files(strat):
    for fmt in ("html", "md", "xlsx"):
        r = client.post(
            "/api/report", json={"problem": strat, "methods": {"topsis": {}}, "format": fmt, "mc_runs": 500}
        )
        assert r.status_code == 200 and len(r.content) > 1000
    xlsx = client.post("/api/export-xlsx", json={"problem": strat}).content
    back = client.post("/api/import", files={"file": ("p.xlsx", xlsx)}).json()
    assert [c["id"] for c in back["criteria"]] == [c["id"] for c in strat["criteria"]]
    assert client.get("/api/template").status_code == 200


def test_errors_are_400(strat):
    bad = dict(strat, criteria=[])
    r = client.post("/api/analyze", json={"problem": bad})
    assert r.status_code == 400 and "critère" in r.json()["detail"]
    assert client.post("/api/analyze", json={"problem": strat, "methods": {"nope": {}}}).status_code == 400


def test_static_index():
    assert "mcdakit" in client.get("/").text


def test_validation_route():
    r = client.get("/api/validation").json()
    assert r["all_ok"] is True and len(r["cases"]) >= 5


def test_report_pdf(strat):
    r = client.post(
        "/api/report", json={"problem": strat, "methods": {"wsm": {}, "topsis": {}}, "format": "pdf", "mc_runs": 500}
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
