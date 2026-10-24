"""API HTTP de mcdakit (FastAPI), déployable telle quelle comme fonction Python sur Vercel.

Toutes les routes sont préfixées par /api. Le corps des requêtes reprend le format de projet décrit dans
docs/FORMAT.md ; les tableaux sont renvoyés sous la forme {"index": [...], "columns": [...], "data": [[...]]}.

Lancement local :  uvicorn api.index:app --reload   (puis ouvrir http://localhost:8000)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcdakit import (DecisionProblem, compare_methods, data_scenarios, delphi, diagnose, full_analysis,  # noqa: E402
                     monte_carlo, resolve_weights, run_methods, to_excel, to_html, to_markdown, value_threshold,
                     weight_profiles, weight_sweep, weight_thresholds)
from mcdakit.io_excel import read_excel, write_excel  # noqa: E402
from mcdakit.methods import METHOD_LABELS, METHOD_LOGIC, METHODS  # noqa: E402
from mcdakit.normalization import NORMALIZATIONS  # noqa: E402
from mcdakit.problem import AGGREGATIONS, OPERATORS  # noqa: E402

EXAMPLES = ROOT / "examples"
LIBRARY = EXAMPLES / "bibliotheque"
MAX_MC = 50000
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

app = FastAPI(title="mcdakit API", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.exception_handler(ValueError)
@app.exception_handler(KeyError)
async def _bad_request(_: Request, exc: Exception):
    msg = exc.args[0] if exc.args else str(exc)
    return JSONResponse({"detail": f"{msg}"}, status_code=400)


# =========================================================================== sérialisation
def _val(v: Any):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    if isinstance(v, (list, tuple)):
        return [_val(x) for x in v]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def table(df: pd.DataFrame | pd.Series | None):
    if df is None:
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return {"index": [str(i) for i in df.index], "columns": [str(c) for c in df.columns],
            "data": [[_val(v) for v in row] for row in df.itertuples(index=False, name=None)]}


def series(s: pd.Series) -> dict:
    return {str(k): _val(v) for k, v in s.items()}


def method_result(key: str, r, labels: pd.Series) -> dict:
    details = {}
    for name, d in r.details.items():
        if isinstance(d, pd.DataFrame) and not name.startswith("P_"):
            details[name] = table(d)
        elif isinstance(d, list):
            details[name] = [list(x) for x in d]
    params = dict(r.params)
    return {"key": key, "label": METHOD_LABELS.get(key, key), "logic": METHOD_LOGIC.get(key, ""),
            "score_label": r.score_label, "scores": series(r.scores), "ranking": series(r.ranking),
            "order": r.order, "best": r.best, "table": table(r.table(labels)), "details": details, "params": params}


# =========================================================================== modèles de requête
class ProblemBody(BaseModel):
    problem: dict


class AnalyzeBody(ProblemBody):
    methods: dict[str, dict] = Field(default_factory=lambda: {"wsm": {}, "topsis": {}})


class SensitivityBody(ProblemBody):
    method: str = "topsis"
    params: dict = Field(default_factory=dict)
    methods: dict[str, dict] | None = None       # pour le critère « accord des méthodes » du diagnostic
    sweep_criterion: str | None = None
    mc_runs: int = 10000
    mc_spread: float | None = 0.2
    profiles: dict[str, dict] = Field(default_factory=dict)
    scenarios: dict[str, list] = Field(default_factory=dict)
    seed: int = 42


class ValueThresholdBody(ProblemBody):
    method: str = "topsis"
    params: dict = Field(default_factory=dict)
    alternative: str
    column: str
    low: float
    high: float


class DelphiItem(BaseModel):
    alternative: str = ""
    column: str = ""
    rounds: list[Any]


class DelphiBody(BaseModel):
    items: list[DelphiItem]
    iqr_threshold: float = 1.0


class ReportBody(AnalyzeBody):
    primary: str | None = None
    format: str = "html"
    mc_runs: int = 5000
    mc_spread: float | None = 0.2
    profiles: dict[str, dict] = Field(default_factory=dict)
    scenarios: dict[str, list] = Field(default_factory=dict)


def _problem(d: dict) -> DecisionProblem:
    return DecisionProblem.from_dict(d)


def _check_method(m: str):
    if m not in METHODS:
        raise ValueError(f"Méthode inconnue : {m}")


# =========================================================================== routes
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    """Listes de choix pour l'interface."""
    return {"methods": METHOD_LABELS, "logic": METHOD_LOGIC, "normalizations": NORMALIZATIONS,
            "aggregations": list(AGGREGATIONS),
            "operators": list(OPERATORS),
            "default_normalization": {"wsm": "minmax", "wpm": "max", "topsis": "vector"}}


@app.get("/api/examples")
def examples():
    """Exemples du menu « Charger un exemple » : les deux cas Stratema (fichiers à la racine de examples/)."""
    return [{"id": p.stem, "name": DecisionProblem.load(p).name}
            for p in sorted(EXAMPLES.glob("*")) if p.suffix in (".json", ".yaml", ".yml")]


@app.get("/api/library")
def library():
    """Bibliothèque de cas (examples/bibliotheque/) : cas publiés et exemple générique, avec leur description."""
    out = []
    for p in sorted(LIBRARY.glob("*")):
        if p.suffix in (".json", ".yaml", ".yml"):
            prob = DecisionProblem.load(p)
            out.append({"id": p.stem, "name": prob.name, "description": prob.description,
                        "published": p.stem.startswith("cas_publie_"),
                        "size": f"{len(prob.alternatives)} alternatives · {len(prob.criteria)} critères"})
    return out


@app.get("/api/examples/{name}")
def example(name: str):
    for folder in (EXAMPLES, LIBRARY):
        for p in folder.glob(f"{name}.*"):
            if p.suffix in (".json", ".yaml", ".yml"):
                return DecisionProblem.load(p).to_dict()
    raise HTTPException(404, "Exemple introuvable")


@app.get("/api/validation")
def validation():
    """Recalcule en direct les cas publiés (mcdakit/benchmarks.py) et les compare aux résultats imprimés."""
    from mcdakit.benchmarks import run_benchmarks
    r = run_benchmarks()
    return {"all_ok": bool(r["conforme"].all() and r["même classement"].all()),
            "cases": [{k: _val(v) if not isinstance(v, list) else [_val(x) for x in v] for k, v in row.items()}
                      for row in r.to_dict(orient="records")]}


@app.post("/api/validate")
def validate(body: ProblemBody):
    """Vérifie le problème et renvoie candidates, admissibilité, matrice et poids."""
    p = _problem(body.problem)
    w, ahp = resolve_weights(p)
    cand = p.candidates()
    feas = p.feasibility(cand)
    out = {"labels": series(cand["name"]),
           "candidates": table(cand.drop(columns=["members"], errors="ignore")),
           "feasibility": table(feas),
           "weights": series(w),
           "ahp": None}
    try:
        out["matrix"] = table(p.decision_matrix())
    except ValueError as exc:
        out["matrix"] = None
        out["matrix_error"] = str(exc)
    if ahp is not None:
        out["ahp"] = {"lambda_max": ahp.lambda_max, "ci": ahp.ci, "ri": ahp.ri, "cr": ahp.cr,
                      "consistent": ahp.consistent, "matrix": table(ahp.matrix),
                      "worst": table(ahp.worst_judgements().set_index("i"))}
    return out


@app.post("/api/delphi")
def delphi_route(body: DelphiBody):
    rows = []
    for it in body.items:
        res = delphi(it.rounds, body.iqr_threshold)
        rows.append({"alternative": it.alternative, "column": it.column, "value": res.value,
                     "status": res.status, "consensus": res.consensus,
                     "rounds": [{"votes": r.votes, "median": r.median, "q1": r.q1, "q3": r.q3, "iqr": r.iqr}
                                for r in res.rounds]})
    return rows


@app.post("/api/analyze")
def analyze(body: AnalyzeBody):
    for m in body.methods:
        _check_method(m)
    p = _problem(body.problem)
    labels = p.labels()
    results = run_methods(p, body.methods)
    out = {"results": {k: method_result(k, r, labels) for k, r in results.items()},
           "labels": series(labels), "comparison": None}
    if len(results) > 1:
        c = compare_methods(results, labels)
        out["comparison"] = {"ranks": table(c["ranks"]), "kendall_tau": table(c["kendall_tau"]),
                             "spearman_rho": table(c["spearman_rho"]), "winners": c["winners"],
                             "same_winner": c["same_winner"], "divergent": c["divergent"]}
    return out


@app.post("/api/sensitivity")
def sensitivity(body: SensitivityBody):
    _check_method(body.method)
    p = _problem(body.problem)
    labels = p.labels()
    params = body.params
    winner = run_methods(p, {body.method: params})[body.method].best
    n_alt = len(p.decision_matrix())
    if n_alt < 2:
        raise ValueError("Il faut au moins deux alternatives admissibles pour une analyse de sensibilité.")
    thr = weight_thresholds(p, body.method, params)
    sweep = None
    if body.sweep_criterion:
        sw = weight_sweep(p, body.method, body.sweep_criterion, params)
        sweep = {"criterion": body.sweep_criterion, "grid": [float(x) for x in sw.index],
                 "scores": {a: [_val(v) for v in sw[a]] for a in sw.columns if a != "gagnant"},
                 "winner": list(sw["gagnant"]), "current": float(resolve_weights(p)[0][body.sweep_criterion])}
    mc = None
    if body.mc_runs > 0:
        mc = monte_carlo(p, body.method, params, n=min(body.mc_runs, MAX_MC), spread=body.mc_spread, seed=body.seed)
    prof = weight_profiles(p, body.method, body.profiles, params) if body.profiles else None
    scen = data_scenarios(p, body.method, body.scenarios, params) if body.scenarios else None
    comp = None
    if body.methods and len(body.methods) > 1:
        comp = compare_methods(run_methods(p, body.methods), labels)
    diag = diagnose(winner, mc, thr, comp, scen)
    return {
        "winner": winner,
        "labels": series(labels),
        "thresholds": table(thr),
        "sweep": sweep,
        "monte_carlo": None if mc is None else {
            "n": mc.n, "spread": mc.spread, "table": table(mc.table(labels)),
            "rank_distribution": table(mc.rank_distribution)},
        "profiles": table(prof),
        "scenarios": table(scen),
        "diagnosis": {"level": diag.level, "reasons": diag.reasons, "winner": diag.winner},
    }


@app.post("/api/value-threshold")
def value_threshold_route(body: ValueThresholdBody):
    _check_method(body.method)
    p = _problem(body.problem)
    r = value_threshold(p, body.method, body.alternative, body.column, body.low, body.high, body.params, steps=41)
    return {"current": r["valeur actuelle"], "threshold": _val(r["seuil"]), "new_winner": r["nouveau gagnant"],
            "curve": [[_val(v), w] for v, w in r["courbe"]]}


@app.post("/api/report")
def report(body: ReportBody):
    for m in body.methods:
        _check_method(m)
    p = _problem(body.problem)
    fa = full_analysis(p, body.methods, primary=body.primary, mc_runs=min(body.mc_runs, MAX_MC),
                       mc_spread=body.mc_spread, scenarios=body.scenarios or None, profiles=body.profiles or None)
    if body.format == "md":
        return Response(to_markdown(fa), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="rapport_mcda.md"'})
    if body.format == "xlsx":
        return Response(to_excel(fa), media_type=XLSX,
                        headers={"Content-Disposition": 'attachment; filename="calculs_mcda.xlsx"'})
    return Response(to_html(fa), media_type="text/html; charset=utf-8")


@app.post("/api/import")
async def import_file(file: UploadFile = File(...)):
    """Convertit un fichier .json, .yaml ou .xlsx en projet (dict)."""
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".xlsx"):
        import io
        p = read_excel(io.BytesIO(raw))
    elif name.endswith((".yaml", ".yml")):
        import yaml
        p = DecisionProblem.from_dict(yaml.safe_load(raw))
    else:
        p = DecisionProblem.from_dict(json.loads(raw))
    return p.to_dict()


@app.post("/api/export-xlsx")
def export_xlsx(body: ProblemBody):
    return Response(write_excel(_problem(body.problem)), media_type=XLSX,
                    headers={"Content-Disposition": 'attachment; filename="projet_mcda.xlsx"'})


@app.get("/api/template")
def template():
    return Response(write_excel(DecisionProblem.load(LIBRARY / "choix_fournisseur.yaml")), media_type=XLSX,
                    headers={"Content-Disposition": 'attachment; filename="modele_mcda.xlsx"'})


# Développement local : servir l'interface statique (sur Vercel, public/ est servi par le CDN).
_PUBLIC = ROOT / "public"


@app.get("/", include_in_schema=False)
def _index():
    return FileResponse(_PUBLIC / "index.html")


@app.get("/app", include_in_schema=False)
def _app_page():
    return FileResponse(_PUBLIC / "app.html")


@app.get("/{path:path}", include_in_schema=False)
def _static(path: str):
    f = (_PUBLIC / path).resolve()
    if f.is_file() and _PUBLIC in f.parents:
        return FileResponse(f)
    raise HTTPException(404)
