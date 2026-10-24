"""Analyse complète et génération de rapport (Markdown, HTML, Excel).

`full_analysis` enchaîne : poids → matrice de décision → méthodes → comparaison → sensibilité →
diagnostic de robustesse. Le résultat (`FullAnalysis`) peut ensuite être rendu en Markdown/HTML
avec le détail de tous les calculs intermédiaires, ou exporté en Excel (une feuille par tableau).
"""
from __future__ import annotations

import html
import io
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .analysis import compare_methods, run_methods
from .methods import METHOD_LABELS, METHOD_LOGIC, MethodResult
from .problem import DecisionProblem
from .sensitivity import (MonteCarloResult, RobustnessDiagnosis, data_scenarios, diagnose, monte_carlo,
                          weight_profiles, weight_thresholds)
from .weights import AHPResult, resolve_weights


@dataclass
class FullAnalysis:
    problem: DecisionProblem
    weights: pd.Series
    ahp: AHPResult | None
    candidates: pd.DataFrame
    feasibility: pd.DataFrame
    matrix: pd.DataFrame
    results: dict[str, MethodResult]
    primary: str
    comparison: dict | None
    thresholds: pd.DataFrame | None
    monte_carlo: MonteCarloResult | None
    profiles: pd.DataFrame | None
    scenarios: pd.DataFrame | None
    diagnosis: RobustnessDiagnosis
    labels: pd.Series = field(default_factory=pd.Series)

    @property
    def winner(self) -> str:
        return self.results[self.primary].best


def full_analysis(problem: DecisionProblem, methods: dict[str, dict] | list[str] | None = None,
                  primary: str | None = None, mc_runs: int = 5000, mc_spread: float | None = 0.2,
                  scenarios: dict[str, list] | None = None, profiles: dict[str, dict] | None = None,
                  seed: int = 42) -> FullAnalysis:
    methods = methods or ["wsm", "topsis"]
    if isinstance(methods, (list, tuple)):
        methods = {m: {} for m in methods}
    primary = primary or next(iter(methods))
    weights, ahp_res = resolve_weights(problem)
    candidates = problem.candidates()
    feas = problem.feasibility(candidates)
    matrix = problem.decision_matrix()
    results = run_methods(problem, methods)
    labels = candidates["name"]
    comparison = compare_methods(results, labels) if len(results) > 1 else None
    params = methods[primary]
    thresholds = weight_thresholds(problem, primary, params) if len(matrix) > 1 else None
    mc = monte_carlo(problem, primary, params, n=mc_runs, spread=mc_spread, seed=seed) if mc_runs and len(matrix) > 1 else None
    prof = weight_profiles(problem, primary, profiles, params) if profiles else None
    scen = data_scenarios(problem, primary, scenarios, params) if scenarios else None
    diag = diagnose(results[primary].best, mc, thresholds, comparison, scen)
    return FullAnalysis(problem, weights, ahp_res, candidates, feas, matrix, results, primary,
                        comparison, thresholds, mc, prof, scen, diag, labels)


# =========================================================================== rendu
def _fmt(v, digits: int = 4) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "oui" if v else "non"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", " ")
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if float(v).is_integer() and abs(v) >= 1000:
            return f"{int(v):,}".replace(",", " ")
        return f"{v:.{digits}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}".replace(",", " ")
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    return str(v)


class _Doc:
    """Petit modèle de document rendu en Markdown ou en HTML."""

    def __init__(self):
        self.blocks: list[tuple] = []

    def h(self, level: int, text: str):
        self.blocks.append(("h", level, text))

    def p(self, text: str):
        self.blocks.append(("p", text))

    def note(self, text: str):
        self.blocks.append(("note", text))

    def table(self, df: pd.DataFrame, index: bool = True, digits: int = 4, caption: str = ""):
        self.blocks.append(("table", df, index, digits, caption))

    def bullets(self, items: list[str]):
        self.blocks.append(("ul", items))

    def markdown(self) -> str:
        out = []
        for b in self.blocks:
            kind = b[0]
            if kind == "h":
                out.append("#" * b[1] + " " + b[2])
            elif kind == "p":
                out.append(b[1])
            elif kind == "note":
                out.append("> " + b[1])
            elif kind == "ul":
                out.append("\n".join(f"- {i}" for i in b[1]))
            elif kind == "table":
                df, index, digits, caption = b[1:]
                cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
                lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                for idx, row in df.iterrows():
                    cells = ([str(idx)] if index else []) + [_fmt(v, digits) for v in row]
                    lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
                if caption:
                    lines.append(f"\n*{caption}*")
                out.append("\n".join(lines))
        return "\n\n".join(out) + "\n"

    def html(self, title: str) -> str:
        e = html.escape
        parts = []
        for b in self.blocks:
            kind = b[0]
            if kind == "h":
                parts.append(f"<h{b[1]}>{e(b[2])}</h{b[1]}>")
            elif kind == "p":
                parts.append(f"<p>{e(b[1])}</p>")
            elif kind == "note":
                parts.append(f'<div class="note">{e(b[1])}</div>')
            elif kind == "ul":
                parts.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b[1]) + "</ul>")
            elif kind == "table":
                df, index, digits, caption = b[1:]
                head = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
                def td(v):
                    cls = "n" if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) else "t"
                    return f'<td class="{cls}">{e(_fmt(v, digits))}</td>'
                rows = "".join(
                    "<tr>" + (f'<td class="t">{e(str(i))}</td>' if index else "") + "".join(td(v) for v in r) + "</tr>"
                    for i, r in df.iterrows())
                cap = f"<caption>{e(caption)}</caption>" if caption else ""
                parts.append(f'<div class="tw"><table>{cap}<thead><tr>' + "".join(f"<th>{e(h)}</th>" for h in head)
                             + f"</tr></thead><tbody>{rows}</tbody></table></div>")
        css = ("body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:2rem auto;padding:0 16px;"
               "color:#1f2328;line-height:1.5}h1{font-size:1.7rem}h2{border-bottom:1px solid #d0d7de;padding-bottom:.3rem;margin-top:2rem}"
               ".tw{overflow-x:auto}table{border-collapse:collapse;margin:.6rem 0 1rem;font-size:.9rem}"
               "th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left}th{background:#f6f8fa}"
               "td.n{text-align:right;font-variant-numeric:tabular-nums}caption{caption-side:bottom;font-style:italic;color:#57606a;padding-top:4px}"
               ".note{background:#eef4ff;border-left:4px solid #3b6fd4;padding:.6rem .9rem;margin:1rem 0}")
        return (f"<!doctype html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>{e(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")


def _labelled(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    out = df.copy()
    if "alternative" not in out.columns:
        out.insert(0, "alternative", labels.reindex(out.index))
    return out


def build_document(fa: FullAnalysis) -> _Doc:
    p, doc, lab = fa.problem, _Doc(), fa.labels
    doc.h(1, p.name)
    doc.p(f"Rapport généré le {date.today():%d/%m/%Y} avec mcdakit.")
    if p.description:
        doc.p(p.description)

    # --- synthèse
    win = fa.winner
    res = fa.results[fa.primary]
    tab = res.table(lab)
    doc.h(2, "Synthèse")
    second = tab.index[1] if len(tab) > 1 else None
    msg = f"Méthode de référence : {METHOD_LABELS.get(fa.primary, fa.primary)}. Alternative recommandée : {lab[win]} ({res.score_label} = {_fmt(res.scores[win])})."
    if second:
        msg += f" Deuxième : {lab[second]} (écart {_fmt(tab.at[second, 'écart au 1er'])})."
    doc.note(msg)
    doc.note(f"Diagnostic : recommandation {fa.diagnosis.level}. " + " ; ".join(fa.diagnosis.reasons) + ".")

    # --- problème
    doc.h(2, "1. Formulation du problème")
    crit = pd.DataFrame([{"id": c.id, "critère": c.name, "sens": "maximiser" if c.direction == "max" else "minimiser",
                          "unité": c.unit, **({"agrégation portefeuille": c.aggregation} if p.portfolio else {})}
                         for c in p.criteria]).set_index("id")
    doc.table(crit, caption="Famille de critères")
    if p.attributes:
        doc.table(pd.DataFrame([vars(a) for a in p.attributes]).set_index("id"), caption="Attributs (hors critères)")
    base_cols = ["name"] + [c for c in p.criterion_ids + p.attribute_ids if c in p.alternatives.columns]
    doc.table(p.alternatives[base_cols].rename(columns={"name": "nom"}),
              caption="Éléments de base" if p.portfolio else "Alternatives")
    if p.portfolio:
        pf = p.portfolio
        size = str(pf.min_size) if pf.min_size == pf.max_size else f"{pf.min_size} à {pf.max_size}"
        doc.p(f"Mode portefeuille : les alternatives sont toutes les combinaisons de {size} éléments. "
              "Chaque colonne d'une combinaison est calculée par la règle d'agrégation indiquée.")
    if p.constraints:
        doc.p("Contraintes d'admissibilité (filtres, et non critères) : " + " ; ".join(map(str, p.constraints)) + ".")
        doc.table(_labelled(fa.feasibility, lab), caption="Vérification des contraintes")
    if p.overrides:
        doc.p("Valeurs surchargées (ex. corrections Delphi) : " +
              " ; ".join(f"{a} × {c} = {_fmt(v)}" for a, ch in p.overrides.items() for c, v in ch.items()) + ".")

    # --- poids
    doc.h(2, "2. Poids des critères")
    spec = p.weights_spec
    doc.p(f"Méthode de pondération : {spec.get('method', 'direct')}.")
    if fa.ahp is not None:
        a = fa.ahp
        doc.table(a.matrix, digits=3, caption="Matrice de comparaisons par paires (AHP)")
        doc.p(f"λmax = {a.lambda_max:.6f} ; IC = {a.ci:.5f} ; IA = {a.ri:.2f} ; RC = {a.cr:.4f} ({a.cr:.2%}) — "
              + ("cohérence acceptable (< 10 %)." if a.consistent else "INCOHÉRENT (≥ 10 %) : réviser les jugements."))
        if not a.consistent:
            doc.table(a.worst_judgements(), index=False, caption="Jugements les plus incohérents")
    wt = pd.DataFrame({"critère": [p.criterion(c).name for c in fa.weights.index], "poids": fa.weights,
                       "poids (%)": fa.weights * 100, "rang": fa.weights.rank(ascending=False, method="min").astype(int)})
    doc.table(wt, digits=6, caption="Poids normalisés (somme = 1)")

    # --- matrice
    doc.h(2, "3. Matrice de décision")
    doc.table(_labelled(fa.matrix, lab), caption=f"{len(fa.matrix)} alternatives admissibles × {fa.matrix.shape[1]} critères")

    # --- méthodes
    doc.h(2, "4. Méthodes d'agrégation")
    for k, (key, r) in enumerate(fa.results.items(), 1):
        doc.h(3, f"4.{k}. {METHOD_LABELS.get(key, key)}")
        doc.p(METHOD_LOGIC.get(key, ""))
        if r.params:
            doc.p("Paramètres : " + "; ".join(f"{a} = {b}" for a, b in r.params.items()))
        for name, d in r.details.items():
            if name.startswith("P_"):
                continue
            if isinstance(d, pd.DataFrame):
                doc.table(d, caption=name)
            elif isinstance(d, list) and name.startswith("Incomparabilités"):
                doc.p(f"{name} : " + (", ".join(f"{a} ? {b}" for a, b in d) if d else "aucune (le préordre partiel est complet)."))
        doc.table(r.table(lab), caption=f"Classement — {METHOD_LABELS.get(key, key)}")

    # --- comparaison
    if fa.comparison:
        c = fa.comparison
        doc.h(2, "5. Comparaison des méthodes")
        doc.table(c["ranks"], caption="Rangs obtenus par méthode")
        doc.table(c["kendall_tau"], digits=3, caption="Corrélation de rang (τ de Kendall)")
        doc.p("Même alternative en tête pour toutes les méthodes." if c["same_winner"] else
              "Les méthodes ne désignent pas la même alternative : " + ", ".join(f"{k} → {lab[v]}" for k, v in c["winners"].items()) + ".")
        if c["divergent"]:
            doc.p("Alternatives dont le rang diffère selon la méthode : " + ", ".join(lab[i] for i in c["divergent"]) +
                  ". Pistes d'explication : logique d'agrégation (somme compensatoire vs distance à l'idéal), normalisation, échelles des données.")

    # --- sensibilité
    doc.h(2, "6. Analyse de sensibilité et robustesse")
    doc.p(f"Tests réalisés sur la méthode de référence ({METHOD_LABELS.get(fa.primary, fa.primary)}). "
          "Pour un critère, les autres poids sont ajustés proportionnellement pour garder une somme de 1.")
    if fa.thresholds is not None:
        doc.table(fa.thresholds, caption="Seuils de basculement du 1er par poids (variation relative = distance au seuil / poids actuel)")
    if fa.monte_carlo is not None:
        mc = fa.monte_carlo
        how = "uniformes sur le simplexe" if mc.spread is None else f"Dirichlet autour des poids de référence, dispersion ≈ {mc.spread:.0%}"
        doc.p(f"Simulation de Monte-Carlo : {mc.n} jeux de poids ({how}).")
        doc.table(mc.table(lab), caption="Fréquence de 1re place et rang moyen")
    if fa.profiles is not None:
        doc.table(fa.profiles, caption="Profils de pondération")
    if fa.scenarios is not None:
        doc.table(fa.scenarios, caption="Scénarios sur les données / hypothèses")
    doc.h(3, "Diagnostic")
    doc.note(f"Recommandation {fa.diagnosis.level}.")
    doc.bullets(fa.diagnosis.reasons)
    doc.p("Règles : fragile si 1er dans moins de 60 % des tirages, si un poids bascule pour moins de 10 % de variation "
          "relative ou si la majorité des scénarios basculent ; robuste si ≥ 90 % des tirages, aucune bascule de poids "
          "en deçà de 50 %, méthodes concordantes et aucun scénario de basculement ; modérément robuste sinon.")

    # --- recommandation
    doc.h(2, "7. Recommandation")
    doc.p(f"Retenir {lab[win]}" + (f" ; alternative à arbitrer : {lab[second]}." if second else "."))
    doc.p("Limites : les résultats dépendent des poids, des évaluations et des hypothèses de modélisation explicitées ci-dessus. "
          "Le modèle éclaire la décision ; il ne se substitue pas au décideur.")
    return doc


def to_markdown(fa: FullAnalysis) -> str:
    return build_document(fa).markdown()


def to_html(fa: FullAnalysis) -> str:
    return build_document(fa).html(fa.problem.name)


def to_excel(fa: FullAnalysis) -> bytes:
    """Classeur Excel : une feuille par tableau de l'analyse."""
    buf = io.BytesIO()
    lab = fa.labels
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        def put(name, df):
            df.to_excel(xw, sheet_name=name[:31])
        put("Poids", pd.DataFrame({"poids": fa.weights}))
        if fa.ahp is not None:
            put("AHP matrice", fa.ahp.matrix)
        put("Candidates", fa.candidates.drop(columns=["members"], errors="ignore"))
        put("Contraintes", fa.feasibility)
        put("Matrice décision", fa.matrix)
        for key, r in fa.results.items():
            put(f"{key} classement", r.table(lab))
            for name, d in r.details.items():
                if isinstance(d, pd.DataFrame) and not name.startswith("P_"):
                    put(f"{key} {name}", d)
        if fa.comparison:
            put("Comparaison rangs", fa.comparison["ranks"])
            put("Kendall tau", fa.comparison["kendall_tau"])
        if fa.thresholds is not None:
            put("Seuils poids", fa.thresholds)
        if fa.monte_carlo is not None:
            put("Monte-Carlo", fa.monte_carlo.table(lab))
            put("MC distribution rangs", fa.monte_carlo.rank_distribution)
        if fa.profiles is not None:
            put("Profils", fa.profiles)
        if fa.scenarios is not None:
            put("Scénarios", fa.scenarios)
    return buf.getvalue()
