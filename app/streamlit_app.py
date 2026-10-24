"""Interface web de mcdakit — aide à la décision multicritère pour tout projet.

Lancer :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcdakit import (DecisionProblem, data_scenarios, delphi, diagnose, full_analysis, monte_carlo,  # noqa: E402
                     resolve_weights, run_methods, compare_methods, to_excel, to_html, to_markdown,
                     value_threshold, weight_profiles, weight_sweep, weight_thresholds)
from mcdakit.io_excel import read_excel, write_excel  # noqa: E402
from mcdakit.methods import METHOD_LABELS, METHOD_LOGIC, PREFERENCE_FUNCTIONS  # noqa: E402
from mcdakit.normalization import NORMALIZATIONS  # noqa: E402
from mcdakit.problem import AGGREGATIONS, OPERATORS  # noqa: E402

st.set_page_config(page_title="mcdakit — Aide à la décision", page_icon="⚖️", layout="wide")

EXAMPLES = {p.stem: p for p in sorted((ROOT / "examples").glob("*")) if p.suffix in (".json", ".yaml", ".yml")}
SAATY = {-9: "9 ◀", -7: "7 ◀", -5: "5 ◀", -3: "3 ◀", -2: "2 ◀", 1: "=", 2: "▶ 2", 3: "▶ 3", 5: "▶ 5", 7: "▶ 7", 9: "▶ 9"}
SAATY_OPTIONS = [-9, -8, -7, -6, -5, -4, -3, -2, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NORM_DEFAULT = {"wsm": "minmax", "wpm": "max", "topsis": "vector"}

ss = st.session_state


# =========================================================================== état
def load_into_state(data: dict) -> None:
    """Charge un problème (dict) dans les tables éditables de la session."""
    crit = pd.DataFrame([{
        "id": c["id"], "nom": c.get("name", c["id"]), "sens": c.get("direction", "max"), "unité": c.get("unit", ""),
        "agrégation": c.get("aggregation", "mean"),
        "préférence": c.get("preference", {}).get("type", "usual"),
        "q": c.get("preference", {}).get("q"), "p": c.get("preference", {}).get("p"),
        "s": c.get("preference", {}).get("s")} for c in data["criteria"]])
    attrs = pd.DataFrame([{"id": a["id"], "nom": a.get("name", a["id"]), "unité": a.get("unit", ""),
                           "agrégation": a.get("aggregation", "sum")} for a in data.get("attributes", [])],
                         columns=["id", "nom", "unité", "agrégation"])
    cols = list(crit["id"]) + list(attrs["id"])
    alts = pd.DataFrame([{"id": a["id"], "nom": a.get("name", a["id"]),
                          **{c: a.get("values", {}).get(c) for c in cols}} for a in data["alternatives"]])
    cons = pd.DataFrame([{"colonne": c["column"], "opérateur": c["op"], "valeur": float(c["value"])}
                         for c in data.get("constraints", [])], columns=["colonne", "opérateur", "valeur"])
    ovr = pd.DataFrame([{"alternative": a, "colonne": c, "valeur": float(v)}
                        for a, ch in data.get("overrides", {}).items() for c, v in ch.items()],
                       columns=["alternative", "colonne", "valeur"])
    ss.latest = {"crit": crit, "attrs": attrs, "alts": alts, "cons": cons, "ovr": ovr}
    ss["name"] = data.get("name", "Nouveau projet")
    ss["description"] = data.get("description", "")
    pf = data.get("portfolio")
    ss["pf_on"] = bool(pf)
    size = (pf or {}).get("size", 2)
    lo, hi = (size.get("min", 2), size.get("max", size.get("min", 2))) if isinstance(size, dict) else (size, size)
    ss["pf_min"], ss["pf_max"] = int(lo), int(hi)
    # poids
    w = data.get("weights", {"method": "equal"})
    if "method" not in w:
        w = {"method": "direct", "values": w}
    ss["wmode"] = {"equal": "Égaux", "direct": "Directs", "roc": "Classement (ROC)", "rank_sum": "Classement (ROC)",
                "ahp": "AHP (comparaisons par paires)"}[w["method"]]
    ids = list(crit["id"])
    ss.w_direct = pd.DataFrame({"critère": ids, "poids": [float(w.get("values", {}).get(c, 1.0)) for c in ids]})
    order = w.get("order", ids)
    ss.w_rank = pd.DataFrame({"critère": ids, "rang": [order.index(c) + 1 if c in order else len(ids) for c in ids]})
    ss.ahp_vals = {}
    if w["method"] == "ahp":
        from mcdakit.weights import ahp, ahp_from_judgements
        res = ahp_from_judgements(ids, w["judgements"]) if "judgements" in w else ahp(w["matrix"], ids)
        a = res.matrix.to_numpy()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                v = a[i, j]
                sv = int(round(v)) if v >= 1 else -int(round(1 / v))
                ss.ahp_vals[(ids[i], ids[j])] = min(SAATY_OPTIONS, key=lambda o: abs(o - sv))
    ss.delphi = pd.DataFrame(columns=["alternative", "colonne", "tour 1", "tour 2", "tour 3"])
    ss.version = ss.get("version", 0) + 1
    ss.init = {}


def editor(name: str, df: pd.DataFrame, colkey: str = "", **kw) -> pd.DataFrame:
    """data_editor dont l'état survit aux reruns ; réinitialisé si le projet ou les colonnes changent."""
    key = f"{name}|{ss.version}|{colkey}"
    if key not in ss.init:
        ss.init[key] = df.reset_index(drop=True).copy()
    out = st.data_editor(ss.init[key], key=key, width="stretch", **kw)
    ss.latest[name] = out
    return out


def bump():
    ss.version += 1
    ss.init = {}


def clean_ids(series: pd.Series) -> list[str]:
    return [str(x).strip() for x in series if pd.notna(x) and str(x).strip()]


def num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else float(v)


def weights_spec(ids: list[str]) -> dict:
    mode = ss.get("wmode", "Égaux")
    if mode == "Égaux":
        return {"method": "equal"}
    if mode == "Directs":
        d = ss.latest.get("w_direct", ss.w_direct)
        vals = {str(r["critère"]): float(r["poids"] or 0) for _, r in d.iterrows()}
        return {"method": "direct", "values": {c: vals.get(c, 0.0) for c in ids}}
    if mode == "Classement (ROC)":
        d = ss.latest.get("w_rank", ss.w_rank)
        ranks = {str(r["critère"]): float(r["rang"] if pd.notna(r["rang"]) else 99) for _, r in d.iterrows()}
        return {"method": "roc", "order": sorted(ids, key=lambda c: (ranks.get(c, 99), ids.index(c)))}
    judg = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            v = ss.ahp_vals.get((a, b), 1)
            judg[f"{a}>{b}"] = v if v >= 1 else f"1/{-v}"
    return {"method": "ahp", "judgements": judg}


def build_dict() -> dict:
    L = ss.latest
    crit_rows = []
    for _, r in L["crit"].iterrows():
        cid = str(r["id"]).strip() if pd.notna(r["id"]) else ""
        if not cid:
            continue
        pref = {"type": r.get("préférence") or "usual"}
        for k in ("q", "p", "s"):
            if num(r.get(k)) is not None:
                pref[k] = num(r.get(k))
        crit_rows.append({"id": cid, "name": r["nom"] if pd.notna(r["nom"]) and r["nom"] else cid,
                          "direction": r["sens"] or "max", "unit": r["unité"] if pd.notna(r["unité"]) else "",
                          "aggregation": r["agrégation"] or "mean", "preference": pref})
    ids = [c["id"] for c in crit_rows]
    attrs = [{"id": str(r["id"]).strip(), "name": r["nom"] if pd.notna(r["nom"]) else str(r["id"]),
              "unit": r["unité"] if pd.notna(r["unité"]) else "", "aggregation": r["agrégation"] or "sum"}
             for _, r in L["attrs"].iterrows() if pd.notna(r["id"]) and str(r["id"]).strip()]
    cols = ids + [a["id"] for a in attrs]
    alts = []
    for _, r in L["alts"].iterrows():
        if pd.isna(r.get("id")) or not str(r["id"]).strip():
            continue
        vals = {c: float(r[c]) for c in cols if c in r.index and num(r[c]) is not None}
        alts.append({"id": str(r["id"]).strip(), "name": r["nom"] if pd.notna(r["nom"]) and r["nom"] else str(r["id"]),
                     "values": vals})
    data = {"name": ss.get("name") or "Projet", "description": ss.get("description", ""), "criteria": crit_rows, "attributes": attrs,
            "alternatives": alts, "weights": weights_spec(ids),
            "constraints": [{"column": r["colonne"], "op": r["opérateur"], "value": float(r["valeur"])}
                            for _, r in L["cons"].iterrows()
                            if pd.notna(r["colonne"]) and pd.notna(r["opérateur"]) and num(r["valeur"]) is not None],
            "overrides": {}}
    for _, r in L["ovr"].iterrows():
        if pd.notna(r["alternative"]) and pd.notna(r["colonne"]) and num(r["valeur"]) is not None:
            data["overrides"].setdefault(str(r["alternative"]).strip(), {})[r["colonne"]] = float(r["valeur"])
    if ss.pf_on:
        data["portfolio"] = {"size": {"min": int(ss.pf_min), "max": int(max(ss.pf_min, ss.pf_max))}}
    return data


@st.cache_data(show_spinner=False)
def cached_mc(problem_json: str, method: str, params_json: str, n: int, spread, seed: int):
    p = DecisionProblem.from_dict(json.loads(problem_json))
    return monte_carlo(p, method, json.loads(params_json), n=n, spread=spread, seed=seed)


@st.cache_data(show_spinner=False)
def cached_thresholds(problem_json: str, method: str, params_json: str):
    p = DecisionProblem.from_dict(json.loads(problem_json))
    return weight_thresholds(p, method, json.loads(params_json))


if "latest" not in ss:
    load_into_state(json.loads(EXAMPLES["stratema_methode2"].read_text(encoding="utf-8")))

# =========================================================================== barre latérale
with st.sidebar:
    st.title("⚖️ mcdakit")
    st.caption("Aide à la décision multicritère — pour n'importe quel projet.")
    st.subheader("Projet")
    choice = st.selectbox("Partir d'un exemple", ["—"] + list(EXAMPLES), key="ex_choice")
    if st.button("Charger l'exemple", disabled=choice == "—", width="stretch"):
        p = DecisionProblem.load(EXAMPLES[choice])
        load_into_state(p.to_dict())
        st.rerun()
    up = st.file_uploader("…ou importer un fichier", type=["json", "yaml", "yml", "xlsx"])
    if up is not None and st.button("Importer", width="stretch"):
        try:
            if up.name.endswith(".xlsx"):
                p = read_excel(up)
            elif up.name.endswith((".yaml", ".yml")):
                import yaml
                p = DecisionProblem.from_dict(yaml.safe_load(up.getvalue()))
            else:
                p = DecisionProblem.from_dict(json.loads(up.getvalue()))
            load_into_state(p.to_dict())
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Import impossible : {exc}")
    if st.button("Nouveau projet vierge", width="stretch"):
        load_into_state({"name": "Nouveau projet", "criteria": [
            {"id": "C1", "name": "Coût", "direction": "min"}, {"id": "C2", "name": "Qualité", "direction": "max"}],
            "alternatives": [{"id": "A1", "name": "Option 1", "values": {"C1": 100, "C2": 3}},
                             {"id": "A2", "name": "Option 2", "values": {"C1": 80, "C2": 2}},
                             {"id": "A3", "name": "Option 3", "values": {"C1": 120, "C2": 5}}]})
        st.rerun()
    st.divider()
    st.markdown("**Démarche**\n1. Définir critères et alternatives\n2. Pondérer (AHP…)\n3. Delphi si besoin\n"
                "4. Comparer les méthodes\n5. Tester la robustesse\n6. Exporter le rapport")

st.text_input("Nom du projet", key="name")
st.text_area("Contexte / décision à prendre", key="description", height=70)

tabs = st.tabs(["1 · Données", "2 · Poids", "3 · Delphi", "4 · Résultats", "5 · Sensibilité", "6 · Rapport"])

# =========================================================================== 1. données
with tabs[0]:
    st.subheader("Critères")
    st.caption("Sens : *max* si une valeur plus élevée est meilleure, *min* sinon (coût, délai…). "
               "Les colonnes préférence/q/p/s ne servent qu'à PROMETHEE (q = seuil d'indifférence, p = de préférence stricte).")
    crit = editor("crit", ss.latest["crit"], num_rows="dynamic", column_config={
        "id": st.column_config.TextColumn("id", required=True, help="Identifiant court, ex. C1"),
        "sens": st.column_config.SelectboxColumn("sens", options=["max", "min"], required=True),
        "agrégation": st.column_config.SelectboxColumn("agrégation", options=list(AGGREGATIONS),
                                                       help="Mode portefeuille : comment combiner les valeurs des éléments"),
        "préférence": st.column_config.SelectboxColumn("préférence", options=list(PREFERENCE_FUNCTIONS)),
        "q": st.column_config.NumberColumn("q"), "p": st.column_config.NumberColumn("p"), "s": st.column_config.NumberColumn("s"),
    })
    crit_ids = clean_ids(crit["id"])

    with st.expander("Attributs hors critères (budget, charge…) utilisables dans les contraintes"):
        attrs = editor("attrs", ss.latest["attrs"], num_rows="dynamic", column_config={
            "agrégation": st.column_config.SelectboxColumn("agrégation", options=list(AGGREGATIONS))})
    attr_ids = clean_ids(attrs["id"])
    all_cols = crit_ids + attr_ids

    st.subheader("Alternatives et évaluations")
    alts_prev = ss.latest["alts"].copy()
    for c in all_cols:
        if c not in alts_prev.columns:
            alts_prev[c] = np.nan
    alts_prev = alts_prev[["id", "nom"] + all_cols]
    editor("alts", alts_prev, colkey="|".join(all_cols), num_rows="dynamic",
           column_config={c: st.column_config.NumberColumn(c, format="%g") for c in all_cols})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Mode portefeuille")
        st.toggle("Choisir une combinaison d'éléments (ex. 2 missions parmi 4)", key="pf_on")
        a, b = st.columns(2)
        a.number_input("Taille min", 1, 20, key="pf_min", disabled=not ss.pf_on)
        b.number_input("Taille max", 1, 20, key="pf_max", disabled=not ss.pf_on)
        if ss.pf_on:
            st.caption("Chaque combinaison devient une alternative ; ses valeurs sont agrégées selon la colonne « agrégation ».")
    with c2:
        st.subheader("Contraintes")
        st.caption("Filtres d'admissibilité (ex. charge ≤ 65). Une alternative qui les viole est exclue.")
        editor("cons", ss.latest["cons"], colkey="|".join(all_cols), num_rows="dynamic", column_config={
            "colonne": st.column_config.SelectboxColumn("colonne", options=all_cols),
            "opérateur": st.column_config.SelectboxColumn("opérateur", options=list(OPERATORS)),
            "valeur": st.column_config.NumberColumn("valeur", format="%g")})

    with st.expander("Surcharges de cellules (ex. valeurs corrigées par Delphi pour une combinaison)"):
        editor("ovr", ss.latest["ovr"], colkey="|".join(all_cols), num_rows="dynamic", column_config={
            "colonne": st.column_config.SelectboxColumn("colonne", options=all_cols),
            "valeur": st.column_config.NumberColumn("valeur", format="%g")})

# =========================================================================== 2. poids
with tabs[1]:
    st.subheader("Pondération des critères")
    st.radio("Méthode", ["Égaux", "Directs", "Classement (ROC)", "AHP (comparaisons par paires)"],
             key="wmode", horizontal=True)
    if ss.wmode == "Directs":
        st.caption("Saisissez des poids quelconques : ils seront normalisés pour sommer à 100 %.")
        base = ss.latest.get("w_direct", ss.w_direct)
        base = pd.DataFrame({"critère": crit_ids, "poids": [float(base.set_index("critère")["poids"].get(c, 1.0)) for c in crit_ids]})
        editor("w_direct", base, colkey="|".join(crit_ids), num_rows="fixed", disabled=["critère"])
    elif ss.wmode == "Classement (ROC)":
        st.caption("Classez simplement les critères (1 = le plus important). Les poids sont déduits par la méthode ROC.")
        base = ss.latest.get("w_rank", ss.w_rank)
        base = pd.DataFrame({"critère": crit_ids, "rang": [int(base.set_index("critère")["rang"].get(c, i + 1)) for i, c in enumerate(crit_ids)]})
        editor("w_rank", base, colkey="|".join(crit_ids), num_rows="fixed", disabled=["critère"],
               column_config={"rang": st.column_config.NumberColumn("rang", min_value=1, step=1)})
    elif ss.wmode.startswith("AHP"):
        st.caption("Pour chaque paire, déplacez le curseur vers le critère le plus important. "
                   "1 = égale importance, 3 = modérée, 5 = forte, 7 = très forte, 9 = extrême.")
        names = dict(zip(crit["id"], crit["nom"]))
        cols = st.columns(2)
        k = 0
        for i, a in enumerate(crit_ids):
            for b in crit_ids[i + 1:]:
                with cols[k % 2]:
                    ss.ahp_vals[(a, b)] = st.select_slider(
                        f"**{names.get(a, a)}** ◀ ▶ **{names.get(b, b)}**", SAATY_OPTIONS[::-1],
                        value=ss.ahp_vals.get((a, b), 1), key=f"ahp|{a}|{b}|{ss.version}",
                        format_func=lambda v: "égal" if v == 1 else (f"◀ {v}" if v > 1 else f"{-v} ▶"))
                k += 1
    try:
        tmp = DecisionProblem.from_dict(build_dict())
        w, ahp_res = resolve_weights(tmp)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(x=[tmp.criterion(c).name for c in w.index], y=w.values * 100,
                         labels={"x": "", "y": "poids (%)"}, text=[f"{v:.1%}" for v in w.values])
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            if ahp_res is not None:
                st.metric("Ratio de cohérence (RC)", f"{ahp_res.cr:.2%}",
                          "cohérent" if ahp_res.consistent else "incohérent (≥ 10 %)",
                          delta_color="normal" if ahp_res.consistent else "inverse")
                st.caption(f"λmax = {ahp_res.lambda_max:.4f} · IC = {ahp_res.ci:.4f} · IA = {ahp_res.ri:.2f}")
                if not ahp_res.consistent:
                    st.warning("Jugements à revoir en priorité :")
                    st.dataframe(ahp_res.worst_judgements(), hide_index=True)
                with st.expander("Matrice AHP"):
                    st.dataframe(ahp_res.matrix.round(3))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Problème incomplet : {exc}")

# =========================================================================== problème courant
try:
    problem_dict = build_dict()
    problem = DecisionProblem.from_dict(problem_dict)
    problem_json = json.dumps(problem_dict, sort_keys=True, default=float)
    labels = problem.labels()
    matrix = problem.decision_matrix()
    ok = True
except Exception as exc:  # noqa: BLE001
    ok = False
    err = str(exc)


def need_problem() -> bool:
    if not ok:
        st.error(f"Le problème n'est pas encore valide : {err}")
    return ok


def lab(i):
    return f"{labels.get(i, i)} ({i})" if labels.get(i, i) != i else i


def short(i):
    """Libellé court pour les graphiques : l'identifiant en mode portefeuille, le nom sinon."""
    return i if ok and problem.portfolio else labels.get(i, i)


# =========================================================================== 3. delphi
with tabs[2]:
    st.subheader("Consultation Delphi")
    st.caption("Pour une évaluation incertaine, saisissez les votes anonymes des experts tour par tour "
               "(ex. « 2, 1, 3, 3, 2 »). La valeur retenue est la médiane du dernier tour ; le consensus est "
               "atteint quand l'écart interquartile est ≤ au seuil.")
    targets = []
    if ok:
        targets = list(problem.alternatives.index) + ([i for i in problem.candidates().index] if problem.portfolio else [])
    iqr = st.number_input("Seuil de consensus (écart interquartile)", 0.0, 10.0, 1.0, 0.5)
    dtab = editor("delphi", ss.delphi, colkey="|".join(targets) + "|".join(crit_ids), num_rows="dynamic", column_config={
        "alternative": st.column_config.SelectboxColumn("alternative", options=targets),
        "colonne": st.column_config.SelectboxColumn("colonne", options=crit_ids)})
    rows, apply = [], []
    for _, r in dtab.iterrows():
        rounds = [r[c] for c in ("tour 1", "tour 2", "tour 3") if pd.notna(r.get(c)) and str(r.get(c)).strip()]
        if not rounds or pd.isna(r.get("alternative")) or pd.isna(r.get("colonne")):
            continue
        try:
            res = delphi(rounds, iqr)
        except Exception as exc:  # noqa: BLE001
            rows.append({"cellule": f"{r['alternative']} × {r['colonne']}", "statut": f"erreur : {exc}"})
            continue
        row = {"cellule": f"{r['alternative']} × {r['colonne']}"}
        for k, rd in enumerate(res.rounds, 1):
            row[f"tour {k}"] = f"méd. {rd.median:g} [{rd.q1:g}–{rd.q3:g}]"
        row.update({"retenu": res.value, "statut": res.status})
        rows.append(row)
        apply.append((r["alternative"], r["colonne"], res.value))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if st.button("Appliquer les valeurs retenues au modèle", type="primary"):
            ss.delphi = dtab
            base_ids = set(problem.alternatives.index) if ok else set()
            ovr = ss.latest["ovr"].copy()
            alts = ss.latest["alts"].copy()
            for a, c, v in apply:
                if a in base_ids:
                    alts.loc[alts["id"] == a, c] = v
                else:
                    ovr = ovr[~((ovr["alternative"] == a) & (ovr["colonne"] == c))]
                    ovr = pd.concat([ovr, pd.DataFrame([{"alternative": a, "colonne": c, "valeur": v}])], ignore_index=True)
            ss.latest["ovr"], ss.latest["alts"] = ovr, alts
            bump()
            st.success("Valeurs appliquées (élément de base modifié, ou surcharge de la combinaison).")
            st.rerun()

# =========================================================================== 4. résultats
with tabs[3]:
    if need_problem():
        st.subheader("Méthodes")
        chosen = st.multiselect("Méthodes à comparer", list(METHOD_LABELS), default=["wsm", "topsis", "promethee"],
                                format_func=METHOD_LABELS.get)
        params = {}
        pcols = st.columns(max(len(chosen), 1))
        for col, m in zip(pcols, chosen):
            with col:
                st.markdown(f"**{METHOD_LABELS[m]}**")
                st.caption(METHOD_LOGIC[m])
                if m in NORM_DEFAULT:
                    opts = list(NORMALIZATIONS)
                    params[m] = {"normalization": st.selectbox("Normalisation", opts, index=opts.index(NORM_DEFAULT[m]),
                                                               key=f"norm_{m}", format_func=lambda o: NORMALIZATIONS[o])}
                else:
                    params[m] = {}
        if chosen:
            ref = st.selectbox("Méthode de référence (recommandation, sensibilité, rapport)", chosen,
                               format_func=METHOD_LABELS.get, key="ref_method")
            params = {ref: params[ref], **{k: v for k, v in params.items() if k != ref}}
            ss.methods = params
            try:
                results = run_methods(problem, params)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                results = {}
            if results:
                best = results[ref].best
                tab = results[ref].table(labels)
                gap = tab["écart au 1er"].iloc[1] if len(tab) > 1 else None
                msg = (f"**Recommandation ({METHOD_LABELS[ref]}) : {lab(best)}** — {results[ref].score_label} = "
                       f"{results[ref].scores[best]:.4f}")
                if gap is not None:
                    msg += f"  \nDeuxième : {lab(tab.index[1])}, écart {gap:.4f}"
                msg += f"  \nAlternatives admissibles : {len(matrix)} / {len(problem.candidates())}"
                st.success(msg)

                st.subheader("Classements")
                rcols = st.columns(len(results))
                for col, (m, r) in zip(rcols, results.items()):
                    with col:
                        t = r.table(labels)
                        fig = px.bar(t.assign(lbl=[short(i) for i in t.index]), x=r.score_label, y="lbl",
                                     orientation="h", title=METHOD_LABELS[m], hover_name="alternative")
                        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=60 + 38 * len(t),
                                          margin=dict(t=40, b=10, l=10, r=10), yaxis_title="")
                        st.plotly_chart(fig, width="stretch")
                        st.dataframe(t.round(4), width="stretch")
                if len(results) > 1:
                    st.subheader("Comparaison des méthodes")
                    comp = compare_methods(results, labels)
                    if comp["same_winner"]:
                        st.success(f"Toutes les méthodes désignent **{labels[best]}**.")
                    else:
                        st.warning("Les méthodes divergent sur le 1er : " +
                                   " · ".join(f"{k} → **{labels[v]}**" for k, v in comp["winners"].items()))
                    c1, c2 = st.columns([3, 2])
                    c1.dataframe(comp["ranks"], width="stretch")
                    c2.dataframe(comp["kendall_tau"].round(3), width="stretch")
                    c2.caption("τ de Kendall : 1 = classements identiques, −1 = inversés.")
                    if comp["divergent"]:
                        st.info("Rangs divergents pour : " + ", ".join(labels[i] for i in comp["divergent"]) +
                                ". Causes typiques : compensation (additive vs surclassement), normalisation, "
                                "seuils de préférence PROMETHEE, valeurs cardinales vs notes.")
                with st.expander("Matrice de décision et détail des calculs"):
                    st.markdown("**Admissibilité**")
                    st.dataframe(pd.concat([problem.candidates().drop(columns=["members"], errors="ignore"),
                                            problem.feasibility()], axis=1), width="stretch")
                    for m, r in results.items():
                        st.markdown(f"### {METHOD_LABELS[m]}")
                        for name, d in r.details.items():
                            if isinstance(d, pd.DataFrame) and not name.startswith("P_"):
                                st.markdown(f"*{name}*")
                                st.dataframe(d.round(4), width="stretch")
                            elif isinstance(d, list):
                                st.markdown(f"*{name}* : " + (", ".join(f"{a} ? {b}" for a, b in d) or "aucune"))

# =========================================================================== 5. sensibilité
with tabs[4]:
    if need_problem() and len(matrix) > 1:
        methods_sel = ss.get("methods") or {"topsis": {}}
        primary = next(iter(methods_sel))
        st.markdown(f"Méthode étudiée : **{METHOD_LABELS[primary]}** (méthode de référence choisie dans l'onglet Résultats).")
        pparams = methods_sel[primary]
        pj = json.dumps(pparams, sort_keys=True)
        winner = run_methods(problem, {primary: pparams})[primary].best
        st.info(f"Gagnant de référence : **{labels[winner]}**. Où la décision bascule-t-elle ?")

        st.subheader("A. Paramètres de préférence : les poids")
        thr = cached_thresholds(problem_json, primary, pj)
        show = thr.copy()
        for c in ("bascule vers (haut)", "bascule vers (bas)"):
            show[c] = show[c].map(lambda i: labels.get(i, i))
        for c, f in (("poids actuel", "{:.1%}"), ("seuil haut", "{:.1%}"), ("seuil bas", "{:.1%}"), ("variation relative min", "{:.0%}")):
            show[c] = show[c].map(lambda v, f=f: "—" if pd.isna(v) else f.format(v))
        st.dataframe(show, width="stretch")
        st.caption("Seuil = poids du critère à partir duquel le 1er change (les autres poids s'ajustent proportionnellement).")
        crit_pick = st.selectbox("Balayer le poids de", crit_ids,
                                 format_func=lambda c: f"{c} — {problem.criterion(c).name}")
        sw = weight_sweep(problem, primary, crit_pick, pparams)
        long = sw.drop(columns="gagnant").reset_index().melt(id_vars=sw.index.name, var_name="alt", value_name="score")
        long["alternative"] = long["alt"].map(short)
        fig = px.line(long, x=sw.index.name, y="score", color="alternative")
        w0 = float(resolve_weights(problem)[0][crit_pick])
        fig.add_vline(x=w0, line_dash="dash", annotation_text="poids actuel")
        fig.update_layout(height=380, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns(3)
        n = c1.number_input("Tirages Monte-Carlo", 500, 100000, 10000, 500)
        uniform = c2.toggle("Aucune information sur les poids (tirage uniforme)", False)
        spread = None if uniform else c3.slider("Dispersion autour des poids", 0.05, 0.6, 0.2, 0.05)
        mc = cached_mc(problem_json, primary, pj, int(n), spread, 42)
        mt = mc.table(labels)
        fig = px.bar(mt.assign(lbl=[short(i) for i in mt.index]), x="lbl", y="fréquence 1er", hover_name="alternative",
                     text=mt["fréquence 1er"].map(lambda v: f"{v:.1%}"), labels={"lbl": ""})
        fig.update_layout(height=320, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        with st.expander("Distribution des rangs"):
            rd = mc.rank_distribution.copy()
            rd.index = [short(i) for i in rd.index]
            st.plotly_chart(px.imshow(rd, text_auto=".0%", aspect="auto", labels={"x": "rang", "y": "", "color": "fréquence"}),
                            width="stretch")

        st.markdown("**Profils de décideurs** — multiplicateurs appliqués aux poids de référence")
        prof_base = pd.DataFrame([{"profil": f"{crit_ids[0]} × 2", **{c: (2.0 if c == crit_ids[0] else 1.0) for c in crit_ids}}])
        prof = editor("profiles", ss.latest.get("profiles", prof_base), colkey="|".join(crit_ids), num_rows="dynamic")
        profiles = {str(r["profil"]): {c: float(r[c]) for c in crit_ids if num(r.get(c)) is not None}
                    for _, r in prof.iterrows() if pd.notna(r.get("profil"))}
        if st.checkbox("Ajouter le profil « poids égaux »", True):
            profiles["Poids égaux"] = {"__equal__": True}
        pt = weight_profiles(problem, primary, profiles, pparams)
        pt["gagnant"] = pt["gagnant"].map(lambda i: labels.get(i, i))
        st.dataframe(pt.round(4), width="stretch")

        st.subheader("B. Données et hypothèses")
        st.caption("Chaque ligne modifie une valeur ; les lignes de même nom de scénario sont appliquées ensemble. "
                   "Modifier un élément de base le ré-agrège dans toutes les combinaisons qui le contiennent.")
        targets = list(problem.alternatives.index) + (list(problem.candidates().index) if problem.portfolio else [])
        scen_df = editor("scen", ss.latest.get("scen", pd.DataFrame(columns=["scénario", "alternative", "colonne", "valeur"])),
                         colkey="|".join(targets + crit_ids + attr_ids), num_rows="dynamic", column_config={
                             "alternative": st.column_config.SelectboxColumn("alternative", options=targets),
                             "colonne": st.column_config.SelectboxColumn("colonne", options=crit_ids + attr_ids),
                             "valeur": st.column_config.NumberColumn("valeur", format="%g")})
        scenarios = {}
        for _, r in scen_df.iterrows():
            if pd.notna(r.get("scénario")) and pd.notna(r.get("alternative")) and pd.notna(r.get("colonne")) and num(r.get("valeur")) is not None:
                scenarios.setdefault(str(r["scénario"]), []).append((r["alternative"], r["colonne"], float(r["valeur"])))
        scen_table = None
        if scenarios:
            scen_table = data_scenarios(problem, primary, scenarios, pparams)
            st.dataframe(scen_table, width="stretch")

        with st.expander("Seuil de basculement d'une valeur"):
            a, b, c, d = st.columns(4)
            tgt = a.selectbox("Alternative / élément", targets, key="vt_alt")
            col = b.selectbox("Colonne", crit_ids + attr_ids, key="vt_col")
            src = problem.alternatives if tgt in problem.alternatives.index else problem.candidates()
            cur = float(src.at[tgt, col]) if col in src.columns and pd.notna(src.at[tgt, col]) else 0.0
            span = abs(cur) if cur else 1.0
            lo = c.number_input("de", value=cur - span, key="vt_lo")
            hi = d.number_input("à", value=cur + span, key="vt_hi")
            if st.button("Chercher le seuil"):
                vt = value_threshold(problem, primary, tgt, col, lo, hi, pparams, steps=41)
                if vt["seuil"] is None:
                    st.success(f"Aucun basculement entre {lo:g} et {hi:g} : le 1er reste {labels[winner]}.")
                else:
                    st.warning(f"Valeur actuelle {vt['valeur actuelle']:g} → basculement vers "
                               f"**{labels.get(vt['nouveau gagnant'], vt['nouveau gagnant'])}** à partir de ≈ {vt['seuil']:.3g}.")

        st.subheader("C. Diagnostic de robustesse")
        comp = compare_methods(run_methods(problem, methods_sel), labels) if len(methods_sel) > 1 else None
        diag = diagnose(winner, mc, thr, comp, scen_table)
        color = {"robuste": st.success, "modérément robuste": st.warning, "fragile": st.error}[diag.level]
        color(f"Recommandation **{diag.level}** : {labels[winner]}")
        for r in diag.reasons:
            st.markdown(f"- {r}")
        ss.sens = {"primary": primary, "profiles": profiles, "scenarios": scenarios, "n": int(n), "spread": spread}
    elif ok:
        st.info("Il faut au moins deux alternatives admissibles.")

# =========================================================================== 6. rapport
with tabs[5]:
    if need_problem():
        st.subheader("Exporter")
        methods_sel = ss.get("methods") or {"wsm": {}, "topsis": {}, "promethee": {}}
        sens = ss.get("sens", {})
        st.caption("Le rapport reprend toutes les étapes : formulation, poids (et cohérence AHP), matrice, calculs "
                   "intermédiaires de chaque méthode, comparaison, sensibilité, diagnostic et recommandation. "
                   "Les réglages pris en compte sont ceux des onglets Résultats et Sensibilité.")
        if st.button("Générer le rapport", type="primary"):
            with st.spinner("Analyse complète…"):
                primary = sens.get("primary", next(iter(methods_sel)))
                ordered = {primary: methods_sel[primary], **{k: v for k, v in methods_sel.items() if k != primary}}
                fa = full_analysis(problem, ordered, primary=primary, mc_runs=sens.get("n", 5000),
                                   mc_spread=sens.get("spread", 0.2), scenarios=sens.get("scenarios") or None,
                                   profiles=sens.get("profiles") or None)
                ss.report = {"html": to_html(fa), "md": to_markdown(fa), "xlsx": to_excel(fa)}
        if "report" in ss:
            r = ss.report
            c1, c2, c3 = st.columns(3)
            c1.download_button("Rapport HTML (imprimable en PDF)", r["html"], "rapport_mcda.html", "text/html", width="stretch")
            c2.download_button("Rapport Markdown", r["md"], "rapport_mcda.md", "text/markdown", width="stretch")
            c3.download_button("Tous les tableaux (Excel)", r["xlsx"], "calculs_mcda.xlsx", width="stretch")
            with st.expander("Aperçu", expanded=True):
                st.markdown(r["md"])
        st.divider()
        st.subheader("Sauvegarder le projet")
        c1, c2 = st.columns(2)
        c1.download_button("Projet (JSON, rechargeable)", json.dumps(problem.to_dict(), ensure_ascii=False, indent=2),
                           "projet_mcda.json", "application/json", width="stretch")
        c2.download_button("Projet (Excel modifiable)", write_excel(problem), "projet_mcda.xlsx", width="stretch")
