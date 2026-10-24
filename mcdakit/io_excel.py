"""Import / export d'un problème au format Excel (modèle à remplir sans écrire de JSON).

Feuilles attendues :
  * « Critères »     : id | nom | sens (max/min) | unité | poids | agrégation
  * « Alternatives » : id | nom | une colonne par critère et par attribut
  * « Attributs »    : id | nom | unité | agrégation                       (optionnelle)
  * « Contraintes »  : colonne | opérateur | valeur                         (optionnelle)
  * « Paramètres »   : clé | valeur  (nom, description, portefeuille_min, portefeuille_max) (optionnelle)
  * « AHP »          : matrice carrée de comparaisons, critères en ligne et colonne (optionnelle ;
                        si présente elle remplace la colonne « poids »)
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .problem import DecisionProblem


def _str(v, default=""):
    return default if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def read_excel(source) -> DecisionProblem:
    sheets = pd.read_excel(source, sheet_name=None)
    norm = {k.strip().lower().replace("è", "e").replace("é", "e"): v for k, v in sheets.items()}
    crit_df = norm.get("criteres")
    alt_df = norm.get("alternatives")
    if crit_df is None or alt_df is None:
        raise ValueError("Le classeur doit contenir les feuilles « Critères » et « Alternatives ».")
    criteria, weights = [], {}
    for _, r in crit_df.iterrows():
        cid = _str(r.get("id"))
        if not cid:
            continue
        criteria.append({"id": cid, "name": _str(r.get("nom"), cid), "direction": _str(r.get("sens"), "max").lower(),
                         "unit": _str(r.get("unité")), "aggregation": _str(r.get("agrégation"), "mean") or "mean"})
        if "poids" in r and pd.notna(r["poids"]):
            weights[cid] = float(r["poids"])
    attributes = []
    if "attributs" in norm:
        for _, r in norm["attributs"].iterrows():
            if _str(r.get("id")):
                attributes.append({"id": _str(r["id"]), "name": _str(r.get("nom"), _str(r["id"])),
                                   "unit": _str(r.get("unité")), "aggregation": _str(r.get("agrégation"), "sum") or "sum"})
    cols = [c["id"] for c in criteria] + [a["id"] for a in attributes]
    alternatives = []
    for _, r in alt_df.iterrows():
        aid = _str(r.get("id"))
        if not aid:
            continue
        alternatives.append({"id": aid, "name": _str(r.get("nom"), aid),
                             "values": {c: r[c] for c in cols if c in r and pd.notna(r[c])}})
    data = {"criteria": criteria, "attributes": attributes, "alternatives": alternatives}
    if "contraintes" in norm:
        data["constraints"] = [{"column": _str(r["colonne"]), "op": _str(r["opérateur"]), "value": r["valeur"]}
                               for _, r in norm["contraintes"].iterrows() if _str(r.get("colonne"))]
    params = {}
    if "parametres" in norm:
        params = {_str(r["clé"]): r["valeur"] for _, r in norm["parametres"].iterrows() if _str(r.get("clé"))}
    data["name"] = _str(params.get("nom"), "Problème importé")
    data["description"] = _str(params.get("description"))
    if pd.notna(params.get("portefeuille_min", float("nan"))):
        lo = int(params["portefeuille_min"])
        hi = int(params.get("portefeuille_max", lo)) if pd.notna(params.get("portefeuille_max", float("nan"))) else lo
        data["portfolio"] = {"size": {"min": lo, "max": hi}}
    if "ahp" in norm:
        m = norm["ahp"].set_index(norm["ahp"].columns[0])
        m.index = m.index.astype(str)
        ids = [c["id"] for c in criteria]
        data["weights"] = {"method": "ahp", "matrix": m.loc[ids, ids].astype(str).values.tolist()}
    elif weights:
        data["weights"] = {"method": "direct", "values": weights}
    else:
        data["weights"] = {"method": "equal"}
    return DecisionProblem.from_dict(data)


def write_excel(problem: DecisionProblem, target: str | Path | None = None) -> bytes:
    """Écrit le problème dans le format modèle. Retourne les octets (et écrit `target` si fourni)."""
    from .weights import resolve_weights
    w, ahp_res = resolve_weights(problem)
    crit = pd.DataFrame([{"id": c.id, "nom": c.name, "sens": c.direction, "unité": c.unit, "poids": w[c.id],
                          "agrégation": c.aggregation}
                         for c in problem.criteria])
    cols = [c for c in problem.criterion_ids + problem.attribute_ids if c in problem.alternatives.columns]
    alts = problem.alternatives[["name"] + cols].rename(columns={"name": "nom"}).reset_index()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        crit.to_excel(xw, sheet_name="Critères", index=False)
        alts.to_excel(xw, sheet_name="Alternatives", index=False)
        if problem.attributes:
            pd.DataFrame([{"id": a.id, "nom": a.name, "unité": a.unit, "agrégation": a.aggregation}
                          for a in problem.attributes]).to_excel(xw, sheet_name="Attributs", index=False)
        pd.DataFrame([{"colonne": c.column, "opérateur": c.op, "valeur": c.value} for c in problem.constraints],
                     columns=["colonne", "opérateur", "valeur"]).to_excel(xw, sheet_name="Contraintes", index=False)
        params = [{"clé": "nom", "valeur": problem.name}, {"clé": "description", "valeur": problem.description}]
        if problem.portfolio:
            params += [{"clé": "portefeuille_min", "valeur": problem.portfolio.min_size},
                       {"clé": "portefeuille_max", "valeur": problem.portfolio.max_size}]
        pd.DataFrame(params).to_excel(xw, sheet_name="Paramètres", index=False)
        if ahp_res is not None:
            ahp_res.matrix.to_excel(xw, sheet_name="AHP")
    data = buf.getvalue()
    if target:
        Path(target).write_bytes(data)
    return data
