"""API de haut niveau : appliquer une ou plusieurs méthodes à un problème et comparer les résultats."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .methods import METHOD_LABELS, METHODS, MethodResult
from .problem import DecisionProblem
from .weights import resolve_weights


def method_params(problem: DecisionProblem, method: str, params: dict | None = None) -> dict:
    """Paramètres effectifs d'une méthode (copie des paramètres fournis)."""
    return dict(params or {})


def evaluate(problem: DecisionProblem, method: str = "topsis", weights: pd.Series | dict | None = None,
             **params) -> MethodResult:
    """Évalue un problème avec une méthode. `weights` remplace la spécification du problème."""
    if method not in METHODS:
        raise ValueError(f"Méthode inconnue : {method}. Choix : {list(METHODS)}")
    matrix = problem.decision_matrix()
    if weights is None:
        w, _ = resolve_weights(problem)
    else:
        w = pd.Series(weights, dtype=float)
        w = w / w.sum()
    return METHODS[method](matrix, w, problem.directions, **method_params(problem, method, params))


def run_methods(problem: DecisionProblem, methods: dict[str, dict] | list[str]) -> dict[str, MethodResult]:
    """Exécute plusieurs méthodes : {"wsm": {"normalization": "none"}, "topsis": {}} ou ["wsm", "topsis"]."""
    if isinstance(methods, (list, tuple)):
        methods = {m: {} for m in methods}
    return {m: evaluate(problem, m, **p) for m, p in methods.items()}


def kendall_tau(a: pd.Series, b: pd.Series) -> float:
    """Tau de Kendall (version b, gère les ex aequo) entre deux vecteurs de scores."""
    idx = list(a.index)
    conc = disc = ta = tb = 0
    for i, j in combinations(idx, 2):
        da = np.sign(a[i] - a[j])
        db = np.sign(b[i] - b[j])
        if da == 0 and db == 0:
            continue
        if da == 0:
            ta += 1
        elif db == 0:
            tb += 1
        elif da == db:
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / denom) if denom else 1.0


def spearman_rho(a: pd.Series, b: pd.Series) -> float:
    ra, rb = a.rank(), b.reindex(a.index).rank()
    if ra.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_methods(results: dict[str, MethodResult], labels: pd.Series | None = None) -> dict:
    """Comparaison des classements : tableau des rangs, accord sur le 1er, corrélations de rang."""
    keys = list(results)
    ranks = pd.DataFrame({METHOD_LABELS.get(k, k): results[k].ranking for k in keys})
    if labels is not None:
        ranks.insert(0, "alternative", labels.reindex(ranks.index))
    first = ranks.columns[1] if labels is not None else ranks.columns[0]
    ranks = ranks.sort_values(first)
    names = [METHOD_LABELS.get(k, k) for k in keys]
    tau = pd.DataFrame(np.eye(len(keys)), index=names, columns=names)
    rho = tau.copy()
    for (i, a), (j, b) in combinations(enumerate(keys), 2):
        t = kendall_tau(results[a].scores, results[b].scores)
        r = spearman_rho(results[a].scores, results[b].scores)
        tau.iloc[i, j] = tau.iloc[j, i] = t
        rho.iloc[i, j] = rho.iloc[j, i] = r
    winners = {METHOD_LABELS.get(k, k): results[k].best for k in keys}
    rank_cols = [c for c in ranks.columns if c != "alternative"]
    spread = ranks[rank_cols].max(axis=1) - ranks[rank_cols].min(axis=1)
    return {
        "ranks": ranks,
        "kendall_tau": tau,
        "spearman_rho": rho,
        "winners": winners,
        "same_winner": len(set(winners.values())) == 1,
        "divergent": list(spread[spread > 0].index),
    }
