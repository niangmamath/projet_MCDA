"""Analyse de sensibilité et de robustesse.

Trois familles de tests (cf. consignes MCDA) :
  * paramètres de préférence : balayage d'un poids, seuils de basculement, profils de décideurs,
    simulation de Monte-Carlo autour des poids ;
  * données / hypothèses : scénarios de modification de cellules, seuil de basculement d'une valeur ;
  * paramètres de méthode : comparer normalisations ou méthodes (voir analysis.compare_methods).

Le moteur `WeightScorer` vectorise chaque méthode pour évaluer des milliers de jeux de poids
en une seule opération (les résultats sont identiques à ceux de `methods.py`, cf. tests).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import evaluate, method_params
from .methods import METHODS
from .normalization import normalize
from .problem import DecisionProblem
from .weights import resolve_weights


# =========================================================================== moteur vectorisé
class WeightScorer:
    """Scores de toutes les alternatives pour une matrice K × m de jeux de poids."""

    def __init__(self, problem: DecisionProblem, method: str, params: dict | None = None):
        if method not in METHODS:
            raise ValueError(f"Méthode inconnue : {method}")
        self.problem = problem
        self.method = method
        self.params = method_params(problem, method, params)
        self.matrix = problem.decision_matrix()
        self.alternatives = list(self.matrix.index)
        self.criteria = list(self.matrix.columns)
        self.base_weights, _ = resolve_weights(problem)
        d = problem.directions.reindex(self.criteria)
        self._is_min = (d == "min").to_numpy()
        # validation + pré-calculs via la méthode de référence
        METHODS[method](self.matrix, self.base_weights, d, **self.params)
        default = {"wsm": "minmax", "topsis": "vector"}[method]
        r, self._oriented = normalize(self.matrix, d, self.params.get("normalization", default))
        self._r = r.to_numpy()

    def scores(self, weights) -> np.ndarray:
        """weights : (m,) ou (K, m). Retourne (n,) ou (K, n)."""
        W = np.atleast_2d(np.asarray(weights, dtype=float))
        W = W / W.sum(axis=1, keepdims=True)
        if self.method == "wsm":
            S = W @ self._r.T
        else:  # topsis
            V = W[:, None, :] * self._r[None, :, :]
            vmax, vmin = V.max(axis=1), V.min(axis=1)
            if self._oriented:
                ideal, anti = vmax, vmin
            else:
                ideal = np.where(self._is_min, vmin, vmax)
                anti = np.where(self._is_min, vmax, vmin)
            dp = np.sqrt(((V - ideal[:, None, :]) ** 2).sum(axis=2))
            dm = np.sqrt(((V - anti[:, None, :]) ** 2).sum(axis=2))
            denom = dp + dm
            S = np.divide(dm, denom, out=np.full_like(dm, 0.5), where=denom > 0)
        return S[0] if np.ndim(weights) == 1 else S

    def winner(self, weights) -> str:
        return self.alternatives[int(np.argmax(self.scores(weights)))]


def _weight_grid(base: np.ndarray, k: int, grid: np.ndarray) -> np.ndarray:
    """Version vectorisée de `set_weight` : une ligne de poids par valeur de la grille pour le critère k."""
    others = base.copy()
    others[k] = 0.0
    tot = others.sum()
    share = others / tot if tot > 0 else np.where(np.arange(len(base)) == k, 0.0, 1.0 / (len(base) - 1))
    W = (1.0 - grid)[:, None] * share[None, :]
    W[:, k] = grid
    return W


def set_weight(weights: pd.Series, criterion: str, value: float) -> pd.Series:
    """Fixe le poids d'un critère et redistribue le reste proportionnellement (somme = 1)."""
    w = weights.astype(float).copy()
    others = w.drop(criterion)
    rest = 1.0 - value
    if others.sum() > 0:
        w[others.index] = others / others.sum() * rest
    else:
        w[others.index] = rest / len(others)
    w[criterion] = value
    return w


# =========================================================================== paramètres de préférence
def weight_sweep(problem: DecisionProblem, method: str, criterion: str, params: dict | None = None,
                 steps: int = 101) -> pd.DataFrame:
    """Scores de chaque alternative quand le poids de `criterion` varie de 0 à 1."""
    sc = WeightScorer(problem, method, params)
    grid = np.linspace(0, 1, steps)
    base = sc.base_weights.reindex(sc.criteria).to_numpy()
    W = _weight_grid(base, sc.criteria.index(criterion), grid)
    S = sc.scores(W)
    df = pd.DataFrame(S, index=pd.Index(grid, name=f"poids {criterion}"), columns=sc.alternatives)
    df["gagnant"] = [sc.alternatives[i] for i in S.argmax(axis=1)]
    return df


def _refine(sc: WeightScorer, criterion: str, a: float, b: float, win_a: str, iters: int = 40) -> float:
    base = sc.base_weights.reindex(sc.criteria).to_numpy()
    k = sc.criteria.index(criterion)
    for _ in range(iters):
        m = (a + b) / 2
        if sc.winner(_weight_grid(base, k, np.array([m]))[0]) == win_a:
            a = m
        else:
            b = m
    return (a + b) / 2


def weight_thresholds(problem: DecisionProblem, method: str, params: dict | None = None,
                      resolution: int = 2001) -> pd.DataFrame:
    """Pour chaque critère, poids à partir duquel le gagnant change (à la hausse et à la baisse)."""
    sc = WeightScorer(problem, method, params)
    base_win = sc.winner(sc.base_weights.to_numpy())
    grid = np.linspace(0, 1, resolution)
    base = sc.base_weights.reindex(sc.criteria).to_numpy()
    rows = []
    for k_c, c in enumerate(sc.criteria):
        w0 = float(sc.base_weights[c])
        W = _weight_grid(base, k_c, grid)
        winners = np.array(sc.alternatives)[sc.scores(W).argmax(axis=1)]
        k0 = int(np.searchsorted(grid, w0))
        up = down = None
        for k in range(k0, resolution):
            if grid[k] > w0 and winners[k] != base_win:
                t = _refine(sc, c, max(w0, grid[k - 1]), grid[k], base_win)
                up = (t, winners[k])
                break
        for k in range(min(k0, resolution - 1), -1, -1):
            if grid[k] < w0 and winners[k] != base_win:
                lo = grid[k + 1] if k + 1 < resolution else w0
                t = _refine(sc, c, min(w0, lo), grid[k], base_win)
                down = (t, winners[k])
                break
        rows.append({
            "critère": c,
            "poids actuel": w0,
            "seuil haut": up[0] if up else np.nan,
            "bascule vers (haut)": up[1] if up else "—",
            "seuil bas": down[0] if down else np.nan,
            "bascule vers (bas)": down[1] if down else "—",
        })
    df = pd.DataFrame(rows).set_index("critère")
    rel_up = (df["seuil haut"] - df["poids actuel"]) / df["poids actuel"]
    rel_down = (df["poids actuel"] - df["seuil bas"]) / df["poids actuel"]
    df["variation relative min"] = pd.concat([rel_up, rel_down], axis=1).min(axis=1)
    return df


def weight_profiles(problem: DecisionProblem, method: str, profiles: dict[str, dict],
                    params: dict | None = None) -> pd.DataFrame:
    """Profils de décideurs : {"Financier": {"C2": 2}, ...} = multiplicateurs appliqués aux poids
    de référence (puis renormalisés). Le profil spécial {"__equal__": True} donne des poids égaux."""
    sc = WeightScorer(problem, method, params)
    ref = pd.Series(sc.scores(sc.base_weights.reindex(sc.criteria).to_numpy()), index=sc.alternatives)
    watch = list(ref.sort_values(ascending=False).index[:3])
    rows = []
    for name, mult in {"Référence": {}, **profiles}.items():
        if mult.get("__equal__"):
            w = pd.Series(1.0, index=sc.criteria)
        else:
            w = sc.base_weights.copy()
            for c, f in mult.items():
                w[c] *= float(f)
        w = w / w.sum()
        s = pd.Series(sc.scores(w.reindex(sc.criteria).to_numpy()), index=sc.alternatives)
        row = {"profil": name, "gagnant": s.idxmax(), "score du gagnant": s.max()}
        row.update({f"score {a}": s[a] for a in watch})
        rows.append(row)
    return pd.DataFrame(rows).set_index("profil")


def dirichlet_concentration(weights: pd.Series, spread: float) -> float:
    """Concentration k telle que le coefficient de variation d'un poids moyen vaille `spread`."""
    wbar = float(weights.mean())
    return max((1 - wbar) / (wbar * spread ** 2) - 1, 1e-3)


@dataclass
class MonteCarloResult:
    """Résultat Monte-Carlo : fréquence de 1re place, rang moyen et distribution des rangs."""
    first_place: pd.Series       # fréquence d'être 1er
    mean_rank: pd.Series
    rank_distribution: pd.DataFrame   # alternatives × rangs (fréquences)
    n: int
    spread: float | None

    def table(self, labels: pd.Series | None = None) -> pd.DataFrame:
        df = pd.DataFrame({"fréquence 1er": self.first_place, "rang moyen": self.mean_rank})
        if labels is not None:
            df.insert(0, "alternative", labels.reindex(df.index))
        return df.sort_values(["fréquence 1er", "rang moyen"], ascending=[False, True])


def monte_carlo(problem: DecisionProblem, method: str, params: dict | None = None, n: int = 10000,
                spread: float | None = 0.2, seed: int | None = 42) -> MonteCarloResult:
    """Tirages de poids autour des poids de référence (loi de Dirichlet).

    spread : coefficient de variation visé pour un poids moyen (0,2 ≈ ±20 %).
             None = tirage uniforme sur le simplexe (aucune information sur les poids).
    """
    sc = WeightScorer(problem, method, params)
    rng = np.random.default_rng(seed)
    base = sc.base_weights.reindex(sc.criteria)
    if spread is None:
        alpha = np.ones(len(base))
    else:
        alpha = np.maximum(base.to_numpy() * dirichlet_concentration(base, spread), 1e-6)
    W = rng.dirichlet(alpha, size=n)
    S = sc.scores(W)
    ranks = (-S).argsort(axis=1).argsort(axis=1) + 1
    alts = sc.alternatives
    dist = pd.DataFrame({r: (ranks == r).mean(axis=0) for r in range(1, len(alts) + 1)}, index=alts)
    return MonteCarloResult(pd.Series((ranks == 1).mean(axis=0), index=alts),
                            pd.Series(ranks.mean(axis=0), index=alts), dist, n, spread)


# =========================================================================== données / hypothèses
def data_scenarios(problem: DecisionProblem, method: str, scenarios: dict[str, list],
                   params: dict | None = None) -> pd.DataFrame:
    """scenarios : {"C3 de C : 4 → 3": [("C", "C3", 3)], ...}. L'alternative peut être un élément de
    base (ré-agrégé dans les portefeuilles) ou une alternative générée."""
    ref = evaluate(problem, method, **(params or {}))
    watch = ref.order[:2]
    rows = []
    for name, changes in {"Référence": [], **scenarios}.items():
        p = problem
        for alt, col, val in changes:
            p = p.with_value(alt, col, val)
        try:
            res = evaluate(p, method, **(params or {}))
        except ValueError as exc:
            rows.append({"scénario": name, "gagnant": f"erreur : {exc}"})
            continue
        row = {"scénario": name, "gagnant": res.best,
               "basculement": "oui" if res.best != ref.best else "non"}
        for a in watch:
            row[f"score {a}"] = res.scores.get(a, np.nan)
        excluded = sorted(set(ref.scores.index) - set(res.scores.index))
        row["exclues (contraintes)"] = ", ".join(excluded) if excluded else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("scénario")


def value_threshold(problem: DecisionProblem, method: str, alternative: str, column: str,
                    low: float, high: float, params: dict | None = None, steps: int = 81) -> dict:
    """Plus proche valeur d'une cellule (dans [low, high]) qui change le gagnant."""
    ref = evaluate(problem, method, **(params or {}))
    base_alt = problem.alternatives if alternative in problem.alternatives.index else problem.candidates()
    current = float(base_alt.at[alternative, column])
    values = np.linspace(low, high, steps)
    winners = []
    for v in values:
        try:
            winners.append(evaluate(problem.with_value(alternative, column, v), method, **(params or {})).best)
        except ValueError:
            winners.append(None)
    changes = [(abs(v - current), v, w) for v, w in zip(values, winners) if w is not None and w != ref.best]
    if not changes:
        return {"valeur actuelle": current, "seuil": None, "nouveau gagnant": None, "courbe": list(zip(values, winners))}
    _, v, w = min(changes)
    # affinage par dichotomie entre la valeur actuelle et v
    a, b = current, v
    for _ in range(30):
        m = (a + b) / 2
        if evaluate(problem.with_value(alternative, column, m), method, **(params or {})).best == ref.best:
            a = m
        else:
            b = m
    return {"valeur actuelle": current, "seuil": (a + b) / 2, "nouveau gagnant": w,
            "courbe": list(zip(values, winners))}


# =========================================================================== diagnostic
@dataclass
class RobustnessDiagnosis:
    """Verdict (robuste / modérément robuste / fragile) et raisons qui le justifient."""
    level: str              # "robuste" | "modérément robuste" | "fragile"
    reasons: list[str]
    winner: str

    def __str__(self) -> str:
        return f"Recommandation {self.level} ({self.winner}) : " + " ; ".join(self.reasons)


def diagnose(winner: str, mc: MonteCarloResult | None = None, thresholds: pd.DataFrame | None = None,
             comparison: dict | None = None, scenarios: pd.DataFrame | None = None,
             robust_freq: float = 0.9, fragile_freq: float = 0.6,
             robust_rel: float = 0.5, fragile_rel: float = 0.1) -> RobustnessDiagnosis:
    """Règles explicites (paramétrables) :
      * fragile  : fréquence Monte-Carlo < fragile_freq OU un poids bascule pour < fragile_rel de
                   variation relative OU une majorité des scénarios de données basculent ;
      * robuste  : fréquence ≥ robust_freq ET toute bascule de poids exige ≥ robust_rel de variation ET
                   toutes les méthodes ont le même gagnant ET aucun scénario ne bascule ;
      * sinon    : modérément robuste.
    """
    reasons, fragile, robust = [], False, True
    if mc is not None:
        f = float(mc.first_place.get(winner, 0.0))
        reasons.append(f"1er dans {f:.0%} des tirages de poids")
        fragile |= f < fragile_freq
        robust &= f >= robust_freq
    if thresholds is not None and len(thresholds):
        rel = thresholds["variation relative min"].dropna()
        if len(rel):
            c = rel.idxmin()
            reasons.append(f"plus petit basculement : poids de {c} modifié de {rel.min():.0%}")
            fragile |= rel.min() < fragile_rel
            robust &= rel.min() >= robust_rel
        else:
            reasons.append("aucun poids isolé ne fait basculer le gagnant")
    if comparison is not None:
        same = comparison["same_winner"]
        reasons.append("toutes les méthodes s'accordent sur le 1er" if same else
                       "les méthodes divergent sur le 1er : " + ", ".join(f"{k} → {v}" for k, v in comparison["winners"].items()))
        robust &= same
    if scenarios is not None and "basculement" in scenarios:
        tested = scenarios.drop(index="Référence", errors="ignore")
        k = int((tested["basculement"] == "oui").sum())
        if len(tested):
            reasons.append(f"{k}/{len(tested)} scénarios de données font basculer le choix")
            fragile |= k > len(tested) / 2
            robust &= k == 0
    level = "fragile" if fragile else "robuste" if robust else "modérément robuste"
    return RobustnessDiagnosis(level, reasons, winner)
