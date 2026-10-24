"""Méthodes d'agrégation multicritère.

Toutes les méthodes ont la même signature :

    method(matrix, weights, directions, **params) -> MethodResult

  * matrix     : DataFrame alternatives × critères (valeurs brutes)
  * weights    : Series critère -> poids (somme 1)
  * directions : Series critère -> "max" | "min"

Elles renvoient les scores, le classement et toutes les matrices intermédiaires (`details`)
afin que chaque calcul soit traçable dans un rapport.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .normalization import normalize


@dataclass
class MethodResult:
    method: str
    scores: pd.Series                      # plus grand = meilleur
    score_label: str
    details: dict = field(default_factory=dict)   # nom -> DataFrame / Series / valeur
    params: dict = field(default_factory=dict)

    @property
    def ranking(self) -> pd.Series:
        """Rang (1 = meilleur), ex aequo partagés."""
        return self.scores.rank(ascending=False, method="min").astype(int)

    @property
    def order(self) -> list[str]:
        return list(self.scores.sort_values(ascending=False).index)

    @property
    def best(self) -> str:
        return self.order[0]

    def table(self, labels: pd.Series | None = None) -> pd.DataFrame:
        df = pd.DataFrame({self.score_label: self.scores, "rang": self.ranking})
        if labels is not None:
            df.insert(0, "alternative", labels.reindex(df.index))
        df = df.sort_values("rang")
        top = df[self.score_label].iloc[0]
        df["écart au 1er"] = top - df[self.score_label]
        return df


def _check(matrix: pd.DataFrame, weights: pd.Series, directions: pd.Series):
    crit = list(matrix.columns)
    w = weights.reindex(crit).astype(float)
    if w.isna().any():
        raise ValueError(f"Poids manquants : {list(w[w.isna()].index)}")
    w = w / w.sum()
    return matrix.astype(float), w, directions.reindex(crit)


# --------------------------------------------------------------------------- somme pondérée
def weighted_sum(matrix, weights, directions, normalization: str = "minmax") -> MethodResult:
    """Somme pondérée (WSM / SAW) : S(a) = Σ w_j · r_j(a). Méthode totalement compensatoire."""
    x, w, d = _check(matrix, weights, directions)
    r, oriented = normalize(x, d, normalization)
    if not oriented and (d == "min").any():
        raise ValueError("Somme pondérée : avec des critères à minimiser, utilisez la normalisation "
                         "'minmax' ou 'max' (les autres ne réorientent pas les critères).")
    contrib = r * w
    scores = contrib.sum(axis=1)
    return MethodResult("Somme pondérée", scores, "score",
                        {"Matrice normalisée": r, "Contributions pondérées (w·r)": contrib},
                        {"normalization": normalization})


# --------------------------------------------------------------------------- produit pondéré
def weighted_product(matrix, weights, directions, normalization: str = "max") -> MethodResult:
    """Produit pondéré (WPM) : S(a) = Π r_j(a)^w_j. Compensation plus faible qu'en somme."""
    x, w, d = _check(matrix, weights, directions)
    r, oriented = normalize(x, d, normalization)
    if (r <= 0).any().any():
        raise ValueError("Produit pondéré : valeurs normalisées strictement positives requises "
                         "(évitez 'minmax' qui produit des zéros ; préférez 'max').")
    exps = w.copy()
    if not oriented:
        exps[d == "min"] *= -1
    factors = r ** exps
    scores = factors.prod(axis=1)
    return MethodResult("Produit pondéré", scores, "score",
                        {"Matrice normalisée": r, "Facteurs r^w": factors},
                        {"normalization": normalization})


# --------------------------------------------------------------------------- TOPSIS
def topsis(matrix, weights, directions, normalization: str = "vector") -> MethodResult:
    """TOPSIS : proximité relative à la solution idéale, Ci* = D⁻ / (D⁺ + D⁻)."""
    x, w, d = _check(matrix, weights, directions)
    r, oriented = normalize(x, d, normalization)
    v = r * w
    if oriented:
        ideal, anti = v.max(), v.min()
    else:
        ideal = pd.Series({c: v[c].min() if d[c] == "min" else v[c].max() for c in v.columns})
        anti = pd.Series({c: v[c].max() if d[c] == "min" else v[c].min() for c in v.columns})
    d_plus = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - anti) ** 2).sum(axis=1))
    denom = (d_plus + d_minus).replace(0, np.nan)
    ci = (d_minus / denom).fillna(0.5)
    refs = pd.DataFrame({"A⁺ (idéal)": ideal, "A⁻ (anti-idéal)": anti}).T
    dist = pd.DataFrame({"D⁺": d_plus, "D⁻": d_minus, "Ci*": ci})
    return MethodResult("TOPSIS", ci, "Ci*",
                        {"Matrice normalisée R": r, "Matrice pondérée V": v,
                         "Solutions de référence": refs, "Distances": dist},
                        {"normalization": normalization})


# --------------------------------------------------------------------------- PROMETHEE
PREFERENCE_FUNCTIONS = {
    "usual": "Usuelle : préférence stricte dès que d > 0",
    "ushape": "En U : préférence stricte si d > q",
    "vshape": "En V : linéaire de 0 à p",
    "level": "Par paliers : 0 si d ≤ q, ½ si q < d ≤ p, 1 sinon",
    "linear": "Linéaire avec indifférence : 0 si d ≤ q, linéaire jusqu'à p",
    "gaussian": "Gaussienne de paramètre s",
}


def preference_degree(dvals: np.ndarray, spec: dict, span: float) -> np.ndarray:
    """Degré de préférence P(d) ∈ [0, 1] pour des écarts orientés d."""
    kind = spec.get("type", "usual")
    q = float(spec.get("q", 0.0))
    p = float(spec.get("p", span if span > 0 else 1.0))
    s = float(spec.get("s", span / 2 if span > 0 else 1.0))
    dv = np.asarray(dvals, dtype=float)
    if kind == "usual":
        return (dv > 0).astype(float)
    if kind == "ushape":
        return (dv > q).astype(float)
    if kind == "vshape":
        return np.clip(dv / p, 0, 1) if p > 0 else (dv > 0).astype(float)
    if kind == "level":
        return np.where(dv > p, 1.0, np.where(dv > q, 0.5, 0.0))
    if kind == "linear":
        if p <= q:
            return (dv > q).astype(float)
        return np.clip((dv - q) / (p - q), 0, 1)
    if kind == "gaussian":
        return np.where(dv > 0, 1 - np.exp(-(dv ** 2) / (2 * s ** 2)), 0.0)
    raise ValueError(f"Fonction de préférence inconnue : {kind}")


def promethee(matrix, weights, directions, preferences: dict | None = None) -> MethodResult:
    """PROMETHEE II : flux net φ = φ⁺ − φ⁻ issu des degrés de préférence par paires.

    Méthode de surclassement : un gros écart sur un critère ne compense qu'en partie (selon la
    fonction de préférence) et les seuils q / p modélisent l'indifférence et la préférence stricte.
    `preferences` : {critère: {"type": "linear", "q": 0, "p": 10}}, défaut « usuelle ».
    """
    x, w, d = _check(matrix, weights, directions)
    preferences = preferences or {}
    alts = list(x.index)
    n = len(alts)
    pi = np.zeros((n, n))
    per_criterion = {}
    for c in x.columns:
        g = x[c].to_numpy()
        diff = g[:, None] - g[None, :]
        if d[c] == "min":
            diff = -diff
        span = float(g.max() - g.min())
        P = preference_degree(diff, preferences.get(c, {"type": "usual"}), span)
        np.fill_diagonal(P, 0.0)
        per_criterion[c] = pd.DataFrame(P, index=alts, columns=alts)
        pi += w[c] * P
    denom = max(n - 1, 1)
    phi_plus = pd.Series(pi.sum(axis=1) / denom, index=alts)
    phi_minus = pd.Series(pi.sum(axis=0) / denom, index=alts)
    net = phi_plus - phi_minus
    flows = pd.DataFrame({"φ⁺": phi_plus, "φ⁻": phi_minus, "φ net": net})
    # PROMETHEE I : paires incomparables (φ⁺ et φ⁻ en désaccord)
    incomparable = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = alts[i], alts[j]
            sp = np.sign(round(phi_plus[a] - phi_plus[b], 12))
            sm = np.sign(round(phi_minus[b] - phi_minus[a], 12))
            if sp * sm < 0:
                incomparable.append((a, b))
    details = {"Indices de préférence π(a,b)": pd.DataFrame(pi, index=alts, columns=alts),
               "Flux": flows, "Incomparabilités PROMETHEE I": incomparable}
    details.update({f"P_{c}(a,b)": m for c, m in per_criterion.items()})
    return MethodResult("PROMETHEE II", net, "φ net", details, {"preferences": preferences})


METHODS = {
    "wsm": weighted_sum,
    "wpm": weighted_product,
    "topsis": topsis,
    "promethee": promethee,
}

METHOD_LABELS = {
    "wsm": "Somme pondérée (WSM/SAW)",
    "wpm": "Produit pondéré (WPM)",
    "topsis": "TOPSIS",
    "promethee": "PROMETHEE II",
}

METHOD_LOGIC = {
    "wsm": "Agrégation additive totalement compensatoire : une faiblesse se rattrape par une force.",
    "wpm": "Agrégation multiplicative : compensation partielle, pénalise les profils déséquilibrés.",
    "topsis": "Distance à un idéal et à un anti-idéal fictifs ; compensatoire, sensible à la normalisation.",
    "promethee": "Surclassement par paires avec seuils d'indifférence/préférence ; compensation limitée.",
}
