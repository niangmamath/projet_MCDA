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
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Les poids doivent être positifs, de somme strictement positive.")
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


METHODS = {
    "wsm": weighted_sum,
    "topsis": topsis,
}

METHOD_LABELS = {
    "wsm": "Somme pondérée (WSM/SAW)",
    "topsis": "TOPSIS",
}

METHOD_LOGIC = {
    "wsm": "Agrégation additive totalement compensatoire : une faiblesse se rattrape par une force.",
    "topsis": "Distance à un idéal et à un anti-idéal fictifs ; compensatoire, sensible à la normalisation.",
}
