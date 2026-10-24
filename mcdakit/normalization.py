"""Normalisations de la matrice de décision.

Chaque fonction renvoie (R, oriented) :
  * oriented=True  : R est « plus c'est grand, mieux c'est » pour tous les critères (sens déjà intégré) ;
  * oriented=False : R garde le sens d'origine, la méthode doit traiter les critères à minimiser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NORMALIZATIONS = {
    "none": "Aucune (valeurs brutes, échelle commune requise)",
    "vector": "Vectorielle  x / √Σx²  (standard TOPSIS)",
    "minmax": "Min-max  (x − min) / (max − min)",
    "max": "Linéaire  x / max  (min / x pour un critère à minimiser)",
    "sum": "Somme  x / Σx",
}


def normalize(matrix: pd.DataFrame, directions: pd.Series, method: str = "minmax") -> tuple[pd.DataFrame, bool]:
    x = matrix.astype(float)
    is_min = directions.reindex(x.columns) == "min"
    if method == "none":
        return x.copy(), False
    if method == "vector":
        norm = np.sqrt((x ** 2).sum())
        return x / norm.replace(0, 1), False
    if method == "sum":
        s = x.sum()
        return x / s.replace(0, 1), False
    if method == "minmax":
        lo, hi = x.min(), x.max()
        span = (hi - lo).replace(0, np.nan)
        r = (x - lo) / span
        r.loc[:, is_min] = 1 - r.loc[:, is_min]
        return r.fillna(1.0), True
    if method == "max":
        if (x <= 0).any().any():
            raise ValueError("La normalisation 'max' exige des valeurs strictement positives.")
        r = x / x.max()
        r.loc[:, is_min] = (x.min() / x).loc[:, is_min]
        return r, True
    raise ValueError(f"Normalisation inconnue : {method}. Choix : {list(NORMALIZATIONS)}")
