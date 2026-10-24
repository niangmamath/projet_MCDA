"""Cas de référence publiés, utilisés pour valider mcdakit.

Chaque cas reprend les données et le résultat imprimés dans la source (article ou ouvrage). Ces cas sont
repris des suites de tests de bibliothèques MCDA établies (pymcdm 1.4, pyrepo-mcda 0.1.15), qui les citent ;
chaque cas renvoie à l'article par son DOI (lien https://doi.org/…).
`run_benchmarks()` recalcule chaque cas avec mcdakit et compare au résultat publié, à la précision de la
publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .methods import topsis, weighted_sum
from .weights import ahp


@dataclass
class Benchmark:
    """Cas publié : données, résultat imprimé, précision et référence (DOI)."""

    key: str
    method: str
    source: str
    doi: str
    published: list[float]
    decimals: int
    compute: Callable[[], np.ndarray]
    note: str = ""
    tolerance: float | None = None  # défaut : une demi-unité de la dernière décimale publiée

    @property
    def tol(self) -> float:
        return self.tolerance if self.tolerance is not None else 0.5 * 10**-self.decimals


def _frame(matrix, types):
    m = pd.DataFrame(np.asarray(matrix, dtype=float))
    m.columns = [f"C{j + 1}" for j in range(m.shape[1])]
    m.index = [f"A{i + 1}" for i in range(m.shape[0])]
    d = pd.Series(["max" if t > 0 else "min" for t in types], index=m.columns)
    return m, d


def _run(fn, matrix, weights, types, **kw) -> np.ndarray:
    m, d = _frame(matrix, types)
    w = np.asarray(weights, dtype=float)
    return fn(m, pd.Series(w / w.sum(), index=m.columns), d, **kw).scores.to_numpy()


# --------------------------------------------------------------------------- données publiées
ERSOY_2021 = [
    [256, 8, 41, 1.6, 1.77, 7347.16],
    [256, 8, 32, 1.0, 1.8, 6919.99],
    [256, 8, 53, 1.6, 1.9, 8400],
    [256, 8, 41, 1.0, 1.75, 6808.9],
    [512, 8, 35, 1.6, 1.7, 8479.99],
    [256, 4, 35, 1.6, 1.7, 7499.99],
]
WANG_CHIN_2011 = [
    [1, 4, 3, 1, 3, 4],
    [1 / 4, 1, 7, 3, 1 / 5, 1],
    [1 / 3, 1 / 7, 1, 1 / 5, 1 / 5, 1 / 6],
    [1, 1 / 3, 5, 1, 1, 1 / 3],
    [1 / 3, 5, 5, 1, 1, 3],
    [1 / 4, 1, 6, 3, 1 / 3, 1],
]
PAPATHANASIOU_AHP_CRIT = [[1, 1, 5, 3], [1, 1, 5, 3], [1 / 5, 1 / 5, 1, 1 / 3], [1 / 3, 1 / 3, 3, 1]]
PAPATHANASIOU_AHP_ALT = [
    [
        [1, 5, 1, 1, 1 / 3, 3],
        [1 / 5, 1, 1 / 3, 1 / 5, 1 / 7, 1],
        [1, 3, 1, 1 / 3, 1 / 5, 1],
        [1, 5, 3, 1, 1 / 3, 3],
        [3, 7, 5, 3, 1, 7],
        [1 / 3, 1, 1, 1 / 3, 1 / 7, 1],
    ],
    [
        [1, 7, 3, 1 / 3, 1 / 3, 1 / 3],
        [1 / 7, 1, 1 / 3, 1 / 7, 1 / 9, 1 / 7],
        [1 / 3, 3, 1, 1 / 5, 1 / 5, 1 / 5],
        [3, 7, 5, 1, 1, 1],
        [3, 9, 5, 1, 1, 1],
        [3, 7, 5, 1, 1, 1],
    ],
    [
        [1, 1 / 9, 1 / 7, 1 / 9, 1, 1 / 5],
        [9, 1, 1, 1, 5, 3],
        [7, 1, 1, 1, 5, 1],
        [9, 1, 1, 1, 7, 3],
        [1, 1 / 5, 1 / 5, 1 / 7, 1, 1 / 3],
        [5, 1 / 3, 1, 1 / 3, 3, 1],
    ],
    [
        [1, 1 / 5, 1 / 5, 1 / 3, 1 / 7, 1 / 5],
        [5, 1, 1, 3, 1 / 3, 1],
        [5, 1, 1, 1, 1 / 3, 1],
        [3, 1 / 3, 1, 1, 1 / 7, 1],
        [7, 3, 3, 7, 1, 5],
        [5, 1, 1, 1, 1 / 5, 1],
    ],
]
RIZAN_2020 = [
    [0.75, 0.50, 0.75, 0, 0, 0, 1],
    [0.75, 1, 0.75, 0, 0, 0, 0.75],
    [0.75, 0.75, 0.75, 0, 0.50, 0.25, 1],
    [0.50, 0.50, 0.75, 1, 0.50, 0, 0.75],
]


def _papathanasiou_ahp():
    crit = ahp(PAPATHANASIOU_AHP_CRIT).weights.to_numpy()
    local = np.column_stack([ahp(m).weights.to_numpy() for m in PAPATHANASIOU_AHP_ALT])
    return _run(weighted_sum, local, crit, [1] * 4, normalization="none")


BENCHMARKS: list[Benchmark] = [
    Benchmark(
        "opricovic2004",
        "TOPSIS (normalisation min-max)",
        "Opricovic & Tzeng (2004), Compromise solution by MCDM methods: a comparative analysis of VIKOR and TOPSIS. European Journal of Operational Research 156(2)",
        "10.1016/S0377-2217(03)00020-1",
        [0.500, 0.617, 0.500],
        3,
        lambda: _run(topsis, np.array([[1, 2, 5], [3000, 3750, 4500]]).T, [0.5, 0.5], [-1, 1], normalization="minmax"),
        "Les valeurs publiées sont obtenues avec la normalisation min-max.",
    ),
    Benchmark(
        "ersoy2021",
        "TOPSIS (normalisation vectorielle)",
        "Ersoy (2021), Equipment selection for an e-commerce company using entropy-based TOPSIS, EDAS and CODAS methods during the COVID-19. LogForum 17(3)",
        "10.17270/J.LOG.2021.603",
        [0.308, 0.254, 0.327, 0.263, 0.857, 0.192],
        3,
        lambda: _run(topsis, ERSOY_2021, [0.405, 0.221, 0.134, 0.199, 0.007, 0.034], [1, 1, 1, 1, -1, -1]),
        "Tolérance 0,001 : les poids entropiques sont publiés arrondis à 3 décimales.",
        tolerance=0.001,
    ),
    Benchmark(
        "wangchin2011",
        "AHP — poids par vecteur propre",
        "Wang & Chin (2011), A linear programming approximation to the eigenvector method in the analytic hierarchy process. Information Sciences 181(23)",
        "10.1016/j.ins.2011.07.009",
        [0.3208, 0.1395, 0.0348, 0.1285, 0.2374, 0.1391],
        4,
        lambda: ahp(WANG_CHIN_2011).weights.to_numpy(),
    ),
    Benchmark(
        "papathanasiou2018_ahp",
        "AHP complet (critères et alternatives par paires)",
        "Papathanasiou & Ploskas (2018), AHP. In Multiple Criteria Decision Aid, Springer",
        "10.1007/978-3-319-91648-4_5",
        [0.117, 0.071, 0.095, 0.212, 0.350, 0.155],
        3,
        _papathanasiou_ahp,
        "Priorités locales et poids par mcdakit.ahp, synthèse par somme pondérée sans normalisation.",
    ),
    Benchmark(
        "rizan2020",
        "Somme pondérée (SAW, normalisation x/max)",
        "Rizan, Wahyuningsih, Pradana & Ramadella (2020), SAW method in supporting the process of admission of new junior high school students. CITSM 2020, IEEE",
        "10.1109/CITSM50537.2020.9268874",
        [0.350, 0.375, 0.825, 0.642],
        3,
        lambda: _run(weighted_sum, RIZAN_2020, [0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0.1], [1] * 7, normalization="max"),
    ),
]


def _rank(v) -> list:
    return pd.Series(v).rank(ascending=False, method="min").tolist()


def run_benchmarks() -> pd.DataFrame:
    """Recalcule chaque cas publié et le compare au résultat imprimé (un tableau, une ligne par cas)."""
    rows = []
    for b in BENCHMARKS:
        got = np.asarray(b.compute(), dtype=float)
        pub = np.asarray(b.published, dtype=float)
        diff = float(np.max(np.abs(got - pub)))
        rows.append(
            {
                "cas": b.key,
                "méthode": b.method,
                "source": b.source,
                "doi": b.doi,
                "url": f"https://doi.org/{b.doi}" if b.doi and b.doi != "—" else "",
                "publié": list(pub),
                "mcdakit": [round(float(x), b.decimals) for x in got],
                "écart max": diff,
                "tolérance": b.tol,
                "même classement": _rank(np.round(got, b.decimals)) == _rank(pub),
                "conforme": diff <= b.tol + 1e-12,
                "note": b.note,
            }
        )
    return pd.DataFrame(rows)
