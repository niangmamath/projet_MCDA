"""Détermination des poids des critères : directs, égaux, par rangs (ROC, RS) et AHP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .problem import DecisionProblem, to_number

# Indices aléatoires de Saaty (n = 1..15)
RANDOM_INDEX = [0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59]

SAATY_SCALE = {
    1: "Importance égale",
    3: "Importance modérée",
    5: "Importance forte",
    7: "Importance très forte",
    9: "Importance extrême",
}


def normalize(weights: dict | pd.Series) -> pd.Series:
    """Ramène des poids positifs quelconques à une somme de 1 (erreur si négatifs ou de somme nulle)."""
    w = pd.Series(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Les poids doivent être positifs.")
    if w.sum() <= 0:
        raise ValueError("La somme des poids doit être strictement positive.")
    return w / w.sum()


def equal_weights(criteria: list[str]) -> pd.Series:
    """Poids égaux 1/n."""
    return pd.Series(1.0 / len(criteria), index=criteria)


def rank_order_centroid(order: list[str]) -> pd.Series:
    """ROC : poids issus d'un simple classement des critères (du plus au moins important)."""
    n = len(order)
    return pd.Series([sum(1.0 / k for k in range(i, n + 1)) / n for i in range(1, n + 1)], index=order)


def rank_sum(order: list[str]) -> pd.Series:
    """Poids « rank-sum » : proportionnels à n − rang + 1, normalisés."""
    n = len(order)
    raw = pd.Series([n - i + 1 for i in range(1, n + 1)], index=order, dtype=float)
    return raw / raw.sum()


@dataclass
class AHPResult:
    """Résultat AHP : matrice, poids, λmax, indice (IC) et ratio (RC) de cohérence."""
    weights: pd.Series
    matrix: pd.DataFrame
    lambda_max: float
    ci: float
    ri: float
    cr: float
    method: str

    @property
    def consistent(self) -> bool:
        """Vrai si RC < 10 % (seuil de Saaty)."""
        return self.cr < 0.10

    def worst_judgements(self, top: int = 3) -> pd.DataFrame:
        """Jugements les plus éloignés du ratio impliqué par les poids (pistes pour réviser l'AHP)."""
        w = self.weights.to_numpy()
        a = self.matrix.to_numpy()
        n = len(w)
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                implied = w[i] / w[j]
                rows.append({
                    "i": self.matrix.index[i], "j": self.matrix.index[j],
                    "jugement": a[i, j], "ratio_implicite": implied,
                    "ecart_log": abs(np.log(a[i, j] / implied)),
                })
        return pd.DataFrame(rows).sort_values("ecart_log", ascending=False).head(top).reset_index(drop=True)


def ahp(matrix, labels: list[str] | None = None, method: str = "eigen") -> AHPResult:
    """Poids AHP à partir d'une matrice de comparaisons par paires (réciproque).

    `matrix` : liste de listes (nombres ou chaînes "1/3") ou DataFrame. method : "eigen" (vecteur
    propre principal, Saaty) ou "geometric" (moyenne géométrique des lignes).
    """
    if isinstance(matrix, pd.DataFrame):
        labels = labels or list(matrix.index)
        a = matrix.map(to_number).to_numpy(dtype=float)
    else:
        a = np.array([[to_number(x) for x in row] for row in matrix], dtype=float)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("La matrice AHP doit être carrée.")
    if (a <= 0).any():
        raise ValueError("La matrice AHP doit être strictement positive.")
    if not np.allclose(a * a.T, 1.0, atol=1e-6):
        raise ValueError("La matrice AHP doit être réciproque (a_ji = 1/a_ij).")
    labels = labels or [f"C{i + 1}" for i in range(n)]

    if method == "eigen":
        vals, vecs = np.linalg.eig(a)
        k = int(np.argmax(vals.real))
        w = np.abs(vecs[:, k].real)
        w = w / w.sum()
        lam = float(vals[k].real)
    elif method == "geometric":
        g = np.prod(a, axis=1) ** (1.0 / n)
        w = g / g.sum()
        lam = float(np.mean((a @ w) / w))
    else:
        raise ValueError("method doit être 'eigen' ou 'geometric'.")

    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX[n - 1] if n <= len(RANDOM_INDEX) else RANDOM_INDEX[-1]
    cr = ci / ri if ri > 0 else 0.0
    return AHPResult(pd.Series(w, index=labels), pd.DataFrame(a, index=labels, columns=labels),
                     lam, ci, ri, cr, method)


def ahp_from_judgements(labels: list[str], judgements: dict) -> AHPResult:
    """Construit la matrice à partir des seuls jugements au-dessus de la diagonale.

    judgements : {("C1", "C2"): 3, ...} ou {"C1>C2": "1/3", ...}
    """
    idx = {c: i for i, c in enumerate(labels)}
    a = np.ones((len(labels), len(labels)))
    for key, val in judgements.items():
        i, j = key if isinstance(key, tuple) else key.split(">")
        v = to_number(val)
        a[idx[i], idx[j]] = v
        a[idx[j], idx[i]] = 1.0 / v
    return ahp(a, labels)


def resolve_weights(problem: DecisionProblem) -> tuple[pd.Series, AHPResult | None]:
    """Applique la spécification de poids d'un problème. Retourne (poids normalisés, détail AHP)."""
    spec = problem.weights_spec or {"method": "equal"}
    method = spec.get("method", "direct")
    crit = problem.criterion_ids
    ahp_result = None
    if method == "equal":
        w = equal_weights(crit)
    elif method == "direct":
        values = spec["values"]
        missing = [c for c in crit if c not in values]
        if missing:
            raise ValueError(f"Poids manquants pour : {missing}.")
        w = normalize({c: to_number(values[c]) for c in crit})
    elif method in ("roc", "rank_sum"):
        order = spec["order"]
        if sorted(order) != sorted(crit):
            raise ValueError("L'ordre des critères doit contenir chaque critère une seule fois.")
        w = (rank_order_centroid if method == "roc" else rank_sum)(order)
    elif method == "ahp":
        if "judgements" in spec:
            ahp_result = ahp_from_judgements(crit, spec["judgements"])
        else:
            ahp_result = ahp(spec["matrix"], crit, spec.get("variant", "eigen"))
        w = ahp_result.weights
    else:
        raise ValueError(f"Méthode de pondération inconnue : {method}.")
    return w.reindex(crit), ahp_result
