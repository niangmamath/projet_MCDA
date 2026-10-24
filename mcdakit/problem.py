"""Modèle de données d'un problème de décision multicritère.

Un problème est décrit par :
  * des critères (sens max/min, règle d'agrégation pour les portefeuilles) ;
  * des attributs optionnels (colonnes qui ne sont pas des critères mais servent aux contraintes, ex. budget) ;
  * des alternatives avec leurs valeurs ;
  * une spécification de poids (directs, égaux, rangs, AHP) ;
  * un mode « portefeuille » optionnel : les alternatives réelles sont alors les combinaisons de k éléments ;
  * des contraintes (filtres d'admissibilité) et des surcharges de cellules (ex. corrections Delphi).

Le format d'échange est un dictionnaire JSON/YAML (voir docs/FORMAT.md).
"""
from __future__ import annotations

import copy
import itertools
import json
import operator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AGGREGATIONS = {
    "sum": np.sum,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "product": np.prod,
}

OPERATORS = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}

MAX_PORTFOLIOS = 20000


def to_number(value: Any) -> float:
    """Convertit 3, "3", "1/3", "0,5" en float."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    return float(Fraction(text)) if "/" in text else float(text)


@dataclass
class Criterion:
    """Critère de décision : identifiant, libellé, sens (max/min), unité et règle d'agrégation en portefeuille."""
    id: str
    name: str = ""
    direction: str = "max"          # "max" (plus c'est haut mieux c'est) ou "min"
    unit: str = ""
    aggregation: str = "mean"       # utilisée seulement en mode portefeuille

    def __post_init__(self):
        self.name = self.name or self.id
        if self.direction not in ("max", "min"):
            raise ValueError(f"Critère {self.id}: sens '{self.direction}' invalide (attendu 'max' ou 'min').")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"Critère {self.id}: agrégation '{self.aggregation}' inconnue {list(AGGREGATIONS)}.")


@dataclass
class Attribute:
    """Colonne hors critères (budget, charge…) : n'entre pas dans le score, sert aux contraintes."""
    id: str
    name: str = ""
    unit: str = ""
    aggregation: str = "sum"

    def __post_init__(self):
        self.name = self.name or self.id
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"Attribut {self.id}: agrégation '{self.aggregation}' inconnue.")


@dataclass
class Constraint:
    """Filtre d'admissibilité « colonne opérateur valeur » (ex. JH <= 65)."""
    column: str
    op: str
    value: float

    def __post_init__(self):
        if self.op not in OPERATORS:
            raise ValueError(f"Opérateur de contrainte '{self.op}' invalide {list(OPERATORS)}.")
        self.value = to_number(self.value)

    def check(self, x: float) -> bool:
        """Vrai si la valeur x respecte la contrainte."""
        return bool(OPERATORS[self.op](x, self.value))

    def __str__(self) -> str:
        return f"{self.column} {self.op} {self.value:g}"


@dataclass
class Portfolio:
    """Mode portefeuille : les alternatives deviennent des combinaisons de min_size à max_size éléments."""
    min_size: int
    max_size: int
    separator: str = "+"

    @classmethod
    def from_spec(cls, spec: dict) -> "Portfolio":
        """Construit le portefeuille depuis {"size": k}, {"size": {"min": a, "max": b}} ou {"size": [a, b]}."""
        size = spec.get("size", 2)
        if isinstance(size, dict):
            lo, hi = int(size.get("min", 1)), int(size.get("max", size.get("min", 1)))
        elif isinstance(size, (list, tuple)):
            lo, hi = int(size[0]), int(size[1])
        else:
            lo = hi = int(size)
        if lo < 1 or hi < lo:
            raise ValueError("Portefeuille : tailles invalides.")
        return cls(lo, hi, spec.get("separator", "+"))

    def to_spec(self) -> dict:
        """Forme sérialisable (inverse de from_spec)."""
        size = self.min_size if self.min_size == self.max_size else {"min": self.min_size, "max": self.max_size}
        return {"size": size, "separator": self.separator}


@dataclass
class DecisionProblem:
    """Problème de décision complet : critères, alternatives et évaluations, poids, contraintes,
    portefeuille et surcharges. Point d'entrée de toute la bibliothèque (voir docs/FORMAT.md)."""
    criteria: list[Criterion]
    alternatives: pd.DataFrame                  # index = id, colonnes = name + critères + attributs
    weights_spec: dict = field(default_factory=lambda: {"method": "equal"})
    attributes: list[Attribute] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    portfolio: Portfolio | None = None
    overrides: dict = field(default_factory=dict)  # {alternative: {colonne: valeur}}
    name: str = "Problème de décision"
    description: str = ""
    analysis: dict = field(default_factory=dict)   # réglages suggérés (méthodes, méthode de référence), facultatif

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, data: dict) -> "DecisionProblem":
        """Construit un problème depuis un dictionnaire au format docs/FORMAT.md (contenu d'un JSON/YAML)."""
        crit_keys = {"id", "name", "direction", "unit", "aggregation"}
        criteria = [Criterion(**{k: v for k, v in c.items() if k in crit_keys}) for c in data["criteria"]]
        attributes = [Attribute(**a) for a in data.get("attributes", [])]
        rows = []
        for alt in data["alternatives"]:
            row = {"id": str(alt["id"]), "name": alt.get("name", str(alt["id"]))}
            row.update({k: to_number(v) for k, v in alt.get("values", {}).items()})
            rows.append(row)
        alts = pd.DataFrame(rows).set_index("id")
        weights = data.get("weights", {"method": "equal"})
        if isinstance(weights, dict) and "method" not in weights:
            weights = {"method": "direct", "values": weights}
        problem = cls(
            criteria=criteria,
            alternatives=alts,
            weights_spec=weights,
            attributes=attributes,
            constraints=[Constraint(**c) for c in data.get("constraints", [])],
            portfolio=Portfolio.from_spec(data["portfolio"]) if data.get("portfolio") else None,
            overrides={str(k): dict(v) for k, v in data.get("overrides", {}).items()},
            name=data.get("name", "Problème de décision"),
            description=data.get("description", ""),
            analysis=dict(data.get("analysis", {})),
        )
        problem.validate()
        return problem

    @classmethod
    def load(cls, path: str | Path) -> "DecisionProblem":
        """Charge un projet .json, .yaml/.yml ou .xlsx."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml
            return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        if suffix in (".xlsx", ".xls"):
            from .io_excel import read_excel
            return read_excel(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        """Sérialise le problème au format docs/FORMAT.md (inverse de from_dict)."""
        cols = self.criterion_ids + self.attribute_ids
        alts = []
        for alt_id, row in self.alternatives.iterrows():
            values = {c: _clean(row[c]) for c in cols if c in row and pd.notna(row[c])}
            alts.append({"id": alt_id, "name": row.get("name", alt_id), "values": values})
        data = {
            "name": self.name,
            "description": self.description,
            "criteria": [vars(c).copy() for c in self.criteria],
            "attributes": [vars(a).copy() for a in self.attributes],
            "alternatives": alts,
            "weights": self.weights_spec,
            "constraints": [vars(c).copy() for c in self.constraints],
            "overrides": self.overrides,
            **({"analysis": self.analysis} if self.analysis else {}),
        }
        if self.portfolio:
            data["portfolio"] = self.portfolio.to_spec()
        return data

    def save(self, path: str | Path) -> None:
        """Enregistre le projet en JSON, YAML ou Excel selon l'extension."""
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def copy(self) -> "DecisionProblem":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ accès
    @property
    def criterion_ids(self) -> list[str]:
        return [c.id for c in self.criteria]

    @property
    def attribute_ids(self) -> list[str]:
        return [a.id for a in self.attributes]

    @property
    def directions(self) -> pd.Series:
        return pd.Series({c.id: c.direction for c in self.criteria})

    def criterion(self, cid: str) -> Criterion:
        for c in self.criteria:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def validate(self) -> None:
        ids = self.criterion_ids + self.attribute_ids
        if len(set(ids)) != len(ids):
            raise ValueError("Identifiants de critères/attributs dupliqués.")
        if not self.criteria:
            raise ValueError("Il faut au moins un critère.")
        if len(self.alternatives) < 1:
            raise ValueError("Il faut au moins une alternative.")
        if self.alternatives.index.duplicated().any():
            raise ValueError("Identifiants d'alternatives dupliqués.")
        missing = [c for c in self.criterion_ids if c not in self.alternatives.columns]
        if missing:
            raise ValueError(f"Valeurs manquantes pour les critères : {missing}.")
        if self.alternatives[self.criterion_ids].isna().any().any():
            bad = self.alternatives[self.criterion_ids].isna().stack()
            raise ValueError(f"Cellules vides : {list(bad[bad].index)}.")
        for c in self.constraints:
            if c.column not in ids:
                raise ValueError(f"Contrainte sur une colonne inconnue : {c.column}.")
        if self.portfolio and self.portfolio.max_size > len(self.alternatives):
            raise ValueError("Taille de portefeuille supérieure au nombre d'éléments.")

    # ------------------------------------------------------------------ matrice de décision
    def candidates(self) -> pd.DataFrame:
        """Toutes les alternatives candidates (avant contraintes), surcharges appliquées.

        En mode portefeuille, chaque combinaison de k éléments est agrégée colonne par colonne.
        """
        cols = self.criterion_ids + [a for a in self.attribute_ids if a in self.alternatives.columns]
        base = self.alternatives
        if self.portfolio is None:
            table = base[["name"] + cols].copy()
        else:
            p = self.portfolio
            rules = {c.id: c.aggregation for c in self.criteria}
            rules.update({a.id: a.aggregation for a in self.attributes})
            n = len(base)
            total = sum(_comb(n, k) for k in range(p.min_size, p.max_size + 1))
            if total > MAX_PORTFOLIOS:
                raise ValueError(f"{total} combinaisons : trop pour une énumération exhaustive (max {MAX_PORTFOLIOS}).")
            rows = {}
            for k in range(p.min_size, p.max_size + 1):
                for combo in itertools.combinations(base.index, k):
                    sub = base.loc[list(combo)]
                    row = {"name": f" {p.separator} ".join(str(base.at[i, "name"]) for i in combo),
                           "members": list(combo)}
                    for col in cols:
                        row[col] = float(AGGREGATIONS[rules[col]](sub[col].to_numpy(dtype=float)))
                    rows[p.separator.join(map(str, combo))] = row
            table = pd.DataFrame.from_dict(rows, orient="index")
        for alt, changes in self.overrides.items():
            if alt not in table.index:
                raise ValueError(f"Surcharge sur une alternative inconnue : {alt}.")
            for col, val in changes.items():
                table.at[alt, col] = to_number(val)
        return table

    def feasibility(self, table: pd.DataFrame | None = None) -> pd.DataFrame:
        """Tableau des contraintes : une colonne booléenne par contrainte + 'admissible'."""
        table = self.candidates() if table is None else table
        out = pd.DataFrame(index=table.index)
        for c in self.constraints:
            out[str(c)] = table[c.column].apply(c.check)
        out["admissible"] = out.all(axis=1) if self.constraints else True
        return out

    def decision_matrix(self) -> pd.DataFrame:
        """Matrice alternatives admissibles × critères, prête pour les méthodes."""
        table = self.candidates()
        feas = self.feasibility(table)
        matrix = table.loc[feas["admissible"], self.criterion_ids].astype(float)
        if matrix.empty:
            raise ValueError("Aucune alternative ne respecte les contraintes.")
        return matrix

    def labels(self) -> pd.Series:
        return self.candidates()["name"]

    # ------------------------------------------------------------------ modifications (scénarios)
    def with_value(self, alternative: str, column: str, value: float) -> "DecisionProblem":
        """Copie du problème où une cellule est modifiée.

        `alternative` peut être un élément de base (sa valeur est alors ré-agrégée dans tous les
        portefeuilles qui le contiennent) ou une alternative générée (surcharge directe).
        """
        p = self.copy()
        if alternative in p.alternatives.index:
            p.alternatives.at[alternative, column] = float(value)
        else:
            p.overrides.setdefault(alternative, {})[column] = float(value)
        return p

    def with_weights(self, weights: dict | pd.Series) -> "DecisionProblem":
        p = self.copy()
        p.weights_spec = {"method": "direct", "values": {k: float(v) for k, v in dict(weights).items()}}
        return p


def _comb(n: int, k: int) -> int:
    from math import comb
    return comb(n, k)


def _clean(v):
    if isinstance(v, (np.floating, float)) and float(v).is_integer():
        return int(v)
    if isinstance(v, np.generic):
        return v.item()
    return v
