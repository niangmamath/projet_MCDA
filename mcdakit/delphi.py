"""Agrégation de consultations Delphi : médiane, quartiles, consensus et stabilité entre tours."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .problem import to_number


def parse_votes(votes) -> list[float]:
    """Accepte [2, 1, 3] ou "2 · 1 · 3" / "2,1,3" / "2;1;3"."""
    if isinstance(votes, str):
        for sep in ("·", ";", "|", " "):
            votes = votes.replace(sep, ",")
        votes = [v for v in votes.split(",") if v.strip()]
    return [to_number(v) for v in votes]


@dataclass
class DelphiRound:
    votes: list[float]
    median: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


@dataclass
class DelphiResult:
    rounds: list[DelphiRound]
    iqr_threshold: float

    @property
    def final(self) -> DelphiRound:
        return self.rounds[-1]

    @property
    def value(self) -> float:
        """Valeur retenue : médiane du dernier tour."""
        return self.final.median

    @property
    def consensus(self) -> bool:
        return self.final.iqr <= self.iqr_threshold

    @property
    def stable(self) -> bool:
        return len(self.rounds) >= 2 and self.rounds[-1].median == self.rounds[-2].median

    @property
    def status(self) -> str:
        if self.consensus and (self.stable or len(self.rounds) == 1):
            return "consensus"
        if self.consensus:
            return "consensus (médiane encore mobile)"
        return "pas de consensus : nouveau tour conseillé"


def delphi(rounds: list, iqr_threshold: float = 1.0) -> DelphiResult:
    """`rounds` : liste de tours, chaque tour étant une liste de votes (ou une chaîne)."""
    out = []
    for r in rounds:
        v = parse_votes(r)
        if not v:
            continue
        arr = np.asarray(v, dtype=float)
        out.append(DelphiRound(v, float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))))
    if not out:
        raise ValueError("Aucun vote fourni.")
    return DelphiResult(out, iqr_threshold)


def delphi_table(items: dict[str, list], iqr_threshold: float = 1.0) -> pd.DataFrame:
    """Synthèse de plusieurs cellules : {"A+D × C3": [tour1, tour2], ...}."""
    rows = []
    for key, rounds in items.items():
        res = delphi(rounds, iqr_threshold)
        row = {"cellule": key}
        for k, r in enumerate(res.rounds, 1):
            row[f"tour {k}"] = " · ".join(f"{x:g}" for x in r.votes)
            row[f"méd. {k} [Q1–Q3]"] = f"{r.median:g} [{r.q1:g}–{r.q3:g}]"
        row["retenu"] = res.value
        row["statut"] = res.status
        rows.append(row)
    return pd.DataFrame(rows)
