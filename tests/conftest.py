from pathlib import Path

import pytest

from mcdakit import DecisionProblem

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def stratema1():
    return DecisionProblem.load(EX / "stratema_methode1.json")


@pytest.fixture
def stratema2():
    return DecisionProblem.load(EX / "stratema_methode2.json")


@pytest.fixture
def fournisseur():
    """Petit problème de test (sans portefeuille, avec une contrainte) défini ici, hors des exemples."""
    return DecisionProblem.from_dict({
        "criteria": [{"id": "COUT", "direction": "min"}, {"id": "DELAI", "direction": "min"},
                     {"id": "QUAL", "direction": "max"}, {"id": "RSE", "direction": "max"}],
        "alternatives": [
            {"id": "F1", "values": {"COUT": 12.5, "DELAI": 15, "QUAL": 82, "RSE": 3}},
            {"id": "F2", "values": {"COUT": 10.9, "DELAI": 30, "QUAL": 74, "RSE": 2}},
            {"id": "F3", "values": {"COUT": 14.0, "DELAI": 7, "QUAL": 90, "RSE": 4}},
            {"id": "F4", "values": {"COUT": 11.8, "DELAI": 20, "QUAL": 79, "RSE": 5}}],
        "weights": {"method": "roc", "order": ["QUAL", "COUT", "DELAI", "RSE"]},
        "constraints": [{"column": "DELAI", "op": "<=", "value": 25}]})
