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
    return DecisionProblem.load(EX / "bibliotheque" / "choix_fournisseur.yaml")
