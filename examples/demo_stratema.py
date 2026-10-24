"""Démonstration : reproduit les calculs des rapports Stratema (méthode 1 et méthode 2) avec mcdakit.

    python examples/demo_stratema.py
"""
from pathlib import Path

import pandas as pd

from mcdakit import (DecisionProblem, data_scenarios, evaluate, full_analysis, monte_carlo, resolve_weights,
                     to_html, weight_profiles, weight_thresholds)

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 160)

# ------------------------------------------------------------ méthode 1 : AHP + somme pondérée
p1 = DecisionProblem.load(HERE / "stratema_methode1.json")
w, ahp = resolve_weights(p1)
print("Poids AHP :", w.round(6).to_dict())
print(f"λmax = {ahp.lambda_max:.6f}  RC = {ahp.cr:.2%}\n")
r1 = evaluate(p1, "wsm", normalization="none")      # notes 1-5 déjà sur une échelle commune
print(r1.table(p1.labels()).round(5), "\n")
print("Seuils de basculement (méthode 1) :")
print(weight_thresholds(p1, "wsm", {"normalization": "none"}).round(4), "\n")

# ------------------------------------------------------------ méthode 2 : TOPSIS
p2 = DecisionProblem.load(HERE / "stratema_methode2.json")
r2 = evaluate(p2, "topsis")
print(r2.table(p2.labels()).round(4), "\n")
print(r2.details["Distances"].round(4), "\n")
print("Seuils de basculement (TOPSIS) :")
print(weight_thresholds(p2, "topsis").round(4), "\n")
print(weight_profiles(p2, "topsis", {
    "Financier (C2 × 2)": {"C2": 2}, "Croissance (C3, C5 × 1,5)": {"C3": 1.5, "C5": 1.5},
    "Relationnel (C4, C6 × 2)": {"C4": 2, "C6": 2}, "Capacité (C1 × 2)": {"C1": 2},
    "Poids égaux": {"__equal__": True}}).round(3), "\n")
scen = {
    "C3 de C : 4 → 3": [("C", "C3", 3)],
    "C3 de D : 2 → 3": [("D", "C3", 3)],
    "C3 de B : 5 → 3": [("B", "C3", 3)],
    "C5 de C : 3 → 2": [("C", "C5", 2)],
    "Montée en compétence B : +5 JH": [("B", "C1", 37)],
    "Montée en compétence B : +10 JH": [("B", "C1", 42)],
    "Marge de B − 20 %": [("B", "C2", 39200 * 0.8)],
}
print(data_scenarios(p2, "topsis", scen).round(4), "\n")
print("Normalisation min-max :", evaluate(p2, "topsis", normalization="minmax").order[:3])
mc = monte_carlo(p2, "topsis", n=20000, spread=0.2)
print(mc.table(p2.labels()).round(3), "\n")

fa = full_analysis(p2, {"topsis": {}, "wsm": {}, "promethee": {}}, scenarios=scen,
                   profiles={"Financier (C2 × 2)": {"C2": 2}, "Poids égaux": {"__equal__": True}})
print(fa.diagnosis)
out = HERE.parent / "docs" / "rapport_exemple_stratema.html"
out.write_text(to_html(fa), encoding="utf-8")
print(f"\nRapport complet : {out}")
