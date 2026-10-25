"""Démonstration courte de mcdakit : entrées → traitement → sorties, en moins d'une minute.

    python examples/demo.py

1. Cas Stratema (méthode 2) chargé depuis un fichier : TOPSIS, recommandation, robustesse, rapport PDF.
2. Le même code sur un tout autre problème, construit en Python : la sélection d'équipement publiée par
   Ersoy (2021), doi:10.17270/J.LOG.2021.603. On retrouve les scores de l'article, puis on change les
   paramètres (poids, normalisation) sans toucher au code de calcul.

Les fichiers produits sont écrits dans demo_output/.
"""

from pathlib import Path

import pandas as pd

from mcdakit import DecisionProblem, evaluate, full_analysis, to_excel, to_pdf, weight_thresholds

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo_output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)


def titre(texte: str) -> None:
    print(f"\n{'=' * 100}\n{texte}\n{'=' * 100}")


# ============================================================================ 1. Stratema
titre("1. ENTRÉES — cas Stratema, méthode 2 (fichier examples/stratema_methode2.json)")
p = DecisionProblem.load(HERE / "stratema_methode2.json")
print(p.name)
print("Critères :", ", ".join(f"{c.id} {c.name} ({c.direction})" for c in p.criteria))
print("Contraintes :", ", ".join(str(c) for c in p.constraints), "| portefeuille de 2 missions parmi 4")
cand = p.candidates()
print(cand[["name", "C1", "C2"]].join(p.feasibility(cand)["admissible"]).to_string())

titre("TRAITEMENT — poids AHP, TOPSIS (normalisation vectorielle)")
res = evaluate(p, "topsis", normalization="vector")
print(res.table(p.labels()).round(4).to_string())

titre("SORTIES — recommandation, seuils de basculement, diagnostic, rapports")
fa = full_analysis(p, {"topsis": {"normalization": "vector"}}, mc_runs=5000)
print("Recommandation :", fa.labels[fa.winner])
print(weight_thresholds(p, "topsis").round(3).to_string())
print("\nDiagnostic :", fa.diagnosis)
(OUT / "rapport_stratema.pdf").write_bytes(to_pdf(fa))
(OUT / "calculs_stratema.xlsx").write_bytes(to_excel(fa))
print(f"Rapport PDF et tableaux Excel écrits dans {OUT}/")

# ============================================================================ 2. Autre problème
titre("2. MÊME CODE, AUTRE PROBLÈME — sélection d'équipement, Ersoy (2021), construit en Python")
ersoy = DecisionProblem.from_dict(
    {
        "name": "Sélection d'équipement (Ersoy 2021)",
        # critères C1 à C6 de l'article (C1-C4 à maximiser, C5-C6 coûts à minimiser)
        "criteria": [{"id": f"C{j}", "direction": "max" if j <= 4 else "min"} for j in range(1, 7)],
        "alternatives": [
            {"id": f"A{i + 1}", "values": {f"C{j + 1}": v for j, v in enumerate(row)}}
            for i, row in enumerate(
                [
                    [256, 8, 41, 1.6, 1.77, 7347.16],
                    [256, 8, 32, 1.0, 1.8, 6919.99],
                    [256, 8, 53, 1.6, 1.9, 8400],
                    [256, 8, 41, 1.0, 1.75, 6808.9],
                    [512, 8, 35, 1.6, 1.7, 8479.99],
                    [256, 4, 35, 1.6, 1.7, 7499.99],
                ]
            )
        ],
        # poids entropiques publiés dans l'article
        "weights": {"C1": 0.405, "C2": 0.221, "C3": 0.134, "C4": 0.199, "C5": 0.007, "C6": 0.034},
    }
)
publie = [0.308, 0.254, 0.327, 0.263, 0.857, 0.192]
r = evaluate(ersoy, "topsis")
comparaison = pd.DataFrame({"publié": publie, "mcdakit": r.scores.to_numpy()}, index=r.scores.index)
print(comparaison.T.round(3).to_string())
ecart = (comparaison["publié"] - comparaison["mcdakit"]).abs().max()
print(
    f"Écart maximal : {ecart:.4f} (poids publiés arrondis à 3 décimales) — même classement :",
    list(comparaison["publié"].rank()) == list(comparaison["mcdakit"].rank()),
)

print("\nParamétrage (illustratif) : poids par simple classement (ROC), deux méthodes, sans changer le code de calcul")
ersoy.weights_spec = {"method": "roc", "order": ["C6", "C1", "C2", "C4", "C3", "C5"]}  # ordre d'importance arbitraire
for methode, params in [("topsis", {}), ("wsm", {"normalization": "minmax"})]:
    print(f"  {methode:6} → classement :", " > ".join(evaluate(ersoy, methode, **params).order))

print(f"\nFin de la démonstration. Fichiers : {', '.join(sorted(f.name for f in OUT.iterdir()))}")
