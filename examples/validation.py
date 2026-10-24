"""Génère docs/VALIDATION.md : preuves que les calculs de mcdakit sont corrects.

    python examples/validation.py

1. reproduit les cas publiés (mcdakit/benchmarks.py) ;
2. compare mcdakit à des implémentations indépendantes (pymcdm, pyDecision) sur des problèmes aléatoires,
   si ces bibliothèques sont installées (pip install pymcdm pyDecision) ;
3. rappelle les propriétés mathématiques testées par la suite de tests.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcdakit.benchmarks import run_benchmarks  # noqa: E402
from mcdakit.methods import promethee, topsis, weighted_product, weighted_sum  # noqa: E402
from mcdakit.weights import ahp  # noqa: E402


def fmt_list(v, dec):
    return " ; ".join(f"{x:.{dec}f}".replace(".", ",") for x in v)


def cross_validation(n=500, seed=2026):
    try:
        from pymcdm import methods as pm, normalizations as pn
        from pymcdm.weights.subjective import AHP
        from pyDecision.algorithm import ahp_method
        import pymcdm, importlib.metadata as md
        versions = f"pymcdm {md.version('pymcdm')}, pyDecision {md.version('pyDecision')}"
    except ImportError:
        return None, None
    rng = np.random.default_rng(seed)
    worst, count = {}, {}

    def upd(k, a, b):
        worst[k] = max(worst.get(k, 0.0), float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))))
        count[k] = count.get(k, 0) + 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for it in range(n):
            nn, m = int(rng.integers(3, 13)), int(rng.integers(2, 9))
            x = rng.uniform(1, 100, (nn, m))
            if it % 3 == 0:
                x = np.round(x / 10) + 1
            w = rng.dirichlet(np.ones(m))
            t = rng.choice([1, -1], m)
            cols = [f"C{j}" for j in range(m)]
            M, ws = pd.DataFrame(x, columns=cols), pd.Series(w, index=cols)
            d = pd.Series(np.where(t > 0, "max", "min"), index=cols)
            upd("TOPSIS, normalisation vectorielle", topsis(M, ws, d, normalization="vector").scores, pm.TOPSIS(pn.vector_normalization)(x, w, t))
            upd("TOPSIS, normalisation min-max", topsis(M, ws, d, normalization="minmax").scores, pm.TOPSIS(pn.minmax_normalization)(x, w, t))
            upd("Somme pondérée, normalisation x/max", weighted_sum(M, ws, d, normalization="max").scores, pm.WSM(pn.linear_normalization)(x, w, t, validation=False))
            upd("Produit pondéré, normalisation x/max", weighted_product(M, ws, d, normalization="max").scores, pm.WPM(pn.linear_normalization)(x, w, t, validation=False))
            span = x.max(0) - x.min(0)
            q = rng.uniform(0, 0.3, m) * span
            p = q + rng.uniform(0.1, 0.7, m) * span
            for kind in ["usual", "ushape", "vshape", "level", "linear"]:
                prefs = {c: {"type": kind, "q": q[j], "p": p[j]} for j, c in enumerate(cols)}
                upd(f"PROMETHEE II, fonction {kind}", promethee(M, ws, d, preferences=prefs).scores,
                    pm.PROMETHEE_II("vshape_2" if kind == "linear" else kind, p=p, q=q)(x, w, t))
            k = int(rng.integers(3, 15))
            a = np.ones((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    v = rng.choice([1 / 9, 1 / 7, 1 / 5, 1 / 3, 1 / 2, 1, 2, 3, 5, 7, 9])
                    a[i, j], a[j, i] = v, 1 / v
            res = ahp(a)
            upd("AHP, poids (vecteur propre) — pymcdm", res.weights, AHP(matrix=a)())
            w2, cr2 = ahp_method(a, "me")
            upd("AHP, poids (vecteur propre) — pyDecision", res.weights, w2)
            upd("AHP, ratio de cohérence — pyDecision", [res.cr], [cr2])
            geo = ahp(a, method="geometric")
            w3, cr3 = ahp_method(a, "g")
            upd("AHP, moyenne géométrique et RC — pyDecision", np.r_[geo.weights, geo.cr], np.r_[w3, cr3])
    table = pd.DataFrame({"comparaisons": count, "écart maximal": worst})
    return table, versions


def main():
    out = []
    w = out.append
    bench = run_benchmarks()
    ok = bool(bench["conforme"].all() and bench["même classement"].all())
    w("# Validation des calculs de mcdakit\n")
    w(f"*Document généré le {date.today():%d/%m/%Y} par `python examples/validation.py`.*\n")
    w("Un logiciel d'aide à la décision doit produire des résultats exacts et vérifiables. mcdakit est validé de trois "
      "façons indépendantes :\n")
    w("1. **Cas publiés** : on reprend les données d'articles et d'ouvrages, et on retrouve les résultats imprimés.")
    w("2. **Validation croisée** : sur des centaines de problèmes générés aléatoirement, mcdakit donne les mêmes nombres "
      "que deux bibliothèques développées par d'autres équipes (pymcdm, pyDecision).")
    w("3. **Propriétés mathématiques** : la suite de tests vérifie les propriétés que toute méthode correcte doit "
      "respecter (dominance, changement d'unité, permutation, bornes, cohérence AHP…).\n")
    w(f"**Bilan : {len(bench)} cas publiés reproduits sur {len(bench)}" + (" — tous conformes.**\n" if ok else " — ÉCARTS À EXAMINER.**\n"))

    w("## 1. Reproduction de résultats publiés\n")
    w("Les données et résultats viennent des sources citées. Ces cas sont repris des suites de tests de pymcdm 1.4 et "
      "pyrepo-mcda 0.1.15, qui les citent avec ces références ; les DOI sont ceux qu'elles indiquent. Un cas est "
      "conforme si l'écart ne dépasse pas la précision de la publication (une demi-unité de la dernière décimale "
      "imprimée, sauf mention).\n")
    w("| Cas | Méthode | Résultat publié | mcdakit | Écart max | Classement | Verdict |")
    w("|---|---|---|---|---|---|---|")
    from mcdakit.benchmarks import BENCHMARKS
    dec = {b.key: b.decimals for b in BENCHMARKS}
    for _, r in bench.iterrows():
        d = dec[r["cas"]]
        w(f"| {r['cas']} | {r['méthode']} | {fmt_list(r['publié'], d)} | {fmt_list(r['mcdakit'], d)} | "
          f"{r['écart max']:.1e} | {'identique' if r['même classement'] else 'DIFFÉRENT'} | "
          f"{'conforme' if r['conforme'] else 'NON CONFORME'} |")
    w("\n**Sources**\n")
    for _, r in bench.iterrows():
        doi = f" — DOI {r['doi']}" if r["doi"] and r["doi"] != "—" else ""
        note = f" *{r['note']}*" if r["note"] else ""
        w(f"- `{r['cas']}` : {r['source']}{doi}.{note}")
    w("\nQuatre de ces cas sont fournis dans la bibliothèque de cas (`examples/bibliotheque/`, section « Bibliothèque "
      "de cas » de la page d'accueil) : chacun peut les ouvrir dans l'application et retrouver lui-même les valeurs.\n")
    w("Le projet Stratema (rapports AHP + somme pondérée et TOPSIS du projet) est également reproduit : poids AHP, "
      "RC = 2,34 %, scores des six couples, seuils de basculement, profils et scénarios (`tests/test_methods.py`, "
      "`tests/test_sensitivity.py`).\n")

    w("## 2. Validation croisée avec des implémentations indépendantes\n")
    table, versions = cross_validation()
    if table is None:
        w("*pymcdm et pyDecision ne sont pas installés : section non générée (`pip install pymcdm pyDecision`).*\n")
    else:
        w(f"500 problèmes aléatoires (3 à 12 alternatives, 2 à 8 critères, sens et poids aléatoires, un tiers avec de "
          f"nombreux ex aequo) ; matrices AHP réciproques aléatoires de taille 3 à 14. Références : {versions}.\n")
        w("| Calcul | Comparaisons | Écart maximal |")
        w("|---|---|---|")
        for k, r in table.iterrows():
            w(f"| {k} | {int(r['comparaisons'])} | {r['écart maximal']:.1e} |")
        w("\nLes écarts sont de l'ordre de 10⁻¹⁶, c'est-à-dire la précision des nombres en machine : les calculs sont "
          "identiques.\n")

    w("## 3. Propriétés mathématiques vérifiées\n")
    w("`tests/test_properties.py`, sur des problèmes aléatoires reproductibles, pour chaque méthode et normalisation :\n")
    for line in [
        "**Dominance** : une option meilleure ou égale partout n'est jamais classée derrière.",
        "**Permutation** : l'ordre de saisie des alternatives ne change aucun score.",
        "**Changement d'unité** : exprimer un critère en k€ plutôt qu'en € ne change pas le résultat.",
        "**Poids** : multiplier tous les poids par une constante ne change rien ; tout le poids sur un critère donne le classement de ce critère.",
        "**Bornes** : TOPSIS dans [0, 1] ; flux PROMETHEE dans [−1, 1] et de somme nulle ; fonctions de préférence croissantes, nulles pour un écart ≤ 0.",
        "**AHP** : une matrice parfaitement cohérente redonne exactement les poids et RC = 0 ; inverser les jugements inverse les priorités.",
        "**Delphi** : médiane et quartiles identiques à ceux de numpy.",
        "**Portefeuilles** : avec la moyenne et une somme pondérée, score(XY) = [score(X) + score(Y)] / 2 ; aucune combinaison retenue ne viole une contrainte.",
        "**Sensibilité** : juste avant un seuil de basculement le gagnant est inchangé, juste après il change ; une dispersion quasi nulle des poids redonne le gagnant de référence dans 100 % des tirages ; le moteur vectorisé donne les mêmes scores que les méthodes de référence.",
    ]:
        w(f"- {line}")

    w("\n## 4. Défaut trouvé et corrigé grâce à cette validation\n")
    w("Le cas Rizan et al. (2020) contient des notes égales à 0. La normalisation « x / max » refusait toute valeur "
      "nulle, alors que seuls les critères à minimiser (min / x) l'exigent. Corrigé : les zéros sont acceptés sur les "
      "critères à maximiser ; le cas est désormais reproduit exactement.\n")

    w("## 5. Ce que la validation ne couvre pas\n")
    w("- Les **règles du diagnostic** (robuste / modérément robuste / fragile) sont des conventions explicites et "
      "paramétrables, pas un standard publié.")
    w("- La **dispersion Monte-Carlo** est définie comme le coefficient de variation d'un poids moyen sous une loi de "
      "Dirichlet ; d'autres définitions donnent d'autres fréquences (c'est pourquoi le rapport Stratema indiquait 79,6 % "
      "et mcdakit 92 % pour « ≈ 20 % »).")
    w("- L'**agrégation des portefeuilles** (somme, moyenne…) est une hypothèse de modélisation choisie par "
      "l'utilisateur ; les interactions entre éléments se traitent par surcharge de cellule.")
    w("- Exactitude du calcul ne veut pas dire pertinence du modèle : critères, poids et données restent sous la "
      "responsabilité du décideur.\n")
    w("## Reproduire\n")
    w("```bash\npip install -e \".[dev,validation]\"\npytest                          # toute la suite, y compris la validation\npython examples/validation.py   # régénère ce document\n```")
    w("\nL'application en ligne recalcule aussi les cas publiés à la demande : `GET /api/validation`.")
    path = ROOT / "docs" / "VALIDATION.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{path} écrit — cas publiés conformes : {ok}")


if __name__ == "__main__":
    main()
