# Validation des calculs de mcdakit

*Document généré le 24/10/2026 par `python examples/validation.py`.*

Un logiciel d'aide à la décision doit produire des résultats exacts et vérifiables. mcdakit est validé de trois façons indépendantes :

1. **Cas publiés** : on reprend les données d'articles et d'ouvrages, et on retrouve les résultats imprimés.
2. **Validation croisée** : sur des centaines de problèmes générés aléatoirement, mcdakit donne les mêmes nombres que deux bibliothèques développées par d'autres équipes (pymcdm, pyDecision).
3. **Propriétés mathématiques** : la suite de tests vérifie les propriétés que toute méthode correcte doit respecter (dominance, changement d'unité, permutation, bornes, cohérence AHP…).

**Bilan : 5 cas publiés reproduits sur 5 — tous conformes.**

## 1. Reproduction de résultats publiés

Les données et résultats viennent des sources citées. Ces cas sont repris des suites de tests de pymcdm 1.4 et pyrepo-mcda 0.1.15, qui les citent avec ces références ; les DOI sont ceux qu'elles indiquent. Un cas est conforme si l'écart ne dépasse pas la précision de la publication (une demi-unité de la dernière décimale imprimée, sauf mention).

| Cas | Méthode | Résultat publié | mcdakit | Écart max | Classement | Verdict |
|---|---|---|---|---|---|---|
| opricovic2004 | TOPSIS (normalisation min-max) | 0,500 ; 0,617 ; 0,500 | 0,500 ; 0,617 ; 0,500 | 2.2e-04 | identique | conforme |
| ersoy2021 | TOPSIS (normalisation vectorielle) | 0,308 ; 0,254 ; 0,327 ; 0,263 ; 0,857 ; 0,192 | 0,307 ; 0,254 ; 0,327 ; 0,262 ; 0,856 ; 0,191 | 7.5e-04 | identique | conforme |
| wangchin2011 | AHP — poids par vecteur propre | 0,3208 ; 0,1395 ; 0,0348 ; 0,1285 ; 0,2374 ; 0,1391 | 0,3208 ; 0,1395 ; 0,0348 ; 0,1285 ; 0,2374 ; 0,1391 | 3.2e-05 | identique | conforme |
| papathanasiou2018_ahp | AHP complet (critères et alternatives par paires) | 0,117 ; 0,071 ; 0,095 ; 0,212 ; 0,350 ; 0,155 | 0,117 ; 0,071 ; 0,095 ; 0,212 ; 0,350 ; 0,155 | 3.8e-04 | identique | conforme |
| rizan2020 | Somme pondérée (SAW, normalisation x/max) | 0,350 ; 0,375 ; 0,825 ; 0,642 | 0,350 ; 0,375 ; 0,825 ; 0,642 | 3.3e-04 | identique | conforme |

**Sources**

- `opricovic2004` : Opricovic & Tzeng (2004), Compromise solution by MCDM methods: a comparative analysis of VIKOR and TOPSIS. European Journal of Operational Research 156(2). *Les valeurs publiées sont obtenues avec la normalisation min-max.*
- `ersoy2021` : Ersoy (2021), Equipment selection for an e-commerce company using entropy-based TOPSIS, EDAS and CODAS methods during the COVID-19. LogForum 17(3). *Tolérance 0,001 : les poids entropiques sont publiés arrondis à 3 décimales.*
- `wangchin2011` : Wang & Chin (2011), A linear programming approximation to the eigenvector method in the analytic hierarchy process. Information Sciences 181(23).
- `papathanasiou2018_ahp` : Papathanasiou & Ploskas (2018), AHP. In Multiple Criteria Decision Aid, Springer — DOI 10.1007/978-3-319-91648-4_5. *Priorités locales et poids par mcdakit.ahp, synthèse par somme pondérée sans normalisation.*
- `rizan2020` : Rizan, Wahyuningsih, Pradana & Ramadella (2020), SAW method in supporting the process of admission of new junior high school students. CITSM 2020, IEEE — DOI 10.1109/CITSM50537.2020.9268874.

Ces études servent uniquement à vérifier le moteur de calcul : elles ne sont pas proposées comme cas dans l'application, qui ne contient que le projet Stratema.

Le projet Stratema (rapports AHP + somme pondérée et TOPSIS du projet) est également reproduit : poids AHP, RC = 2,34 %, scores des six couples, seuils de basculement, profils et scénarios (`tests/test_methods.py`, `tests/test_sensitivity.py`).

## 2. Validation croisée avec des implémentations indépendantes

500 problèmes aléatoires (3 à 12 alternatives, 2 à 8 critères, sens et poids aléatoires, un tiers avec de nombreux ex aequo) ; matrices AHP réciproques aléatoires de taille 3 à 14. Références : pymcdm 1.4.0, pyDecision 5.1.8.

| Calcul | Comparaisons | Écart maximal |
|---|---|---|
| TOPSIS, normalisation vectorielle | 500 | 4.4e-16 |
| TOPSIS, normalisation min-max | 500 | 2.2e-16 |
| Somme pondérée, normalisation x/max | 500 | 4.4e-16 |
| AHP, poids (vecteur propre) — pymcdm | 500 | 0.0e+00 |
| AHP, poids (vecteur propre) — pyDecision | 500 | 0.0e+00 |
| AHP, ratio de cohérence — pyDecision | 500 | 0.0e+00 |
| AHP, moyenne géométrique et RC — pyDecision | 500 | 1.6e-15 |

Les écarts sont de l'ordre de 10⁻¹⁶, c'est-à-dire la précision des nombres en machine : les calculs sont identiques.

## 3. Propriétés mathématiques vérifiées

`tests/test_properties.py`, sur des problèmes aléatoires reproductibles, pour chaque méthode et normalisation :

- **Dominance** : une option meilleure ou égale partout n'est jamais classée derrière.
- **Permutation** : l'ordre de saisie des alternatives ne change aucun score.
- **Changement d'unité** : exprimer un critère en k€ plutôt qu'en € ne change pas le résultat.
- **Poids** : multiplier tous les poids par une constante ne change rien ; tout le poids sur un critère donne le classement de ce critère.
- **Bornes** : TOPSIS dans [0, 1] ; matrices normalisées dans [0, 1].
- **AHP** : une matrice parfaitement cohérente redonne exactement les poids et RC = 0 ; inverser les jugements inverse les priorités.
- **Delphi** : médiane et quartiles identiques à ceux de numpy.
- **Portefeuilles** : avec la moyenne et une somme pondérée, score(XY) = [score(X) + score(Y)] / 2 ; aucune combinaison retenue ne viole une contrainte.
- **Sensibilité** : juste avant un seuil de basculement le gagnant est inchangé, juste après il change ; une dispersion quasi nulle des poids redonne le gagnant de référence dans 100 % des tirages ; le moteur vectorisé donne les mêmes scores que les méthodes de référence.

## 4. Défaut trouvé et corrigé grâce à cette validation

Le cas Rizan et al. (2020) contient des notes égales à 0. La normalisation « x / max » refusait toute valeur nulle, alors que seuls les critères à minimiser (min / x) l'exigent. Corrigé : les zéros sont acceptés sur les critères à maximiser ; le cas est désormais reproduit exactement.

## 5. Ce que la validation ne couvre pas

- Les **règles du diagnostic** (robuste / modérément robuste / fragile) sont des conventions explicites et paramétrables, pas un standard publié.
- La **dispersion Monte-Carlo** est définie comme le coefficient de variation d'un poids moyen sous une loi de Dirichlet ; d'autres définitions donnent d'autres fréquences (c'est pourquoi le rapport Stratema indiquait 79,6 % et mcdakit 92 % pour « ≈ 20 % »).
- L'**agrégation des portefeuilles** (somme, moyenne…) est une hypothèse de modélisation choisie par l'utilisateur ; les interactions entre éléments se traitent par surcharge de cellule.
- Exactitude du calcul ne veut pas dire pertinence du modèle : critères, poids et données restent sous la responsabilité du décideur.

## Reproduire

```bash
pip install -e ".[dev,validation]"
pytest                          # toute la suite, y compris la validation
python examples/validation.py   # régénère ce document
```

L'application en ligne recalcule aussi les cas publiés à la demande : `GET /api/validation`.
