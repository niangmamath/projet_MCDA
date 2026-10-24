# mcdakit — aide à la décision multicritère pour n'importe quel projet

`mcdakit` est une brique logicielle **générique et réutilisable** d'aide à la décision multicritère (MCDA).
Elle transforme une décision « intuitive » en une démarche structurée et traçable :

```
Entrées                          Traitement                                   Sorties
────────────────────────────     ──────────────────────────────────────────   ─────────────────────────────
alternatives, critères,      →   pondération (égale, directe, ROC, AHP)     →  scores, classements,
performances, sens (max/min),    Delphi (consensus d'experts)                  comparaison des méthodes,
poids, contraintes,              portefeuilles (combinaisons de k éléments)    seuils de basculement,
paramètres de méthode            normalisation + méthode MCDA                  Monte-Carlo, diagnostic de
                                 analyse de sensibilité                        robustesse, rapport HTML/MD/Excel
```

Aucun résultat, alternative ou critère n'est codé en dur : tout vient du fichier de projet (JSON, YAML ou Excel)
ou de l'interface web.

| Composant | Rôle |
|---|---|
| `mcdakit/` | bibliothèque Python (numpy + pandas), importable dans une future plateforme |
| `api/index.py` | API HTTP (FastAPI) qui expose la bibliothèque — fonction Python Vercel |
| `public/` | site web : page d'accueil (`index.html`) et application (`app.html`, servie sur `/app`) en HTML/CSS/JS sans build |
| `mcdakit` (commande) | CLI pour automatiser / scripter des analyses |
| `examples/` | les deux cas Stratema (menu « Charger un exemple ») ; `examples/bibliotheque/` : cas publiés et exemple générique (bibliothèque de cas) |
| `tests/` | tests (bibliothèque et API), dont la reproduction chiffrée des deux rapports Stratema |

---

## 1. Installation

Prérequis : Python ≥ 3.10.

```bash
git clone <ce dépôt> mcdakit && cd mcdakit
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[web,dev,validation]"                   # bibliothèque + API web + tests (+ bibliothèques de validation croisée, Python ≥ 3.11)
```

Dépendances : `numpy`, `pandas` (calcul), `openpyxl` (Excel), `pyyaml` (YAML) ; `fastapi`,
`python-multipart` (et `uvicorn` en local) pour l'API web ; `pytest`/`httpx` pour les tests. L'interface charge
Plotly depuis cdnjs. Aucune bibliothèque MCDA externe : toutes les méthodes
sont implémentées et testées dans le dépôt.

## 2. Utilisation

### a) Interface web (pour tout le monde)

En ligne : https://projet-mcda.vercel.app (accueil et guide) et https://projet-mcda.vercel.app/app (application).
Liens directs : `/app?exemple=<id>` charge un exemple, `/app?nouveau=1` ouvre un projet vierge. En local :

```bash
uvicorn api.index:app --reload        # puis ouvrir http://localhost:8000 (accueil) ou /app
```

L'interface propose six onglets qui suivent la démarche MCDA :

1. **Données** — critères (sens, unité), alternatives et évaluations dans des tableaux éditables ;
   attributs hors critères (budget, charge…), contraintes d'admissibilité, mode portefeuille.
2. **Poids** — égaux, directs, simple classement (ROC) ou **AHP** par curseurs (échelle de Saaty) avec
   ratio de cohérence en direct et désignation des jugements à revoir.
3. **Delphi** — saisie des votes par tour, médiane / quartiles / consensus, application au modèle en un clic.
4. **Résultats** — plusieurs méthodes côte à côte, choix de la normalisation, comparaison des classements
   (τ de Kendall), détail de tous les calculs intermédiaires.
5. **Sensibilité** — seuils de basculement des poids, courbes de balayage, Monte-Carlo, profils de
   décideurs, scénarios sur les données, seuil d'une valeur, **diagnostic robuste / modérément robuste / fragile**.
6. **Rapport** — export HTML (imprimable en PDF), Markdown, Excel (tous les tableaux) ; sauvegarde du projet.

Le menu « Charger un exemple » propose les deux cas Stratema ; les autres cas (cas publiés, choix d'un fournisseur)
sont dans la **bibliothèque de cas** de la page d'accueil (`/#bibliotheque`). On peut aussi partir d'un projet vierge ou importer un fichier `.json`, `.yaml` ou `.xlsx`
(modèle Excel : `mcdakit template modele.xlsx`).

### b) Ligne de commande

```bash
# analyse complète + rapport (HTML, .md ou .xlsx selon l'extension)
mcdakit run examples/stratema_methode2.json -m topsis -m wsm -m promethee --report rapport.html

# une seule méthode avec paramètres
mcdakit rank examples/stratema_methode1.json -m wsm:normalization=none

# outils autonomes
mcdakit ahp "1,1/2,3;2,1,4;1/3,1/4,1" --labels Coût,Qualité,Délai
mcdakit delphi "2,1,3,3,2" "2,2,2,3,2"
mcdakit template mon_projet.xlsx
```

Options de `run` : `--primary` (méthode de référence pour la sensibilité), `--mc` (nombre de tirages),
`--spread` (dispersion des poids), `--scenarios scen.json` (`{"nom": [["alternative", "colonne", valeur]]}`),
`--profiles prof.json` (`{"nom": {"C2": 2}}`).

### c) Bibliothèque Python

```python
from mcdakit import DecisionProblem, evaluate, full_analysis, weight_thresholds, to_html

problem = DecisionProblem.load("examples/bibliotheque/choix_fournisseur.yaml")

res = evaluate(problem, "topsis")                 # ou "wsm", "wpm", "promethee"
print(res.table(problem.labels()))                # scores, rangs, écarts au 1er
print(res.details["Distances"])                   # calculs intermédiaires

print(weight_thresholds(problem, "topsis"))       # où la décision bascule

fa = full_analysis(problem, {"topsis": {}, "promethee": {}},
                   scenarios={"Délai F3 + 10 j": [("F3", "DELAI", 17)]})
print(fa.diagnosis)                               # robuste / modérément robuste / fragile
open("rapport.html", "w").write(to_html(fa))
```

Un problème peut aussi être construit en mémoire avec `DecisionProblem.from_dict({...})`.

## 3. Démonstration

```bash
python examples/demo_stratema.py
```

reproduit l'intégralité des calculs des deux rapports Stratema et écrit
`docs/rapport_exemple_stratema.html`. Résultats retrouvés à 10⁻⁴ près :

| Élément du rapport | mcdakit |
|---|---|
| Poids AHP, λmax = 6,145258, RC = 2,34 % | identiques |
| Méthode 1 : BC 3,8746 > BD 3,8675 > CD > AB > AC > AD | identique |
| Méthode 2 : TOPSIS BC 0,7228 > BD 0,6559 > AB 0,5986 > … | identique, D⁺/D⁻ compris |
| Seuils TOPSIS : C2 24,5 % · C3 17,1 % · C4 13,9 % · C5 44,1 % · C6 29,3 % · C1 47,1 % | identiques |
| Profils (financier → BD, égaux → CD) et 8 scénarios de données (C5 de C → BD) | identiques |
| Normalisation min-max → BD 0,7019 | identique |

### Validation

Les calculs sont validés de trois façons (détail et chiffres dans [`docs/VALIDATION.md`](docs/VALIDATION.md)) :

- **10 cas publiés** (articles et manuels : Opricovic & Tzeng 2004, Ziemba 2017 et 2019, Papathanasiou & Ploskas 2018,
  Wang & Chin 2011…) reproduits à la précision de la publication — `mcdakit/benchmarks.py`, `tests/test_literature.py` ;
- **validation croisée** avec deux bibliothèques indépendantes (pymcdm, pyDecision) sur des centaines de problèmes
  aléatoires : écarts de l'ordre de 10⁻¹⁶ — `tests/test_cross_validation.py` ;
- **propriétés mathématiques** (dominance, changement d'unité, permutation, bornes, cohérence AHP, seuils de
  basculement…) — `tests/test_properties.py`.

`python examples/validation.py` régénère le rapport ; l'API recalcule les cas publiés à la demande (`GET /api/validation`).
Quatre de ces cas sont dans la bibliothèque de cas (`examples/bibliotheque/`), ouvrable depuis la page d'accueil.

## 4. Format d'un projet

Voir [`docs/FORMAT.md`](docs/FORMAT.md) (JSON/YAML/Excel) et [`docs/METHODES.md`](docs/METHODES.md)
(formules, hypothèses et règles du diagnostic). Exemple minimal :

```yaml
name: Choix d'un logiciel
criteria:
  - {id: PRIX, name: Prix annuel, direction: min, unit: "€"}
  - {id: FONC, name: Couverture fonctionnelle, direction: max}
alternatives:
  - {id: X, values: {PRIX: 12000, FONC: 4}}
  - {id: Y, values: {PRIX: 9000,  FONC: 3}}
weights: {PRIX: 1, FONC: 2}          # normalisés automatiquement
```

## 5. Architecture du code

```
mcdakit/
├── problem.py        modèle de données, validation, portefeuilles, contraintes, surcharges, (dé)sérialisation
├── weights.py        poids égaux / directs / ROC / rank-sum / AHP (vecteur propre ou moyenne géométrique, RC)
├── delphi.py         médiane, quartiles, consensus (IQR) et stabilité entre tours
├── normalization.py  none, vector, minmax, max, sum (avec ou sans réorientation des critères à minimiser)
├── methods.py        WSM, WPM, TOPSIS, PROMETHEE II (+ incomparabilités PROMETHEE I) — signature commune
├── analysis.py       evaluate, run_methods, compare_methods (τ de Kendall, ρ de Spearman)
├── sensitivity.py    moteur vectorisé, balayage, seuils, profils, Monte-Carlo, scénarios, diagnostic
├── report.py         analyse complète + rendu Markdown / HTML / Excel
├── io_excel.py       import / export du modèle Excel
└── cli.py            commande `mcdakit`
api/index.py          API FastAPI (Vercel)
public/               page d'accueil (index.html, landing.css, img/) et application (app.html, app.js, style.css)
```

**Ajouter une méthode** : écrire une fonction `ma_methode(matrix, weights, directions, **params) -> MethodResult`
dans `methods.py` et l'enregistrer dans `METHODS` ; elle devient disponible dans l'API, la CLI, l'interface et
les rapports (et apparaît dans l'interface web via `/api/meta`). Pour la rendre utilisable par la sensibilité vectorisée, ajouter sa branche dans
`WeightScorer.scores` (un test vérifie l'égalité avec la méthode de référence).

## 6. Tests

```bash
pytest
```

Tests (98) : propriétés mathématiques (cohérence AHP, flux PROMETHEE de somme nulle, respect de la dominance par
toutes les méthodes), aller-retour JSON/Excel, égalité moteur vectorisé ↔ méthodes, et reproduction des
résultats publiés dans les deux rapports Stratema, et appels de toutes les routes de l'API.

## 7. Déploiement sur Vercel

Le dépôt est prêt pour Vercel, sans configuration supplémentaire :

- `public/` est servi tel quel par le CDN (interface) ;
- `api/index.py` est détecté comme application FastAPI et reçoit toutes les requêtes `/api/*` ;
  `vercel.json` y inclut les dossiers `mcdakit/`, `examples/` et `public/` ;
- `requirements.txt` liste les dépendances installées au build.

Étapes : importer le dépôt GitHub dans Vercel (*Add New → Project*), le preset « FastAPI » est détecté automatiquement, déployer.
Chaque `git push` sur `main` redéploie automatiquement. La documentation interactive de l'API est servie sur
`/api/docs`.

### Routes de l'API

| Route | Rôle |
|---|---|
| `GET /api/validation` | recalcule les cas publiés et les compare aux résultats imprimés |
| `GET /api/meta` | méthodes, normalisations, fonctions de préférence disponibles |
| `GET /api/examples` | cas Stratema du menu « Charger un exemple » |
| `GET /api/library` | bibliothèque de cas (cas publiés, exemple générique) |
| `GET /api/examples/{id}` | contenu d'un exemple (menu ou bibliothèque) |
| `POST /api/validate` | contrôle du projet : candidates, admissibilité, matrice, poids, cohérence AHP |
| `POST /api/analyze` | résultats de plusieurs méthodes + comparaison |
| `POST /api/sensitivity` | seuils, balayage, Monte-Carlo, profils, scénarios, diagnostic |
| `POST /api/value-threshold` | seuil de basculement d'une valeur |
| `POST /api/delphi` | synthèse des tours Delphi |
| `POST /api/report` | rapport `html`, `md` ou `xlsx` |
| `POST /api/import`, `POST /api/export-xlsx`, `GET /api/template` | fichiers JSON / YAML / Excel |

Les corps de requête contiennent `{"problem": <projet au format docs/FORMAT.md>, ...}`.

## 8. Limites

- Les portefeuilles sont énumérés exhaustivement (jusqu'à 20 000 combinaisons) ; au-delà, une approche
  d'optimisation serait nécessaire.
- L'agrégation d'un portefeuille (somme, moyenne, min, max) suppose l'additivité ; les interactions
  (synergies, redondances) se traitent par surcharge de cellules, idéalement issues d'un Delphi.
- Le diagnostic de robustesse applique des règles explicites et paramétrables (`diagnose`), pas une vérité
  absolue. Le modèle éclaire la décision ; il ne se substitue pas au décideur.

Licence MIT.
