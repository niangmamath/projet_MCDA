# Format d'un projet mcdakit

Un projet est un dictionnaire (JSON ou YAML). Seuls `criteria` et `alternatives` sont obligatoires.

```jsonc
{
  "name": "Titre de la décision",
  "description": "Contexte, décideur, décision à prendre",

  "criteria": [
    {
      "id": "C1",                    // identifiant unique (obligatoire)
      "name": "Charge",              // libellé
      "direction": "min",            // "max" : plus haut = mieux ; "min" : plus bas = mieux
      "unit": "jours-homme",
      "aggregation": "sum"           // mode portefeuille : sum | mean | min | max | product
    }
  ],

  "attributes": [                    // colonnes qui ne sont PAS des critères (servent aux contraintes)
    {"id": "BUDGET", "name": "Budget", "unit": "€", "aggregation": "sum"}
  ],

  "alternatives": [
    {"id": "A", "name": "Option A", "values": {"C1": 30, "BUDGET": 12000}}
  ],

  "weights": ...,                    // voir ci-dessous (défaut : poids égaux)

  "portfolio": {"size": 2},          // optionnel : alternatives = combinaisons de 2 éléments
                                     // ou {"size": {"min": 1, "max": 3}}

  "constraints": [                   // filtres d'admissibilité : <=, <, >=, >, ==, !=
    {"column": "BUDGET", "op": "<=", "value": 50000}
  ],

  "overrides": {                     // valeurs imposées pour une alternative (ex. Delphi sur un couple)
    "A+D": {"C3": 2}
  },

  "analysis": {                      // facultatif : méthodes cochées à l'ouverture dans l'application
    "primary": "topsis",
    "methods": {"topsis": {"normalization": "vector"}, "wsm": {"normalization": "minmax"}}
  }
}
```

## Poids

| Forme | Exemple |
|---|---|
| directs (normalisés) | `{"C1": 3, "C2": 1}` ou `{"method": "direct", "values": {...}}` |
| égaux | `{"method": "equal"}` |
| classement (ROC) | `{"method": "roc", "order": ["C2", "C1", "C3"]}` (du plus au moins important) ; `rank_sum` aussi |
| AHP — matrice | `{"method": "ahp", "matrix": [["1", "1/3"], ["3", "1"]]}` ; `"variant": "geometric"` possible |
| AHP — jugements | `{"method": "ahp", "judgements": {"C1>C2": "1/3", "C1>C3": 5, ...}}` (au-dessus de la diagonale) |

## Mode portefeuille

Avec `"portfolio": {"size": k}`, les alternatives listées deviennent des **éléments** et le modèle compare
toutes les combinaisons de `k` éléments (identifiants `A+B`, `A+C`…). Chaque colonne d'une combinaison est
calculée avec la règle `aggregation` du critère/attribut. Les contraintes portent sur les valeurs agrégées
(ex. charge totale ≤ 65). Les `overrides` permettent de corriger une combinaison précise (interaction,
synergie, Delphi).

## Excel

`mcdakit template modele.xlsx` génère un classeur à remplir :

- **Critères** : id, nom, sens, unité, poids, agrégation
- **Alternatives** : id, nom, puis une colonne par critère/attribut
- **Attributs**, **Contraintes** (colonne, opérateur, valeur), **Paramètres** (nom, description,
  portefeuille_min, portefeuille_max) — facultatives
- **AHP** — facultative : matrice carrée de comparaisons (remplace la colonne poids)
