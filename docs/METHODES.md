# Méthodes implémentées

Notations : alternatives *a*, critères *j*, valeur brute x_aj, poids w_j (Σ w_j = 1).

## Pondération

- **AHP** (Saaty) : poids = vecteur propre principal normalisé de la matrice réciproque A
  (variante : moyenne géométrique des lignes). IC = (λmax − n)/(n − 1), RC = IC / IA (IA de Saaty).
  RC < 10 % : cohérence acceptable. En cas d'incohérence, `worst_judgements()` liste les jugements
  a_ij les plus éloignés du ratio w_i/w_j impliqué.
- **ROC** : w_i = (1/n) Σ_{k=i..n} 1/k pour le critère classé i-ème. Utile quand le décideur sait classer
  mais pas quantifier.

## Delphi

Pour chaque tour : médiane, Q1, Q3 (interpolation linéaire), IQR = Q3 − Q1. Consensus si IQR ≤ seuil
(1 par défaut) ; stabilité si la médiane ne change plus. Valeur retenue = médiane du dernier tour.

## Normalisations

| nom | formule | réoriente les critères « min » |
|---|---|---|
| none | x | non |
| vector | x / √Σx² | non (TOPSIS traite le sens dans l'idéal) |
| minmax | (x − min)/(max − min), puis 1 − r pour « min » | oui |
| max | x / max ; min / x pour « min » | oui |
| sum | x / Σx | non |

## Méthodes d'agrégation

| Méthode | Score | Logique |
|---|---|---|
| Somme pondérée (WSM) | S(a) = Σ w_j r_aj | totalement compensatoire |
| TOPSIS | Ci* = D⁻ / (D⁺ + D⁻), distances euclidiennes à l'idéal / anti-idéal de V = w·R | compensatoire, dépend de la normalisation |

Toutes respectent la dominance (testé). Les différences de classement entre méthodes viennent de :
la logique d'agrégation (somme compensatoire vs distance à l'idéal), la normalisation (étirement des écarts par
min-max) et le traitement cardinal vs ordinal des données.

## Sensibilité

- **Variation d'un poids** : w_j ← v, les autres poids sont mis à l'échelle (1 − v)/(1 − w_j) pour garder Σ = 1.
- **Seuil de basculement** : grille fine (pas 0,0005) puis dichotomie ; donne le poids exact à partir
  duquel le 1er change, vers le haut et vers le bas, et la variation relative |seuil − w_j| / w_j.
- **Monte-Carlo** : poids ~ Dirichlet(k·w). k est choisi pour qu'un poids moyen ait un coefficient de
  variation égal à `spread` (0,2 ≈ ±20 %) ; `spread=None` = tirage uniforme sur le simplexe. Sorties :
  fréquence de 1re place, rang moyen, distribution des rangs.
- **Profils** : multiplicateurs appliqués aux poids de référence puis renormalisés (ou poids égaux).
- **Scénarios de données** : modification de cellules (élément de base → ré-agrégé dans tous les
  portefeuilles ; combinaison → surcharge). Les alternatives devenues inadmissibles sont signalées.
- **Seuil d'une valeur** : balayage puis dichotomie sur une cellule.

Toutes les analyses portant sur les poids utilisent un moteur vectorisé (`WeightScorer`) qui évalue des
milliers de jeux de poids d'un coup ; un test garantit l'égalité avec les fonctions de référence.

## Diagnostic de robustesse (`diagnose`)

| Niveau | Règle (seuils paramétrables) |
|---|---|
| fragile | 1er dans < 60 % des tirages **ou** un poids bascule pour < 10 % de variation relative **ou** la majorité des scénarios de données basculent |
| robuste | ≥ 90 % des tirages **et** toute bascule de poids exige ≥ 50 % de variation **et** toutes les méthodes ont le même 1er **et** aucun scénario ne bascule |
| modérément robuste | autres cas |
