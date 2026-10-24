"""Interface en ligne de commande.

    mcdakit run problem.json -m wsm:normalization=none -m topsis --report rapport.html
    mcdakit ahp "1,1/2,3;2,1,4;1/3,1/4,1" --labels C1,C2,C3
    mcdakit delphi "2,1,3,3,2" "2,2,2,3,2"
    mcdakit template modele.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .analysis import evaluate
from .delphi import delphi
from .methods import METHOD_LABELS, METHODS
from .problem import DecisionProblem
from .report import full_analysis, to_excel, to_html, to_markdown
from .weights import ahp


def _method_spec(text: str) -> tuple[str, dict]:
    """"topsis" ou "wsm:normalization=none"."""
    name, _, rest = text.partition(":")
    if name not in METHODS:
        raise argparse.ArgumentTypeError(f"méthode inconnue {name} ({', '.join(METHODS)})")
    params = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    return name, params


def cmd_run(args) -> int:
    problem = DecisionProblem.load(args.problem)
    methods = dict(args.method or [("wsm", {}), ("topsis", {}), ("promethee", {})])
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8")) if args.scenarios else None
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8")) if args.profiles else None
    fa = full_analysis(problem, methods, primary=args.primary, mc_runs=args.mc, mc_spread=args.spread,
                       scenarios=scenarios, profiles=profiles)
    pd.set_option("display.width", 160)
    for key, res in fa.results.items():
        print(f"\n=== {METHOD_LABELS[key]} ===")
        print(res.table(fa.labels).round(4).to_string())
    print(f"\nRecommandation ({METHOD_LABELS[fa.primary]}) : {fa.labels[fa.winner]}")
    print(f"Diagnostic : {fa.diagnosis}")
    if args.report:
        out = Path(args.report)
        if out.suffix.lower() in (".html", ".htm"):
            out.write_text(to_html(fa), encoding="utf-8")
        elif out.suffix.lower() == ".xlsx":
            out.write_bytes(to_excel(fa))
        else:
            out.write_text(to_markdown(fa), encoding="utf-8")
        print(f"Rapport écrit : {out}")
    return 0


def cmd_rank(args) -> int:
    problem = DecisionProblem.load(args.problem)
    name, params = args.method
    res = evaluate(problem, name, **params)
    print(res.table(problem.labels()).round(4).to_string())
    return 0


def cmd_ahp(args) -> int:
    rows = [r.split(",") for r in args.matrix.split(";")]
    labels = args.labels.split(",") if args.labels else None
    res = ahp(rows, labels)
    print(pd.DataFrame({"poids": res.weights}).round(6).to_string())
    print(f"λmax = {res.lambda_max:.6f}  IC = {res.ci:.5f}  IA = {res.ri:.2f}  RC = {res.cr:.2%}  "
          f"{'cohérent' if res.consistent else 'INCOHÉRENT'}")
    return 0


def cmd_delphi(args) -> int:
    res = delphi(args.rounds, args.iqr)
    for k, r in enumerate(res.rounds, 1):
        print(f"Tour {k} : votes {r.votes}  médiane {r.median:g}  [Q1 {r.q1:g} – Q3 {r.q3:g}]  IQR {r.iqr:g}")
    print(f"Valeur retenue : {res.value:g} — {res.status}")
    return 0


def cmd_template(args) -> int:
    from .io_excel import write_excel
    src = Path(__file__).resolve().parent.parent / "examples" / "choix_fournisseur.yaml"
    problem = DecisionProblem.load(args.source or src)
    write_excel(problem, args.output)
    print(f"Modèle Excel écrit : {args.output}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mcdakit", description="Aide à la décision multicritère")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse complète (méthodes, comparaison, sensibilité, rapport)")
    r.add_argument("problem", help="fichier .json, .yaml ou .xlsx")
    r.add_argument("-m", "--method", action="append", type=_method_spec,
                   help="méthode[:param=valeur,...] (répétable). Défaut : wsm, topsis, promethee")
    r.add_argument("--primary", help="méthode de référence pour la sensibilité (défaut : la première)")
    r.add_argument("--mc", type=int, default=5000, help="nombre de tirages Monte-Carlo (0 pour désactiver)")
    r.add_argument("--spread", type=float, default=0.2, help="dispersion relative des poids tirés")
    r.add_argument("--scenarios", help="JSON {nom: [[alternative, colonne, valeur], ...]}")
    r.add_argument("--profiles", help="JSON {nom: {critère: multiplicateur}}")
    r.add_argument("--report", help="rapport à écrire (.md, .html ou .xlsx)")
    r.set_defaults(func=cmd_run)

    k = sub.add_parser("rank", help="classement avec une seule méthode")
    k.add_argument("problem")
    k.add_argument("-m", "--method", type=_method_spec, default=("topsis", {}))
    k.set_defaults(func=cmd_rank)

    a = sub.add_parser("ahp", help="poids et cohérence d'une matrice AHP")
    a.add_argument("matrix", help='lignes séparées par ";" et valeurs par "," ex. "1,3;1/3,1"')
    a.add_argument("--labels")
    a.set_defaults(func=cmd_ahp)

    d = sub.add_parser("delphi", help="synthèse des tours d'une consultation Delphi")
    d.add_argument("rounds", nargs="+", help='un argument par tour, ex. "2,1,3,3,2"')
    d.add_argument("--iqr", type=float, default=1.0, help="seuil de consensus sur l'écart interquartile")
    d.set_defaults(func=cmd_delphi)

    t = sub.add_parser("template", help="génère un modèle Excel à remplir")
    t.add_argument("output")
    t.add_argument("--source", help="problème à convertir (défaut : exemple fournisseur)")
    t.set_defaults(func=cmd_template)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
