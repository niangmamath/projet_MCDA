"""mcdakit — boîte à outils générique d'aide à la décision multicritère (MCDA).

Usage minimal :

    from mcdakit import DecisionProblem, evaluate, full_analysis, to_markdown
    problem = DecisionProblem.load("examples/stratema_methode2.json")
    print(evaluate(problem, "topsis").table(problem.labels()))
    print(to_markdown(full_analysis(problem, ["wsm", "topsis"])))
"""
from .analysis import compare_methods, evaluate, kendall_tau, run_methods
from .delphi import delphi, delphi_table
from .methods import METHODS, MethodResult, topsis, weighted_sum
from .problem import Attribute, Constraint, Criterion, DecisionProblem, Portfolio
from .report import FullAnalysis, full_analysis, to_excel, to_html, to_markdown
from .sensitivity import (WeightScorer, data_scenarios, diagnose, monte_carlo, value_threshold,
                          weight_profiles, weight_sweep, weight_thresholds)
from .weights import ahp, ahp_from_judgements, rank_order_centroid, resolve_weights

__version__ = "1.0.0"
