"""Reproduction des résultats publiés (articles et ouvrages) — voir mcdakit/benchmarks.py."""
import numpy as np
import pytest

from mcdakit.benchmarks import BENCHMARKS, run_benchmarks


@pytest.mark.parametrize("bench", BENCHMARKS, ids=[b.key for b in BENCHMARKS])
def test_published_values(bench):
    got = np.asarray(bench.compute(), dtype=float)
    assert got.shape == (len(bench.published),)
    assert np.max(np.abs(got - np.asarray(bench.published))) <= bench.tol + 1e-12


def test_all_rankings_match():
    r = run_benchmarks()
    assert r["conforme"].all() and r["même classement"].all()
