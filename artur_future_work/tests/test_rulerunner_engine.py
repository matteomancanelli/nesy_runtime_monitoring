"""Tests for the symbolic RuleRunner executor.

Three layers:
1. Hand-picked traces per operator — easy to debug template bugs.
2. The IJCNN 2014 §III worked example for `a ∨ ◇b`.
3. Randomized equivalence sweep vs SymbolicDFAMonitor — the actual
   correctness oracle for the template + executor + end-of-trace logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.runner import random_traces
from src.monitors.base import Verdict
from src.monitors.rulerunner.engine import RuleEngine
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ----------------- IJCNN worked example -----------------


def test_ijcnn_worked_example_trace() -> None:
    """Paper §III: monitoring `a ∨ ◇b` over [c, a, b] yields SUCCESS at cell 3."""
    eng = RuleEngine.from_formula("a | F b")
    assert eng.step({"a": False, "b": False}) is Verdict.UNDECIDED
    assert eng.step({"a": True, "b": False}) is Verdict.UNDECIDED
    assert eng.step({"a": False, "b": True}) is Verdict.SATISFY


def test_ijcnn_worked_example_a_observed_at_cell_one() -> None:
    """If `a` is observed at the first cell, [a]T fires, OR resolves T."""
    eng = RuleEngine.from_formula("a | F b")
    assert eng.step({"a": True, "b": False}) is Verdict.SATISFY


def test_absorbing_after_satisfy() -> None:
    """Once decided, subsequent step() calls keep returning the same verdict."""
    eng = RuleEngine.from_formula("F a")
    assert eng.step({"a": True}) is Verdict.SATISFY
    assert eng.step({"a": False}) is Verdict.SATISFY
    assert eng.final_verdict() is Verdict.SATISFY


# ----------------- per-operator hand traces -----------------


def test_atom_satisfies_and_violates() -> None:
    sat = RuleEngine.from_formula("a")
    assert sat.run([{"a": True}]) is Verdict.SATISFY
    vio = RuleEngine.from_formula("a")
    assert vio.run([{"a": False}]) is Verdict.VIOLATE


def test_not_atom() -> None:
    eng = RuleEngine.from_formula("!a")
    assert eng.run([{"a": True}]) is Verdict.VIOLATE
    eng.reset()
    assert eng.run([{"a": False}]) is Verdict.SATISFY


def test_eventually_fires_on_first_occurrence() -> None:
    eng = RuleEngine.from_formula("F a")
    trace = [{"a": False}, {"a": False}, {"a": True}, {"a": False}]
    assert eng.run(trace) is Verdict.SATISFY


def test_eventually_violates_at_end_of_trace_when_never_seen() -> None:
    eng = RuleEngine.from_formula("F a")
    assert eng.run([{"a": False}, {"a": False}]) is Verdict.VIOLATE


def test_always_violates_on_first_failure() -> None:
    eng = RuleEngine.from_formula("G a")
    trace = [{"a": True}, {"a": True}, {"a": False}]
    assert eng.run(trace) is Verdict.VIOLATE


def test_always_satisfies_at_end_when_never_violated() -> None:
    eng = RuleEngine.from_formula("G a")
    assert eng.run([{"a": True}, {"a": True}]) is Verdict.SATISFY


def test_until_resolves_when_b_arrives_and_a_held() -> None:
    eng = RuleEngine.from_formula("a U b")
    trace = [{"a": True, "b": False}, {"a": True, "b": False}, {"a": False, "b": True}]
    assert eng.run(trace) is Verdict.SATISFY


def test_until_violates_when_a_drops_before_b() -> None:
    eng = RuleEngine.from_formula("a U b")
    trace = [{"a": True, "b": False}, {"a": False, "b": False}]
    assert eng.run(trace) is Verdict.VIOLATE


def test_strong_next_violates_at_end_of_trace() -> None:
    """X φ at cell i needs cell i+1 to exist; if it doesn't, F."""
    eng = RuleEngine.from_formula("X a")
    assert eng.run([{"a": True}]) is Verdict.VIOLATE


def test_weak_next_satisfies_at_end_of_trace() -> None:
    eng = RuleEngine.from_formula("WX a")
    assert eng.run([{"a": True}]) is Verdict.SATISFY


def test_strong_next_satisfies_when_next_cell_holds() -> None:
    eng = RuleEngine.from_formula("X a")
    assert eng.run([{"a": False}, {"a": True}]) is Verdict.SATISFY


# ----------------- randomized equivalence sweep -----------------

# See engine.py "Known limitation" section: the IJCNN single-literal
# encoding cannot disambiguate concurrent instances of a temporal
# subformula. Formulas that nest a temporal operator under F/G/U/R
# trigger this, so the equivalence sweep is restricted to flat-temporal
# and temporal-under-propositional formulas. The three formulas that
# exercise the limitation are marked xfail below so the regression is
# visible without breaking the suite.
_SWEEP_FORMULAS = [
    # atoms and propositional
    "a", "!a", "a & b", "a | b", "a -> b",
    "(a & b) | (!a & !b)",  # a <-> b
    # single-temporal
    "F a", "G a", "X a", "WX a",
    "a U b", "a R b",
    # temporal-under-propositional (safe — the temporal operator is the
    # *outer* one over a propositional combination of atoms)
    "F (a & b)", "G (a | b)",
    "F ((a & b) | (a & c))",            # IJCNN 2014 family, n=4
    "F ((a & b) | (a & c) | (a & d))",  # n=6
    "(F a) -> (F b)",
    "G (!a | !b)",  # mutual exclusion
]

_NESTED_TEMPORAL_XFAIL = [
    "F (a & X b)",     # X nested under propositional under F
    "G (a -> F b)",    # F nested under -> under G (response pattern)
    "G (a -> X b)",    # X nested under -> under G (chain response)
]


def _stable_seed(formula: str) -> int:
    """Deterministic across processes. Python's built-in `hash()` is
    randomised per interpreter session (PYTHONHASHSEED), which would
    make the rare nested-temporal mismatch absent in some sessions and
    flip xfail-strict tests to XPASS."""
    import hashlib
    return int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)


def _sweep_one(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    atoms = ("a", "b", "c", "d")
    traces = random_traces(atoms, trace_length=12, n_traces=80, rng=rng)
    engine = RuleEngine.from_formula(formula)
    dfa = SymbolicDFAMonitor.compile(formula)
    for trace in traces:
        engine_v = engine.run(trace)
        dfa_v = dfa.run(trace)
        assert engine_v is dfa_v, (
            f"Mismatch on formula {formula!r}, trace {trace}: "
            f"engine={engine_v} vs dfa={dfa_v}"
        )


@pytest.mark.parametrize("formula", _SWEEP_FORMULAS)
def test_engine_matches_symbolic_dfa(formula: str) -> None:
    """For every flat-temporal formula in the sweep, RuleEngine agrees
    with SymbolicDFAMonitor on every random trace (80 traces × 12 cells)."""
    _sweep_one(formula)


@pytest.mark.parametrize("formula", _NESTED_TEMPORAL_XFAIL)
@pytest.mark.xfail(
    strict=True,
    reason="Nested temporal under F/G/U/R: IJCNN 2014's single-literal "
           "encoding cannot disambiguate concurrent instances of the inner "
           "temporal subformula. See engine.py 'Known limitation' section.",
)
def test_engine_diverges_on_nested_temporal(formula: str) -> None:
    """xfail-tracked formulas: the engine is expected to disagree with the
    DFA monitor. Marked strict so any future fix flips this to PASS and
    surfaces the regression."""
    _sweep_one(formula)


def test_engine_matches_symbolic_dfa_short_traces() -> None:
    """Short traces (length 1-3) are most likely to expose end-of-trace bugs."""
    rng = np.random.default_rng(seed=7)  # fixed int is fine: not formula-keyed
    atoms = ("a", "b")
    formulas = ["X a", "WX a", "F a", "G a", "a U b", "a R b", "a -> F b"]
    for L in (1, 2, 3):
        traces = random_traces(atoms, trace_length=L, n_traces=40, rng=rng)
        for f in formulas:
            eng = RuleEngine.from_formula(f)
            dfa = SymbolicDFAMonitor.compile(f)
            for trace in traces:
                assert eng.run(trace) is dfa.run(trace), (
                    f"Mismatch on {f!r}, L={L}, trace={trace}"
                )
