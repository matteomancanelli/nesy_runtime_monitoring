"""Tests for the CILP-encoded RuleRunner network.

The CILP network and the symbolic `RuleEngine` should agree on every
per-cell verdict: each rule maps to one hidden unit, sign activation
matches set-membership semantics, OR-accumulation across iterations
matches the engine's eval-loop fixed-point.

We piggyback on the engine's sweep formulas and traces. The known
nested-temporal limitation (see CLAUDE.md § Paradigm 2) is inherited
from the rule system itself; the same xfail set applies here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.runner import random_traces
from src.monitors.base import Verdict
from src.monitors.rulerunner.cilp import CILPRunner
from src.monitors.rulerunner.engine import RuleEngine
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ----------------- IJCNN worked example -----------------


def test_ijcnn_worked_example_cilp() -> None:
    """The paper's §III trace: SATISFY at cell 3."""
    r = CILPRunner.from_formula("a | F b")
    assert r.step({"a": False, "b": False}) is Verdict.UNDECIDED
    assert r.step({"a": True, "b": False}) is Verdict.UNDECIDED
    assert r.step({"a": False, "b": True}) is Verdict.SATISFY


def test_absorbing_after_satisfy_cilp() -> None:
    r = CILPRunner.from_formula("F a")
    assert r.step({"a": True}) is Verdict.SATISFY
    assert r.step({"a": False}) is Verdict.SATISFY  # absorbing
    assert r.final_verdict() is Verdict.SATISFY


# ----------------- per-operator hand traces -----------------


def test_atom() -> None:
    assert CILPRunner.from_formula("a").run([{"a": True}]) is Verdict.SATISFY
    assert CILPRunner.from_formula("a").run([{"a": False}]) is Verdict.VIOLATE


def test_not() -> None:
    assert CILPRunner.from_formula("!a").run([{"a": True}]) is Verdict.VIOLATE
    assert CILPRunner.from_formula("!a").run([{"a": False}]) is Verdict.SATISFY


def test_eventually_satisfies() -> None:
    r = CILPRunner.from_formula("F a")
    assert r.run([{"a": False}, {"a": False}, {"a": True}]) is Verdict.SATISFY


def test_eventually_end_of_trace_violates() -> None:
    r = CILPRunner.from_formula("F a")
    assert r.run([{"a": False}, {"a": False}]) is Verdict.VIOLATE


def test_until_satisfies_and_violates() -> None:
    r = CILPRunner.from_formula("a U b")
    assert r.run([{"a": True, "b": False}, {"a": False, "b": True}]) is Verdict.SATISFY
    r.reset()
    assert r.run([{"a": True, "b": False}, {"a": False, "b": False}]) is Verdict.VIOLATE


def test_strong_next() -> None:
    r = CILPRunner.from_formula("X a")
    assert r.run([{"a": True}]) is Verdict.VIOLATE  # no next cell
    r.reset()
    assert r.run([{"a": False}, {"a": True}]) is Verdict.SATISFY


def test_weak_next() -> None:
    r = CILPRunner.from_formula("WX a")
    assert r.run([{"a": True}]) is Verdict.SATISFY  # no next cell


# ----------------- equivalence sweeps -----------------

# Same formula sets as test_rulerunner_engine.py; deliberately kept in
# sync so any divergence is a CILP-encoding bug, not a rule-system bug.
_SWEEP_FORMULAS = [
    "a", "!a", "a & b", "a | b", "a -> b",
    "(a & b) | (!a & !b)",
    "F a", "G a", "X a", "WX a",
    "a U b", "a R b",
    "F (a & b)", "G (a | b)",
    "F ((a & b) | (a & c))",
    "F ((a & b) | (a & c) | (a & d))",
    "(F a) -> (F b)",
    "G (!a | !b)",
]

_NESTED_TEMPORAL_XFAIL = [
    "F (a & X b)",
    "G (a -> F b)",
    "G (a -> X b)",
]


def _stable_seed(formula: str) -> int:
    """Deterministic across processes (Python's `hash()` is randomised
    per session via PYTHONHASHSEED)."""
    import hashlib
    return int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)


def _cilp_vs_engine_sweep(formula: str) -> None:
    """CILP and RuleEngine must produce identical verdicts on every trace."""
    rng = np.random.default_rng(seed=_stable_seed(formula))
    atoms = ("a", "b", "c", "d")
    traces = random_traces(atoms, trace_length=12, n_traces=40, rng=rng)
    cilp = CILPRunner.from_formula(formula)
    engine = RuleEngine.from_formula(formula)
    for trace in traces:
        c = cilp.run(trace)
        e = engine.run(trace)
        assert c is e, (
            f"CILP/Engine mismatch on {formula!r}, trace {trace}: "
            f"cilp={c} engine={e}"
        )


def _cilp_vs_dfa_sweep(formula: str, n_traces: int = 40) -> None:
    """End-to-end check: CILP matches the canonical DFA monitor too."""
    rng = np.random.default_rng(seed=_stable_seed(formula))
    atoms = ("a", "b", "c", "d")
    traces = random_traces(atoms, trace_length=12, n_traces=n_traces, rng=rng)
    cilp = CILPRunner.from_formula(formula)
    dfa = SymbolicDFAMonitor.compile(formula)
    for trace in traces:
        c = cilp.run(trace)
        d = dfa.run(trace)
        assert c is d, (
            f"CILP/DFA mismatch on {formula!r}, trace {trace}: "
            f"cilp={c} dfa={d}"
        )


@pytest.mark.parametrize("formula", _SWEEP_FORMULAS)
def test_cilp_matches_engine(formula: str) -> None:
    _cilp_vs_engine_sweep(formula)


@pytest.mark.parametrize("formula", _SWEEP_FORMULAS)
def test_cilp_matches_symbolic_dfa(formula: str) -> None:
    _cilp_vs_dfa_sweep(formula)


@pytest.mark.parametrize("formula", _NESTED_TEMPORAL_XFAIL)
@pytest.mark.xfail(
    strict=True,
    reason="Inherited from the rule system: IJCNN's single-literal "
           "encoding cannot disambiguate concurrent instances of nested "
           "temporal subformulae. See CLAUDE.md § Paradigm 2.",
)
def test_cilp_diverges_on_nested_temporal(formula: str) -> None:
    # Use the same 80-trace budget as the engine's nested-temporal xfail so
    # the rare divergent trace is deterministically inside the sample.
    _cilp_vs_dfa_sweep(formula, n_traces=80)
