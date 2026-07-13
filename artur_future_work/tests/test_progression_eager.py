"""Eager residual-DFA construction (Part 2a).

The eager, table-driven monitor must be verdict-for-verdict identical to the
lazy engine and to ``SymbolicDFAMonitor`` on the full sweep (including the
nested-temporal formulas), the construction must terminate, and the
cost-of-correctness metrics must be well-formed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.benchmarks.runner import random_traces
from src.monitors.progression import (
    ProgressionEngine,
    ProgressionRuleRunnerEagerMonitor,
    build_progression_dfa,
)
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

_ALL = [
    "a", "!a", "a & b", "a | b", "a -> b",
    "(a & b) | (!a & !b)",
    "F a", "G a", "X a", "WX a",
    "a U b", "a R b",
    "F (a & b)", "G (a | b)",
    "F ((a & b) | (a & c))",
    "(F a) -> (F b)",
    "G (!a | !b)",
    # nested temporal — the fix must handle these in the eager form too
    "F (a & X b)",
    "G (a -> F b)",
    "G (a -> X b)",
    "F (a & X (b & X c))",
    "G (a -> X (F b))",
    "(a U (b & X c))",
    "a U (b U c)",
]


def _stable_seed(formula: str) -> int:
    return int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)


@pytest.mark.parametrize("formula", _ALL)
def test_eager_matches_lazy_and_dfa(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), trace_length=12, n_traces=80, rng=rng)
    eager = ProgressionRuleRunnerEagerMonitor.compile(formula)
    lazy = ProgressionEngine.compile(formula)
    dfa = SymbolicDFAMonitor.compile(formula)
    for trace in traces:
        ev, lv, dv = eager.run(trace), lazy.run(trace), dfa.run(trace)
        assert ev is dv, f"eager vs dfa mismatch on {formula!r}, trace {trace}"
        assert lv is dv, f"lazy vs dfa mismatch on {formula!r}, trace {trace}"


@pytest.mark.parametrize("formula", _ALL)
def test_construction_terminates_and_metrics(formula: str) -> None:
    dfa = build_progression_dfa(formula)
    assert dfa.n_states >= 1
    assert dfa.n_states == len(dfa.states)
    assert dfa.initial == 0
    # every state has aligned tables
    assert len(dfa.trans) == dfa.n_states
    assert len(dfa.relevant) == dfa.n_states
    # transitions land in-range
    for tr in dfa.trans:
        for j in tr.values():
            assert 0 <= j < dfa.n_states
    # metrics are positive and self-consistent
    assert dfa.n_roots >= 1
    assert dfa.n_closure >= dfa.n_roots
    assert dfa.n_input_sub >= 1


def test_eager_matches_dfa_short_traces() -> None:
    rng = np.random.default_rng(seed=11)
    formulas = ["X a", "WX a", "F a", "G a", "a U b", "a R b",
                "F (a & X b)", "G (a -> F b)"]
    for L in (1, 2, 3):
        traces = random_traces(("a", "b"), trace_length=L, n_traces=40, rng=rng)
        for f in formulas:
            eager = ProgressionRuleRunnerEagerMonitor.compile(f)
            dfa = SymbolicDFAMonitor.compile(f)
            for trace in traces:
                assert eager.run(trace) is dfa.run(trace), (
                    f"Mismatch on {f!r}, L={L}, trace={trace}"
                )


def test_state_count_matches_minimal_dfa_on_response() -> None:
    """G(a -> F b): the residual system oscillates between {G(a->Fb)} and
    {Fb, G(a->Fb)} — 2 reachable residual states, matching the 2-state
    minimal DFA. (Sanity that the eager construction is not over-splitting.)"""
    dfa = build_progression_dfa("G (a -> F b)")
    assert dfa.n_states == 2


def test_alphabet_blowup_cap() -> None:
    """A guard over many atoms must refuse rather than enumerate 2^k."""
    # F(a0 & (a1 | ... )) style guard over 6 atoms is fine (2^6 = 64);
    # force the cap low to exercise the guard.
    with pytest.raises(ValueError, match="alphabet-blowup"):
        build_progression_dfa("F ((a & b) | (a & c) | (a & d))", max_guard_atoms=2)
