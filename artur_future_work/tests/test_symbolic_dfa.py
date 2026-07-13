"""Tests for src/monitors/symbolic_dfa.py — step semantics, early termination, reset."""

import pytest
from src.monitors.base import Verdict
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def monitor(formula: str) -> SymbolicDFAMonitor:
    return SymbolicDFAMonitor.compile(formula)


# ---------------------------------------------------------------------------
# Basic SATISFY / VIOLATE (via run())
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace,expected", [
    ([{"a": True}],                     Verdict.SATISFY),
    ([{"a": False}, {"a": True}],       Verdict.SATISFY),
    ([{"a": False}, {"a": False}],      Verdict.VIOLATE),
])
def test_eventually(trace, expected):
    assert monitor("F a").run(trace) == expected


@pytest.mark.parametrize("trace,expected", [
    ([{"a": True}, {"a": True}],        Verdict.SATISFY),
    ([{"a": True}, {"a": False}],       Verdict.VIOLATE),
    ([{"a": False}],                    Verdict.VIOLATE),
])
def test_globally(trace, expected):
    assert monitor("G a").run(trace) == expected


@pytest.mark.parametrize("trace,expected", [
    ([{"a": True, "b": False}, {"a": False, "b": True}],  Verdict.SATISFY),
    ([{"a": False, "b": True}],                           Verdict.SATISFY),
    ([{"a": False, "b": False}],                          Verdict.VIOLATE),
    ([{"a": True, "b": False}, {"a": False, "b": False}], Verdict.VIOLATE),
])
def test_until(trace, expected):
    assert monitor("a U b").run(trace) == expected


@pytest.mark.parametrize("trace,expected", [
    ([{"a": True, "b": False}, {"a": False, "b": True}],   Verdict.SATISFY),
    ([{"a": False, "b": False}, {"a": False, "b": False}], Verdict.SATISFY),
    ([{"a": True, "b": False}, {"a": True, "b": False}],   Verdict.VIOLATE),
])
def test_response(trace, expected):
    assert monitor("G(a -> F b)").run(trace) == expected


# ---------------------------------------------------------------------------
# Early termination: step() decides before trace end
# ---------------------------------------------------------------------------


def test_early_satisfy_eventually():
    m = monitor("F a")
    assert m.step({"a": False}) == Verdict.UNDECIDED
    assert m.step({"a": True}) == Verdict.SATISFY   # accepting sink reached


def test_early_violate_globally():
    m = monitor("G a")
    assert m.step({"a": True}) == Verdict.UNDECIDED
    assert m.step({"a": False}) == Verdict.VIOLATE   # trap reached


def test_no_early_termination_response():
    # G(a -> F b) has neither trap nor accepting sink
    m = monitor("G(a -> F b)")
    for obs in [{"a": True, "b": False}, {"a": False, "b": True}]:
        v = m.step(obs)
        assert v == Verdict.UNDECIDED, f"unexpected early verdict {v} on {obs}"


# ---------------------------------------------------------------------------
# final_verdict() — binary end-of-trace check
# ---------------------------------------------------------------------------


def test_final_verdict_satisfy():
    m = monitor("G(a -> F b)")
    m.step({"a": True, "b": False})
    m.step({"a": False, "b": True})
    assert m.final_verdict() == Verdict.SATISFY


def test_final_verdict_violate():
    m = monitor("G(a -> F b)")
    m.step({"a": True, "b": False})
    m.step({"a": True, "b": False})
    assert m.final_verdict() == Verdict.VIOLATE


# ---------------------------------------------------------------------------
# reset() — state is cleared between traces
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    m = monitor("F a")
    m.step({"a": True})          # now in accepting sink
    m.reset()
    # after reset, first step on a=False should be UNDECIDED, not SATISFY
    assert m.step({"a": False}) == Verdict.UNDECIDED


def test_run_calls_reset():
    m = monitor("F a")
    assert m.run([{"a": True}]) == Verdict.SATISFY
    # second run on a different trace must start fresh
    assert m.run([{"a": False}]) == Verdict.VIOLATE


# ---------------------------------------------------------------------------
# batch_run()
# ---------------------------------------------------------------------------


def test_batch_run():
    m = monitor("F a")
    traces = [
        [{"a": True}],
        [{"a": False}],
        [{"a": False}, {"a": True}],
    ]
    results = m.batch_run(traces)
    assert results == [Verdict.SATISFY, Verdict.VIOLATE, Verdict.SATISFY]
