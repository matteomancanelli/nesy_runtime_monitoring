"""Smoke tests for the Monitor-interface adapter.

The heavy correctness work is in test_rulerunner_cilp.py (sweep against
the engine and the DFA monitor). Here we just verify the Monitor ABC
plumbing: `compile` returns a working instance, `step`/`final_verdict`/
`reset` delegate correctly, and the inherited `run`/`batch_run` work.
"""

from __future__ import annotations

from src.monitors.base import Monitor, Verdict
from src.monitors.rulerunner import RuleRunnerMonitor


def test_is_a_monitor_subclass() -> None:
    assert issubclass(RuleRunnerMonitor, Monitor)


def test_compile_returns_instance() -> None:
    m = RuleRunnerMonitor.compile("F a")
    assert isinstance(m, RuleRunnerMonitor)


def test_run_via_inherited_default() -> None:
    """The base Monitor.run loops step() and falls back to final_verdict."""
    m = RuleRunnerMonitor.compile("F a")
    assert m.run([{"a": False}, {"a": True}]) is Verdict.SATISFY
    assert m.run([{"a": False}, {"a": False}]) is Verdict.VIOLATE


def test_batch_run_via_inherited_default() -> None:
    m = RuleRunnerMonitor.compile("F a")
    traces = [
        [{"a": False}, {"a": True}],
        [{"a": False}, {"a": False}],
        [{"a": True}],
    ]
    assert m.batch_run(traces) == [Verdict.SATISFY, Verdict.VIOLATE, Verdict.SATISFY]


def test_reset_returns_to_initial_state() -> None:
    m = RuleRunnerMonitor.compile("F a")
    m.step({"a": True})  # absorbs to SATISFY
    m.reset()
    # After reset, a fresh trace that should violate yields VIOLATE.
    assert m.run([{"a": False}]) is Verdict.VIOLATE


def test_ijcnn_worked_example_via_monitor() -> None:
    m = RuleRunnerMonitor.compile("a | F b")
    assert m.run([
        {"a": False, "b": False},
        {"a": True, "b": False},
        {"a": False, "b": True},
    ]) is Verdict.SATISFY
