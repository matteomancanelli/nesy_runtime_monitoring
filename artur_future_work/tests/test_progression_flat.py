"""Flat CILP network for the progression-based RuleRunner (Part 2b).

The batched, multi-hot-root network must be verdict-for-verdict identical to
the eager / lazy / symbolic monitors on the full sweep (including
nested-temporal), and ``batch_run`` must equal ``[run(t) ...]`` on both CPU
and CUDA.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.benchmarks.runner import random_traces
from src.monitors.progression import (
    ProgressionRuleRunnerEagerMonitor,
    ProgressionRuleRunnerMonitor,
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
def test_flat_matches_eager_and_dfa(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), trace_length=12, n_traces=80, rng=rng)
    flat = ProgressionRuleRunnerMonitor.compile(formula)
    eager = ProgressionRuleRunnerEagerMonitor.compile(formula)
    dfa = SymbolicDFAMonitor.compile(formula)
    for trace in traces:
        fv, ev, dv = flat.run(trace), eager.run(trace), dfa.run(trace)
        assert fv is dv, f"flat vs dfa mismatch on {formula!r}, trace {trace}"
        assert fv is ev, f"flat vs eager mismatch on {formula!r}, trace {trace}"


@pytest.mark.parametrize("formula", _ALL)
def test_flat_batch_equals_sequential_cpu(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula) ^ 0x5A5A)
    traces = random_traces(("a", "b", "c", "d"), trace_length=10, n_traces=64, rng=rng)
    flat = ProgressionRuleRunnerMonitor.compile(formula, device="cpu")
    seq = [flat.run(t) for t in traces]
    bat = flat.batch_run(traces)
    assert bat == seq, f"batch != sequential on {formula!r}"


def test_flat_ragged_batch() -> None:
    """Mixed trace lengths (incl. length 0 and 1) in one batch."""
    flat = ProgressionRuleRunnerMonitor.compile("G (a -> F b)")
    traces = [
        [],
        [{"a": True, "b": False}],
        [{"a": True, "b": False}, {"a": False, "b": True}],
        [{"a": True, "b": False}, {"a": False, "b": False}],
    ]
    assert flat.batch_run(traces) == [flat.run(t) for t in traces]


def test_flat_counterexample() -> None:
    phi_b = "F (a & X b)"
    A = [{"a": True, "b": False}, {"a": False, "b": False}, {"a": False, "b": True}]
    B = [{"a": False, "b": False}, {"a": True, "b": False}, {"a": False, "b": True}]
    from src.monitors.base import Verdict

    assert ProgressionRuleRunnerMonitor.compile(phi_b).run(A) is Verdict.VIOLATE
    assert ProgressionRuleRunnerMonitor.compile(phi_b).run(B) is Verdict.SATISFY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("formula", ["G (a -> F b)", "F (a & X b)", "a U b"])
def test_flat_batch_equals_sequential_cuda(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula) ^ 0xC0DA)
    traces = random_traces(("a", "b", "c", "d"), trace_length=10, n_traces=64, rng=rng)
    flat = ProgressionRuleRunnerMonitor.compile(formula, device="cuda")
    assert flat.effective_device == "cuda"
    seq = [flat.run(t) for t in traces]
    bat = flat.batch_run(traces)
    assert bat == seq, f"CUDA batch != sequential on {formula!r}"
