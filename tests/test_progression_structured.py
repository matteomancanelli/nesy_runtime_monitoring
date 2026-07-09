"""Structured (per-closure-node) progression RuleRunner (Part 3).

The structured monitor must be verdict-for-verdict identical to the flat /
eager / lazy / symbolic monitors on the full sweep (including nested-temporal
and the cross-root ``(X a) & (X ~a)`` collapse), ``batch_run`` must equal
``[run(t) ...]`` on CPU and CUDA, and its per-node evaluation subnetworks /
exposed closure must be well-formed (the local-learning substrate).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.benchmarks.runner import random_traces
from src.monitors.base import Verdict
from src.monitors.progression import (
    ProgressionRuleRunnerEagerMonitor,
    ProgressionRuleRunnerMonitor,
    ProgressionRuleRunnerStructuredMonitor,
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
    # nested temporal — the fix must handle these in the structured form too
    "F (a & X b)",
    "G (a -> F b)",
    "G (a -> X b)",
    "F (a & X (b & X c))",
    "G (a -> X (F b))",
    "(a U (b & X c))",
    "a U (b U c)",
    # cross-root canonicalization collapse -> VIOLATE no single root sees
    "(X a) & (X !a)",
]


def _stable_seed(formula: str) -> int:
    return int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)


@pytest.mark.parametrize("formula", _ALL)
def test_structured_matches_eager_flat_and_dfa(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), trace_length=12, n_traces=80, rng=rng)
    struct = ProgressionRuleRunnerStructuredMonitor.compile(formula)
    flat = ProgressionRuleRunnerMonitor.compile(formula)
    eager = ProgressionRuleRunnerEagerMonitor.compile(formula)
    dfa = SymbolicDFAMonitor.compile(formula)
    for trace in traces:
        sv = struct.run(trace)
        assert sv is dfa.run(trace), f"structured vs dfa on {formula!r}, {trace}"
        assert sv is flat.run(trace), f"structured vs flat on {formula!r}, {trace}"
        assert sv is eager.run(trace), f"structured vs eager on {formula!r}, {trace}"


@pytest.mark.parametrize("formula", _ALL)
def test_structured_batch_equals_sequential_cpu(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula) ^ 0x5A5A)
    traces = random_traces(("a", "b", "c", "d"), trace_length=10, n_traces=64, rng=rng)
    struct = ProgressionRuleRunnerStructuredMonitor.compile(formula, device="cpu")
    seq = [struct.run(t) for t in traces]
    bat = struct.batch_run(traces)
    assert bat == seq, f"batch != sequential on {formula!r}"


def test_structured_short_traces_match_dfa() -> None:
    """Length 1..3 stress the end-of-trace ``last`` bit and empty-word path."""
    rng = np.random.default_rng(seed=11)
    formulas = ["X a", "WX a", "F a", "G a", "a U b", "a R b",
                "F (a & X b)", "G (a -> F b)"]
    for L in (1, 2, 3):
        traces = random_traces(("a", "b"), trace_length=L, n_traces=40, rng=rng)
        for f in formulas:
            struct = ProgressionRuleRunnerStructuredMonitor.compile(f)
            dfa = SymbolicDFAMonitor.compile(f)
            for trace in traces:
                assert struct.run(trace) is dfa.run(trace), (
                    f"Mismatch on {f!r}, L={L}, trace={trace}"
                )


def test_structured_ragged_batch() -> None:
    """Mixed trace lengths (incl. length 0 and 1) in one batch."""
    struct = ProgressionRuleRunnerStructuredMonitor.compile("G (a -> F b)")
    traces = [
        [],
        [{"a": True, "b": False}],
        [{"a": True, "b": False}, {"a": False, "b": True}],
        [{"a": True, "b": False}, {"a": False, "b": False}],
    ]
    assert struct.batch_run(traces) == [struct.run(t) for t in traces]


def test_structured_counterexample() -> None:
    phi_b = "F (a & X b)"
    A = [{"a": True, "b": False}, {"a": False, "b": False}, {"a": False, "b": True}]
    B = [{"a": False, "b": False}, {"a": True, "b": False}, {"a": False, "b": True}]
    Mon = ProgressionRuleRunnerStructuredMonitor
    assert Mon.compile(phi_b).run(A) is Verdict.VIOLATE
    assert Mon.compile(phi_b).run(B) is Verdict.SATISFY


def test_structured_cross_root_violate() -> None:
    """``(X a) & (X ~a)`` progresses to ``a & ~a`` = FALSE — a VIOLATE that
    neither root produces alone. Exercises the global-canonicalization path the
    per-node evaluation deliberately does not attempt."""
    struct = ProgressionRuleRunnerStructuredMonitor.compile("(X a) & (X !a)")
    # unsatisfiable next cell -> VIOLATE; single cell has no successor for the
    # strong X either -> also VIOLATE.
    assert struct.run([{"a": True}, {"a": False}]) is Verdict.VIOLATE
    assert struct.run([{"a": False}]) is Verdict.VIOLATE


def test_structured_effective_device_is_cpu() -> None:
    struct = ProgressionRuleRunnerStructuredMonitor.compile(
        "G (a -> F b)", device="cpu"
    )
    assert struct.effective_device == "cpu"


def test_structured_exposes_per_node_subnetworks() -> None:
    """The local-learning substrate: one eval subnetwork per non-atom closure
    node, and the closure/roots are exposed in bottom-up order."""
    struct = ProgressionRuleRunnerStructuredMonitor.compile("G (a -> F b)")
    net = struct._net
    dfa = build_progression_dfa("G (a -> F b)")
    n_non_atom = sum(1 for n in dfa.closure if n.op.name != "ATOM")
    assert len(net.eval_layers) == n_non_atom
    # closure is bottom-up: every child precedes its parent
    pos = {n.key: i for i, n in enumerate(dfa.closure)}
    for n in dfa.closure:
        for c in n.args:
            assert pos[c.key] < pos[n.key]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("formula", ["G (a -> F b)", "F (a & X b)", "a U b"])
def test_structured_batch_equals_sequential_cuda(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula) ^ 0xC0DA)
    traces = random_traces(("a", "b", "c", "d"), trace_length=10, n_traces=64, rng=rng)
    struct = ProgressionRuleRunnerStructuredMonitor.compile(formula, device="cuda")
    assert struct.effective_device == "cuda"
    seq = [struct.run(t) for t in traces]
    bat = struct.batch_run(traces)
    assert bat == seq, f"CUDA batch != sequential on {formula!r}"
