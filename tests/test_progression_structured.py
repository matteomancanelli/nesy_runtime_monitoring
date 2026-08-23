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
    build_factorized_progression_graph,
)
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

_ALL = [
    "a",
    "!a",
    "a & b",
    "a | b",
    "a -> b",
    "(a & b) | (!a & !b)",
    "F a",
    "G a",
    "X a",
    "WX a",
    "a U b",
    "a R b",
    "F (a & b)",
    "G (a | b)",
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

    for trace in traces[:10]:
        struct.reset()
        dfa.reset()
        for obs in trace:
            assert struct.step(obs) is dfa.step(obs), (
                f"online structured vs dfa on {formula!r}, {obs}"
            )


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
    formulas = [
        "X a",
        "WX a",
        "F a",
        "G a",
        "a U b",
        "a R b",
        "F (a & X b)",
        "G (a -> F b)",
    ]
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


def test_structured_mixed_batch_preserves_empty_trace_semantics() -> None:
    struct = ProgressionRuleRunnerStructuredMonitor.compile("G a")
    traces = [[], [{"a": False}], [{"a": True}]]
    assert struct.batch_run(traces) == [struct.run(trace) for trace in traces]


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
    """Evaluation and recurrence expose syntactically-owned subnetworks."""
    struct = ProgressionRuleRunnerStructuredMonitor.compile("G (a -> F b)")
    net = struct._net
    graph = build_factorized_progression_graph("G (a -> F b)")
    assert set(net.eval_net) == {node.key for node in graph.closure}
    assert set(net.react_net) == {root.key for root in graph.roots}
    # closure is bottom-up: every child precedes its parent
    pos = {n.key: i for i, n in enumerate(graph.closure)}
    for n in graph.closure:
        for c in n.args:
            assert pos[c.key] < pos[n.key]


def test_reactivation_modules_depend_on_one_root_only() -> None:
    """No progression clause recognizes the complete aggregate root set."""
    net = ProgressionRuleRunnerStructuredMonitor.compile("F (a & X b) & G c")._net
    for source, root in enumerate(net.roots):
        W_ih = net.react_net[root.key][0]
        used_root_columns = set(
            torch.nonzero(W_ih[:, : net.n_roots], as_tuple=False)[:, 1].tolist()
        )
        assert used_root_columns <= {source}


def test_local_reactivation_matches_factorized_graph() -> None:
    """Every reachable aggregate transition is the union of root modules."""
    net = ProgressionRuleRunnerStructuredMonitor.compile(
        "F (a & X b) & G (c -> F a)"
    )._net
    graph = net.graph
    for state_index, state in enumerate(graph.states):
        relevant = tuple(
            sorted({atom for root in state for atom in graph.relevant[root]})
        )
        for symbol, expected_index in graph.trans[state_index].items():
            root_state = torch.full((1, net.n_roots), -1.0)
            for root in state:
                root_state[0, root] = 1.0
            atoms = torch.full((1, net.n_atoms), -1.0)
            for j, atom in enumerate(relevant):
                if (symbol >> j) & 1:
                    atoms[0, net.atom_index[atom]] = 1.0
            react_input = torch.cat([root_state, atoms], dim=1)
            actual = torch.full_like(root_state, -1.0)
            for root in net.roots:
                output = net._forward(net.react_net[root.key], react_input)
                actual = torch.maximum(actual, output[:, : net.n_roots])
            actual_set = frozenset(
                torch.nonzero(actual[0] > 0, as_tuple=False).flatten().tolist()
            )
            assert actual_set == graph.states[expected_index]


def test_global_head_only_labels_factorized_states() -> None:
    net = ProgressionRuleRunnerStructuredMonitor.compile("(X a) & (X !a)")._net
    for index, state in enumerate(net.graph.states):
        row = torch.full((1, net.n_roots), -1.0)
        for root in state:
            row[0, root] = 1.0
        labels = net._forward(net.label_layer, row)
        assert bool(labels[0, 0] > 0) is (index in net.graph.accepting_sinks)
        assert bool(labels[0, 1] > 0) is (index in net.graph.trap_states)


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
