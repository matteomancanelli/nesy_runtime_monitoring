"""DeepDFAMonitorScan: the parallel prefix-scan batched run must be verdict-for-
verdict identical to the sequential DeepDFA and the symbolic DFA.

The scan folds the whole batched trace into O(log L) matmuls instead of L
per-cell bmm()s (Phase 0.6). It is a performance path only — it must not change
any verdict. These tests pin that equivalence on the standard formula sweep, the
state-scaling family, ragged/edge-case batches, and the low-memory fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.formulas import STATE_SCALING_DEADLINES, bounded_response
from src.benchmarks.runner import random_traces
from src.monitors import deep_dfa
from src.monitors.deep_dfa import DeepDFAMonitor, DeepDFAMonitorScan
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

_FORMULAS = [
    "a", "F a", "G a", "X a", "WX a", "a U b", "a R b",
    "F (a & b)", "G (a | b)", "F ((a & b) | (a & c))",
    "(F a) -> (F b)", "G (!a | !b)",
    "F (a & X b)", "G (a -> F b)", "G (a -> X b)",  # nested temporal — DeepDFA exact
]


@pytest.mark.parametrize("formula", _FORMULAS)
def test_scan_matches_sequential_and_symbolic(formula: str) -> None:
    rng = np.random.default_rng(0)
    traces = random_traces(("a", "b", "c", "d"), 40, 30, rng=rng)
    scan = DeepDFAMonitorScan.compile(formula)
    seq = DeepDFAMonitor.compile(formula, mode="dense")
    sym = SymbolicDFAMonitor.compile(formula)

    v_scan = scan.batch_run(traces, early_termination=False)
    v_seq = seq.batch_run(traces, early_termination=False)
    v_sym = [sym.run(t) for t in traces]
    assert v_scan == v_seq, f"scan != sequential on {formula!r}"
    assert v_scan == v_sym, f"scan != symbolic on {formula!r}"
    # early_termination flag must not change the reconstructed verdicts
    assert scan.batch_run(traces, early_termination=True) == v_scan


@pytest.mark.parametrize("k", STATE_SCALING_DEADLINES[:4])
def test_scan_on_state_scaling_family(k: int) -> None:
    """Larger automata (bounded response) — scan stays exact as |Q| grows."""
    f = bounded_response(k)
    rng = np.random.default_rng(k)
    traces = random_traces(f.atoms, 60, 40, rng=rng)
    scan = DeepDFAMonitorScan.compile(f.formula)
    sym = SymbolicDFAMonitor.compile(f.formula)
    v_sym = [sym.run(t) for t in traces]
    assert scan.batch_run(traces, early_termination=False) == v_sym


def test_scan_edge_cases() -> None:
    scan = DeepDFAMonitorScan.compile("G (a -> F b)")
    seq = DeepDFAMonitor.compile("G (a -> F b)", mode="dense")
    assert scan.batch_run([]) == []
    # single trace, length-1, and ragged lengths in one batch
    ragged = [[{"a": True, "b": False}],
              [{"a": False, "b": True}] * 7,
              [{"a": True, "b": False}] * 3 + [{"a": False, "b": True}]]
    assert scan.batch_run(ragged) == seq.batch_run(ragged)


def test_scan_memory_fallback(monkeypatch) -> None:
    """Past the memory cap the scan falls back to the sequential loop, unchanged."""
    monkeypatch.setattr(deep_dfa, "SCAN_MEM_LIMIT_BYTES", 0)  # force fallback
    scan = DeepDFAMonitorScan.compile("F ((a & b) | (a & c))")
    seq = DeepDFAMonitor.compile("F ((a & b) | (a & c))", mode="dense")
    rng = np.random.default_rng(1)
    traces = random_traces(("a", "b", "c"), 30, 20, rng=rng)
    assert (scan.batch_run(traces, early_termination=False)
            == seq.batch_run(traces, early_termination=False))
