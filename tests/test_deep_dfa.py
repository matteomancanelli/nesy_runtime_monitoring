"""Tests for the DeepDFA monitor (Paradigm 3).

DeepDFA is the canonical, exactly-correct neural monitor: it must agree
with SymbolicDFAMonitor on *every* trace, including the nested-temporal
formulas where RuleRunner diverges (no xfails here). We also check that
the dense and factored representations agree on crisp traces, that the
batched GPU path matches sequential runs, and that the soft (fractional)
transition matrix is row-stochastic and exact on read-once guards.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.benchmarks.runner import random_traces
from src.formula.compiler import compile_ltlf
from src.monitors.base import Verdict
from src.monitors.deep_dfa import DeepDFAMonitor, DeepDFATensor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


def _stable_seed(formula: str) -> int:
    return int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)


_FORMULAS = [
    "a", "!a", "a & b", "a | b", "a -> b",
    "(a & b) | (!a & !b)",
    "F a", "G a", "X a", "WX a", "a U b", "a R b",
    "F (a & b)", "G (a | b)",
    "F ((a & b) | (a & c))",
    "F ((a & b) | (a & c) | (a & d))",
    "(F a) -> (F b)", "G (!a | !b)",
    # nested temporal — DeepDFA is exact, unlike RuleRunner
    "F (a & X b)", "G (a -> F b)", "G (a -> X b)",
]


# ----------------- step semantics & verdicts -----------------


def test_eventually_early_satisfy() -> None:
    m = DeepDFAMonitor.compile("F a")
    assert m.step({"a": False}) is Verdict.UNDECIDED
    assert m.step({"a": True}) is Verdict.SATISFY
    assert m.step({"a": False}) is Verdict.SATISFY  # absorbing


def test_always_early_violate() -> None:
    m = DeepDFAMonitor.compile("G a")
    assert m.step({"a": True}) is Verdict.UNDECIDED
    assert m.step({"a": False}) is Verdict.VIOLATE


def test_strong_vs_weak_next_end_of_trace() -> None:
    assert DeepDFAMonitor.compile("X a").run([{"a": True}]) is Verdict.VIOLATE
    assert DeepDFAMonitor.compile("WX a").run([{"a": True}]) is Verdict.SATISFY


def test_reset() -> None:
    m = DeepDFAMonitor.compile("F a")
    m.run([{"a": True}])
    m.reset()
    assert m.run([{"a": False}]) is Verdict.VIOLATE


def test_nested_temporal_is_exact() -> None:
    """The response pattern G(a -> F b): exact, where RuleRunner diverges."""
    m = DeepDFAMonitor.compile("G (a -> F b)")
    good = [{"a": True, "b": False}, {"a": False, "b": True}]
    assert m.run(good) is Verdict.SATISFY
    m.reset()
    bad = [{"a": False, "b": False}, {"a": True, "b": False}]
    assert m.run(bad) is Verdict.VIOLATE


# ----------------- equivalence vs symbolic DFA -----------------


@pytest.mark.parametrize("formula", _FORMULAS)
def test_dense_matches_symbolic_dfa(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), 12, 60, rng=rng)
    deep = DeepDFAMonitor.compile(formula, mode="dense")
    dfa = SymbolicDFAMonitor.compile(formula)
    for t in traces:
        assert deep.run(t) is dfa.run(t), f"dense/DFA mismatch on {formula!r}"


@pytest.mark.parametrize("formula", _FORMULAS)
def test_factored_matches_symbolic_dfa(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), 12, 60, rng=rng)
    deep = DeepDFAMonitor.compile(formula, mode="factored")
    dfa = SymbolicDFAMonitor.compile(formula)
    for t in traces:
        assert deep.run(t) is dfa.run(t), f"factored/DFA mismatch on {formula!r}"


# ----------------- dense vs factored -----------------


@pytest.mark.parametrize("formula", _FORMULAS)
def test_dense_equals_factored(formula: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), 12, 40, rng=rng)
    dense = DeepDFAMonitor.compile(formula, mode="dense")
    fac = DeepDFAMonitor.compile(formula, mode="factored")
    for t in traces:
        assert dense.run(t) is fac.run(t), f"dense != factored on {formula!r}"


# ----------------- batched GPU path -----------------


@pytest.mark.parametrize("mode", ["dense", "factored"])
@pytest.mark.parametrize("formula", _FORMULAS)
def test_batch_run_matches_sequential(formula: str, mode: str) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(("a", "b", "c", "d"), 10, 32, rng=rng)
    m = DeepDFAMonitor.compile(formula, mode=mode)
    batched = m.batch_run(traces)
    sequential = [m.run(t) for t in traces]
    assert batched == sequential, f"batch != sequential on {formula!r} ({mode})"


def test_batch_run_empty() -> None:
    assert DeepDFAMonitor.compile("F a").batch_run([]) == []


# ----------------- soft (fractional) transition matrix -----------------


def test_soft_matrix_row_stochastic() -> None:
    """Outgoing guard probabilities partition the assignment space -> rows sum to 1."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    p = torch.tensor([0.3, 0.7, 0.4])  # P(a), P(b), P(c)
    M = dt.soft_matrix(p)
    assert torch.allclose(M.sum(dim=1), torch.ones(dt.n_states), atol=1e-6)


def test_soft_matrix_exact_on_read_once() -> None:
    """`a & (b | c)` is read-once, so the factored soft prob is exact."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    pa, pb, pc = 0.5, 0.5, 0.5
    p = torch.tensor([pa, pb, pc])
    M = dt.soft_matrix(p)
    init = dt.state_idx[dt.dfa.initial]
    leave = pa * (1 - (1 - pb) * (1 - pc))  # 0.5 * 0.75 = 0.375
    # The non-self-loop edge from init carries the leave probability.
    off_diag = M[init].clone()
    off_diag[init] = 0.0
    assert torch.isclose(off_diag.sum(), torch.tensor(leave), atol=1e-6)


def test_soft_matrix_crisp_is_one_hot() -> None:
    """With crisp 0/1 input the soft matrix is a permutation-like 0/1 matrix."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    p = torch.tensor([1.0, 1.0, 0.0])  # a&b true -> satisfies
    M = dt.soft_matrix(p)
    assert torch.all((M == 0) | (M == 1))


# ----------------- scalability: factored avoids 2^|AP| -----------------


def test_factored_handles_large_alphabet() -> None:
    """Factored mode must not build a 2^|AP| tensor (would OOM at n=24)."""
    big = "F (" + " | ".join(f"(a & a{i})" for i in range(1, 24)) + ")"
    m = DeepDFAMonitor.compile(big, mode="factored")
    assert not hasattr(m._dt, "T")  # no dense tensor
    # Sanity: a trace where a & a1 both hold is satisfied.
    obs_sat = {"a": True, "a1": True}
    assert m.run([obs_sat]) is Verdict.SATISFY
