"""Tests for the DeepDFA monitor (Paradigm 3).

DeepDFA is the canonical, exactly-correct neural monitor: it must agree
with SymbolicDFAMonitor on *every* trace, including the nested-temporal
formulas where RuleRunner diverges (no xfails here). We also check that
the dense and factored representations agree on crisp traces, that the
batched GPU path matches sequential runs, and that the exact fractional path
preserves mass, matches brute-force WMC, and remains connected to autograd.
"""

from __future__ import annotations

import hashlib
import itertools

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


# ----------------- recursive fractional approximation -----------------


def test_recursive_matrix_row_stochastic_on_factored_read_once_guard() -> None:
    """Outgoing guard probabilities partition the assignment space -> rows sum to 1."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    p = torch.tensor([0.3, 0.7, 0.4])  # P(a), P(b), P(c)
    M = dt.recursive_matrix(p)
    assert torch.allclose(M.sum(dim=1), torch.ones(dt.n_states), atol=1e-6)


def test_recursive_matrix_exact_on_read_once() -> None:
    """`a & (b | c)` is read-once, so the factored soft prob is exact."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    pa, pb, pc = 0.5, 0.5, 0.5
    p = torch.tensor([pa, pb, pc])
    M = dt.recursive_matrix(p)
    init = dt.state_idx[dt.dfa.initial]
    leave = pa * (1 - (1 - pb) * (1 - pc))  # 0.5 * 0.75 = 0.375
    # The non-self-loop edge from init carries the leave probability.
    off_diag = M[init].clone()
    off_diag[init] = 0.0
    assert torch.isclose(off_diag.sum(), torch.tensor(leave), atol=1e-6)


def test_recursive_matrix_crisp_is_one_hot() -> None:
    """With crisp 0/1 input the soft matrix is a permutation-like 0/1 matrix."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    p = torch.tensor([1.0, 1.0, 0.0])  # a&b true -> satisfies
    M = dt.recursive_matrix(p)
    assert torch.all((M == 0) | (M == 1))


# ----------------- vectorized exact cube/WMC matrix -----------------


def test_exact_matrix_row_stochastic() -> None:
    """Disjoint cube cover -> exact transition rows sum to 1."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    p = torch.tensor([0.3, 0.7, 0.4])
    M = dt.exact_matrix(p)
    assert torch.allclose(M.sum(dim=1), torch.ones(dt.n_states), atol=1e-6)


def test_exact_matrix_is_one_hot_on_crisp() -> None:
    """Crisp 0/1 input -> each cube is 0/1 and cubes are disjoint -> 0/1 matrix."""
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    for p in (torch.tensor([1.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 1.0])):
        M = dt.exact_matrix(p)
        assert torch.all((M == 0) | (M == 1))


@pytest.mark.parametrize(
    "formula", ["F ((a & b) | (a & c))", "F ((a & b) | (a & c) | (a & d))", "G (a | b)"]
)
def test_exact_matrix_equals_recursive_matrix_on_read_once(formula: str) -> None:
    """The orthogonal-cube matrix equals the recursive approximation on
    read-once guards, for *both* crisp and fractional inputs."""
    dt = DeepDFATensor(compile_ltlf(formula), mode="factored")
    rng = np.random.default_rng(_stable_seed(formula))
    for _ in range(20):
        p = torch.tensor(rng.random(dt.n_atoms), dtype=torch.float32)
        assert torch.allclose(dt.exact_matrix(p), dt.recursive_matrix(p), atol=1e-6)


def test_exact_matrix_batched_matches_unbatched() -> None:
    dt = DeepDFATensor(compile_ltlf("F ((a & b) | (a & c))"), mode="factored")
    P = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.2, 0.5, 0.9]])
    batched = dt.exact_matrix(P)
    for i in range(P.shape[0]):
        assert torch.allclose(batched[i], dt.exact_matrix(P[i]), atol=1e-6)


def test_crisp_and_soft_matrix_names_remain_compatible() -> None:
    """Historical names retain their old meanings for downstream callers."""
    dt = DeepDFATensor(compile_ltlf("F (a & b)"), mode="factored")
    p = torch.tensor([0.3, 0.7])
    assert torch.equal(dt.crisp_matrix(p), dt.exact_matrix(p))
    assert torch.equal(dt.soft_matrix(p), dt.recursive_matrix(p))


def test_exact_matrix_row_stochastic_on_non_read_once_guard() -> None:
    """Exact WMC preserves mass even when Boolean subexpressions overlap."""
    formula = "F ((a & b) | (a & c) | (b & c))"
    dt = DeepDFATensor(compile_ltlf(formula), mode="factored")
    p = torch.tensor([0.31, 0.57, 0.83])
    M = dt.exact_matrix(p)
    assert torch.allclose(M.sum(dim=1), torch.ones(dt.n_states), atol=1e-6)


def test_tensor_acceptance_matches_bruteforce_non_read_once() -> None:
    """One-cell exact score equals enumeration of all Bernoulli valuations."""
    formula = "F ((a & b) | (a & c) | (b & c))"
    dt = DeepDFATensor(compile_ltlf(formula), mode="factored")
    monitor = DeepDFAMonitor(dt)
    p = torch.tensor([[0.31, 0.57, 0.83]])

    expected = 0.0
    for bits in itertools.product((False, True), repeat=dt.n_atoms):
        obs = dict(zip(dt.atoms, bits))
        weight = 1.0
        for probability, bit in zip(p[0].tolist(), bits):
            weight *= probability if bit else 1.0 - probability
        if dt.dfa.step(dt.dfa.initial, obs) in dt.dfa.accepting:
            expected += weight

    actual = monitor.acceptance_probability_tensor(p)
    assert torch.isclose(actual, torch.tensor(expected), atol=1e-6)


@pytest.mark.parametrize(
    "formula",
    [
        "F ((a & b) | (a & c) | (b & c))",  # non-read-once guard
        "G (a -> X (b | c))",  # obligation carried across cells
        "(a U b) & F (c & a)",  # two interacting temporal operators
    ],
)
@pytest.mark.parametrize("length", [1, 2, 3])
def test_trace_marginal_matches_bruteforce_multi_cell(
    formula: str, length: int
) -> None:
    """The acceptance score is the marginal over whole random traces.

    The single-cell check above exercises Proposition "exact guard WMC" only:
    one transition matrix against one weighted valuation sum.  The trace-marginal
    proposition has content only from length 2 on, where the propagated state
    distribution has to keep the per-cell weights correctly coupled through
    delta*.  A single cell also hides the non-read-once defect entirely, so the
    recursive approximation would pass the one-cell test on these formulas too.

    Here the reference enumerates *every* crisp trace of the given length,
    weights it by the independent-Bernoulli probability of its cells, and sums
    the weights of the accepted ones.
    """
    dt = DeepDFATensor(compile_ltlf(formula), mode="factored")
    monitor = DeepDFAMonitor(dt)
    generator = torch.Generator().manual_seed(_stable_seed(f"{formula}:{length}"))
    P = torch.rand(1, length, dt.n_atoms, generator=generator)

    valuations = list(itertools.product((False, True), repeat=dt.n_atoms))
    expected = 0.0
    for cells in itertools.product(valuations, repeat=length):
        weight = 1.0
        state = dt.dfa.initial
        for position, bits in enumerate(cells):
            for atom_index, bit in enumerate(bits):
                probability = float(P[0, position, atom_index])
                weight *= probability if bit else 1.0 - probability
            state = dt.dfa.step(state, dict(zip(dt.atoms, bits)))
        if state in dt.dfa.accepting:
            expected += weight

    actual = monitor.acceptance_probability_tensor(P)
    assert torch.isclose(actual, torch.tensor([expected]), atol=1e-5)


def test_tensor_acceptance_preserves_autograd() -> None:
    """The public tensor API backpropagates through exact guard WMC."""
    formula = "F ((a & b) | (a & c) | (b & c))"
    monitor = DeepDFAMonitor.compile(formula, mode="factored")
    p = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)

    score = monitor.acceptance_probability_tensor(p)
    score.backward()

    assert score.grad_fn is not None
    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))
    assert torch.allclose(p.grad, torch.full_like(p, 0.5), atol=1e-6)


def test_tensor_acceptance_batch_lengths_and_python_wrappers_agree() -> None:
    monitor = DeepDFAMonitor.compile("F a", mode="factored")
    P = torch.tensor(
        [
            [[0.2], [0.5]],
            [[0.7], [0.9]],
        ]
    )
    scores = monitor.acceptance_probability_tensor(P, lengths=torch.tensor([2, 1]))
    wrapped = monitor.batch_acceptance_probability(
        [[{"a": 0.2}, {"a": 0.5}], [{"a": 0.7}]]
    )
    assert torch.allclose(scores, torch.tensor(wrapped), atol=1e-6)


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -0.01, 1.01],
)
def test_tensor_acceptance_rejects_invalid_probabilities(bad_value: float) -> None:
    """The public probabilistic boundary accepts Bernoulli parameters only."""
    monitor = DeepDFAMonitor.compile("F a", mode="factored")
    p = torch.tensor([[bad_value]], dtype=torch.float32)
    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        monitor.acceptance_probability_tensor(p)


@pytest.mark.parametrize("threshold", [float("nan"), -0.01, 1.01])
def test_soft_verdict_rejects_invalid_threshold(threshold: float) -> None:
    monitor = DeepDFAMonitor.compile("F a", mode="factored")
    with pytest.raises(ValueError, match="threshold"):
        monitor.soft_verdict([{"a": 0.5}], threshold=threshold)


def test_dense_artifact_stats_match_allocated_tensor() -> None:
    monitor = DeepDFAMonitor.compile("F (a & b)", mode="dense")
    stats = monitor.artifact_stats

    assert stats.mode == "dense"
    assert stats.n_atoms == 2
    assert stats.alphabet_size == 4
    assert stats.n_states == monitor._dt.n_states
    assert stats.n_transitions == len(monitor._dt.dfa.transitions)
    assert stats.dense_tensor_elements == monitor._dt.T.numel()
    assert stats.dense_tensor_bytes == (
        monitor._dt.T.numel() * monitor._dt.T.element_size()
    )
    assert stats.cube_count is None
    assert stats.cube_tensor_bytes is None


def test_factored_artifact_stats_match_allocated_cube_tensors() -> None:
    monitor = DeepDFAMonitor.compile("F ((a & b) | (a & c))", mode="factored")
    stats = monitor.artifact_stats
    dt = monitor._dt

    assert stats.mode == "factored"
    assert stats.dense_tensor_elements is None
    assert stats.dense_tensor_bytes is None
    assert stats.cube_count == dt._cube_flat.numel()
    assert stats.cube_mask_elements == 2 * stats.cube_count * stats.n_atoms
    assert stats.cube_tensor_bytes == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (dt._cube_rt, dt._cube_rf, dt._cube_flat)
    )


# ----------------- scalability: factored avoids 2^|AP| -----------------


def test_factored_handles_large_alphabet() -> None:
    """Factored mode must not build a 2^|AP| tensor (would OOM at n=24)."""
    big = "F (" + " | ".join(f"(a & a{i})" for i in range(1, 24)) + ")"
    m = DeepDFAMonitor.compile(big, mode="factored")
    assert not hasattr(m._dt, "T")  # no dense tensor
    # Sanity: a trace where a & a1 both hold is satisfied.
    obs_sat = {"a": True, "a1": True}
    assert m.run([obs_sat]) is Verdict.SATISFY
