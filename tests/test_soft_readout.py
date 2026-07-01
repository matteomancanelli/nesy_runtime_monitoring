"""Tests for the Phase 1.2 soft-consumption paths.

Covers:
  * DeepDFA's marginal acceptance-probability readout (Option A) — the
    crisp-input anchor to the symbolic oracle, [0,1] bounds, batch ==
    single-trace, ragged-length masking, mode guard.
  * The symbolic threshold baseline (`threshold_trace`).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.noise import BetaNoise, threshold_trace, true_verdicts
from src.benchmarks.runner import random_traces
from src.monitors.base import Verdict
from src.monitors.deep_dfa import DeepDFAMonitor, DeepDFAMonitorFactored

# A read-once formula (small |Q|) and a couple of atoms.
FORMULA = "F(a & b)"
ATOMS = ("a", "b")


def _factored(formula: str = FORMULA) -> DeepDFAMonitorFactored:
    return DeepDFAMonitorFactored.compile(formula)


def _as_soft(crisp_trace):
    return [{a: float(v) for a, v in obs.items()} for obs in crisp_trace]


# ---------------------------------------------------------------------------
# Crisp-input anchor: acceptance probability agrees with the oracle
# ---------------------------------------------------------------------------


def test_crisp_acceptance_matches_oracle():
    rng = np.random.default_rng(0)
    traces = random_traces(ATOMS, trace_length=12, n_traces=60, rng=rng)
    verdicts = true_verdicts(FORMULA, traces)

    mon = _factored()
    for trace, verdict in zip(traces, verdicts):
        p = mon.acceptance_probability(_as_soft(trace))
        # crisp inputs keep the distribution one-hot, so p is exactly 0 or 1
        assert p == pytest.approx(0.0) or p == pytest.approx(1.0)
        expected = 1.0 if verdict is Verdict.SATISFY else 0.0
        assert p == pytest.approx(expected)


def test_soft_verdict_matches_oracle_on_crisp():
    rng = np.random.default_rng(1)
    traces = random_traces(ATOMS, trace_length=12, n_traces=60, rng=rng)
    verdicts = true_verdicts(FORMULA, traces)
    mon = _factored()
    got = [mon.soft_verdict(_as_soft(t)) for t in traces]
    assert got == verdicts


# ---------------------------------------------------------------------------
# Fractional inputs: bounds + batch consistency
# ---------------------------------------------------------------------------


def test_fractional_acceptance_in_unit_interval():
    rng = np.random.default_rng(2)
    crisp = random_traces(ATOMS, trace_length=10, n_traces=40, rng=rng)
    soft = BetaNoise(0.5).corrupt_all(crisp, rng)
    mon = _factored()
    for t in soft:
        p = mon.acceptance_probability(t)
        assert 0.0 <= p <= 1.0


def test_batch_matches_single_trace():
    rng = np.random.default_rng(3)
    crisp = random_traces(ATOMS, trace_length=15, n_traces=50, rng=rng)
    soft = BetaNoise(0.4).corrupt_all(crisp, rng)
    mon = _factored()
    batched = mon.batch_acceptance_probability(soft)
    single = [mon.acceptance_probability(t) for t in soft]
    assert batched == pytest.approx(single, abs=1e-5)


def test_batch_ragged_lengths_freeze_ended_traces():
    # Different-length traces: the shorter one's distribution must be frozen
    # once it ends, so batched == per-trace.
    rng = np.random.default_rng(4)
    soft = [
        _as_soft(random_traces(ATOMS, trace_length=L, n_traces=1, rng=rng)[0])
        for L in (3, 7, 1, 10)
    ]
    mon = _factored()
    batched = mon.batch_acceptance_probability(soft)
    single = [mon.acceptance_probability(t) for t in soft]
    assert batched == pytest.approx(single, abs=1e-5)


def test_empty_trace_is_initial_acceptance():
    mon = _factored()
    # F(a & b): initial state is non-accepting -> empty trace rejects.
    assert mon.acceptance_probability([]) == pytest.approx(0.0)
    assert mon.batch_acceptance_probability([[]]) == pytest.approx([0.0])


# ---------------------------------------------------------------------------
# Mode guard
# ---------------------------------------------------------------------------


def test_dense_mode_rejects_soft_readout():
    mon = DeepDFAMonitor.compile(FORMULA)  # default dense
    with pytest.raises(ValueError, match="factored"):
        mon.acceptance_probability([{"a": 0.5, "b": 0.5}])
    with pytest.raises(ValueError, match="factored"):
        mon.batch_acceptance_probability([[{"a": 0.5, "b": 0.5}]])


# ---------------------------------------------------------------------------
# Symbolic threshold baseline
# ---------------------------------------------------------------------------


def test_threshold_trace_boundary():
    soft = [{"a": 0.5, "b": 0.49}, {"a": 0.9, "b": 0.1}]
    crisp = threshold_trace(soft)  # default 0.5 threshold, >= is True
    assert crisp == [{"a": True, "b": False}, {"a": True, "b": False}]


def test_threshold_trace_identity_on_crisp():
    crisp = [{"a": True, "b": False}, {"a": False, "b": True}]
    soft = [{a: float(v) for a, v in obs.items()} for obs in crisp]
    assert threshold_trace(soft) == crisp


def test_threshold_then_symbolic_matches_direct_run():
    # Thresholding a crisp-valued soft trace then labeling == oracle directly.
    rng = np.random.default_rng(5)
    crisp = random_traces(ATOMS, trace_length=12, n_traces=40, rng=rng)
    soft = [_as_soft(t) for t in crisp]
    thresholded = [threshold_trace(t) for t in soft]
    assert true_verdicts(FORMULA, thresholded) == true_verdicts(FORMULA, crisp)


# ---------------------------------------------------------------------------
# Normalized readout (Phase 1.4): raw score is not row-stochastic on
# non-read-once guards; normalize=True forces a valid [0, 1] value.
# ---------------------------------------------------------------------------

# 2-of-3 majority under F: the accepting-edge guard is non-read-once.
_MAJORITY = "F((a & b) | (b & c) | (a & c))"


def test_normalize_equals_raw_on_read_once():
    # F(a & b) is read-once: soft_matrix is row-stochastic (mass == 1), so
    # normalization is a no-op.
    rng = np.random.default_rng(6)
    crisp = random_traces(ATOMS, trace_length=10, n_traces=40, rng=rng)
    soft = BetaNoise(0.5).corrupt_all(crisp, rng)
    mon = _factored()
    raw = mon.batch_acceptance_probability(soft, normalize=False)
    norm = mon.batch_acceptance_probability(soft, normalize=True)
    assert norm == pytest.approx(raw, abs=1e-6)


def test_raw_exceeds_one_but_normalized_is_bounded_on_non_read_once():
    rng = np.random.default_rng(7)
    crisp = random_traces(("a", "b", "c"), trace_length=4, n_traces=200, rng=rng)
    soft = BetaNoise(0.4).corrupt_all(crisp, rng)
    mon = DeepDFAMonitorFactored.compile(_MAJORITY)
    raw = np.array(mon.batch_acceptance_probability(soft, normalize=False))
    norm = np.array(mon.batch_acceptance_probability(soft, normalize=True))
    # the non-stochastic rows make the raw score overshoot 1 (the finding)
    assert raw.max() > 1.0
    # normalization pulls every score back into [0, 1]
    assert norm.min() >= 0.0 and norm.max() <= 1.0 + 1e-6
    # and it actually changes the values (not a no-op here)
    assert not np.allclose(raw, norm)


def test_normalize_batch_matches_single_non_read_once():
    rng = np.random.default_rng(8)
    crisp = random_traces(("a", "b", "c"), trace_length=5, n_traces=30, rng=rng)
    soft = BetaNoise(0.3).corrupt_all(crisp, rng)
    mon = DeepDFAMonitorFactored.compile(_MAJORITY)
    batched = mon.batch_acceptance_probability(soft, normalize=True)
    single = [mon.acceptance_probability(t, normalize=True) for t in soft]
    assert batched == pytest.approx(single, abs=1e-5)
