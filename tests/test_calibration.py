"""Tests for the Phase 1.3 accuracy + calibration metrics.

Two groups:
  * The metric functions themselves, checked against hand-computed values
    (accuracy, reliability binning, ECE/MCE, Brier, ROC-AUC) and their
    edge/error cases.
  * The non-read-once calibration formula: it verifies that the majority
    guard genuinely makes DeepDFA's soft acceptance probability *inexact*
    on fractional inputs (so calibration is an empirical result), while the
    read-once references stay exact (the hollow-identity contrast).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.benchmarks.calibration import (
    brier_score,
    expected_calibration_error,
    max_calibration_error,
    reliability_curve,
    roc_auc,
    verdict_accuracy,
    verdict_labels,
)
from src.benchmarks.formulas import CALIBRATION_SUITE
from src.monitors.base import Verdict
from src.monitors.deep_dfa import DeepDFAMonitorFactored

S, V, U = Verdict.SATISFY, Verdict.VIOLATE, Verdict.UNDECIDED


# ---------------------------------------------------------------------------
# Verdict labels / accuracy
# ---------------------------------------------------------------------------


def test_verdict_labels_maps_satisfy_to_true():
    assert list(verdict_labels([S, V, S, V])) == [True, False, True, False]


def test_verdict_labels_rejects_undecided():
    with pytest.raises(ValueError, match="UNDECIDED"):
        verdict_labels([S, U])


def test_verdict_accuracy_basic():
    assert verdict_accuracy([S, V, S, V], [S, V, V, V]) == pytest.approx(0.75)
    assert verdict_accuracy([S, S], [S, S]) == pytest.approx(1.0)


def test_verdict_accuracy_length_and_empty_errors():
    with pytest.raises(ValueError, match="length mismatch"):
        verdict_accuracy([S], [S, V])
    with pytest.raises(ValueError, match="empty"):
        verdict_accuracy([], [])


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------


def test_reliability_curve_bins_and_stats():
    probs = [0.1, 0.4, 0.6, 0.9]
    labels = [0, 1, 0, 1]
    bins = reliability_curve(probs, labels, n_bins=2)
    assert len(bins) == 2
    lo, hi = bins
    assert (lo.lo, lo.hi, lo.count) == (0.0, 0.5, 2)
    assert lo.mean_confidence == pytest.approx(0.25)
    assert lo.accuracy == pytest.approx(0.5)
    assert (hi.lo, hi.hi, hi.count) == (0.5, 1.0, 2)
    assert hi.mean_confidence == pytest.approx(0.75)
    assert hi.accuracy == pytest.approx(0.5)


def test_reliability_curve_empty_bins_are_nan_with_fixed_grid():
    bins = reliability_curve([0.9, 0.95], [1, 1], n_bins=10)
    assert len(bins) == 10
    populated = [b for b in bins if b.count]
    assert len(populated) == 1  # both land in the last bin
    assert populated[0].count == 2
    # empty bins keep their edges but report NaN stats
    empty = bins[0]
    assert empty.count == 0
    assert np.isnan(empty.mean_confidence) and np.isnan(empty.accuracy)


def test_reliability_boundary_and_one():
    # p == 0.5 lands in the upper bin; p == 1.0 lands in the last bin.
    bins = reliability_curve([0.5, 1.0], [1, 1], n_bins=2)
    assert bins[0].count == 0
    assert bins[1].count == 2


def test_reliability_accepts_verdict_labels():
    bins = reliability_curve([0.2, 0.8], [V, S], n_bins=2)
    assert bins[0].accuracy == pytest.approx(0.0)
    assert bins[1].accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ECE / MCE / Brier / AUC — hand-computed
# ---------------------------------------------------------------------------


def test_ece_mce_hand_computed():
    probs = [0.1, 0.4, 0.6, 0.9]
    labels = [0, 1, 0, 1]
    # both bins have |acc - conf| = 0.25, weights 0.5 each
    assert expected_calibration_error(probs, labels, n_bins=2) == pytest.approx(0.25)
    assert max_calibration_error(probs, labels, n_bins=2) == pytest.approx(0.25)


def test_perfect_calibration_is_zero():
    probs = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)
    assert max_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)
    assert brier_score(probs, labels) == pytest.approx(0.0)
    assert roc_auc(probs, labels) == pytest.approx(1.0)


def test_brier_hand_computed():
    probs = [0.1, 0.4, 0.6, 0.9]
    labels = [0, 1, 0, 1]
    # (0.01 + 0.36 + 0.36 + 0.01) / 4
    assert brier_score(probs, labels) == pytest.approx(0.185)


def test_roc_auc_hand_computed():
    # classic example: AUC = 0.75
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_roc_auc_handles_ties():
    # tied scores across classes count as 0.5
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan():
    assert np.isnan(roc_auc([0.2, 0.8], [1, 1]))
    assert np.isnan(roc_auc([0.2, 0.8], [0, 0]))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_probs_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error([1.5], [1])


def test_probs_labels_length_mismatch_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        brier_score([0.1, 0.2], [1])


def test_empty_calibration_rejected():
    with pytest.raises(ValueError, match="empty"):
        brier_score([], [])


# ---------------------------------------------------------------------------
# The non-read-once formula: calibration must be an empirical result
# ---------------------------------------------------------------------------


def _exact_majority_accept(pa: float, pb: float, pc: float) -> float:
    """True P(2-of-3 majority) over independent bits — brute force."""
    total = 0.0
    for a, b, c in itertools.product((0, 1), repeat=3):
        if (a and b) or (b and c) or (a and c):
            pr = (pa if a else 1 - pa) * (pb if b else 1 - pb) * (pc if c else 1 - pc)
            total += pr
    return total


def test_majority3_is_flagged_non_read_once():
    maj = CALIBRATION_SUITE[0]
    assert maj.name == "majority3"
    assert maj.read_once is False
    # the read-once references are the contrast
    assert all(f.read_once for f in CALIBRATION_SUITE[1:])


def test_majority3_soft_readout_is_inexact_on_fractional_inputs():
    # Single cell of F(majority): the soft product assumes atom independence,
    # but the majority guard is non-read-once, so it diverges from the true
    # marginal. This is exactly why calibration on it is a real result.
    mon = DeepDFAMonitorFactored.compile(CALIBRATION_SUITE[0].formula)
    pa, pb, pc = 0.6, 0.5, 0.7
    soft = mon.acceptance_probability([{"a": pa, "b": pb, "c": pc}])
    exact = _exact_majority_accept(pa, pb, pc)
    assert abs(soft - exact) > 0.05  # ~0.086 gap — genuinely miscalibrated


def test_majority3_soft_readout_is_exact_on_crisp_inputs():
    # Crisp 0/1 inputs are exact for *any* guard (read-once or not).
    mon = DeepDFAMonitorFactored.compile(CALIBRATION_SUITE[0].formula)
    for a, b, c in itertools.product((0.0, 1.0), repeat=3):
        soft = mon.acceptance_probability([{"a": a, "b": b, "c": c}])
        exact = _exact_majority_accept(a, b, c)
        assert soft == pytest.approx(exact)


def test_read_once_reference_soft_readout_is_exact():
    # ijcnn_n4 guard is read-once after MONA factoring: soft == exact marginal
    # even on fractional inputs (the hollow-identity case).
    ijcnn = CALIBRATION_SUITE[2]
    mon = DeepDFAMonitorFactored.compile(ijcnn.formula)
    # exact marginal for F(OR_i a0 & ai) over one cell = P(a0 & (OR ai))
    pv = {a: 0.3 + 0.1 * i for i, a in enumerate(ijcnn.atoms)}
    soft = mon.acceptance_probability([pv])
    p0 = pv[ijcnn.atoms[0]]
    p_any = 1.0 - np.prod([1.0 - pv[a] for a in ijcnn.atoms[1:]])
    exact = p0 * p_any
    assert soft == pytest.approx(exact, abs=1e-6)
