"""Accuracy + calibration metrics for Capability Exp A (Phase 1.3).

Capability Exp A scores two things as the noise level ε grows:

  1. **Verdict accuracy** — does the binary verdict (SATISFY/VIOLATE) still
     match the oracle? Both the brittle Symbolic-threshold baseline and the
     soft DeepDFA readout produce a verdict, so accuracy is paradigm-agnostic.

  2. **Calibration** of DeepDFA's *acceptance probability* — when the soft
     readout says "P(accept) = 0.7", does it actually accept ~70 % of the
     time? Only the soft paradigm emits such a confidence; the symbolic
     baseline cannot, which is the point of the experiment. Reported as a
     reliability curve + Expected Calibration Error (ECE), with Brier score
     and ROC-AUC as summary scalars.

⚠ **The formula matters.** On a read-once-guard formula (the IJCNN family)
DeepDFA's ``soft_matrix`` is *exact*, so P(accept) is the true marginal by
construction — perfect calibration is then a hollow identity, not a result.
The calibration claim must therefore be made on a **non-read-once-guard**
formula (see ``CALIBRATION_SUITE`` in :mod:`src.benchmarks.formulas`, e.g.
the 2-of-3 majority ``(a&b)|(b&c)|(a&c)``), where the independence-assuming
soft product genuinely over/under-counts and the confidence must be
calibrated empirically. See CLAUDE.md § Phase 1.3 and the Phase 3.3 finding.

No sklearn dependency: ECE / reliability / AUC / Brier are implemented over
numpy (+ scipy for tie-aware ranks).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from src.monitors.base import Verdict

# ---------------------------------------------------------------------------
# Label / verdict helpers
# ---------------------------------------------------------------------------


def verdict_labels(verdicts: Sequence[Verdict]) -> np.ndarray:
    """Map end-of-trace verdicts to boolean labels (SATISFY -> True).

    ``final_verdict`` is always binary (never UNDECIDED), but we guard
    against an accidental UNDECIDED so a silent mislabel can't creep in.
    """
    out = np.empty(len(verdicts), dtype=bool)
    for i, v in enumerate(verdicts):
        if v is Verdict.UNDECIDED:
            raise ValueError(
                "UNDECIDED verdict cannot be scored; expected a binary "
                "end-of-trace verdict (SATISFY/VIOLATE)"
            )
        out[i] = v is Verdict.SATISFY
    return out


def verdict_accuracy(
    predicted: Sequence[Verdict], truth: Sequence[Verdict]
) -> float:
    """Fraction of traces whose predicted verdict equals the oracle verdict."""
    if len(predicted) != len(truth):
        raise ValueError(
            f"length mismatch: {len(predicted)} predicted vs {len(truth)} truth"
        )
    if not predicted:
        raise ValueError("cannot compute accuracy over an empty set")
    return float(np.mean([p is t for p, t in zip(predicted, truth)]))


# ---------------------------------------------------------------------------
# Calibration of a probabilistic acceptance readout
# ---------------------------------------------------------------------------


def _as_probs_labels(
    probs: Sequence[float], labels: Sequence[bool | Verdict]
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        raise ValueError("cannot compute calibration over an empty set")
    if p.ndim != 1:
        raise ValueError(f"probs must be 1-D, got shape {p.shape}")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probs must lie in [0, 1]")
    # labels may be bools/0-1 or Verdicts
    if len(labels) and isinstance(labels[0], Verdict):
        y = verdict_labels(labels)  # type: ignore[arg-type]
    else:
        y = np.asarray(labels, dtype=bool)
    if p.shape != y.shape:
        raise ValueError(
            f"probs/labels length mismatch: {p.shape} vs {y.shape}"
        )
    return p, y


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram.

    ``mean_confidence`` is the average predicted P(accept) of the samples in
    ``[lo, hi)``; ``accuracy`` is the empirical fraction that actually
    accepted (the oracle label). A perfectly calibrated model has
    ``mean_confidence == accuracy`` in every populated bin. Empty bins are
    returned with ``count == 0`` so the curve keeps a fixed x-grid.
    """

    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_curve(
    probs: Sequence[float],
    labels: Sequence[bool | Verdict],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin predictions into ``n_bins`` equal-width bins over [0, 1].

    The positive class is SATISFY, and ``probs`` is P(accept), so we bin by
    the acceptance probability directly (the positive-class-calibration
    convention that is natural when the confidence *is* P(positive)).
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    p, y = _as_probs_labels(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rightmost bin is closed so p == 1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            mean_conf = float(p[mask].mean())
            acc = float(y[mask].mean())
        else:
            mean_conf = float("nan")
            acc = float("nan")
        bins.append(
            ReliabilityBin(float(edges[b]), float(edges[b + 1]), count, mean_conf, acc)
        )
    return bins


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[bool | Verdict],
    n_bins: int = 10,
) -> float:
    """ECE: sample-weighted mean gap between confidence and accuracy per bin.

    ``ECE = Σ_b (n_b / N) · |acc_b − conf_b|`` over populated bins. 0 is
    perfectly calibrated; the max is 1.
    """
    p, y = _as_probs_labels(probs, labels)
    bins = reliability_curve(p, y, n_bins)
    n = p.size
    return sum(
        (b.count / n) * abs(b.accuracy - b.mean_confidence)
        for b in bins
        if b.count
    )


def max_calibration_error(
    probs: Sequence[float],
    labels: Sequence[bool | Verdict],
    n_bins: int = 10,
) -> float:
    """MCE: the worst per-bin |accuracy − confidence| gap over populated bins."""
    p, y = _as_probs_labels(probs, labels)
    bins = reliability_curve(p, y, n_bins)
    gaps = [abs(b.accuracy - b.mean_confidence) for b in bins if b.count]
    return max(gaps) if gaps else 0.0


def brier_score(
    probs: Sequence[float], labels: Sequence[bool | Verdict]
) -> float:
    """Mean squared error between P(accept) and the 0/1 oracle label."""
    p, y = _as_probs_labels(probs, labels)
    return float(np.mean((p - y.astype(float)) ** 2))


def roc_auc(probs: Sequence[float], labels: Sequence[bool | Verdict]) -> float:
    """ROC-AUC via the tie-aware Mann–Whitney rank statistic.

    Measures ranking/discrimination (independent of calibration): the
    probability that a random SATISFY trace gets a higher confidence than a
    random VIOLATE trace. Returns NaN if the labels are single-class (AUC
    undefined). scipy's ``rankdata`` averages tied ranks so ties count 0.5.
    """
    p, y = _as_probs_labels(probs, labels)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
