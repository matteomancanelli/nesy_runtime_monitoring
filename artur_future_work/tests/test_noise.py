"""Tests for the Phase 1.1 noise generator + oracle (src/benchmarks/noise.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.noise import (
    BetaNoise,
    BitFlipNoise,
    true_verdicts,
)
from src.monitors.base import Verdict


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def test_oracle_hand_checked_eventually():
    # F a: SATISFY iff a is true at some cell.
    traces = [
        [{"a": False}, {"a": True}],   # a occurs -> SATISFY
        [{"a": False}, {"a": False}],  # never    -> VIOLATE
    ]
    assert true_verdicts("F a", traces) == [Verdict.SATISFY, Verdict.VIOLATE]


def test_oracle_binary_never_undecided():
    traces = [[{"a": bool(i % 2)} for i in range(5)] for _ in range(4)]
    verdicts = true_verdicts("G a", traces)
    assert all(v in (Verdict.SATISFY, Verdict.VIOLATE) for v in verdicts)


# ---------------------------------------------------------------------------
# BitFlipNoise
# ---------------------------------------------------------------------------


def test_bitflip_eps0_identity():
    trace = [{"a": True, "b": False}, {"a": False, "b": True}]
    out = BitFlipNoise(0.0).corrupt(trace, _rng())
    assert out == [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]


def test_bitflip_eps1_all_flipped():
    trace = [{"a": True, "b": False}, {"a": False, "b": True}]
    out = BitFlipNoise(1.0).corrupt(trace, _rng())
    assert out == [{"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}]


def test_bitflip_output_is_crisp():
    trace = [{"a": bool(i % 2), "b": bool((i + 1) % 2)} for i in range(50)]
    out = BitFlipNoise(0.5).corrupt(trace, _rng())
    assert all(v in (0.0, 1.0) for obs in out for v in obs.values())


def test_bitflip_flip_rate_tracks_eps():
    eps = 0.3
    trace = [{"a": True} for _ in range(20_000)]
    out = BitFlipNoise(eps).corrupt(trace, _rng(123))
    flipped = np.mean([obs["a"] == 0.0 for obs in out])
    assert abs(flipped - eps) < 0.02


# ---------------------------------------------------------------------------
# BetaNoise
# ---------------------------------------------------------------------------


def test_beta_eps0_identity():
    trace = [{"a": True, "b": False}, {"a": False, "b": True}]
    out = BetaNoise(0.0).corrupt(trace, _rng())
    assert out == [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]


def test_beta_values_in_unit_interval():
    trace = [{"a": bool(i % 2)} for i in range(5_000)]
    out = BetaNoise(0.4).corrupt(trace, _rng())
    assert all(0.0 <= obs["a"] <= 1.0 for obs in out)


@pytest.mark.parametrize("bit, eps", [(True, 0.4), (False, 0.4), (True, 0.8)])
def test_beta_mean_tracks_target(bit, eps):
    trace = [{"a": bit} for _ in range(20_000)]
    out = BetaNoise(eps).corrupt(trace, _rng(7))
    empirical = np.mean([obs["a"] for obs in out])
    target = (1.0 - eps) * (1.0 if bit else 0.0) + eps * 0.5
    assert abs(empirical - target) < 0.02


def test_beta_variance_grows_with_eps():
    trace = [{"a": True} for _ in range(20_000)]
    low = np.var([o["a"] for o in BetaNoise(0.1).corrupt(trace, _rng(1))])
    high = np.var([o["a"] for o in BetaNoise(0.6).corrupt(trace, _rng(1))])
    assert high > low


# ---------------------------------------------------------------------------
# Structure + determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [BitFlipNoise(0.4), BetaNoise(0.4)])
def test_structure_preserved(model):
    trace = [{"a": True, "b": False, "c": True} for _ in range(10)]
    out = model.corrupt(trace, _rng())
    assert len(out) == len(trace)
    assert all(set(obs) == {"a", "b", "c"} for obs in out)


@pytest.mark.parametrize("model", [BitFlipNoise(0.4), BetaNoise(0.4)])
def test_determinism_same_seed(model):
    trace = [{"a": bool(i % 2), "b": bool((i + 1) % 3)} for i in range(30)]
    a = model.corrupt(trace, _rng(99))
    b = model.corrupt(trace, _rng(99))
    assert a == b


@pytest.mark.parametrize("model", [BitFlipNoise(0.4), BetaNoise(0.4)])
def test_corrupt_all_matches_sequential(model):
    traces = [[{"a": bool(i % 2)} for i in range(8)] for _ in range(5)]
    batched = model.corrupt_all(traces, _rng(5))
    rng = _rng(5)
    sequential = [model.corrupt(t, rng) for t in traces]
    assert batched == sequential


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_eps_rejected(bad):
    with pytest.raises(ValueError):
        BitFlipNoise(bad)
    with pytest.raises(ValueError):
        BetaNoise(bad)


def test_invalid_concentration_rejected():
    with pytest.raises(ValueError):
        BetaNoise(0.4, concentration=0.0)
