"""batch_run() must equal the sequential run() per trace, on CPU and CUDA.

The vectorised cross-trace path in `CILPRunner.batch_run` parallelises the
trace axis with batched matmuls but must reproduce the single-trace verdicts
exactly — including per-trace early termination and end-of-trace resolution.
This includes the nested-temporal formulas where the encoding diverges from the
DFA: batch_run must diverge identically to run(), not differently.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.monitors.base import Verdict
from src.monitors.rulerunner import RuleRunnerMonitor
from src.monitors.rulerunner.cilp import CILPRunner

FORMULAS = [
    "F(a & b)",
    "a U b",
    "G a",
    "F a",
    "G(a -> F b)",        # nested (diverges from DFA) — must diverge identically
    "F(a & X b)",
    "G(a -> X b)",
    "X(X a)",
    "a U (b & X c)",
    "F((a & b) | (a & c) | (a & d))",
]


def _seed(formula: str) -> np.random.Generator:
    h = int(hashlib.md5(formula.encode()).hexdigest(), 16) & 0xFFFFFFFF
    return np.random.default_rng(h)


def _random_traces(atoms, rng, n, max_len):
    """n traces of length in [0, max_len] (includes empty traces)."""
    out = []
    for _ in range(n):
        length = int(rng.integers(0, max_len + 1))
        out.append(
            [{a: bool(rng.integers(0, 2)) for a in atoms} for _ in range(length)]
        )
    return out


def _check(formula: str, device: str) -> None:
    atoms = CILPRunner.from_formula(formula)._rs.atoms
    traces = _random_traces(atoms, _seed(formula), 60, 7)
    sequential = [CILPRunner.from_formula(formula).run(t) for t in traces]
    batched = CILPRunner.from_formula(formula, device=device).batch_run(traces)
    assert batched == sequential


@pytest.mark.parametrize("formula", FORMULAS)
def test_batch_matches_sequential_cpu(formula: str) -> None:
    _check(formula, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("formula", FORMULAS)
def test_batch_matches_sequential_cuda(formula: str) -> None:
    _check(formula, "cuda")


def test_monitor_wrapper_batch_matches_sequential() -> None:
    formula = "G(a -> F b)"
    atoms = CILPRunner.from_formula(formula)._rs.atoms
    traces = _random_traces(atoms, _seed(formula), 40, 6)
    mon = RuleRunnerMonitor.compile(formula)
    assert mon.batch_run(traces) == [mon.compile(formula).run(t) for t in traces]


def test_empty_batch_and_empty_traces() -> None:
    runner = CILPRunner.from_formula("F(a & b)")
    assert runner.batch_run([]) == []
    # length-0 traces: end-of-trace verdict on the initial state, same as run().
    empty = [[], []]
    assert runner.batch_run(empty) == [
        CILPRunner.from_formula("F(a & b)").run(t) for t in empty
    ]


def test_mixed_lengths_and_early_termination() -> None:
    # `G a` traps the moment `a` is false; mixing decided-early and run-to-end
    # traces exercises the per-trace replay.
    runner = CILPRunner.from_formula("G a")
    traces = [
        [{"a": True}, {"a": True}],            # undecided -> end-of-trace SATISFY
        [{"a": True}, {"a": False}, {"a": True}],  # VIOLATE at cell 1
        [],                                    # empty -> end-of-trace
        [{"a": False}],                        # VIOLATE at cell 0
    ]
    assert runner.batch_run(traces) == [
        CILPRunner.from_formula("G a").run(t) for t in traces
    ]
    assert runner.batch_run(traces)[1] is Verdict.VIOLATE
