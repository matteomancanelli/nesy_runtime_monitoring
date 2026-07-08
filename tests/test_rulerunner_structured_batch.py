"""Structured (IJCNN 2015 Fig. 5) batch_run() must equal sequential run().

Mirrors `test_rulerunner_batch.py` for the structured variant: the vectorised
cross-trace path in `StructuredCILPRunner.batch_run` parallelises the trace axis
with a batched per-node sweep on the requested device, but must reproduce the
single-trace verdicts exactly — including per-trace early termination, the
end-of-trace resolution, and the nested-temporal formulas where the encoding
diverges from the DFA (batch_run must diverge identically to run()).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.monitors.base import Verdict
from src.monitors.rulerunner import StructuredRuleRunnerMonitor
from src.monitors.rulerunner.structured import StructuredCILPRunner

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
    atoms = StructuredCILPRunner.from_formula(formula)._rs.atoms
    traces = _random_traces(atoms, _seed(formula), 60, 7)
    sequential = [StructuredCILPRunner.from_formula(formula).run(t) for t in traces]
    batched = StructuredCILPRunner.from_formula(formula, device=device).batch_run(
        traces
    )
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
    atoms = StructuredCILPRunner.from_formula(formula)._rs.atoms
    traces = _random_traces(atoms, _seed(formula), 40, 6)
    mon = StructuredRuleRunnerMonitor.compile(formula)
    assert mon.batch_run(traces) == [mon.compile(formula).run(t) for t in traces]


def test_effective_device_reports_cpu_by_default() -> None:
    assert StructuredRuleRunnerMonitor.compile("F(a & b)").effective_device == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_effective_device_reports_cuda() -> None:
    mon = StructuredRuleRunnerMonitor.compile("F(a & b)", device="cuda")
    assert mon.effective_device == "cuda"


def test_empty_batch_and_empty_traces() -> None:
    runner = StructuredCILPRunner.from_formula("F(a & b)")
    assert runner.batch_run([]) == []
    empty = [[], []]
    assert runner.batch_run(empty) == [
        StructuredCILPRunner.from_formula("F(a & b)").run(t) for t in empty
    ]


def test_mixed_lengths_and_early_termination() -> None:
    runner = StructuredCILPRunner.from_formula("G a")
    traces = [
        [{"a": True}, {"a": True}],
        [{"a": True}, {"a": False}, {"a": True}],
        [],
        [{"a": False}],
    ]
    assert runner.batch_run(traces) == [
        StructuredCILPRunner.from_formula("G a").run(t) for t in traces
    ]
    assert runner.batch_run(traces)[1] is Verdict.VIOLATE
