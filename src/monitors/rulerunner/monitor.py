"""Monitor-interface wrapper for the paradigm-2 RuleRunner pipeline.

`RuleRunnerMonitor` is the public face of the package — what
[experiments/exp1_single_trace.py](experiments/exp1_single_trace.py),
[exp2](experiments/exp2_formula_complexity.py), and
[exp3](experiments/exp3_batch_size.py) put in their `MONITORS` list.
It is a thin adapter over `CILPRunner` that exposes the `compile`,
`step`, `final_verdict`, `reset` primitives the `Monitor` ABC
requires; the default `run` and `batch_run` from the base class then
work as-is.

If a future step needs vectorised per-cell batching for Exp 3, the
right place is to override `batch_run` here — `CILPRunner` already
builds the weight matrices as torch tensors, so adding a batch
dimension to the activation vector is the natural extension.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.rulerunner.cilp import CILPRunner


class RuleRunnerMonitor(Monitor):
    """Paradigm 2 — formula → parse tree → rules → CILP network."""

    def __init__(self, runner: CILPRunner) -> None:
        self._runner = runner

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "RuleRunnerMonitor":
        return cls(CILPRunner.from_formula(formula, device=device))

    def step(self, obs: Observation) -> Verdict:
        return self._runner.step(obs)

    def final_verdict(self) -> Verdict:
        return self._runner.final_verdict()

    def reset(self) -> None:
        self._runner.reset()

    def batch_run(
        self,
        traces: Iterable[Iterable[Observation]],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Vectorised cross-trace path (CPU/CUDA). Delegates to the runner,
        which parallelises the trace axis with batched matmuls; identical
        verdicts to the sequential default. ``early_termination`` is accepted
        for interface parity but does not change the compute — the batched
        path always advances every trace through all its cells (see
        ``CILPRunner.batch_run``)."""
        return self._runner.batch_run(traces, early_termination=early_termination)
