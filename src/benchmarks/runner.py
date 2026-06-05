"""Timing harness for cross-paradigm benchmarks.

Methodology follows IJCNN 2014: measure total wall time for batch_run(),
then divide by n_traces * trace_length (total potential cells). This
captures early-termination advantages naturally — a paradigm that
terminates early spends less total time and earns a lower per-cell cost.

Traces are randomly generated (independent uniform Bernoulli per atom
per step). Compilation is excluded from timing — all three paradigms
share the same ltlf2dfa compilation step, so it is not part of the
monitoring comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.benchmarks.formulas import BenchmarkFormula
from src.monitors.base import Monitor


@dataclass
class TimingResult:
    monitor_name: str
    formula_name: str
    n_leaves: int
    trace_length: int
    n_traces: int
    n_repeats: int
    mean_s_per_cell: float
    std_s_per_cell: float


def random_traces(
    atoms: tuple[str, ...],
    trace_length: int,
    n_traces: int,
    rng: np.random.Generator,
) -> list[list[dict[str, bool]]]:
    """Generate n_traces independent random traces of the given length."""
    bits = rng.integers(0, 2, size=(n_traces, trace_length, len(atoms)), dtype=np.int8)
    return [
        [{atom: bool(bits[t, s, i]) for i, atom in enumerate(atoms)}
            for s in range(trace_length)]
        for t in range(n_traces)
    ]


def time_monitor(
    monitor_cls: type[Monitor],
    formula: BenchmarkFormula,
    trace_length: int,
    n_traces: int = 100,
    n_repeats: int = 7,
    n_warmup: int = 3,
    seed: int = 42,
) -> TimingResult:
    """Time monitor_cls on formula over randomly generated traces.

    Args:
        monitor_cls:    Monitor subclass to benchmark (not yet compiled).
        formula:        BenchmarkFormula from the registry.
        trace_length:   Number of steps per trace.
        n_traces:       Number of traces per timed run.
        n_repeats:      Number of timed repetitions; mean/std reported.
        n_warmup:       Untimed warm-up runs before measurement.
        seed:           RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    traces = random_traces(formula.atoms, trace_length, n_traces, rng)

    monitor = monitor_cls.compile(formula.formula)

    # warm-up: prime caches without contributing to measurements
    for _ in range(n_warmup):
        monitor.batch_run(traces)

    total_cells = n_traces * trace_length
    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        monitor.batch_run(traces)
        times.append(time.perf_counter() - t0)

    per_cell = [t / total_cells for t in times]
    return TimingResult(
        monitor_name=monitor_cls.__name__,
        formula_name=formula.name,
        n_leaves=formula.n_leaves,
        trace_length=trace_length,
        n_traces=n_traces,
        n_repeats=n_repeats,
        mean_s_per_cell=float(np.mean(per_cell)),
        std_s_per_cell=float(np.std(per_cell)),
    )


def results_to_df(results: list[TimingResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
