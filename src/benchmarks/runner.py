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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    device: str = "cpu"
    early_termination: bool = True
    gpu_name: str = ""


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
    device: str = "cpu",
    early_termination: bool = True,
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
        device:         "cpu" or "cuda". Passed to every monitor's compile()
                        (the symbolic DFA ignores it); RuleRunner and DeepDFA
                        run their batched matmuls on it.
        early_termination: when False, every monitor processes all cells of
                        every trace (no early give-up). Use False for the
                        per-cell-cost figures so the early-terminating IJCNN
                        family does not let crisp monitors give up after a
                        couple of cells while batched monitors do the full
                        pass (the early-termination confound; CLAUDE.md
                        Phase 0.1). The batched monitors already process all
                        cells, so this only affects the crisp symbolic walk.
    """
    rng = np.random.default_rng(seed)
    traces = random_traces(formula.atoms, trace_length, n_traces, rng)

    monitor = monitor_cls.compile(formula.formula, device=device)

    # CUDA kernel launches are asynchronous, so time.perf_counter() would stop
    # before the GPU finishes unless we synchronize. We sync ONCE around the
    # whole timed region (after the full batch_run), never per cell (Phase 0.3
    # measurement hygiene) — a per-cell sync would serialize the batched path
    # and inflate the very GPU advantage Exp 3 is meant to measure.
    cuda_sync = device == "cuda" and torch.cuda.is_available()
    # Stamp the GPU model so results from different machines (laptop vs Colab vs
    # server) are attributable and comparable. Empty on CPU.
    gpu_name = torch.cuda.get_device_name() if cuda_sync else ""

    # warm-up: prime caches without contributing to measurements
    for _ in range(n_warmup):
        monitor.batch_run(traces, early_termination=early_termination)
    if cuda_sync:
        torch.cuda.synchronize()  # drain warm-up before the first measurement

    total_cells = n_traces * trace_length
    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        monitor.batch_run(traces, early_termination=early_termination)
        if cuda_sync:
            torch.cuda.synchronize()  # ensure all kernels finished before t1
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
        device=device,
        early_termination=early_termination,
        gpu_name=gpu_name,
    )


def results_to_df(results: list[TimingResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


# ---------------------------------------------------------------------------
# Incremental / resumable persistence
# ---------------------------------------------------------------------------
#
# Each experiment is a grid of (monitor, x-value) cells. A cell can take a long
# time (factored DeepDFA at n=32, 1024-trace batches), so a run may be killed
# midway (OOM, timeout, disconnect). To make runs resumable we flush every cell
# to the CSV as soon as it finishes, and identify cells by a canonical key so a
# later run can skip whatever is already on disk. Delete the CSV to start fresh.


def result_key(
    monitor_name: str, formula_name: str, trace_length: int, n_traces: int
) -> tuple[str, str, int, int]:
    """Canonical identity of one timed cell, used for resume de-duplication.

    Within any single experiment this 4-tuple is unique: exp1 varies
    trace_length, exp2 varies formula_name, exp3 varies n_traces; the rest are
    fixed. (Config knobs like n_repeats/device are deliberately not part of the
    key — changing them mid-grid and resuming would mix settings; delete the
    CSV to recompute from scratch in that case.)
    """
    return (monitor_name, formula_name, int(trace_length), int(n_traces))


def append_result(result: TimingResult, csv_path: str | Path) -> None:
    """Append one result row to csv_path, writing the header only if new.

    Incremental persistence: each timed cell is flushed to disk the moment it
    finishes, so a run killed midway leaves a valid partial CSV that a later run
    can resume from (see load_completed).
    """
    path = Path(csv_path)
    row = pd.DataFrame([asdict(result)])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def reset_if_stale(csv_path: str | Path, early_termination: bool) -> None:
    """Delete csv_path if it was produced under a different measurement mode.

    The early-termination setting (Phase 0.1) changes *what is measured* — a
    forced full-trace pass is a different workload than an early-give-up walk —
    so partial results from the other mode must not be resumed or mixed. A
    legacy CSV without the column predates Phase 0.1 (its implicit mode was the
    confounded early-terminating one), so it is treated as stale unless the new
    run also requests early_termination=True. The file is removed so the run
    recomputes from scratch with a consistent schema.
    """
    path = Path(csv_path)
    if not path.exists():
        return
    prior = pd.read_csv(path)
    if "early_termination" in prior.columns:
        prior_mode = bool(prior["early_termination"].iloc[0]) if len(prior) else None
    else:
        prior_mode = True  # legacy CSVs were measured with early termination on
    if prior_mode is not None and prior_mode != early_termination:
        path.unlink()


def load_completed(csv_path: str | Path) -> set[tuple[str, str, int, int]]:
    """Return the set of result_key()s already present in csv_path.

    Empty if the file does not exist. Used to skip cells computed by a prior
    (possibly interrupted) run so experiments resume where they left off.
    """
    path = Path(csv_path)
    if not path.exists():
        return set()
    prior = pd.read_csv(path)
    return {
        result_key(r.monitor_name, r.formula_name, r.trace_length, r.n_traces)
        for r in prior.itertuples(index=False)
    }
