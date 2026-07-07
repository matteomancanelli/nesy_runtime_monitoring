"""Experiment 3: throughput vs batch size.

Fix formula and trace length; vary the number of traces processed in
a single batch_run() call (batch size = 1, 2, 4, ..., 1024).

Expected story:
  - Symbolic DFA: throughput scales linearly with batch size (sequential
    batch_run — no parallelism, constant time per trace).
  - DeepDFA (when added): throughput increases super-linearly up to GPU
    saturation, then flattens — this is DeepDFA's key advantage.

Metric: time per trace = total_wall_time / batch_size.
(Equivalently: time per cell = time per trace / trace_length.)

Run:
    python experiments/exp3_batch_size.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.formulas import IJCNN_SUITE
from src.benchmarks.runner import (
    append_result,
    load_completed,
    reset_if_stale,
    result_key,
    time_monitor,
)
from src.monitors.deep_dfa import DeepDFAMonitor, DeepDFAMonitorFactored
from src.monitors.rulerunner import RuleRunnerMonitor, StructuredRuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ijcnn_n8 -> 8 atoms -> dense alphabet 2^8 = 256 symbols. This is the
# cross-trace batching showcase: both RuleRunner and DeepDFA do batched
# matmuls per cell over the whole batch. DeepDFA does one matmul/cell;
# RuleRunner does depth+1 (its intrinsic within-step sequential cost), so a
# fair, vectorised comparison isolates that architectural difference rather
# than Python per-trace overhead.
# DeepDFAMonitor is dense (the batching showcase); DeepDFAMonitorFactored is a
# reference line. StructuredRuleRunnerMonitor is CPU/sequential — it CANNOT
# batch cross-trace, so its time-per-trace stays flat as batch grows; that is a
# deliberate contrast for this experiment, not a defect (docs/EXPERIMENT_MAP.md).
MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    StructuredRuleRunnerMonitor,
    DeepDFAMonitor,
    DeepDFAMonitorFactored,
]

FORMULA      = IJCNN_SUITE[2]    # ijcnn_n8 — enough atoms to be interesting
TRACE_LENGTH = 1_000
BATCH_SIZES  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_REPEATS    = 7
N_WARMUP     = 3
SEED         = 42

# Phase 0.1 — kill the early-termination confound. ijcnn_n8 early-terminates
# almost immediately on random traces, so without this the crisp symbolic walk
# is timed giving up after a couple of cells while the batched monitors do the
# full pass. Forcing all cells makes the cross-trace comparison fair. State
# this explicitly in the paper.
EARLY_TERMINATION = False

# Tensor monitors run their batched matmuls here; the symbolic DFA ignores it.
# Auto-uses the GPU when one is available — the parallel-execution axis Exp 3
# is meant to exercise.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "exp3_batch_size.csv"
reset_if_stale(csv_path, EARLY_TERMINATION)  # drop CSVs from the other mode
completed = load_completed(csv_path)   # resume: skip cells already on disk

total = len(MONITORS) * len(BATCH_SIZES)
with tqdm(total=total, desc="exp3") as pbar:
    for monitor_cls in MONITORS:
        for batch_size in BATCH_SIZES:
            key = result_key(
                monitor_cls.__name__, FORMULA.name, TRACE_LENGTH, batch_size, DEVICE
            )
            if key in completed:
                pbar.set_postfix(
                    monitor=monitor_cls.__name__, batch=batch_size, skip=True
                )
                pbar.update()
                continue
            r = time_monitor(
                monitor_cls, FORMULA,
                trace_length=TRACE_LENGTH,
                n_traces=batch_size,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
                device=DEVICE,
                early_termination=EARLY_TERMINATION,
            )
            append_result(r, csv_path)   # flush immediately for resumability
            pbar.set_postfix(monitor=monitor_cls.__name__, batch=batch_size)
            pbar.update()

print(f"Saved (incremental): {csv_path}")
df = pd.read_csv(csv_path)
df["mean_s_per_trace"] = df["mean_s_per_cell"] * df["trace_length"]

# ---------------------------------------------------------------------------
# Plot (decoupled: the LEAD absolute-time-per-trace panel + the demoted speedup
# panel live in experiments/plots.py, so figures can be re-generated from the
# CSV — and a CPU CSV and GPU CSV overlaid — without re-running the sweep).
# ---------------------------------------------------------------------------

from experiments.plots import plot_exp3  # noqa: E402

plot_exp3(csv_path)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(df[["monitor_name", "n_traces", "mean_s_per_trace"]].to_string(index=False))
