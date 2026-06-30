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

import matplotlib.pyplot as plt
import numpy as np
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
                monitor_cls.__name__, FORMULA.name, TRACE_LENGTH, batch_size
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

# Derive time-per-trace from time-per-cell * trace_length
df["mean_s_per_trace"] = df["mean_s_per_cell"] * df["trace_length"]
df["std_s_per_trace"]  = df["std_s_per_cell"]  * df["trace_length"]

# ---------------------------------------------------------------------------
# Plot (Phase 0.3: LEAD with absolute time-per-trace; the speedup panel is
# demoted/annotated. Each monitor's speedup is normalised to ITS OWN batch=1
# time, so cross-monitor speedup comparison is misleading — RuleRunner's
# batch-1 number is catastrophically slow, which inflates its apparent
# "speedup". The absolute panel is the honest lead figure.)
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

baseline_times: dict[str, float] = {}

for monitor_name, group in df.groupby("monitor_name"):
    group = group.sort_values("n_traces")
    x = group["n_traces"]
    y_ms = group["mean_s_per_trace"] * 1e3
    yerr_ms = group["std_s_per_trace"] * 1e3
    baseline_times[monitor_name] = group["mean_s_per_trace"].iloc[0]

    ax1.errorbar(x, y_ms, yerr=yerr_ms, marker="o", label=monitor_name, capsize=3)

    speedup = baseline_times[monitor_name] / group["mean_s_per_trace"]
    ax2.plot(x, speedup, marker="o", label=monitor_name)

# Reference: ideal linear speedup
x_ref = np.array(BATCH_SIZES)
ax2.plot(x_ref, x_ref / x_ref[0], linestyle="--", color="gray", label="ideal linear")

for ax, ylabel, title in [
    (ax1, "Time per trace (ms)", "LEAD: absolute time per trace"),
    (ax2, "Speedup vs own batch=1", "Speedup (demoted — per-monitor baseline)"),
]:
    ax.set_xlabel("Batch size (number of traces)")
    ax.set_xscale("log", base=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(BATCH_SIZES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

ax2.set_yscale("log", base=2)
ax2.text(
    0.5, -0.32,
    "Caveat: each curve is normalised to its OWN batch=1 time, so curves are\n"
    "NOT comparable across monitors (RuleRunner's batch-1 is catastrophically\n"
    "slow, inflating its apparent speedup). Read absolute times from the left.",
    transform=ax2.transAxes, ha="center", va="top", fontsize=7.5, color="0.35",
)

et_note = "early termination OFF (all cells processed)" if not EARLY_TERMINATION \
    else "early termination ON"
fig.suptitle(f"Exp 3 — batch size ({et_note})", y=1.02)

plot_path = RESULTS_DIR / "exp3_batch_size.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved: {plot_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(df[["monitor_name", "n_traces", "mean_s_per_trace", "std_s_per_trace"]].to_string(index=False))
