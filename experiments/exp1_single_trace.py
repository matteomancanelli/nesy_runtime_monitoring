"""Experiment 1: per-step cost vs trace length.

Fix formula to G(a -> F b) (no trap/sink, so no early termination).
Vary trace length from 1k to 10k cells, matching IJCNN 2014.
Expected story: all paradigms flat — per-step cost is constant.

Run:
    python experiments/exp1_single_trace.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.formulas import TRACE_LENGTH_SUITE
from src.benchmarks.runner import (
    append_result,
    load_completed,
    result_key,
    time_monitor,
)
from src.monitors.deep_dfa import DeepDFAMonitor
from src.monitors.rulerunner import RuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# G(a -> F b) has 2 atoms, so DeepDFA's dense 2^|AP| alphabet is tiny (4).
MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    DeepDFAMonitor,
]

FORMULA = TRACE_LENGTH_SUITE[0]          # G(a -> F b) — no early termination
TRACE_LENGTHS = [1_000, 2_000, 3_000, 4_000, 5_000,
                 6_000, 7_000, 8_000, 9_000, 10_000]
N_TRACES  = 100
N_REPEATS = 7
N_WARMUP  = 3
SEED      = 42

# Tensor monitors (RuleRunner, DeepDFA) run their batched matmuls here;
# the symbolic DFA ignores it. Auto-uses the GPU when one is available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "exp1_single_trace.csv"
completed = load_completed(csv_path)   # resume: skip cells already on disk

total = len(MONITORS) * len(TRACE_LENGTHS)
with tqdm(total=total, desc="exp1") as pbar:
    for monitor_cls in MONITORS:
        for tl in TRACE_LENGTHS:
            key = result_key(monitor_cls.__name__, FORMULA.name, tl, N_TRACES)
            if key in completed:
                pbar.set_postfix(monitor=monitor_cls.__name__, tl=tl, skip=True)
                pbar.update()
                continue
            r = time_monitor(
                monitor_cls, FORMULA,
                trace_length=tl,
                n_traces=N_TRACES,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
                device=DEVICE,
            )
            append_result(r, csv_path)   # flush immediately for resumability
            pbar.set_postfix(monitor=monitor_cls.__name__, tl=tl)
            pbar.update()

print(f"Saved (incremental): {csv_path}")
df = pd.read_csv(csv_path)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

for monitor_name, group in df.groupby("monitor_name"):
    group = group.sort_values("trace_length")
    ax.errorbar(
        group["trace_length"] / 1_000,
        group["mean_s_per_cell"] * 1e6,
        yerr=group["std_s_per_cell"] * 1e6,
        marker="o",
        label=monitor_name,
        capsize=3,
    )

ax.set_xlabel("Monitored cells (×10³)")
ax.set_ylabel("Avg time per cell (µs)")
ax.set_title(f"Impact of trace length — formula: {FORMULA.formula}")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

plot_path = RESULTS_DIR / "exp1_single_trace.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(df[["monitor_name", "trace_length", "mean_s_per_cell", "std_s_per_cell"]].to_string(index=False))
