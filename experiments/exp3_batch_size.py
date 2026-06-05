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
from tqdm import tqdm

from src.benchmarks.formulas import IJCNN_SUITE
from src.benchmarks.runner import results_to_df, time_monitor
from src.monitors.deep_dfa import DeepDFAMonitor
from src.monitors.rulerunner import RuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ijcnn_n8 -> 8 atoms -> dense alphabet 2^8 = 256 symbols. This is the
# DeepDFA batching showcase: batch_run does one bmm per cell over the whole
# batch. To exercise the GPU, set device="cuda" in DeepDFAMonitor.compile.
MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    DeepDFAMonitor,
]

FORMULA      = IJCNN_SUITE[2]    # ijcnn_n8 — enough atoms to be interesting
TRACE_LENGTH = 1_000
BATCH_SIZES  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_REPEATS    = 7
N_WARMUP     = 3
SEED         = 42

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = []
total = len(MONITORS) * len(BATCH_SIZES)
with tqdm(total=total, desc="exp3") as pbar:
    for monitor_cls in MONITORS:
        for batch_size in BATCH_SIZES:
            r = time_monitor(
                monitor_cls, FORMULA,
                trace_length=TRACE_LENGTH,
                n_traces=batch_size,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
            )
            results.append(r)
            pbar.set_postfix(monitor=monitor_cls.__name__, batch=batch_size)
            pbar.update()

df = results_to_df(results)

# Derive time-per-trace from time-per-cell * trace_length
df["mean_s_per_trace"] = df["mean_s_per_cell"] * df["trace_length"]
df["std_s_per_trace"]  = df["std_s_per_cell"]  * df["trace_length"]

csv_path = RESULTS_DIR / "exp3_batch_size.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ---------------------------------------------------------------------------
# Plot (two panels: time per trace and normalised speedup)
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
    (ax1, "Time per trace (ms)", "Impact of batch size — time per trace"),
    (ax2, "Speedup vs batch=1", "Throughput scaling"),
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

plot_path = RESULTS_DIR / "exp3_batch_size.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(df[["monitor_name", "n_traces", "mean_s_per_trace", "std_s_per_trace"]].to_string(index=False))
