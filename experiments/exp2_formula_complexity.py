"""Experiment 2: per-step cost vs formula complexity (number of leaves).

Reproduces and extends Figure 2 from Perotti et al. IJCNN 2014.
IJCNN 2014 compared only RuleRunner variants (base/sparse/gpu).
We add the symbolic DFA baseline and, later, DeepDFA.

Formula family: F( OR_{i=1}^{n-1} (a0 & ai) ), n = 2, 4, 8, 16, 32 leaves.
Expected story: symbolic DFA flat; RuleRunner linear; DeepDFA flat
with GPU overhead amortised for large n.

Run:
    python experiments/exp2_formula_complexity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
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
from src.formula.compiler import compile_ltlf
from src.monitors.deep_dfa import (
    DeepDFAMonitorDense,
    DeepDFAMonitorFactored,
    DeepDFATensor,
)
from src.monitors.rulerunner import RuleRunnerMonitor, StructuredRuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# Phase 0.2 — DeepDFA's dual finding. The IJCNN family scales atoms up to n=32.
# DeepDFAMonitorFactored / DeepDFAMonitorDense are reusable subclasses defined in
# src/monitors/deep_dfa.py (shared with exp1/3/5):
#
#   * Factored mode never materializes the 2^|AP| tensor: each guard is a small
#     disjoint cube cover, and the per-cell transition is a vectorized mask
#     reduction (no per-cell sympy closures). Its per-cell cost stays *flat* in
#     |AP| — this is the curve that scales. Exact for crisp traces.
#
#   * Dense mode stores the full (|Q|, 2^|AP|, |Q|) transition tensor. Its
#     per-cell compute is a single gather (fast where it fits), but storage is
#     exponential in |AP| — the alphabet-blowup *finding*. We run it only up to
#     DENSE_MAX_LEAVES and visualize the memory wall analytically (3rd panel),
#     so its curve shows "fast but memory-walls out" against factored's "flat
#     and scales". Beyond the cap, building 2^n symbols is infeasible (n=32 is
#     4.3e9 symbols / ~64 GB at |Q|=2), so we skip it.

# Largest leaf count for which the 2^|AP| dense tensor is still feasible to
# build/store. Above this the dense variant is skipped (it would OOM); the
# memory panel shows the wall analytically for all n.
DENSE_MAX_LEAVES = 16


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# RuleRunnerMonitor (CILP) + StructuredRuleRunnerMonitor (Fig. 5 variant) give
# the intra-RuleRunner comparison from IJCNN 2014 alongside the symbolic and
# DeepDFA baselines (see docs/EXPERIMENT_MAP.md).
MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    StructuredRuleRunnerMonitor,
    DeepDFAMonitorFactored,
    DeepDFAMonitorDense,
]

TRACE_LENGTH = 5_000
N_TRACES     = 100
N_REPEATS    = 7
N_WARMUP     = 3
SEED         = 42

# Phase 0.1 — kill the early-termination confound. The IJCNN family
# `◇(⋁(a0 ∧ ai))` early-terminates almost immediately on random traces, so a
# crisp monitor that gives up after ~2 cells would be timed against batched
# monitors that process all cells: not the same workload. Forcing all cells to
# be processed makes the per-cell cost an apples-to-apples comparison. State
# this explicitly in the paper.
EARLY_TERMINATION = False

# Tensor monitors run their batched matmuls here; the symbolic DFA ignores it.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "exp2_formula_complexity.csv"
reset_if_stale(csv_path, EARLY_TERMINATION)  # drop CSVs from the other mode
completed = load_completed(csv_path)   # resume: skip cells already on disk

total = len(MONITORS) * len(IJCNN_SUITE)
with tqdm(total=total, desc="exp2") as pbar:
    for monitor_cls in MONITORS:
        for formula in IJCNN_SUITE:
            # Dense materializes 2^|AP| symbols — infeasible past the cap.
            dense_oom = (
                monitor_cls is DeepDFAMonitorDense
                and formula.n_leaves > DENSE_MAX_LEAVES
            )
            if dense_oom:
                pbar.set_postfix(
                    monitor=monitor_cls.__name__, n=formula.n_leaves, skip="OOM"
                )
                pbar.update()
                continue
            key = result_key(
                monitor_cls.__name__, formula.name, TRACE_LENGTH, N_TRACES
            )
            if key in completed:
                pbar.set_postfix(
                    monitor=monitor_cls.__name__, n=formula.n_leaves, skip=True
                )
                pbar.update()
                continue
            r = time_monitor(
                monitor_cls, formula,
                trace_length=TRACE_LENGTH,
                n_traces=N_TRACES,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
                device=DEVICE,
                early_termination=EARLY_TERMINATION,
            )
            append_result(r, csv_path)   # flush immediately for resumability
            pbar.set_postfix(monitor=monitor_cls.__name__, n=formula.n_leaves)
            pbar.update()

print(f"Saved (incremental): {csv_path}")
df = pd.read_csv(csv_path)

# ---------------------------------------------------------------------------
# Transition-representation memory vs n — the alphabet-blowup *finding*.
# Computed analytically (no need to build the infeasible dense tensors): the
# dense tensor is (|Q|, 2^n, |Q|) float32; the factored cube masks are
# (n_cubes, n) require-true + require-false float32. |Q| and the cube count
# come from the (cheap) DFA compilation, which never materializes 2^n.
# ---------------------------------------------------------------------------

FLOAT_BYTES = 4
mem = []
for formula in IJCNN_SUITE:
    dfa = compile_ltlf(formula.formula)
    n_q = len(dfa.states)
    dt = DeepDFATensor(dfa, mode="factored")
    dense_bytes = n_q * n_q * (2 ** formula.n_leaves) * FLOAT_BYTES
    factored_bytes = (dt._cube_rt.numel() + dt._cube_rf.numel()) * FLOAT_BYTES
    mem.append((formula.n_leaves, dense_bytes, factored_bytes))

mem_n = [m[0] for m in mem]
dense_gb = [m[1] / 1e9 for m in mem]
factored_gb = [m[2] / 1e9 for m in mem]

# ---------------------------------------------------------------------------
# Plot: two timing panels (raw + per-leaf, matching IJCNN 2014) + memory wall
# ---------------------------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.2))

for monitor_name, group in df.groupby("monitor_name"):
    group = group.sort_values("n_leaves")
    x = group["n_leaves"]
    y_us = group["mean_s_per_cell"] * 1e6
    yerr_us = group["std_s_per_cell"] * 1e6

    ax1.errorbar(x, y_us, yerr=yerr_us, marker="o", label=monitor_name, capsize=3)
    ax2.errorbar(x, y_us / x, yerr=yerr_us / x, marker="o", label=monitor_name, capsize=3)

all_n = [f.n_leaves for f in IJCNN_SUITE]
for ax, ylabel, title in [
    (ax1, "Avg time per cell (µs)", "Impact of number of leaves"),
    (ax2, "Avg time per cell per leaf (µs)", "Averaged impact of number of leaves"),
]:
    ax.set_xlabel("Leaves (atoms) — exponential scale")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(all_n)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

# Memory panel: dense 2^n wall vs factored mask growth (log-y).
ax3.plot(mem_n, dense_gb, marker="o", color="tab:red",
         label="DeepDFA dense  $|Q|^2\\,2^{|AP|}$")
ax3.plot(mem_n, factored_gb, marker="o", color="tab:green",
         label="DeepDFA factored (cube masks)")
ax3.axhline(4.0, color="gray", linestyle=":", label="4 GB VRAM (laptop)")
ax3.axvline(DENSE_MAX_LEAVES, color="tab:red", linestyle="--", alpha=0.4)
ax3.set_yscale("log")
ax3.set_xlabel("Leaves (atoms)")
ax3.set_ylabel("Transition representation (GB)")
ax3.set_title("Alphabet-blowup finding: dense $2^{|AP|}$ memory wall")
ax3.set_xticks(all_n)
ax3.legend(fontsize=8)
ax3.grid(True, which="both", linestyle="--", alpha=0.4)

et_note = "early termination OFF (all cells processed)" if not EARLY_TERMINATION \
    else "early termination ON"
fig.suptitle(
    f"Exp 2 — formula complexity ({et_note}); "
    f"dense timed for n≤{DENSE_MAX_LEAVES}, factored for all n",
    y=1.02,
)

plot_path = RESULTS_DIR / "exp2_formula_complexity.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close(fig)

print("\nTransition-representation memory (GB):")
for n, d, f in zip(mem_n, dense_gb, factored_gb):
    print(f"  n={n:2d}  dense={d:.3e}  factored={f:.3e}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(df[["monitor_name", "n_leaves", "mean_s_per_cell", "std_s_per_cell"]].to_string(index=False))
