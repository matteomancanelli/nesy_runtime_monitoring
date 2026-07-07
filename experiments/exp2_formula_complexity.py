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
from src.monitors.deep_dfa import (
    DeepDFAMonitorDense,
    DeepDFAMonitorFactored,
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
                monitor_cls.__name__, formula.name, TRACE_LENGTH, N_TRACES, DEVICE
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
# Plot (decoupled: the two timing panels + the analytic alphabet-blowup memory
# wall are drawn by experiments/plots.py from the CSV, so figures can be
# re-generated — and CPU/GPU CSVs overlaid — without re-running the sweep).
# ---------------------------------------------------------------------------

from experiments.plots import plot_exp2  # noqa: E402

plot_exp2(csv_path)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
_cols = ["monitor_name", "n_leaves", "mean_s_per_cell", "std_s_per_cell"]
print(df[_cols].to_string(index=False))
