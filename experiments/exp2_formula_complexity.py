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
from tqdm import tqdm

from src.benchmarks.formulas import IJCNN_SUITE
from src.benchmarks.runner import results_to_df, time_monitor
from src.monitors.deep_dfa import DeepDFAMonitor
from src.monitors.rulerunner import RuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


# The IJCNN family scales atoms up to n=32. DeepDFA's *dense* alphabet is
# 2^|AP|, which is infeasible beyond ~16 atoms (this is DeepDFA's structural
# scaling weakness; see CLAUDE.md § Paradigm 3). We therefore time DeepDFA in
# *factored* mode here, which never materializes the 2^|AP| tensor and is
# exact for crisp traces. (To also show the dense blowup, add a dense variant
# capped at small n.)
class DeepDFAMonitorFactored(DeepDFAMonitor):
    @classmethod
    def compile(cls, formula: str) -> DeepDFAMonitor:
        return DeepDFAMonitor.compile(formula, mode="factored")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    DeepDFAMonitorFactored,
]

TRACE_LENGTH = 5_000
N_TRACES     = 100
N_REPEATS    = 7
N_WARMUP     = 3
SEED         = 42

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = []
total = len(MONITORS) * len(IJCNN_SUITE)
with tqdm(total=total, desc="exp2") as pbar:
    for monitor_cls in MONITORS:
        for formula in IJCNN_SUITE:
            r = time_monitor(
                monitor_cls, formula,
                trace_length=TRACE_LENGTH,
                n_traces=N_TRACES,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
            )
            results.append(r)
            pbar.set_postfix(monitor=monitor_cls.__name__, n=formula.n_leaves)
            pbar.update()

df = results_to_df(results)
csv_path = RESULTS_DIR / "exp2_formula_complexity.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ---------------------------------------------------------------------------
# Plot (two panels: raw time and time normalised per leaf, matching IJCNN 2014)
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for monitor_name, group in df.groupby("monitor_name"):
    group = group.sort_values("n_leaves")
    x = group["n_leaves"]
    y_us = group["mean_s_per_cell"] * 1e6
    yerr_us = group["std_s_per_cell"] * 1e6

    ax1.errorbar(x, y_us, yerr=yerr_us, marker="o", label=monitor_name, capsize=3)
    ax2.errorbar(x, y_us / x, yerr=yerr_us / x, marker="o", label=monitor_name, capsize=3)

for ax, ylabel, title in [
    (ax1, "Avg time per cell (µs)", "Impact of number of leaves"),
    (ax2, "Avg time per cell per leaf (µs)", "Averaged impact of number of leaves"),
]:
    ax.set_xlabel("Leaves (atoms) — exponential scale")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(group["n_leaves"]))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

plot_path = RESULTS_DIR / "exp2_formula_complexity.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(df[["monitor_name", "n_leaves", "mean_s_per_cell", "std_s_per_cell"]].to_string(index=False))
