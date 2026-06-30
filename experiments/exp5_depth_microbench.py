"""Experiment 5 (Phase 0.5): within-step depth micro-benchmark.

Isolates the *within-step* parallelism axis explicitly, independent of formula
breadth (Exp 2's leaf count) and cross-trace batching (Exp 3's batch size).

Design (CLAUDE.md § Benchmark Design):
  - Fix formula breadth at ijcnn_n8 (8 leaves) and wrap it in nested X:
        ijcnn_n8,  X(ijcnn_n8),  X(X(ijcnn_n8)),  ... up to DEPTH_MAX.
    Each extra X adds one parse-tree level without changing breadth.
  - Fix batch size = 1 (no cross-trace batching), so the only thing varying is
    the per-cell within-step cost as a function of parse-tree depth. Trace
    length is held fixed and only amortises per-call overhead — see the
    TRACE_LENGTH note below for why we use a longer single trace rather than the
    literal single cell sketched in CLAUDE.md.
  - Report per-cell time vs. depth.

Expected curves:
  - Symbolic DFA: FLAT. Depth is absorbed into the DFA state count, not a loop —
    one state lookup regardless of depth.
  - RuleRunner: LINEAR in depth. Its convergence loop runs depth+1 evaluation
    passes per cell — the within-step sequential cost IJCNN 2014 claims to
    parallelise.
  - DeepDFA: FLAT. One matmul per cell regardless of depth (depth is absorbed
    into |Q|, not into a loop).

This is the cleanest empirical support for the "one matmul per cell" framing,
separating the within-step depth cost from Exp 2's breadth (leaf) scaling.

Note on RuleRunner correctness: nesting X over the F-rooted ijcnn formula hits
the documented nested-temporal limitation (CLAUDE.md), so RuleRunner's
*verdicts* are not trusted here. As in Exp 1, that does not affect the timing
methodology — we measure per-cell wall time, which is well-defined regardless
of verdict correctness.

Run:
    python experiments/exp5_depth_microbench.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.formulas import IJCNN_SUITE, BenchmarkFormula
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

# Depth is the within-step axis: RuleRunner (CILP) runs depth+1 convergence
# passes per cell; the structured RuleRunner is the second within-step data
# point; DeepDFA (dense + factored) and Symbolic stay flat (one matmul / one
# lookup regardless of depth). See docs/EXPERIMENT_MAP.md.
MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    StructuredRuleRunnerMonitor,
    DeepDFAMonitor,
    DeepDFAMonitorFactored,
]

BASE_FORMULA = IJCNN_SUITE[2]   # ijcnn_n8 — fixed breadth (8 leaves)
DEPTHS       = list(range(0, 11))   # nested-X depth 0..10

# The design isolates within-step cost at batch=1 (no cross-trace batching) and
# a single depth axis (depth lives only in the formula). CLAUDE.md sketches
# trace_length=1, but a single cell is dominated by fixed per-call Python
# overhead (compile dispatch, encode) which buries the depth signal — verified
# empirically: at len=1 the RuleRunner depth trend is pure noise, at len>=500 it
# is a clean rise while Symbolic/DeepDFA stay flat. Trace length does NOT
# contaminate the depth axis (it adds no breadth and no cross-trace batching),
# it only amortises the call overhead, so we default to a longer single trace.
TRACE_LENGTH = 500
BATCH_SIZE   = 1
N_REPEATS    = 25
N_WARMUP     = 5
SEED         = 42

# Per-cell-cost methodology: process the cell rather than measure how fast a
# monitor gives up (Phase 0.1). With a single cell this barely matters, but we
# keep it consistent with Exp 2/3.
EARLY_TERMINATION = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def nested_x(base: BenchmarkFormula, depth: int) -> BenchmarkFormula:
    """Wrap ``base`` in ``depth`` nested X operators (breadth unchanged)."""
    formula = "X(" * depth + base.formula + ")" * depth
    return BenchmarkFormula(
        name=f"{base.name}_xdepth{depth}",
        formula=formula,
        atoms=base.atoms,
        n_leaves=base.n_leaves,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "exp5_depth_microbench.csv"
reset_if_stale(csv_path, EARLY_TERMINATION)  # drop CSVs from the other mode
completed = load_completed(csv_path)          # resume: skip cells already on disk

total = len(MONITORS) * len(DEPTHS)
with tqdm(total=total, desc="exp5") as pbar:
    for monitor_cls in MONITORS:
        for depth in DEPTHS:
            formula = nested_x(BASE_FORMULA, depth)
            key = result_key(
                monitor_cls.__name__, formula.name, TRACE_LENGTH, BATCH_SIZE
            )
            if key in completed:
                pbar.set_postfix(
                    monitor=monitor_cls.__name__, depth=depth, skip=True
                )
                pbar.update()
                continue
            r = time_monitor(
                monitor_cls, formula,
                trace_length=TRACE_LENGTH,
                n_traces=BATCH_SIZE,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
                device=DEVICE,
                early_termination=EARLY_TERMINATION,
            )
            append_result(r, csv_path)   # flush immediately for resumability
            pbar.set_postfix(monitor=monitor_cls.__name__, depth=depth)
            pbar.update()

print(f"Saved (incremental): {csv_path}")
df = pd.read_csv(csv_path)

# Recover the nested-X depth from the formula name (..._xdepth{d}).
df["depth"] = df["formula_name"].str.extract(r"_xdepth(\d+)$").astype(int)

# ---------------------------------------------------------------------------
# Plot: per-cell time vs nested-X depth
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

for monitor_name, group in df.groupby("monitor_name"):
    group = group.sort_values("depth")
    ax.errorbar(
        group["depth"],
        group["mean_s_per_cell"] * 1e6,           # microseconds per cell
        yerr=group["std_s_per_cell"] * 1e6,
        marker="o", capsize=3, label=monitor_name,
    )

ax.set_xlabel("Nested-X depth (parse-tree levels over fixed ijcnn_n8 breadth)")
ax.set_ylabel("Time per cell (µs)")
ax.set_title("Exp 5 — within-step cost vs parse-tree depth")
ax.set_xticks(DEPTHS)
ax.set_yscale("log")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

et_note = "early termination OFF" if not EARLY_TERMINATION else "early termination ON"
fig.suptitle(
    f"batch=1, trace_len=1, breadth=ijcnn_n8 ({et_note}, device={DEVICE})",
    y=0.98, fontsize=9,
)

plot_path = RESULTS_DIR / "exp5_depth_microbench.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(
    df.sort_values(["monitor_name", "depth"])[
        ["monitor_name", "depth", "mean_s_per_cell", "std_s_per_cell"]
    ].to_string(index=False)
)
