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
from src.monitors.deep_dfa import (
    DeepDFAMonitor,
    DeepDFAMonitorFactored,
    DeepDFAMonitorScan,
)
from src.monitors.progression import (
    ProgressionRuleRunnerMonitor,
    ProgressionRuleRunnerStructuredMonitor,
)
from src.monitors.rulerunner import RuleRunnerMonitor, StructuredRuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# G(a -> F b) has 2 atoms, so DeepDFA's dense 2^|AP| alphabet is tiny (4).
# DeepDFAMonitor is the dense default; DeepDFAMonitorFactored and the structured
# RuleRunner are added as within-paradigm reference lines (see docs/EXPERIMENT_MAP.md).
# G(a->F b) is nested-temporal, so the ORIGINAL RuleRunner's verdicts are wrong
# here (timing stays fair — no trap/sink, so early termination never fires); the
# Progression RR pair is the corrected paradigm 2 and is verdict-exact.
MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    StructuredRuleRunnerMonitor,
    ProgressionRuleRunnerMonitor,
    ProgressionRuleRunnerStructuredMonitor,
    DeepDFAMonitor,
    DeepDFAMonitorFactored,
    DeepDFAMonitorScan,   # parallel prefix-scan — long trace is its best case
]

FORMULA = TRACE_LENGTH_SUITE[0]          # G(a -> F b) — no early termination
TRACE_LENGTHS = [1_000, 2_000, 3_000, 4_000, 5_000,
                 6_000, 7_000, 8_000, 9_000, 10_000]
N_TRACES  = 100
N_REPEATS = 5
N_WARMUP  = 1
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
            # label BEFORE the call: one time_monitor can run for minutes and the
            # bar only advances after it returns.
            pbar.set_postfix(monitor=monitor_cls.__name__, tl=tl, run="...")
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
# Plot (decoupled: plotting lives in experiments/plots.py so figures can be
# re-generated from the CSV without re-running the sweep).
# ---------------------------------------------------------------------------

from experiments.plots import plot_exp1  # noqa: E402

plot_exp1(csv_path)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
_cols = ["monitor_name", "trace_length", "mean_s_per_cell", "std_s_per_cell"]
print(df[_cols].to_string(index=False))
