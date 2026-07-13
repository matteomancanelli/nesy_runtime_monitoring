"""Experiment 7: state blowup as a shared weakness (symbolic vs DeepDFA).

``STATE_BLOWUP_SUITE`` = F(a & X^k b), a *genuinely exponential* |Q| = 2^k + 1
with a tiny alphabet (|AP| = 2). Symbolic must compile and store 2^k states;
DeepDFA's transition tensor is O(|Q|^2) — so the blowup hits *both* paradigms
(a shared weakness, good for the neutrality mandate), unlike the alphabet
blowup which only hits DeepDFA. Measured per-cell time (symbolic ~flat,
DeepDFA-dense rising O(|Q|^2)) + an analytic memory panel drawn by plots.py.

Distinct from exp6 (``STATE_SCALING_SUITE``/bounded_response, |Q| only linear
in the deadline k): this family is the exponential instrument, exp6 the
controlled-growth one.

The soft-divergence panel that used to share this script (non-read-once
guards vs the exact marginal) moved to the future-work fork:
artur_future_work/experiments/exp7_richer_family.py.

Run:
    python experiments/exp7_state_blowup.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.formulas import STATE_BLOWUP_SUITE
from src.benchmarks.runner import (
    append_result,
    load_completed,
    reset_if_stale,
    result_key,
    time_monitor,
)
from src.formula.compiler import compile_ltlf
from src.monitors.deep_dfa import DeepDFAMonitorDense, DeepDFAMonitorFactored
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLOWUP_MONITORS = [
    SymbolicDFAMonitor,
    DeepDFAMonitorDense,
    DeepDFAMonitorFactored,
]
TRACE_LENGTH = 500
BATCH_SIZE = 64
N_REPEATS = 5
N_WARMUP = 3
SEED = 42
EARLY_TERMINATION = False  # per-cell cost (Phase 0.1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
BLOWUP_CSV = RESULTS_DIR / "exp7_stateblowup.csv"


# ---------------------------------------------------------------------------
# State-blowup timing (symbolic vs DeepDFA — shared |Q| weakness)
# ---------------------------------------------------------------------------


def run_state_blowup() -> pd.DataFrame:
    # Stamp measured |Q| into n_leaves so it is the plot x-axis (mirrors exp6).
    formulas = []
    for base in STATE_BLOWUP_SUITE:
        n_q = len(compile_ltlf(base.formula).states)
        formulas.append(replace(base, n_leaves=n_q))
    formulas.sort(key=lambda f: f.n_leaves)

    print("State-blowup formulas (depth k -> |Q|):")
    for f in formulas:
        print(f"  {f.name:12s} |Q|={f.n_leaves}")

    reset_if_stale(BLOWUP_CSV, EARLY_TERMINATION)
    completed = load_completed(BLOWUP_CSV)
    total = len(BLOWUP_MONITORS) * len(formulas)
    with tqdm(total=total, desc="exp7 state-blowup") as pbar:
        for monitor_cls in BLOWUP_MONITORS:
            for formula in formulas:
                key = result_key(
                    monitor_cls.__name__, formula.name, TRACE_LENGTH, BATCH_SIZE
                )
                if key in completed:
                    pbar.update()
                    continue
                # label BEFORE the call: one time_monitor can run for minutes and the
                # bar only advances after it returns.
                pbar.set_postfix(
                    monitor=monitor_cls.__name__, q=formula.n_leaves, run="..."
                )
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
                append_result(r, BLOWUP_CSV)
                pbar.set_postfix(monitor=monitor_cls.__name__, q=formula.n_leaves)
                pbar.update()
    print(f"Saved (incremental): {BLOWUP_CSV}")
    return pd.read_csv(BLOWUP_CSV)


# ---------------------------------------------------------------------------
# Run + plot (decoupled)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    blow = run_state_blowup()

    from experiments.plots import plot_exp7_stateblowup

    plot_exp7_stateblowup(BLOWUP_CSV)

    print("\nState-blowup — per-cell time vs |Q|:")
    print(blow[["monitor_name", "n_leaves", "mean_s_per_cell"]]
          .rename(columns={"n_leaves": "Q"}).to_string(index=False))
