"""Experiment 7: richer benchmark family (Phase 3.3).

Two findings the IJCNN family cannot show, each on a new formula family
(src/benchmarks/formulas.py):

  * **Panel 1 — probabilistic divergence (non-read-once).** On a non-read-once
    guard, DeepDFA's ``soft_matrix`` assumes atom independence and therefore
    *over-counts* the true marginal acceptance probability. The threshold family
    ``NON_READ_ONCE_SUITE`` (at-least-k-of-n, atom multiplicity growing 2→3→4→6)
    turns the single ``majority3`` data point into a **curve**: we sweep a shared
    input probability p and compare the soft score to the brute-force exact
    marginal (``characterize.exact_marginal_trace``). The realistic Declare
    template ``alt_response`` (multi-state, also non-read-once) is the same
    measurement over a short trace. This is a *finding*, not an identity — on the
    read-once IJCNN family the soft path is exact and the gap would be zero.

  * **Panel 2 — state-blowup neutrality.** ``STATE_BLOWUP_SUITE`` = F(a & X^k b),
    a *genuinely exponential* |Q| = 2^k + 1 with a tiny alphabet (|AP| = 2).
    Symbolic must compile and store 2^k states; DeepDFA's transition tensor is
    O(|Q|^2) — so the blowup hits *both* paradigms (a shared weakness, good for
    the neutrality mandate), unlike the alphabet blowup which only hits DeepDFA.
    Measured per-cell time (symbolic ~flat, DeepDFA-dense rising O(|Q|^2)) + an
    analytic memory panel drawn by plots.py.

Panel 1 needs no GPU (it is a numeric-divergence measurement); Panel 2's timing
uses the GPU when available. Both are resumable.

Run:
    python experiments/exp7_richer_family.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.characterize import exact_marginal_trace, guard_read_once
from src.benchmarks.formulas import (
    NON_READ_ONCE_SUITE,
    STATE_BLOWUP_SUITE,
)
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

# Panel 1 — divergence sweep. Shared per-atom probability p; deterministic (a
# constant-p input has no randomness), so no seed loop. alt_response is measured
# over a short trace (it is multi-state); the threshold family over a single cell.
P_GRID = tuple(round(p, 3) for p in np.linspace(0.05, 0.95, 19))
ALT_TRACE_LEN = 4  # alt_response only; 2 atoms → 2^(2*4)=256 brute-force terms

# Panel 2 — state-blowup timing (the symbolic-vs-DeepDFA shared-weakness trio).
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
DIVERGENCE_CSV = RESULTS_DIR / "exp7_divergence.csv"
BLOWUP_CSV = RESULTS_DIR / "exp7_stateblowup.csv"


# ---------------------------------------------------------------------------
# Panel 1: probabilistic divergence (non-read-once)
# ---------------------------------------------------------------------------


def _trace_len_for(name: str) -> int:
    return ALT_TRACE_LEN if name == "alt_response" else 1


def divergence_rows() -> list[dict]:
    rows = []
    for f in NON_READ_ONCE_SUITE:
        mult = max(guard_read_once(compile_ltlf(f.formula))[1].values(), default=1)
        dd = DeepDFAMonitorFactored.compile(f.formula)
        L = _trace_len_for(f.name)
        for p in P_GRID:
            trace = [{a: float(p) for a in f.atoms} for _ in range(L)]
            exact = exact_marginal_trace(f.formula, trace)
            soft_raw = dd.acceptance_probability(trace, normalize=False)
            soft_norm = dd.acceptance_probability(trace, normalize=True)
            rows.append({
                "formula": f.name,
                "n_atoms": len(f.atoms),
                "multiplicity": int(mult),
                "trace_len": L,
                "p": float(p),
                "exact": float(exact),
                "soft_raw": float(soft_raw),
                "soft_norm": float(soft_norm),
                "gap_raw": float(soft_raw - exact),
                "gap_norm": float(soft_norm - exact),
            })
    return rows


def run_divergence() -> pd.DataFrame:
    # Resume unless the p-grid or formula set changed (a different workload).
    if DIVERGENCE_CSV.exists():
        prev = pd.read_csv(DIVERGENCE_CSV)
        same_p = set(np.round(prev["p"], 3)) == set(P_GRID)
        same_f = set(prev["formula"]) == {f.name for f in NON_READ_ONCE_SUITE}
        if same_p and same_f:
            print(f"exp7 divergence: reusing {DIVERGENCE_CSV}")
            return prev
        print("exp7 divergence: config changed — recomputing.")
    df = pd.DataFrame(divergence_rows())
    df.to_csv(DIVERGENCE_CSV, index=False)
    print(f"Saved: {DIVERGENCE_CSV}")
    return df


# ---------------------------------------------------------------------------
# Panel 2: state-blowup timing (symbolic vs DeepDFA — shared |Q| weakness)
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
    div = run_divergence()
    blow = run_state_blowup()

    from experiments.plots import plot_exp7_divergence, plot_exp7_stateblowup

    plot_exp7_divergence(DIVERGENCE_CSV)
    plot_exp7_stateblowup(BLOWUP_CSV)

    # Console summary.
    print("\nDivergence — max soft over-count over p (raw vs normalized):")
    summary = (
        div.groupby(["formula", "multiplicity"])
        .agg(max_gap_raw=("gap_raw", "max"), max_abs_gap_norm=("gap_norm",
             lambda s: s.abs().max()))
        .reset_index()
        .sort_values("multiplicity")
    )
    print(summary.to_string(index=False))

    print("\nState-blowup — per-cell time vs |Q|:")
    print(blow[["monitor_name", "n_leaves", "mean_s_per_cell"]]
          .rename(columns={"n_leaves": "Q"}).to_string(index=False))
