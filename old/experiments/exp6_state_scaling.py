"""Experiment 6: per-cell cost vs automaton size (|Q|).

The promising direction for inverting the "symbolic always wins" trend. Exp 2
scales the *alphabet* (|AP|); this scales the *state space* (|Q|) while keeping
the alphabet tiny, using the bounded-response family
    G( a -> (b | X b | ... | X^k b) )
whose minimal DFA size grows ~linearly with the deadline k (|AP| = 2 fixed, so
the dense 2^|AP| = 4 tensor stays feasible even at large |Q|).

Why this might invert the trend:
  - The symbolic walk only ever evaluates the current state's out-edges, so its
    per-cell cost is ~flat in |Q| (still a lookup).
  - DeepDFA's batched step is a matmul whose work is O(|Q|^2) per trace; as |Q|
    grows, that real arithmetic finally amortizes the fixed per-launch overhead
    that dominates at the tiny IJCNN automata — so the batched matmul can catch
    up to, and potentially overtake, the symbolic walk.

Three DeepDFA variants are compared so the regimes are visible:
  - dense (sequential): one bmm per cell (the Exp 2/3 path).
  - scan: folds the whole trace into O(log L) matmuls (kills the per-launch
    overhead, but multiplies arithmetic by ~|Q|*log L — so it wins at SMALL |Q|
    and loses at large |Q|, the opposite regime; included to make that explicit).
  - factored: the cube-mask crisp path.

RuleRunner: bounded response nests X under -> under G, so the ORIGINAL RuleRunner
hits its nested-temporal limitation (verdicts untrusted — kept only for the
throughput comparison, timing stays well-defined). The Progression RR pair is
verdict-exact on this family and directly relevant to the |Q| story: its residual
closure grows with |Q|, so progression-flat is a second "scales with |Q|"
paradigm-2 curve (flat per cell — depth/|Q| absorbed into the state count),
while the original RR pays depth+1 within-step passes that grow with the deadline.

Run on a GPU (Colab) for the intended measurement — the overhead story is a
GPU/launch-latency effect.

Run:
    python experiments/exp6_state_scaling.py
"""

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.formulas import STATE_SCALING_SUITE
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

MONITORS = [
    SymbolicDFAMonitor,
    RuleRunnerMonitor,
    StructuredRuleRunnerMonitor,
    ProgressionRuleRunnerMonitor,
    ProgressionRuleRunnerStructuredMonitor,
    DeepDFAMonitorDense,
    DeepDFAMonitorScan,
    DeepDFAMonitorFactored,
]

# Progression's own wall (the dual of DeepDFA-dense's 2^|AP| wall in exp2, and
# the "price" CLAUDE.md flags for the corrected paradigm 2). On this family the
# *residual* closure is not quotiented by full LTLf language equivalence:
# `simplify` can leave multiple right-language-equivalent residuals even though
# the minimal DFA for this family grows only linearly.  Exact raw counts and
# build times depend on the normalization version and belong in regenerated
# evaluation artifacts, not in this source comment.  The qualitative wall is
# stable: one SymPy Boolean simplification is paid per discovered residual,
# entirely inside `compile`, before a single cell is timed.
#
# Capping at |Q| <= 14 (k <= 12) keeps six points per progression curve and the
# whole experiment near ~20 min. Past the cap the two progression monitors are
# skipped, exactly as DeepDFAMonitorDense is skipped past DENSE_MAX_LEAVES in
# exp2 — the wall is a *finding* to report, not a bug to hide. The symbolic /
# original-RR / DeepDFA curves still span the full |Q| = 4..20 sweep.
PROGRESSION_MAX_Q = 14

_CLOSURE_WALL = (
    ProgressionRuleRunnerMonitor,
    ProgressionRuleRunnerStructuredMonitor,
)

TRACE_LENGTH = 500
BATCH_SIZE = 256  # fixed batch; |Q| is the swept axis
N_REPEATS = 5
N_WARMUP = 1
SEED = 42
EARLY_TERMINATION = False  # per-cell cost (Phase 0.1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Record measured |Q| separately from the declared deadline parameter.
# ---------------------------------------------------------------------------

FORMULAS = []
for base in STATE_SCALING_SUITE:
    n_q = len(compile_ltlf(base.formula).states)
    FORMULAS.append(replace(base, dfa_states=n_q))
FORMULAS.sort(key=lambda f: f.dfa_states or 0)

print("State-scaling formulas (deadline -> |Q|):")
for f in FORMULAS:
    print(f"  {f.name:16s} |Q|={f.dfa_states}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "exp6_state_scaling.csv"
reset_if_stale(csv_path, EARLY_TERMINATION)
completed = load_completed(csv_path)

total = len(MONITORS) * len(FORMULAS)
with tqdm(total=total, desc="exp6") as pbar:
    for monitor_cls in MONITORS:
        for formula in FORMULAS:
            # Progression's own wall — see PROGRESSION_MAX_Q above.
            closure_wall = (
                monitor_cls in _CLOSURE_WALL
                and formula.dfa_states is not None
                and formula.dfa_states > PROGRESSION_MAX_Q
            )
            if closure_wall:
                pbar.set_postfix(
                    monitor=monitor_cls.__name__,
                    q=formula.dfa_states,
                    skip="closure",
                )
                pbar.update()
                continue
            key = result_key(
                monitor_cls.__name__, formula.name, TRACE_LENGTH, BATCH_SIZE
            )
            if key in completed:
                pbar.set_postfix(
                    monitor=monitor_cls.__name__, q=formula.dfa_states, skip=True
                )
                pbar.update()
                continue
            # label BEFORE the call: one time_monitor can run for minutes and the
            # bar only advances after it returns.
            pbar.set_postfix(
                monitor=monitor_cls.__name__, q=formula.dfa_states, run="..."
            )
            r = time_monitor(
                monitor_cls,
                formula,
                trace_length=TRACE_LENGTH,
                n_traces=BATCH_SIZE,
                n_repeats=N_REPEATS,
                n_warmup=N_WARMUP,
                seed=SEED,
                device=DEVICE,
                early_termination=EARLY_TERMINATION,
            )
            append_result(r, csv_path)
            pbar.set_postfix(monitor=monitor_cls.__name__, q=formula.dfa_states)
            pbar.update()

print(f"Saved (incremental): {csv_path}")
df = pd.read_csv(csv_path)

# ---------------------------------------------------------------------------
# Plot (decoupled)
# ---------------------------------------------------------------------------

from experiments.plots import plot_exp6  # noqa: E402

plot_exp6(csv_path)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print(
    df[["monitor_name", "dfa_states", "mean_s_per_cell", "std_s_per_cell"]]
    .rename(columns={"dfa_states": "Q"})
    .to_string(index=False)
)
