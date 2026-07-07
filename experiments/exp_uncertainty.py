"""Capability Exp A: monitoring under perceptual uncertainty (Phase 1.4).

The first capability that *justifies* going neuro-symbolic. A neural
perceptor emits per-atom probabilities, not booleans. We corrupt clean
traces at a controllable noise level ε (Phase 1.1 models) and ask, as ε
grows:

  1. **Verdict accuracy** — does the SATISFY/VIOLATE verdict still match the
     oracle (the clean-trace symbolic verdict)? Three monitors:
       * **Symbolic (threshold)** — the brittle baseline: it *must* collapse
         each probability to a bit at 0.5 and walk the crisp DFA. It cannot
         emit a confidence at all.
       * **DeepDFA (soft, raw)** — the settled Option-A readout: propagate the
         full state distribution through ``soft_matrix`` (no mid-trace argmax)
         and read ``q_final @ accepting``.
       * **DeepDFA (soft, normalized)** — the same, divided by the total
         propagated mass. Identical to raw on read-once guards; on a
         non-read-once guard it repairs the score into [0, 1] (see below).

  2. **Calibration** of DeepDFA's acceptance score — the capability the
     symbolic monitor *fundamentally lacks*. Reliability diagram + ECE, with
     Brier and ROC-AUC as summary scalars (Phase 1.3 metrics).

**Two honest findings this experiment surfaces (do not oversell):**

  * *Accuracy:* for **unbiased** perceptual noise, thresholding the Beta mean
    is near Bayes-optimal per cell and the DFA is then exact, so the symbolic
    baseline is *strong* — the soft monitor is **competitive, not dominant**,
    on the hard verdict. The NeSy payoff here is the **calibrated confidence**,
    which symbolic cannot produce, not a higher accuracy. Under ``BitFlipNoise``
    the soft monitor sees the same crisp flipped bits, so its accuracy is
    *identical* to symbolic — the "no free lunch when information is destroyed"
    control.

  * *Non-read-once normalization defect:* ``soft_matrix`` is only row-stochastic
    when every DFA guard is read-once. On the ``majority3`` guard
    ``(a&b)|(b&c)|(a&c)`` the rows sum to ~1.16, so the **raw** acceptance score
    is not a valid probability (it exceeds 1). Normalization repairs the range;
    the diagnostic panel plots the raw max-score / fraction-over-1 vs ε. This is
    a stronger form of the read-once caveat (Phase 3.3) and connects to the
    Phase 3.1 theory question of what the "correct" probabilistic verdict is.

This is a **new accuracy/calibration harness**: it reuses only ``random_traces``
and the symbolic oracle, not the timing ``runner.py``.

Run:
    python experiments/exp_uncertainty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.benchmarks.calibration import (
    brier_score,
    expected_calibration_error,
    roc_auc,
    verdict_accuracy,
)
from src.benchmarks.formulas import CALIBRATION_SUITE, BenchmarkFormula
from src.benchmarks.noise import (
    BetaNoise,
    BitFlipNoise,
    NoiseModel,
    threshold_trace,
    true_verdicts,
)
from src.benchmarks.runner import random_traces
from src.monitors.base import Verdict
from src.monitors.deep_dfa import DeepDFAMonitorFactored
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPS_SWEEP = tuple(round(x, 3) for x in np.linspace(0.0, 0.8, 9))
N_TRACES = 3_000
N_NOISE_SEEDS = 3          # average metrics over independent corruptions
TRACE_SEED = 42            # clean-trace generation (fixed across ε)
BETA_CONCENTRATION = 10.0
REP_EPS = 0.4              # ε at which the reliability diagram is drawn
N_BINS = 10

# Trace length is balanced so accuracy is not a majority-class artifact, but
# floored at MIN_LEN: the non-read-once normalization defect (raw score > 1)
# only compounds over ≥2 cells, and single-cell "monitoring" is degenerate.
# So majority3 (which balances at L=1) is monitored over MIN_LEN cells, where
# the defect is visible; the imbalance this introduces is shown via the
# majority-class baseline line on the accuracy panels.
MIN_LEN = 3

# Noise models are (name, factory(eps)) so ε is swept uniformly.
NOISE_MODELS: dict[str, "callable[[float], NoiseModel]"] = {
    "beta": lambda e: BetaNoise(e, concentration=BETA_CONCENTRATION),
    "bitflip": lambda e: BitFlipNoise(e),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "exp_uncertainty.csv"


# ---------------------------------------------------------------------------
# Balanced trace length: pick L so the clean-label positive rate is ~0.5,
# so "accuracy" is not dominated by a majority-class baseline. F-formulas
# accept more with longer traces, G/response fewer — pick the closest L.
# ---------------------------------------------------------------------------


def pick_trace_length(
    formula: BenchmarkFormula,
    target: float = 0.5,
    max_len: int = 12,
    n: int = 2_000,
    min_len: int = MIN_LEN,
) -> tuple[int, float]:
    best_len, best_rate = min_len, 1.0
    probe_rng = np.random.default_rng(9_999)
    for length in range(min_len, max_len + 1):
        crisp = random_traces(formula.atoms, length, n, probe_rng)
        rate = np.mean(
            [v is Verdict.SATISFY for v in true_verdicts(formula.formula, crisp)]
        )
        if abs(rate - target) < abs(best_rate - target):
            best_len, best_rate = length, float(rate)
    return best_len, best_rate


# ---------------------------------------------------------------------------
# One (formula, noise, ε) evaluation, averaged over noise seeds.
# ---------------------------------------------------------------------------


def _verdicts(scores: np.ndarray) -> list[Verdict]:
    # clip is a no-op for the 0.5 decision but keeps the type honest
    return [Verdict.SATISFY if s >= 0.5 else Verdict.VIOLATE for s in scores]


def evaluate(
    formula: BenchmarkFormula,
    trace_length: int,
    make_noise: "callable[[float], NoiseModel]",
    eps: float,
    crisp: list,
    labels: list[Verdict],
) -> dict:
    """Metrics for one cell of the sweep, averaged over N_NOISE_SEEDS draws."""
    sym = SymbolicDFAMonitor.compile(formula.formula)
    dd = DeepDFAMonitorFactored.compile(formula.formula, device=DEVICE)

    acc_sym, acc_raw, acc_norm = [], [], []
    ece_raw, ece_norm = [], []
    brier_raw, brier_norm = [], []
    auc_raw, auc_norm = [], []
    max_raw, frac_over1 = [], []

    for s in range(N_NOISE_SEEDS):
        model = make_noise(eps)
        soft = model.corrupt_all(crisp, np.random.default_rng(1_000 + s))

        # Symbolic: threshold then crisp walk (the brittle baseline).
        sym_v = [sym.run(threshold_trace(t)) for t in soft]
        acc_sym.append(verdict_accuracy(sym_v, labels))

        # DeepDFA soft, both readouts.
        raw = np.asarray(dd.batch_acceptance_probability(soft, normalize=False))
        norm = np.asarray(dd.batch_acceptance_probability(soft, normalize=True))
        # a real monitor's confidence saturates at 1; the invalidity of the
        # raw score is captured separately by max_raw / frac_over1.
        raw_conf = np.clip(raw, 0.0, 1.0)

        acc_raw.append(verdict_accuracy(_verdicts(raw), labels))
        acc_norm.append(verdict_accuracy(_verdicts(norm), labels))
        ece_raw.append(expected_calibration_error(raw_conf, labels, N_BINS))
        ece_norm.append(expected_calibration_error(norm, labels, N_BINS))
        brier_raw.append(brier_score(raw_conf, labels))
        brier_norm.append(brier_score(norm, labels))
        auc_raw.append(roc_auc(raw_conf, labels))
        auc_norm.append(roc_auc(norm, labels))
        max_raw.append(float(raw.max()))
        frac_over1.append(float(np.mean(raw > 1.0 + 1e-9)))

    return {
        "formula": formula.name,
        "read_once": formula.read_once,
        "trace_length": trace_length,
        "eps": eps,
        "sym_acc": np.mean(acc_sym),
        "raw_acc": np.mean(acc_raw),
        "norm_acc": np.mean(acc_norm),
        "raw_ece": np.mean(ece_raw),
        "norm_ece": np.mean(ece_norm),
        "raw_brier": np.mean(brier_raw),
        "norm_brier": np.mean(brier_norm),
        "raw_auc": np.nanmean(auc_raw),
        "norm_auc": np.nanmean(auc_norm),
        "raw_max_score": np.mean(max_raw),
        "raw_frac_over1": np.mean(frac_over1),
    }


# ---------------------------------------------------------------------------
# Run the sweep (resumable: skip (formula, noise, eps) rows already on disk)
# ---------------------------------------------------------------------------

# Clean traces + oracle labels are fixed per formula across ε and noise.
lengths = {f.name: pick_trace_length(f) for f in CALIBRATION_SUITE}

# Resume only if the on-disk trace lengths match the current config; a config
# change (balance target / MIN_LEN / seed) makes the old rows a different
# workload, so drop them rather than silently mixing.
done = None
done_keys: set = set()
if CSV_PATH.exists():
    prev = pd.read_csv(CSV_PATH)
    stale = any(
        f.name in set(prev["formula"])
        and prev.loc[prev["formula"] == f.name, "trace_length"].iloc[0]
        != lengths[f.name][0]
        for f in CALIBRATION_SUITE
    )
    if stale:
        print("Config changed (trace lengths differ) — recomputing from scratch.")
    else:
        done = prev
        done_keys = set(zip(prev["formula"], prev["noise"], prev["eps"].round(3)))
clean = {}
labels_by_formula = {}
for f in CALIBRATION_SUITE:
    L, _ = lengths[f.name]
    rng = np.random.default_rng(TRACE_SEED)
    crisp = random_traces(f.atoms, L, N_TRACES, rng)
    clean[f.name] = crisp
    labels_by_formula[f.name] = true_verdicts(f.formula, crisp)

rows = []
total = len(CALIBRATION_SUITE) * len(NOISE_MODELS) * len(EPS_SWEEP)
with tqdm(total=total, desc="exp_uncertainty") as pbar:
    for f in CALIBRATION_SUITE:
        L, rate = lengths[f.name]
        for noise_name, make_noise in NOISE_MODELS.items():
            for eps in EPS_SWEEP:
                if (f.name, noise_name, round(eps, 3)) in done_keys:
                    pbar.update()
                    continue
                row = evaluate(
                    f, L, make_noise, eps, clean[f.name], labels_by_formula[f.name]
                )
                row["noise"] = noise_name
                row["pos_rate"] = rate
                rows.append(row)
                pbar.set_postfix(formula=f.name, noise=noise_name, eps=eps)
                pbar.update()

df_new = pd.DataFrame(rows)
df = pd.concat([done, df_new], ignore_index=True) if done is not None else df_new
df = df.sort_values(["formula", "noise", "eps"]).reset_index(drop=True)
df.to_csv(CSV_PATH, index=False)
print(f"Saved: {CSV_PATH}")

# ---------------------------------------------------------------------------
# Plot (decoupled: both the accuracy figure and the calibration figure are
# drawn by experiments/plots.py from the CSV, so they can be re-generated
# without re-running the sweep).
# ---------------------------------------------------------------------------

from experiments.plots import plot_uncertainty  # noqa: E402

plot_uncertainty(CSV_PATH)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print("\nTrace lengths (balanced to ~0.5 positive rate):")
for f in CALIBRATION_SUITE:
    L, rate = lengths[f.name]
    print(f"  {f.name:10s} L={L:2d}  pos_rate={rate:.2f}  "
          f"read_once={f.read_once}")

print("\nBeta-noise summary (accuracy + ECE):")
cols = ["formula", "eps", "sym_acc", "raw_acc", "norm_acc",
        "raw_ece", "norm_ece", "raw_max_score", "raw_frac_over1"]
print(df[df["noise"] == "beta"][cols].to_string(index=False))
