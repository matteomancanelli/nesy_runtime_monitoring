"""Plotting for all experiments, decoupled from the runs.

Every experiment script (exp1/2/3/5, exp_uncertainty) writes a CSV and nothing
else about the figures; this module turns those CSVs into PNGs. That split means
you can re-style a plot without re-running the (slow) sweep: edit here and call
the relevant ``plot_*`` function, or run this file to regenerate every figure
from whatever CSVs are currently in ``results/``.

Two design points worth knowing:

* **Log y-axis on the timing panels.** Per-cell costs span >2 orders of
  magnitude (Symbolic ~0.1 µs vs DeepDFA ~15 µs), so a linear axis flattens the
  cheap paradigms onto the x-axis. All timing panels use a log y-scale.

* **CPU vs GPU overlays.** Each timing row is stamped with ``device`` and
  ``gpu_name`` (Colab was run both on CPU and on a Tesla T4). Every ``plot_*``
  function accepts one path OR a list of paths: pass several CSVs (e.g. a CPU
  run and a GPU run) and the curves are split by hardware config. The dedicated
  ``plot_device_comparison`` faceting draws, per monitor, the CPU curve against
  the GPU curve for a fixed experiment.

Run:
    python experiments/plots.py                    # regenerate everything
    python experiments/plots.py exp3               # just one experiment
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NOTE: the heavy stack (ltlf2dfa/MONA, torch) is imported *lazily* inside the
# two functions that analytically recompute something from the formulas (the
# exp2 memory-wall panel and the exp_uncertainty reliability diagram). Every
# other plot is driven entirely by the CSV, so re-styling a timing figure needs
# only matplotlib/pandas — no torch, no MONA. This is what makes "act on the
# plot only" cheap.

RESULTS_DIR = ROOT / "results"
DENSE_MAX_LEAVES = 16
FLOAT_BYTES = 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_paths(csv_paths: str | Path | list) -> list[Path]:
    """Normalise a single path or a list of paths to a list of Paths."""
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    return [Path(p) for p in csv_paths]


def config_label(device: str, gpu_name: str | float) -> str:
    """Human-readable hardware label used to split/annotate curves.

    ``cpu`` -> "CPU"; ``cuda`` -> "GPU (Tesla T4)" (or just "GPU" if the name is
    missing). This is the grouping key for CPU-vs-GPU comparisons.
    """
    if str(device) == "cuda":
        missing = gpu_name is None or isinstance(gpu_name, float)
        name = "" if missing else str(gpu_name)
        return f"GPU ({name})" if name else "GPU"
    return "CPU"


def load_timing(csv_paths: str | Path | list) -> pd.DataFrame:
    """Load and concatenate one or more timing CSVs, adding derived columns.

    Adds:
      * ``config``          — hardware label from (device, gpu_name).
      * ``mean_s_per_trace``/``std_s_per_trace`` — per-cell * trace_length.
      * ``us_per_cell``/``us_err`` — microseconds, the plotting unit.
    Missing ``device``/``gpu_name`` columns (very old CSVs) default to CPU.
    """
    frames = []
    for path in _as_paths(csv_paths):
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no timing CSVs found among {csv_paths}")
    df = pd.concat(frames, ignore_index=True)
    if "device" not in df:
        df["device"] = "cpu"
    if "gpu_name" not in df:
        df["gpu_name"] = ""
    df["config"] = [config_label(d, g) for d, g in zip(df["device"], df["gpu_name"])]
    df["mean_s_per_trace"] = df["mean_s_per_cell"] * df["trace_length"]
    df["std_s_per_trace"] = df["std_s_per_cell"] * df["trace_length"]
    df["us_per_cell"] = df["mean_s_per_cell"] * 1e6
    df["us_err"] = df["std_s_per_cell"] * 1e6
    return df


def _series_label(monitor: str, config: str, multi_config: bool) -> str:
    """Legend label: include the hardware only when >1 config is on the plot."""
    return f"{monitor} — {config}" if multi_config else monitor


# One dashing per hardware config so CPU vs GPU is distinguishable when both are
# overlaid on the same axes (colour still encodes the monitor).
_CONFIG_DASH = ["-", "--", ":", "-."]


def _config_styles(configs: list[str]) -> dict[str, str]:
    n = len(_CONFIG_DASH)
    return {c: _CONFIG_DASH[i % n] for i, c in enumerate(sorted(configs))}


def _memory_curves() -> tuple[list[int], list[float], list[float]]:
    """Analytic transition-representation memory (GB) for the IJCNN family.

    Dense tensor is |Q|^2 * 2^|AP| float32; factored cube masks are the
    require-true/require-false integer masks. Computed from the (cheap) DFA
    compilation, which never materialises 2^n — same as the experiment.

    Needs the heavy stack (ltlf2dfa/MONA + torch); imported lazily so the timing
    plots stay dependency-light. Raises ImportError if unavailable.
    """
    from src.benchmarks.formulas import IJCNN_SUITE
    from src.formula.compiler import compile_ltlf
    from src.monitors.deep_dfa import DeepDFATensor

    mem_n, dense_gb, factored_gb = [], [], []
    for formula in IJCNN_SUITE:
        dfa = compile_ltlf(formula.formula)
        n_q = len(dfa.states)
        dt = DeepDFATensor(dfa, mode="factored")
        dense_bytes = n_q * n_q * (2 ** formula.n_leaves) * FLOAT_BYTES
        factored_bytes = (dt._cube_rt.numel() + dt._cube_rf.numel()) * FLOAT_BYTES
        mem_n.append(formula.n_leaves)
        dense_gb.append(dense_bytes / 1e9)
        factored_gb.append(factored_bytes / 1e9)
    return mem_n, dense_gb, factored_gb


# ---------------------------------------------------------------------------
# Exp 1 — per-cell cost vs trace length
# ---------------------------------------------------------------------------


def plot_exp1(csv_paths=None, out: Path | None = None) -> Path:
    csv_paths = csv_paths or RESULTS_DIR / "exp1_single_trace.csv"
    df = load_timing(csv_paths)
    configs = sorted(df["config"].unique())
    multi = len(configs) > 1
    dash = _config_styles(configs)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (monitor, cfg), g in df.groupby(["monitor_name", "config"]):
        g = g.sort_values("trace_length")
        ax.errorbar(
            g["trace_length"] / 1_000, g["us_per_cell"], yerr=g["us_err"],
            marker="o", capsize=3, ls=dash[cfg],
            label=_series_label(monitor, cfg, multi),
        )

    ax.set_xlabel("Monitored cells (×10³)")
    ax.set_ylabel("Avg time per cell (µs)")
    ax.set_yscale("log")            # (2) fast paradigms would collapse on linear
    formula = df["formula_name"].iloc[0]
    ax.set_title(f"Exp 1 — impact of trace length ({formula})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    out = out or RESULTS_DIR / "exp1_single_trace.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Exp 2 — per-cell cost vs formula complexity (+ analytic memory wall)
# ---------------------------------------------------------------------------


def plot_exp2(csv_paths=None, out: Path | None = None) -> Path:
    csv_paths = csv_paths or RESULTS_DIR / "exp2_formula_complexity.csv"
    df = load_timing(csv_paths)
    all_n = sorted(df["n_leaves"].unique())
    configs = sorted(df["config"].unique())
    multi = len(configs) > 1
    dash = _config_styles(configs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.4))

    for (monitor, cfg), g in df.groupby(["monitor_name", "config"]):
        g = g.sort_values("n_leaves")
        x = g["n_leaves"]
        lbl = _series_label(monitor, cfg, multi)
        ax1.errorbar(x, g["us_per_cell"], yerr=g["us_err"], marker="o",
                     capsize=3, ls=dash[cfg], label=lbl)
        ax2.errorbar(x, g["us_per_cell"] / x, yerr=g["us_err"] / x, marker="o",
                     capsize=3, ls=dash[cfg], label=lbl)

    for ax, ylabel, title in [
        (ax1, "Avg time per cell (µs)", "Time per cell vs formula size"),
        (ax2, "Avg time per cell per leaf (µs)", "Time per cell, per leaf"),
    ]:
        ax.set_xlabel("Leaves (atoms)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")        # (2) symbolic (~0.1µs) vs DeepDFA (~µs..ms)
        ax.set_xticks(all_n)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)

    # Memory panel is analytic (recomputed from the formulas, not the CSV) so it
    # needs the heavy stack; degrade to a note if it is unavailable.
    try:
        mem_n, dense_gb, factored_gb = _memory_curves()
        ax3.plot(mem_n, dense_gb, marker="o", color="tab:red",
                 label="DeepDFA dense  $|Q|^2\\,2^{|AP|}$")
        ax3.plot(mem_n, factored_gb, marker="o", color="tab:green",
                 label="DeepDFA factored (cube masks)")
        ax3.axhline(4.0, color="gray", ls=":", label="4 GB VRAM (laptop)")
        ax3.axvline(DENSE_MAX_LEAVES, color="tab:red", ls="--", alpha=0.4)
        ax3.set_yscale("log")
        ax3.set_ylabel("Transition representation (GB)")
        ax3.set_title("Alphabet-blowup: dense $2^{|AP|}$ memory wall")
        ax3.set_xticks(mem_n)
        ax3.legend(fontsize=8)
    except ImportError:
        ax3.text(0.5, 0.5, "memory panel needs ltlf2dfa + torch\n(recomputed "
                 "analytically, not in the CSV)", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=9, color="0.4")
        ax3.set_title("Alphabet-blowup memory wall (unavailable)")
    ax3.set_xlabel("Leaves (atoms)")
    ax3.grid(True, which="both", ls="--", alpha=0.4)

    fig.suptitle("Exp 2 — formula complexity", y=1.02)
    out = out or RESULTS_DIR / "exp2_formula_complexity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Exp 3 — throughput vs batch size
# ---------------------------------------------------------------------------


def plot_exp3(csv_paths=None, out: Path | None = None) -> Path:
    csv_paths = csv_paths or RESULTS_DIR / "exp3_batch_size.csv"
    df = load_timing(csv_paths)
    batch_sizes = sorted(df["n_traces"].unique())
    configs = sorted(df["config"].unique())
    multi = len(configs) > 1
    dash = _config_styles(configs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    for (monitor, cfg), g in df.groupby(["monitor_name", "config"]):
        g = g.sort_values("n_traces")
        x = g["n_traces"]
        lbl = _series_label(monitor, cfg, multi)
        ax1.errorbar(x, g["mean_s_per_trace"] * 1e3, yerr=g["std_s_per_trace"] * 1e3,
                     marker="o", capsize=3, ls=dash[cfg], label=lbl)
        baseline = g["mean_s_per_trace"].iloc[0]
        ax2.plot(x, baseline / g["mean_s_per_trace"], marker="o", ls=dash[cfg],
                 label=lbl)

    x_ref = np.array(batch_sizes)
    ax2.plot(x_ref, x_ref / x_ref[0], ls="--", color="gray", label="ideal linear")

    for ax, ylabel, title in [
        (ax1, "Time per trace (ms)", "LEAD: absolute time per trace"),
        (ax2, "Speedup vs own batch=1", "Speedup (per-monitor baseline)"),
    ]:
        ax.set_xlabel("Batch size (number of traces)")
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(batch_sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_yscale("log")           # (2) time-per-trace spans orders of magnitude
    ax2.set_yscale("log", base=2)
    ax2.text(
        0.5, -0.32,
        "Caveat: each curve is normalised to its OWN batch=1 time, so curves are\n"
        "NOT comparable across monitors. Read absolute times from the left panel.",
        transform=ax2.transAxes, ha="center", va="top", fontsize=7.5, color="0.35",
    )

    fig.suptitle("Exp 3 — batch size", y=1.02)
    out = out or RESULTS_DIR / "exp3_batch_size.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Exp 5 — within-step cost vs nested-X depth
# ---------------------------------------------------------------------------


def plot_exp5(csv_paths=None, out: Path | None = None) -> Path:
    csv_paths = csv_paths or RESULTS_DIR / "exp5_depth_microbench.csv"
    df = load_timing(csv_paths)
    df["depth"] = df["formula_name"].str.extract(r"_xdepth(\d+)$").astype(int)
    depths = sorted(df["depth"].unique())
    configs = sorted(df["config"].unique())
    multi = len(configs) > 1
    dash = _config_styles(configs)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (monitor, cfg), g in df.groupby(["monitor_name", "config"]):
        g = g.sort_values("depth")
        ax.errorbar(
            g["depth"], g["us_per_cell"], yerr=g["us_err"],
            marker="o", capsize=3, ls=dash[cfg],
            label=_series_label(monitor, cfg, multi),
        )

    ax.set_xlabel("Nested-X depth (parse-tree levels over fixed ijcnn_n8 breadth)")
    ax.set_ylabel("Time per cell (µs)")
    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_title("Exp 5 — within-step cost vs parse-tree depth")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    out = out or RESULTS_DIR / "exp5_depth_microbench.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# CPU vs GPU comparison — one subplot per monitor, for a fixed experiment
# ---------------------------------------------------------------------------

# x-axis column + human label for each timing experiment.
_EXP_XAXIS = {
    "exp1": ("trace_length", "Trace length (cells)"),
    "exp2": ("n_leaves", "Leaves (atoms)"),
    "exp3": ("n_traces", "Batch size (traces)"),
    "exp5": ("depth", "Nested-X depth"),
}


def plot_device_comparison(
    csv_paths, experiment: str, out: Path | None = None, monitors=None,
) -> Path | None:
    """Facet CPU vs GPU per monitor for one timing experiment.

    ``csv_paths`` must together contain >1 hardware config (e.g. a CPU CSV and a
    GPU CSV, or one accumulated CSV with both). Each subplot is one monitor with
    its CPU and GPU curves overlaid, answering "what does the GPU buy this
    paradigm?" directly. Returns None (and warns) if only one config is present.
    """
    df = load_timing(csv_paths)
    if experiment == "exp5":
        df["depth"] = df["formula_name"].str.extract(r"_xdepth(\d+)$").astype(int)
    xcol, xlabel = _EXP_XAXIS[experiment]

    configs = sorted(df["config"].unique())
    if len(configs) < 2:
        print(f"[{experiment}] only one config present ({configs}); "
              "device comparison needs both a CPU and a GPU CSV — skipping.")
        return None

    names = monitors or sorted(df["monitor_name"].unique())
    ncol = min(3, len(names))
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow),
                             squeeze=False)
    for i, monitor in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        sub = df[df["monitor_name"] == monitor]
        for cfg, g in sub.groupby("config"):
            g = g.sort_values(xcol)
            ax.errorbar(g[xcol], g["us_per_cell"], yerr=g["us_err"],
                        marker="o", capsize=3, label=cfg)
        ax.set_title(monitor, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time per cell (µs)")
        ax.set_yscale("log")
        if experiment in ("exp3",):
            ax.set_xscale("log", base=2)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)
    for j in range(len(names), nrow * ncol):      # hide unused axes
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"{experiment} — CPU vs GPU per monitor", y=1.01)
    out = out or RESULTS_DIR / f"{experiment}_device_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Capability Exp A — accuracy + calibration vs perceptual noise
# ---------------------------------------------------------------------------


def plot_uncertainty(csv_paths=None, out_dir: Path | None = None,
                     rep_eps: float = 0.4) -> list[Path]:
    """Accuracy + calibration figures for Capability Exp A, from the CSV.

    Everything except the reliability-diagram panel is driven purely by the CSV
    columns (``read_once`` tells us which formula is non-read-once, so we never
    import the formula registry). The reliability panel needs the raw per-trace
    scores, which are not stored in the CSV, so it is recomputed from the monitor
    — lazily imported and skipped with a note if torch/MONA are unavailable.
    """
    csv_paths = csv_paths or RESULTS_DIR / "exp_uncertainty.csv"
    frames = [pd.read_csv(p) for p in _as_paths(csv_paths) if Path(p).exists()]
    if not frames:
        raise FileNotFoundError(f"no uncertainty CSV found among {csv_paths}")
    df = pd.concat(frames, ignore_index=True)
    out_dir = out_dir or RESULTS_DIR
    N_BINS = 10

    # Stable formula ordering from the CSV; keep the read_once flag alongside.
    fmeta = (df[["formula", "read_once"]].drop_duplicates()
             .set_index("formula")["read_once"].to_dict())
    formula_names = list(fmeta)
    noise_names = sorted(df["noise"].unique())

    # --- Figure 1: verdict accuracy vs eps (rows = noise, cols = formula) ---
    fig, axes = plt.subplots(
        len(noise_names), len(formula_names),
        figsize=(4.6 * len(formula_names), 3.6 * len(noise_names)), squeeze=False,
    )
    for r, noise_name in enumerate(noise_names):
        for c, fname in enumerate(formula_names):
            ax = axes[r][c]
            g = df[(df["formula"] == fname) & (df["noise"] == noise_name)]
            g = g.sort_values("eps")
            if g.empty:
                ax.axis("off")
                continue
            ax.plot(g["eps"], g["sym_acc"], marker="s", label="Symbolic (threshold)")
            ax.plot(g["eps"], g["raw_acc"], marker="o", label="DeepDFA soft (raw)")
            ax.plot(g["eps"], g["norm_acc"], marker="^", ls="--",
                    label="DeepDFA soft (norm)")
            rate = float(g["pos_rate"].iloc[0])
            ax.axhline(max(rate, 1.0 - rate), color="gray", ls=":", alpha=0.6,
                       label="majority-class baseline")
            ax.set_ylim(0.45, 1.02)
            ro = "read-once" if fmeta[fname] else "NON-read-once"
            ax.set_title(f"{fname}  ({ro})")
            ax.set_xlabel("noise level ε")
            if c == 0:
                ax.set_ylabel(f"{noise_name} noise\nverdict accuracy")
            ax.grid(True, ls="--", alpha=0.4)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Capability Exp A — verdict accuracy vs perceptual noise", y=1.02)
    fig.tight_layout()
    acc_path = out_dir / "exp_uncertainty_accuracy.png"
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {acc_path}")

    # The non-read-once formula the calibration figure focuses on.
    non_read_once = [n for n, ro in fmeta.items() if not ro]
    majority = non_read_once[0] if non_read_once else formula_names[0]

    fig, (axr, axe, axd) = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) Reliability diagram — recomputed from the monitor (raw scores are not
    # in the CSV). Needs torch/MONA; degrade to a note if unavailable.
    try:
        reliability = _reliability_bins(df, majority, rep_eps, N_BINS)
        axr.plot([0, 1], [0, 1], color="gray", ls=":", label="perfectly calibrated")
        for xs, ys, lab, ece, style in reliability:
            axr.plot(xs, ys, style, label=f"{lab} (ECE={ece:.3f})")
        axr.legend(fontsize=8)
    except ImportError:
        axr.text(0.5, 0.5, "reliability panel needs the monitor\n(raw scores not "
                 "stored in the CSV)", ha="center", va="center",
                 transform=axr.transAxes, fontsize=9, color="0.4")
    axr.set_xlabel("mean predicted confidence")
    axr.set_ylabel("empirical accuracy")
    axr.set_title(f"Reliability — {majority} (non-read-once)\nBeta ε={rep_eps}")
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.grid(True, ls="--", alpha=0.4)

    # (b) ECE vs eps (Beta) — straight from the CSV.
    for fname in formula_names:
        g = df[(df["formula"] == fname) & (df["noise"] == "beta")].sort_values("eps")
        if g.empty:
            continue
        tag = "read-once" if fmeta[fname] else "NON-read-once"
        axe.plot(g["eps"], g["raw_ece"], marker="o", label=f"{fname} raw ({tag})")
        if not fmeta[fname]:
            axe.plot(g["eps"], g["norm_ece"], marker="^", ls="--",
                     label=f"{fname} norm")
    axe.set_xlabel("noise level ε")
    axe.set_ylabel("Expected Calibration Error")
    axe.set_title("ECE vs noise (Beta)\nonly the soft paradigm emits a confidence")
    axe.legend(fontsize=8)
    axe.grid(True, ls="--", alpha=0.4)

    # (c) Non-read-once defect — straight from the CSV.
    g = df[(df["formula"] == majority) & (df["noise"] == "beta")].sort_values("eps")
    axd.plot(g["eps"], g["raw_max_score"], marker="o", color="tab:red",
             label="raw max score")
    axd.axhline(1.0, color="gray", ls=":", label="valid-probability ceiling")
    axd2 = axd.twinx()
    axd2.plot(g["eps"], g["raw_frac_over1"], marker="s", color="tab:purple",
              label="fraction of scores > 1")
    axd2.set_ylabel("fraction of traces with raw score > 1", color="tab:purple")
    axd.set_xlabel("noise level ε")
    axd.set_ylabel("raw acceptance score (max)", color="tab:red")
    axd.set_title(f"Non-read-once defect — {majority}\nraw rows not stochastic")
    axd.legend(fontsize=8, loc="upper left")
    axd2.legend(fontsize=8, loc="lower right")
    axd.grid(True, ls="--", alpha=0.4)

    fig.suptitle("Capability Exp A — calibration (symbolic cannot emit a confidence)",
                 y=1.03)
    fig.tight_layout()
    cal_path = out_dir / "exp_uncertainty_calibration.png"
    fig.savefig(cal_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cal_path}")
    return [acc_path, cal_path]


def _reliability_bins(df: pd.DataFrame, formula_name: str, rep_eps: float, n_bins: int):
    """Recompute reliability-curve points for the majority formula at rep_eps.

    Raw scores are not stored in the CSV, so this re-runs the soft monitor on a
    fresh corrupted trace set. Lazily imports the heavy stack; raises ImportError
    if unavailable so the caller can degrade gracefully.
    """
    from src.benchmarks.calibration import (
        expected_calibration_error,
        reliability_curve,
    )
    from src.benchmarks.formulas import CALIBRATION_SUITE
    from src.benchmarks.noise import BetaNoise, true_verdicts
    from src.benchmarks.runner import random_traces
    from src.monitors.deep_dfa import DeepDFAMonitorFactored

    spec = next(f for f in CALIBRATION_SUITE if f.name == formula_name)
    L = int(df.loc[df["formula"] == formula_name, "trace_length"].iloc[0])
    crisp = random_traces(spec.atoms, L, 3_000, np.random.default_rng(42))
    labels = true_verdicts(spec.formula, crisp)
    soft = BetaNoise(rep_eps, concentration=10.0).corrupt_all(
        crisp, np.random.default_rng(1_000))
    dd = DeepDFAMonitorFactored.compile(spec.formula)
    raw = np.clip(dd.batch_acceptance_probability(soft, normalize=False), 0.0, 1.0)
    norm = np.asarray(dd.batch_acceptance_probability(soft, normalize=True))

    out = []
    for scores, lab, style in [(raw, "raw", "o-"), (norm, "normalized", "^--")]:
        bins = reliability_curve(scores, labels, n_bins)
        xs = [b.mean_confidence for b in bins if b.count]
        ys = [b.accuracy for b in bins if b.count]
        ece = expected_calibration_error(scores, labels, n_bins)
        out.append((xs, ys, lab, ece, style))
    return out


# ---------------------------------------------------------------------------
# CLI: regenerate figures from the CSVs currently in results/
# ---------------------------------------------------------------------------

_PLOTTERS = {
    "exp1": plot_exp1,
    "exp2": plot_exp2,
    "exp3": plot_exp3,
    "exp5": plot_exp5,
    "uncertainty": plot_uncertainty,
}


def main(argv: list[str]) -> None:
    which = argv or list(_PLOTTERS)
    for name in which:
        fn = _PLOTTERS.get(name)
        if fn is None:
            print(f"unknown experiment '{name}'; choices: {list(_PLOTTERS)}")
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[{name}] skipped: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
