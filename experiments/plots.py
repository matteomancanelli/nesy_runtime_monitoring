"""Plotting for all experiments, decoupled from the runs.

Every experiment script (exp1/2/3/5, exp_uncertainty) writes a CSV and nothing
else about the figures; this module turns those CSVs into PNGs. That split means
you can re-style a plot without re-running the (slow) sweep: edit here and call
the relevant ``plot_*`` function, or run this file to regenerate every figure
from whatever CSVs are currently in ``results/``.

Conventions (so the whole paper reads as one system):

* **One file per plot.** No side-by-side composites — each logical plot is its
  own PNG so it can be dropped into LaTeX independently. Functions that used to
  draw a multi-panel figure now return a *list* of the files they wrote.

* **Consistent colours + labels.** Every paradigm has one fixed colour and one
  display name across every figure (``MONITOR_STYLE``), from the Okabe–Ito
  colourblind-safe palette (validated: worst adjacent CVD ΔE ≈ 37). Hardware
  config (CPU vs GPU) is encoded by *line style*, never by colour, so colour
  always means "which monitor".

* **Log y-axis on the timing panels.** Per-cell costs span >2 orders of
  magnitude (Symbolic ~0.1 µs vs DeepDFA ~15 µs); a linear axis flattens the
  cheap paradigms onto the x-axis.

* **CPU vs GPU.** Each timing row is stamped with ``device``/``gpu_name`` (Colab
  is run both on CPU and on a Tesla T4). Every ``plot_*`` accepts one path OR a
  list of paths: pass several CSVs and curves split by hardware. ``plot_device_
  comparison`` draws, per monitor, CPU vs GPU; ``plot_device_speedup`` draws the
  CPU/GPU speed-up ratio per monitor.

The heavy stack (ltlf2dfa/MONA, torch) is imported lazily — only the analytic
memory-wall and the reliability diagram need it — so re-styling a timing figure
needs just matplotlib/pandas.

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

RESULTS_DIR = ROOT / "results"
DENSE_MAX_LEAVES = 16
FLOAT_BYTES = 4

# ---------------------------------------------------------------------------
# Canonical monitor identity: one display name + one colour, used everywhere.
# Okabe–Ito palette (colourblind-safe). DeepDFAMonitor (the dense default in
# exp1/3/5) and DeepDFAMonitorDense (exp2) are the same paradigm variant, so
# they share a name and colour. Hardware is a line style, never a colour.
# ---------------------------------------------------------------------------

MONITOR_STYLE: dict[str, tuple[str, str]] = {
    "SymbolicDFAMonitor":          ("Symbolic DFA",           "#0072B2"),  # blue
    "RuleRunnerMonitor":           ("RuleRunner (flat)",      "#E69F00"),  # orange
    "StructuredRuleRunnerMonitor": ("RuleRunner (structured)", "#CC79A7"),  # purple
    "DeepDFAMonitor":              ("DeepDFA (dense)",        "#D55E00"),  # vermillion
    "DeepDFAMonitorDense":         ("DeepDFA (dense)",        "#D55E00"),
    "DeepDFAMonitorFactored":      ("DeepDFA (factored)",     "#009E73"),  # green
    "DeepDFAMonitorScan":          ("DeepDFA (scan)",         "#56B4E9"),  # sky blue
}
# Draw order for legends (canonical, not alphabetical).
_MONITOR_ORDER = [
    "SymbolicDFAMonitor", "RuleRunnerMonitor", "StructuredRuleRunnerMonitor",
    "DeepDFAMonitor", "DeepDFAMonitorDense", "DeepDFAMonitorFactored",
    "DeepDFAMonitorScan",
]
_FALLBACK_COLORS = ["#56B4E9", "#000000", "#999999", "#F0E442"]


def style_for(monitor: str, seen: dict[str, str] | None = None) -> tuple[str, str]:
    """(display label, colour) for a monitor class name; stable fallback if new."""
    if monitor in MONITOR_STYLE:
        return MONITOR_STYLE[monitor]
    seen = seen if seen is not None else {}
    if monitor not in seen:
        seen[monitor] = _FALLBACK_COLORS[len(seen) % len(_FALLBACK_COLORS)]
    return monitor, seen[monitor]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_paths(csv_paths: str | Path | list) -> list[Path]:
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    return [Path(p) for p in csv_paths]


def config_label(device: str, gpu_name: str | float) -> str:
    """Hardware label used to split/annotate curves: "CPU" or "GPU (Tesla T4)"."""
    if str(device) == "cuda":
        missing = gpu_name is None or isinstance(gpu_name, float)
        name = "" if missing else str(gpu_name)
        return f"GPU ({name})" if name else "GPU"
    return "CPU"


def load_timing(csv_paths: str | Path | list) -> pd.DataFrame:
    """Load and concatenate one or more timing CSVs, adding derived columns."""
    frames = [pd.read_csv(p) for p in _as_paths(csv_paths) if Path(p).exists()]
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


# Line style per hardware config: CPU solid, GPU dashed, extras after.
_CONFIG_DASH = ["-", "--", ":", "-."]
_CONFIG_MARKER = ["o", "s", "^", "D"]


def _config_styles(configs: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(configs, key=lambda c: (c != "CPU", c))  # CPU first -> solid
    dash = {c: _CONFIG_DASH[i % len(_CONFIG_DASH)] for i, c in enumerate(ordered)}
    mark = {c: _CONFIG_MARKER[i % len(_CONFIG_MARKER)] for i, c in enumerate(ordered)}
    return dash, mark


def _ordered_monitors(df: pd.DataFrame) -> list[str]:
    present = set(df["monitor_name"])
    canon = [m for m in _MONITOR_ORDER if m in present]
    extra = sorted(m for m in present if m not in _MONITOR_ORDER)
    # de-dup while keeping order (DeepDFAMonitor/Dense map to same label)
    return list(dict.fromkeys(canon + extra))


def _new_ax(figsize=(7.0, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save(fig, ax, out: Path, legend=True) -> Path:
    if legend:
        ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def _draw_timing(ax, df, xcol, ynorm=None) -> None:
    """Draw per-monitor timing curves (µs/cell) vs xcol on a log-y axis.

    ``ynorm``: optional column to divide by (e.g. n_leaves for per-leaf cost).
    Colour = monitor; line style = hardware config; legend labels include the
    config only when more than one is present.
    """
    configs = sorted(df["config"].unique())
    multi = len(configs) > 1
    dash, mark = _config_styles(configs)
    seen: dict[str, str] = {}
    for monitor in _ordered_monitors(df):
        label, color = style_for(monitor, seen)
        for cfg in sorted(df.loc[df["monitor_name"] == monitor, "config"].unique()):
            g = df[(df["monitor_name"] == monitor) & (df["config"] == cfg)]
            g = g.sort_values(xcol)
            y = g["us_per_cell"]
            yerr = g["us_err"]
            if ynorm is not None:
                y = y / g[ynorm]
                yerr = yerr / g[ynorm]
            lbl = f"{label} — {cfg}" if multi else label
            ax.errorbar(g[xcol], y, yerr=yerr, marker=mark[cfg], ms=6, lw=2,
                        capsize=3, color=color, ls=dash[cfg], label=lbl)
    ax.set_yscale("log")


# ---------------------------------------------------------------------------
# Exp 1 — per-cell cost vs trace length  (1 file)
# ---------------------------------------------------------------------------


def plot_exp1(csv_paths=None, out_dir: Path | None = None) -> list[Path]:
    df = load_timing(csv_paths or RESULTS_DIR / "exp1_single_trace.csv")
    out_dir = out_dir or RESULTS_DIR
    df["kcells"] = df["trace_length"] / 1_000

    fig, ax = _new_ax()
    _draw_timing(ax, df, "kcells")
    ax.set_xlabel("Monitored cells (×10³)")
    ax.set_ylabel("Avg time per cell (µs)")
    formula = df["formula_name"].iloc[0]
    ax.set_title(f"Exp 1 — time per cell vs trace length ({formula})")
    return [_save(fig, ax, out_dir / "exp1_time_per_cell.png")]


# ---------------------------------------------------------------------------
# Exp 2 — per-cell cost vs formula complexity + memory wall  (3 files)
# ---------------------------------------------------------------------------


def _memory_curves() -> tuple[list[int], list[float], list[float]]:
    """Analytic transition-representation memory (GB). Lazily needs ltlf2dfa+torch."""
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


def plot_exp2(csv_paths=None, out_dir: Path | None = None) -> list[Path]:
    df = load_timing(csv_paths or RESULTS_DIR / "exp2_formula_complexity.csv")
    out_dir = out_dir or RESULTS_DIR
    all_n = sorted(df["n_leaves"].unique())
    outs = []

    # (1) time per cell
    fig, ax = _new_ax()
    _draw_timing(ax, df, "n_leaves")
    ax.set_xlabel("Leaves (atoms)")
    ax.set_ylabel("Avg time per cell (µs)")
    ax.set_xticks(all_n)
    ax.set_title("Exp 2 — time per cell vs formula size")
    outs.append(_save(fig, ax, out_dir / "exp2_time_per_cell.png"))

    # (2) time per cell, per leaf
    fig, ax = _new_ax()
    _draw_timing(ax, df, "n_leaves", ynorm="n_leaves")
    ax.set_xlabel("Leaves (atoms)")
    ax.set_ylabel("Avg time per cell per leaf (µs)")
    ax.set_xticks(all_n)
    ax.set_title("Exp 2 — time per cell, per leaf")
    outs.append(_save(fig, ax, out_dir / "exp2_time_per_cell_per_leaf.png"))

    # (3) analytic memory wall (dense/factored colours match the timing curves)
    fig, ax = _new_ax()
    try:
        mem_n, dense_gb, factored_gb = _memory_curves()
        _, dense_c = MONITOR_STYLE["DeepDFAMonitorDense"]
        _, fact_c = MONITOR_STYLE["DeepDFAMonitorFactored"]
        ax.plot(mem_n, dense_gb, marker="o", ms=6, lw=2, color=dense_c,
                label="DeepDFA (dense)  $|Q|^2\\,2^{|AP|}$")
        ax.plot(mem_n, factored_gb, marker="^", ms=6, lw=2, color=fact_c,
                label="DeepDFA (factored) cube masks")
        ax.axhline(4.0, color="gray", ls=":", label="4 GB VRAM")
        ax.axvline(DENSE_MAX_LEAVES, color=dense_c, ls="--", alpha=0.4)
        ax.set_yscale("log")
        ax.set_ylabel("Transition representation (GB)")
        ax.set_xticks(mem_n)
        ax.set_title("Exp 2 — alphabet-blowup: dense $2^{|AP|}$ memory wall")
        legend = True
    except ImportError:
        ax.text(0.5, 0.5, "memory panel needs ltlf2dfa + torch\n(recomputed "
                "analytically, not stored in the CSV)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
        ax.set_title("Exp 2 — memory wall (deps unavailable)")
        legend = False
    ax.set_xlabel("Leaves (atoms)")
    outs.append(_save(fig, ax, out_dir / "exp2_memory_wall.png", legend=legend))
    return outs


# ---------------------------------------------------------------------------
# Exp 3 — throughput vs batch size  (2 files)
# ---------------------------------------------------------------------------


def plot_exp3(csv_paths=None, out_dir: Path | None = None) -> list[Path]:
    df = load_timing(csv_paths or RESULTS_DIR / "exp3_batch_size.csv")
    out_dir = out_dir or RESULTS_DIR
    batch_sizes = sorted(df["n_traces"].unique())
    configs = sorted(df["config"].unique())
    multi = len(configs) > 1
    dash, mark = _config_styles(configs)
    outs = []

    # (1) LEAD: absolute time per trace (ms), log-y
    fig, ax = _new_ax(figsize=(7.5, 4.6))
    seen: dict[str, str] = {}
    for monitor in _ordered_monitors(df):
        label, color = style_for(monitor, seen)
        for cfg in sorted(df.loc[df["monitor_name"] == monitor, "config"].unique()):
            g = df[(df["monitor_name"] == monitor) & (df["config"] == cfg)]
            g = g.sort_values("n_traces")
            lbl = f"{label} — {cfg}" if multi else label
            ax.errorbar(g["n_traces"], g["mean_s_per_trace"] * 1e3,
                        yerr=g["std_s_per_trace"] * 1e3, marker=mark[cfg], ms=6,
                        lw=2, capsize=3, color=color, ls=dash[cfg], label=lbl)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Batch size (number of traces)")
    ax.set_ylabel("Time per trace (ms)")
    ax.set_title("Exp 3 — absolute time per trace")
    outs.append(_save(fig, ax, out_dir / "exp3_time_per_trace.png"))

    # (2) speed-up vs own batch=1 (demoted; per-monitor baseline)
    fig, ax = _new_ax(figsize=(7.5, 4.6))
    seen = {}
    for monitor in _ordered_monitors(df):
        label, color = style_for(monitor, seen)
        for cfg in sorted(df.loc[df["monitor_name"] == monitor, "config"].unique()):
            g = df[(df["monitor_name"] == monitor) & (df["config"] == cfg)]
            g = g.sort_values("n_traces")
            baseline = g["mean_s_per_trace"].iloc[0]
            lbl = f"{label} — {cfg}" if multi else label
            ax.plot(g["n_traces"], baseline / g["mean_s_per_trace"], marker=mark[cfg],
                    ms=6, lw=2, color=color, ls=dash[cfg], label=lbl)
    x_ref = np.array(batch_sizes)
    ax.plot(x_ref, x_ref / x_ref[0], ls="--", color="gray", label="ideal linear")
    ax.set_yscale("log", base=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Batch size (number of traces)")
    ax.set_ylabel("Speedup vs own batch=1")
    ax.set_title("Exp 3 — speedup (per-monitor baseline; read absolute times above)")
    outs.append(_save(fig, ax, out_dir / "exp3_speedup.png"))
    return outs


# ---------------------------------------------------------------------------
# Exp 5 — within-step cost vs nested-X depth  (1 file)
# ---------------------------------------------------------------------------


def plot_exp5(csv_paths=None, out_dir: Path | None = None) -> list[Path]:
    df = load_timing(csv_paths or RESULTS_DIR / "exp5_depth_microbench.csv")
    out_dir = out_dir or RESULTS_DIR
    df["depth"] = df["formula_name"].str.extract(r"_xdepth(\d+)$").astype(int)

    fig, ax = _new_ax()
    _draw_timing(ax, df, "depth")
    ax.set_xlabel("Nested-X depth (parse-tree levels over fixed ijcnn_n8 breadth)")
    ax.set_ylabel("Time per cell (µs)")
    ax.set_xticks(sorted(df["depth"].unique()))
    ax.set_title("Exp 5 — within-step cost vs parse-tree depth")
    return [_save(fig, ax, out_dir / "exp5_depth.png")]


# ---------------------------------------------------------------------------
# Exp 6 — per-cell cost vs automaton size |Q|  (1 file)
# ---------------------------------------------------------------------------


def plot_exp6(csv_paths=None, out_dir: Path | None = None) -> list[Path]:
    df = load_timing(csv_paths or RESULTS_DIR / "exp6_state_scaling.csv")
    out_dir = out_dir or RESULTS_DIR

    fig, ax = _new_ax(figsize=(7.5, 4.6))
    _draw_timing(ax, df, "n_leaves")  # n_leaves holds the measured |Q| (exp6)
    ax.set_xlabel("DFA states |Q|  (bounded-response deadline)")
    ax.set_ylabel("Avg time per cell (µs)")
    ax.set_title("Exp 6 — time per cell vs automaton size |Q|")
    return [_save(fig, ax, out_dir / "exp6_state_scaling.png")]


# ---------------------------------------------------------------------------
# CPU vs GPU comparisons  (one file per monitor + one speedup file)
# ---------------------------------------------------------------------------

_EXP_XAXIS = {
    "exp1": ("trace_length", "Trace length (cells)", False),
    "exp2": ("n_leaves", "Leaves (atoms)", False),
    "exp3": ("n_traces", "Batch size (traces)", True),
    "exp5": ("depth", "Nested-X depth", False),
    "exp6": ("n_leaves", "DFA states |Q|", False),
}


def _prep_device_df(csv_paths, experiment: str):
    df = load_timing(csv_paths)
    if experiment == "exp5":
        df["depth"] = df["formula_name"].str.extract(r"_xdepth(\d+)$").astype(int)
    xcol, xlabel, logx = _EXP_XAXIS[experiment]
    return df, xcol, xlabel, logx


def plot_device_comparison(csv_paths, experiment: str,
                           out_dir: Path | None = None, monitors=None) -> list[Path]:
    """One file per monitor: its CPU curve vs its GPU curve (same colour, CPU
    solid/circle, GPU dashed/square). Needs both a CPU and a GPU CSV."""
    df, xcol, xlabel, logx = _prep_device_df(csv_paths, experiment)
    out_dir = out_dir or RESULTS_DIR
    configs = sorted(df["config"].unique())
    if len(configs) < 2:
        print(f"[{experiment}] only one config present ({configs}); device "
              "comparison needs a CPU CSV and a GPU CSV — skipping.")
        return []
    dash, mark = _config_styles(configs)

    names = monitors or _ordered_monitors(df)
    outs = []
    seen: dict[str, str] = {}
    for monitor in names:
        label, color = style_for(monitor, seen)
        sub = df[df["monitor_name"] == monitor]
        if sub.empty:
            continue
        fig, ax = _new_ax(figsize=(6.5, 4.4))
        for cfg in sorted(sub["config"].unique()):
            g = sub[sub["config"] == cfg].sort_values(xcol)
            ax.errorbar(g[xcol], g["us_per_cell"], yerr=g["us_err"], marker=mark[cfg],
                        ms=6, lw=2, capsize=3, color=color, ls=dash[cfg], label=cfg)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(sub[xcol].unique()))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time per cell (µs)")
        ax.set_title(f"{experiment} — {label}: CPU vs GPU")
        safe = label.replace(" ", "_").replace("(", "").replace(")", "")
        outs.append(_save(fig, ax, out_dir / f"{experiment}_device_{safe}.png"))
    return outs


def plot_device_speedup(csv_paths, experiment: str,
                        out_dir: Path | None = None) -> list[Path]:
    """One file: GPU speed-up (CPU time / GPU time) per monitor vs the x-axis.

    Answers "how much does the GPU actually buy each paradigm?" directly — the
    interesting comparison that falls out of running the same sweep on both.
    Needs both a CPU and a GPU CSV.
    """
    df, xcol, xlabel, logx = _prep_device_df(csv_paths, experiment)
    out_dir = out_dir or RESULTS_DIR
    gpu_cfgs = [c for c in df["config"].unique() if c != "CPU"]
    if "CPU" not in set(df["config"]) or not gpu_cfgs:
        print(f"[{experiment}] need both a CPU CSV and a GPU CSV for the "
              "device-speedup plot — skipping.")
        return []
    gpu_cfg = sorted(gpu_cfgs)[0]

    fig, ax = _new_ax(figsize=(7.0, 4.6))
    seen: dict[str, str] = {}
    for monitor in _ordered_monitors(df):
        label, color = style_for(monitor, seen)
        cpu = (df[(df["monitor_name"] == monitor) & (df["config"] == "CPU")]
               .set_index(xcol)["mean_s_per_cell"])
        gpu = (df[(df["monitor_name"] == monitor) & (df["config"] == gpu_cfg)]
               .set_index(xcol)["mean_s_per_cell"])
        common = sorted(set(cpu.index) & set(gpu.index))
        if not common:
            continue
        ratio = [cpu[x] / gpu[x] for x in common]
        ax.plot(common, ratio, marker="o", ms=6, lw=2, color=color, label=label)
    ax.axhline(1.0, color="gray", ls=":", label="parity (GPU = CPU)")
    ax.set_yscale("log", base=2)
    if logx:
        ax.set_xscale("log", base=2)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Speedup  CPU time / {gpu_cfg} time")
    ax.set_title(f"{experiment} — GPU speedup per monitor")
    return [_save(fig, ax, out_dir / f"{experiment}_device_speedup.png")]


# ---------------------------------------------------------------------------
# Capability Exp A — accuracy + calibration vs perceptual noise
# ---------------------------------------------------------------------------

# Fixed colours for the three uncertainty series (reuse the paradigm palette:
# Symbolic = blue, DeepDFA-factored = green; the two soft read-outs are the same
# monitor, so raw = green and norm = vermillion to tell them apart).
_UNC_STYLE = {
    "sym":  ("Symbolic (threshold)",   "#0072B2", "s", "-"),
    "raw":  ("DeepDFA soft (raw)",     "#009E73", "o", "-"),
    "norm": ("DeepDFA soft (norm)",    "#D55E00", "^", "--"),
}
# Colours for per-formula lines (distinct from the paradigm palette).
_FORMULA_COLORS = ["#E69F00", "#56B4E9", "#CC79A7", "#000000"]


def plot_uncertainty(csv_paths=None, out_dir: Path | None = None,
                     rep_eps: float = 0.4) -> list[Path]:
    """Capability Exp A figures, one file per plot.

    Accuracy: one file per (noise model, formula). Calibration: reliability,
    ECE, and the two non-read-once defect quantities (split — no dual axis) as
    separate files. Everything but the reliability diagram is CSV-driven; the
    reliability panel recomputes raw scores from the monitor (lazy import).
    """
    csv_paths = csv_paths or RESULTS_DIR / "exp_uncertainty.csv"
    frames = [pd.read_csv(p) for p in _as_paths(csv_paths) if Path(p).exists()]
    if not frames:
        raise FileNotFoundError(f"no uncertainty CSV found among {csv_paths}")
    df = pd.concat(frames, ignore_index=True)
    out_dir = out_dir or RESULTS_DIR
    N_BINS = 10

    fmeta = (df[["formula", "read_once"]].drop_duplicates()
             .set_index("formula")["read_once"].to_dict())
    formula_names = list(fmeta)
    noise_names = sorted(df["noise"].unique())
    fcolor = {f: _FORMULA_COLORS[i % len(_FORMULA_COLORS)]
              for i, f in enumerate(formula_names)}
    outs = []

    # --- accuracy vs eps: one file per (noise, formula) ---
    for noise_name in noise_names:
        for fname in formula_names:
            g = df[(df["formula"] == fname) & (df["noise"] == noise_name)]
            g = g.sort_values("eps")
            if g.empty:
                continue
            fig, ax = _new_ax(figsize=(6.0, 4.2))
            for key, col in [("sym", "sym_acc"), ("raw", "raw_acc"),
                             ("norm", "norm_acc")]:
                lab, color, marker, ls = _UNC_STYLE[key]
                ax.plot(g["eps"], g[col], marker=marker, ms=6, lw=2, ls=ls,
                        color=color, label=lab)
            rate = float(g["pos_rate"].iloc[0])
            ax.axhline(max(rate, 1.0 - rate), color="gray", ls=":", alpha=0.7,
                       label="majority-class baseline")
            ax.set_ylim(0.45, 1.02)
            ro = "read-once" if fmeta[fname] else "NON-read-once"
            ax.set_xlabel("noise level ε")
            ax.set_ylabel("verdict accuracy")
            ax.set_title(f"Accuracy vs {noise_name} noise — {fname} ({ro})")
            fname_out = f"exp_uncertainty_accuracy_{noise_name}_{fname}.png"
            outs.append(_save(fig, ax, out_dir / fname_out))

    non_read_once = [n for n, ro in fmeta.items() if not ro]
    majority = non_read_once[0] if non_read_once else formula_names[0]

    # --- reliability diagram (recomputed; graceful if deps missing) ---
    fig, ax = _new_ax(figsize=(5.6, 5.2))
    try:
        reliability = _reliability_bins(df, majority, rep_eps, N_BINS)
        ax.plot([0, 1], [0, 1], color="gray", ls=":", label="perfectly calibrated")
        for xs, ys, key, ece in reliability:
            lab, color, marker, _ = _UNC_STYLE[key]
            ls = "-" if key == "raw" else "--"
            ax.plot(xs, ys, marker=marker, ms=6, lw=2, ls=ls, color=color,
                    label=f"{key} (ECE={ece:.3f})")
        legend = True
    except ImportError:
        ax.text(0.5, 0.5, "reliability panel needs the monitor\n(raw scores not "
                "stored in the CSV)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
        legend = False
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability — {majority} (non-read-once), Beta ε={rep_eps}")
    outs.append(_save(fig, ax, out_dir / "exp_uncertainty_reliability.png",
                      legend=legend))

    # --- ECE vs eps (Beta), per formula ---
    fig, ax = _new_ax(figsize=(6.5, 4.4))
    for fname in formula_names:
        g = df[(df["formula"] == fname) & (df["noise"] == "beta")].sort_values("eps")
        if g.empty:
            continue
        tag = "read-once" if fmeta[fname] else "NON-read-once"
        ax.plot(g["eps"], g["raw_ece"], marker="o", ms=6, lw=2, color=fcolor[fname],
                label=f"{fname} raw ({tag})")
        if not fmeta[fname]:
            ax.plot(g["eps"], g["norm_ece"], marker="^", ms=6, lw=2, ls="--",
                    color=fcolor[fname], label=f"{fname} norm")
    ax.set_xlabel("noise level ε")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE vs noise (Beta) — only the soft paradigm emits a confidence")
    outs.append(_save(fig, ax, out_dir / "exp_uncertainty_ece.png"))

    # --- non-read-once defect: split into two single-axis files (no twinx) ---
    g = df[(df["formula"] == majority) & (df["noise"] == "beta")].sort_values("eps")
    fig, ax = _new_ax(figsize=(6.0, 4.2))
    ax.plot(g["eps"], g["raw_max_score"], marker="o", ms=6, lw=2, color="#D55E00",
            label="raw max acceptance score")
    ax.axhline(1.0, color="gray", ls=":", label="valid-probability ceiling")
    ax.set_xlabel("noise level ε")
    ax.set_ylabel("raw acceptance score (max)")
    ax.set_title(f"Non-read-once defect — {majority}: raw score exceeds 1")
    outs.append(_save(fig, ax, out_dir / "exp_uncertainty_defect_maxscore.png"))

    fig, ax = _new_ax(figsize=(6.0, 4.2))
    ax.plot(g["eps"], g["raw_frac_over1"], marker="s", ms=6, lw=2, color="#CC79A7",
            label="fraction of traces with raw score > 1")
    ax.set_xlabel("noise level ε")
    ax.set_ylabel("fraction of traces with raw score > 1")
    ax.set_title(f"Non-read-once defect — {majority}: overshoot prevalence")
    outs.append(_save(fig, ax, out_dir / "exp_uncertainty_defect_fracover1.png"))
    return outs


def _reliability_bins(df: pd.DataFrame, formula_name: str, rep_eps: float, n_bins: int):
    """Recompute reliability-curve points; lazy heavy imports, raises ImportError."""
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
    for scores, key in [(raw, "raw"), (norm, "norm")]:
        bins = reliability_curve(scores, labels, n_bins)
        xs = [b.mean_confidence for b in bins if b.count]
        ys = [b.accuracy for b in bins if b.count]
        ece = expected_calibration_error(scores, labels, n_bins)
        out.append((xs, ys, key, ece))
    return out


# ---------------------------------------------------------------------------
# CLI: regenerate figures from the CSVs currently in results/
# ---------------------------------------------------------------------------

_PLOTTERS = {
    "exp1": plot_exp1,
    "exp2": plot_exp2,
    "exp3": plot_exp3,
    "exp5": plot_exp5,
    "exp6": plot_exp6,
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
