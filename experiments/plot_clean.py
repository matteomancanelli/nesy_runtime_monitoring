"""Clean re-plot of Exp 2 and Exp 3 straight from the committed CSVs.

For presentation/slides: keeps only axis labels + legend. No titles,
suptitles, captions, or annotation notes. Data is read as-is from
results/exp2_formula_complexity.csv and results/exp3_batch_size.csv
(no re-run, no recomputation of the analytic memory panel).

Run:
    python experiments/plot_clean.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.formulas import IJCNN_SUITE
from src.formula.compiler import compile_ltlf
from src.monitors.deep_dfa import DeepDFATensor

RESULTS_DIR = ROOT / "results"
DENSE_MAX_LEAVES = 16
FLOAT_BYTES = 4


# ---------------------------------------------------------------------------
# Exp 2 — formula complexity (two timing panels + memory-wall panel)
# ---------------------------------------------------------------------------

def plot_exp2() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp2_formula_complexity.csv")
    all_n = [f.n_leaves for f in IJCNN_SUITE]

    # Analytic transition-representation memory (same as the experiment).
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.2))

    for monitor_name, group in df.groupby("monitor_name"):
        group = group.sort_values("n_leaves")
        x = group["n_leaves"]
        y_us = group["mean_s_per_cell"] * 1e6
        yerr_us = group["std_s_per_cell"] * 1e6
        ax1.errorbar(x, y_us, yerr=yerr_us, marker="o", label=monitor_name, capsize=3)
        ax2.errorbar(x, y_us / x, yerr=yerr_us / x, marker="o", label=monitor_name, capsize=3)

    for ax, ylabel, title in [
        (ax1, "Avg time per cell (µs)", "Time per cell vs formula size"),
        (ax2, "Avg time per cell per leaf (µs)", "Time per cell per leaf"),
    ]:
        ax.set_xlabel("Leaves (atoms) — exponential scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(all_n)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    ax3.plot(mem_n, dense_gb, marker="o", color="tab:red",
             label="DeepDFA dense  $|Q|^2\\,2^{|AP|}$")
    ax3.plot(mem_n, factored_gb, marker="o", color="tab:green",
             label="DeepDFA factored (cube masks)")
    ax3.axhline(4.0, color="gray", linestyle=":", label="4 GB VRAM (laptop)")
    ax3.axvline(DENSE_MAX_LEAVES, color="tab:red", linestyle="--", alpha=0.4)
    ax3.set_yscale("log")
    ax3.set_xlabel("Leaves (atoms)")
    ax3.set_ylabel("Transition representation (GB)")
    ax3.set_title("DeepDFA memory: dense $2^{|AP|}$ wall vs factored")
    ax3.set_xticks(all_n)
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle("Exp 2 — formula complexity", y=1.02)

    out = RESULTS_DIR / "exp2_formula_complexity_clean.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Exp 3 — batch size (absolute time-per-trace + speedup)
# ---------------------------------------------------------------------------

def plot_exp3() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp3_batch_size.csv")
    df["mean_s_per_trace"] = df["mean_s_per_cell"] * df["trace_length"]
    df["std_s_per_trace"] = df["std_s_per_cell"] * df["trace_length"]
    batch_sizes = sorted(df["n_traces"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for monitor_name, group in df.groupby("monitor_name"):
        group = group.sort_values("n_traces")
        x = group["n_traces"]
        y_ms = group["mean_s_per_trace"] * 1e3
        yerr_ms = group["std_s_per_trace"] * 1e3
        ax1.errorbar(x, y_ms, yerr=yerr_ms, marker="o", label=monitor_name, capsize=3)

        baseline = group["mean_s_per_trace"].iloc[0]
        speedup = baseline / group["mean_s_per_trace"]
        ax2.plot(x, speedup, marker="o", label=monitor_name)

    x_ref = np.array(batch_sizes)
    ax2.plot(x_ref, x_ref / x_ref[0], linestyle="--", color="gray", label="ideal linear")

    for ax, ylabel, title in [
        (ax1, "Time per trace (ms)", "Absolute time per trace"),
        (ax2, "Speedup vs own batch=1", "Speedup vs batch size"),
    ]:
        ax.set_xlabel("Batch size (number of traces)")
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(batch_sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
    ax2.set_yscale("log", base=2)

    fig.suptitle("Exp 3 — batch size", y=1.02)

    out = RESULTS_DIR / "exp3_batch_size_clean.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_exp2()
    plot_exp3()
