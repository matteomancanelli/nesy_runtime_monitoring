# `results/` — experiment CSVs and figures

This directory holds the **CSV outputs** of the experiments (the source of
truth) and the **PNG figures** rendered from them. Running and plotting are
decoupled: an experiment script writes/updates its CSV; `experiments/plots.py`
turns CSVs into figures. You can re-style or regenerate any figure without
re-running a sweep.

We run the sweeps on **Google Colab** (both CPU and a GPU runtime — usually a
Tesla T4) and develop locally in the `nesy-monitoring` conda/pip env. There is
no Docker in this project.

## CSV files

| CSV | Experiment | X-axis |
|---|---|---|
| `exp1_single_trace.csv` | per-cell cost vs trace length | `trace_length` |
| `exp2_formula_complexity.csv` | per-cell cost vs formula size (IJCNN family) | `n_leaves` |
| `exp3_batch_size.csv` | throughput vs batch size (`ijcnn_n8`) | `n_traces` |
| `exp5_depth_microbench.csv` | within-step cost vs nested-X depth | depth (from `formula_name`) |
| `exp_uncertainty.csv` | accuracy + calibration vs perceptual noise | `eps` |

### Timing CSV columns (exp1/2/3/5)

`monitor_name, formula_name, n_leaves, trace_length, n_traces, n_repeats,
mean_s_per_cell, std_s_per_cell, device, early_termination, gpu_name`

- **`device`** is `cpu` or `cuda`, and **`gpu_name`** records the GPU model
  (e.g. `Tesla T4`), so every row is attributable to the hardware it ran on.
  This is what makes the CPU-vs-GPU figures possible.
- **`early_termination`** is the measurement mode (Phase 0.1). A CSV measured in
  one mode is auto-dropped if the script is re-run in the other mode
  (`reset_if_stale`) — the two are different workloads and must not be mixed.
- Per-cell cost = `total_wall_time / (n_traces × trace_length)`; time-per-trace
  = `mean_s_per_cell × trace_length`.

### Provenance / accumulating runs

The resume key includes `device`, so a **CPU run and a GPU run of the same cell
coexist** in one CSV instead of the second being skipped. Two workflows both
work:

1. **One accumulating CSV.** Run the script on the CPU runtime, then on the GPU
   runtime, pointing at the same file — you end up with both `device` values in
   one CSV.
2. **One CSV per run** (what we usually do on Colab). Download each run and keep
   them side by side, e.g. `exp3_batch_size_cpu.csv`, `exp3_batch_size_t4.csv`.
   The plot functions accept a *list* of CSVs and merge them.

## Figures

**One file per plot** — no side-by-side composites, so each can be dropped into
LaTeX independently. Colours and display names are consistent across every
figure (Okabe–Ito palette; colour = paradigm, line style = CPU vs GPU).

Regenerate everything from the current CSVs:

```bash
python experiments/plots.py            # all experiments
python experiments/plots.py exp3       # just one
```

Or from code, for CPU-vs-GPU work:

```python
from experiments.plots import (
    plot_exp3, plot_device_comparison, plot_device_speedup,
)

# overlay a CPU CSV and a GPU CSV (CPU solid, GPU dashed):
plot_exp3(["results/exp3_batch_size_cpu.csv", "results/exp3_batch_size_t4.csv"])

# one file per monitor, CPU vs GPU:
plot_device_comparison([".._cpu.csv", ".._t4.csv"], "exp3")

# GPU speed-up (CPU time / GPU time) per monitor, one file:
plot_device_speedup([".._cpu.csv", ".._t4.csv"], "exp3")
```

Files written (per experiment):

- exp1 → `exp1_time_per_cell.png`
- exp2 → `exp2_time_per_cell.png`, `exp2_time_per_cell_per_leaf.png`,
  `exp2_memory_wall.png`
- exp3 → `exp3_time_per_trace.png`, `exp3_speedup.png`
- exp5 → `exp5_depth.png`
- uncertainty → `exp_uncertainty_accuracy_{noise}_{formula}.png` (one per cell),
  `exp_uncertainty_reliability.png`, `exp_uncertainty_ece.png`,
  `exp_uncertainty_defect_maxscore.png`, `exp_uncertainty_defect_fracover1.png`
- device comparison → `{exp}_device_{Monitor}.png`, `{exp}_device_speedup.png`

The analytic memory-wall panel (exp2) and the reliability diagram (uncertainty)
recompute from the formulas, so they need `ltlf2dfa`/MONA + `torch`; if those are
missing the panel degrades to a note. Every other figure needs only
matplotlib/pandas.

PNGs are regenerable artifacts — if `results/` has no PNGs (e.g. after a fresh
clone), just run the command above.
