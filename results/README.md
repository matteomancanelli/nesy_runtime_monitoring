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
| `exp6_state_scaling.csv` | per-cell cost vs automaton size `|Q|` (bounded response) | `n_leaves` (holds the measured `|Q|`) |
| `exp_uncertainty.csv` | accuracy + calibration vs perceptual noise | `eps` |

## Experiment × parameter × plot matrix

Every experiment fixes all parameters but one (the swept axis) and runs the same
`MONITORS` list. Each timing experiment can be run on **CPU** or **GPU (cuda)**
— pass `device` via the auto-selected `DEVICE` — and the two runs are kept as
separate CSVs and overlaid at plot time (§ *CPU vs GPU*). CPU-only monitors
(Symbolic, RuleRunner-structured) always record `cpu` in either run.

| Exp | Swept axis (values) | Fixed params | Monitors | Output plots (one file each) |
|---|---|---|---|---|
| **exp1** trace length | `trace_length` ∈ 1k…10k | `G(a→Fb)`, n_traces=100 | Symbolic, RuleRunner(flat), RuleRunner(structured), DeepDFA dense, factored, **scan** | `exp1_time_per_cell.png` |
| **exp2** formula size | `n_leaves` ∈ {2,4,8,16,32} | IJCNN family, len=5000, n_traces=100 | Symbolic, RuleRunner(flat/structured), DeepDFA factored, dense (≤16 leaves) | `exp2_time_per_cell.png`, `exp2_time_per_cell_per_leaf.png`, `exp2_memory_wall.png` |
| **exp3** batch size | `n_traces` ∈ {1,2,…,1024} | `ijcnn_n8`, len=1000 | Symbolic, RuleRunner(flat/structured), DeepDFA dense, factored, **scan** | `exp3_time_per_trace.png`, `exp3_speedup.png` |
| **exp5** parse-tree depth | nested-X depth ∈ {0…10} | `ijcnn_n8`, batch=1, len=500 | Symbolic, RuleRunner(flat/structured), DeepDFA dense, factored | `exp5_depth.png` |
| **exp6** automaton size | `|Q|` (deadlines {2,4,8,16,32,64}) | bounded response, batch=256, len=500 | Symbolic, DeepDFA dense, **scan**, factored | `exp6_state_scaling.png` |
| **exp_uncertainty** noise | `ε` ∈ [0,0.8] × noise {beta, bitflip} × formula {majority3, response, ijcnn_n4} | N=3000 traces, 3 seeds | Symbolic-threshold, DeepDFA soft (raw), DeepDFA soft (norm) | `exp_uncertainty_accuracy_{noise}_{formula}.png` (6), `exp_uncertainty_reliability.png`, `exp_uncertainty_ece.png`, `exp_uncertainty_defect_maxscore.png`, `exp_uncertainty_defect_fracover1.png` |

**Cross-cutting CPU-vs-GPU plots** (any timing experiment, given both a CPU CSV
and a GPU CSV):

| Function | Output |
|---|---|
| `plot_exp{1,2,3,5,6}([cpu.csv, gpu.csv])` | the standard plot with CPU (solid) and GPU (dashed) overlaid |
| `plot_device_comparison([cpu, gpu], "exp3")` | `exp3_device_{Monitor}.png` — one file per monitor, CPU vs GPU |
| `plot_device_speedup([cpu, gpu], "exp3")` | `exp3_device_speedup.png` — CPU-time / GPU-time per monitor |

### Timing CSV columns (exp1/2/3/5/6)

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

### Truthful device labeling

Each row's `device`/`gpu_name` record the device the monitor **actually**
computed on, not the one requested. The symbolic DFA walk and the structured
RuleRunner are pure Python and always run on the CPU — they stamp `cpu` even in
a GPU run (there is no tensor op to place on a GPU), so a CSV never claims a GPU
run that did not happen.

### CPU vs GPU: one CSV per run

Keep **one CSV per run** (the Colab workflow): run the whole sweep on the CPU
runtime, download it as e.g. `exp3_batch_size_cpu.csv`; run again on the GPU
runtime, download `exp3_batch_size_t4.csv`. The plot functions accept a *list*
of CSVs and merge them, splitting curves by the truthful `(device, gpu_name)`.
CPU-only monitors that appear in both files (same truthful `cpu` config) are
de-duplicated (their measurements averaged) so each curve is drawn once. Do
**not** point two runs at the same CSV file: the resume logic keys on
`(monitor, formula, trace_length, n_traces)` — not device — so the second run
would skip every already-present cell.

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
- exp6 → `exp6_state_scaling.png`
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
