# Results layout

Experiment scripts write CSVs to `results/` directly; runs from different
machines are then sorted by hand into per-device folders (the CSVs are
resumable, so mixing devices in one file would silently skip cells — keep them
separate):

```
results/
├── cpu/       one CSV per experiment — Colab CPU runtime
├── gpu/       one CSV per experiment — Colab GPU runtime (Tesla T4)
└── figures/   PNGs rendered from the CSVs
    ├── (root)     merged CPU+GPU overlays (solid = CPU, dashed = GPU)
    ├── gpu_only/  GPU session alone — the clean lead figures for the paper
    └── device/    per-monitor CPU-vs-GPU comparisons + GPU-speedup summary
```

| CSV | Experiment | Axis |
|---|---|---|
| `exp1_single_trace.csv` | exp1 | per-cell cost vs trace length |
| `exp2_formula_complexity.csv` | exp2 | per-cell cost vs formula breadth (+ memory wall) |
| `exp3_batch_size.csv` | exp3 | time per trace vs batch size |
| `exp5_depth_microbench.csv` | exp5 | per-cell cost vs nested-X depth |
| `exp6_state_scaling.csv` | exp6 | per-cell cost vs \|Q\| (linear family) |
| `exp7_stateblowup.csv` | exp7 | per-cell cost vs \|Q\| = 2^k + 1 (exponential family) |

Regenerate every figure without re-running the sweeps:

```bash
python experiments/plots.py              # from whatever CSVs are in results/
python experiments/make_all_plots.py     # the merged / gpu_only / device sets
```

Uncertainty/calibration results (Capability Exp A, exp7 soft divergence) moved
to `artur_future_work/results/` with the rest of the probabilistic-monitoring
thread.
