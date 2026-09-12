# Results layout

# Active results layout

Only artifacts produced by the redesigned E0+ evaluation belong here. The July
CPU/GPU CSVs, figures, and corresponding scripts have moved to `old/`; they are
historical diagnostics and are not submission evidence.

```
results/
├── rq1/       frozen exact semantic characterization (CSV + JSON manifest)
├── e2/        fresh-process instrumentation smoke (not paper measurements)
├── rq2/       Phase E3 cost-of-correctness raw records and summaries
├── rq3/       Phase E4 structural-scaling raw records and summaries
└── rq4/       Phase E5 cross-architecture CPU candidate and CUDA audit
```

`rq1/rq1_characterization.{csv,json}` is not a timing result. It is the
versioned exact-product semantic gate for original, bounded-default,
bounded-exact-online, and progression RuleRunner. Regenerate or validate it
with:

```bash
python experiments/rq1_semantic_characterization.py
python experiments/rq1_semantic_characterization.py --check
```

`e2/e2_instrumentation_smoke.{csv,json}` validates cold-compilation stages,
representation statistics, resource supervision, and peak-memory collection
for all twelve monitor configurations. Regenerate it with:

```bash
python experiments/e2_instrumentation_smoke.py
```

It is explicitly labeled as an infrastructure smoke artifact and must not be
used as final performance evidence.

`rq2/` contains the Phase E3 cost-of-correctness run: native compilation,
paired batch-1/batch-64 runtime, separately timed offline certificates,
exhaustive decision-lag rows, summaries, a manifest, and an overview figure.
Regenerate all raw measurements or only derived summaries with:

```bash
python experiments/rq2_cost_of_correctness.py
python experiments/rq2_cost_of_correctness.py --summarize-existing
```

The checked-in artifact is labeled as a controlled CPU-run candidate. Repeat
it on the final submission machine before copying absolute times into the paper.

`rq3/` contains the five Phase E4 structural panels, their compiled artifact
statistics, bootstrap summaries, manifest, and overview figure. Regenerate it
with `python experiments/rq3_structural_scaling.py`. Its absolute CPU timings
have the same final-machine caveat.

`rq4/` contains raw native-compilation and runtime rows for capacity
end-to-end, capacity predecoded/native-interface, and deployment end-to-end
modes; absolute bootstrap summaries; explicitly symbolic-relative cold-start
comparisons; Pareto membership; a manifest; and an overview figure. Regenerate
the full requested CPU/CUDA grid with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-rq4 \
  python experiments/rq4_cross_architecture.py
```

The current CPU block is a controlled local candidate. This host has no CUDA,
so requested accelerator cells are explicit `unsupported` rows; they are not
GPU measurements. Run the same command on the final CUDA machine before making
accelerator claims.

The archived suite is documented in `old/README.md`. Uncertainty/calibration
results remain in `artur_future_work/results/` with the probabilistic-monitoring
thread.
