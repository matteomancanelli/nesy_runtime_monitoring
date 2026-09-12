# Archived benchmark suite

This directory contains the benchmark, experiment, and result material from the
pre-E0 evaluation. It was moved here intact on 2026-08-25 because those scripts
and measurements predate the current RuleRunner repairs, balanced IJCNN corpus,
result schemas, and isolated compilation protocol.

The archive is retained for provenance only:

- `experiments/`: the numbered exp1/2/3/5/6/7 scripts and old plotters;
- `results/`: the July CPU/GPU CSVs and their generated figures;
- `docs/`: the superseded experiment map and findings based on those runs;
- `scripts/`: the old all-experiments launcher;
- `notebooks/`: the old Colab entry point.

Do not import these scripts into new experiments or quote their measurements in
the paper. The active evaluation starts with
`docs/EXPERIMENTAL_EVALUATION_PLAN.md`, uses `src/benchmarks/schema.py`, and
writes versioned artifacts under the top-level `results/` directory.
