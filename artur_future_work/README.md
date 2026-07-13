# NeSy LTLf Monitoring — Future Work

Self-contained fork of the `nesy_runtime_monitoring` project (fork point:
parent commit `57d74e3`, 2026-07-13) holding the research threads deferred from
the ICLR foundation paper:

- **Probabilistic monitoring** — soft/probabilistic observations, calibrated
  verdict confidence, and the theory of verdicts on probabilistic traces
  (a fully drafted section: `latex/5b_probabilistic_verdict.tex`).
- **Specification adaptation** — gradient-based correction of a wrong spec
  (planned, unimplemented).
- **Decision-diagram (WMC) monitor** — exact, calibrated, differentiable soft
  transitions (design note in `docs/`).

**Start with [CLAUDE.md](CLAUDE.md)** — it records the established findings
(so they are not re-derived), the sharp edges, and a concrete continuation
plan per thread.

## Setup

```bash
conda env create -f environment.yml     # env: nesy-monitoring-future
conda activate nesy-monitoring-future
python -m pytest                        # run from this folder
```

Requires MONA on PATH for `ltlf2dfa` (`apt-get install mona`). While this
folder lives inside the parent repo, always run from this folder — the
`conftest.py` shim pins `import src` to the local copy.

## Contents

| Path | What |
|---|---|
| `src/` | All four monitor implementations + benchmarks (noise, calibration, characterize) |
| `tests/` | Full test suite (~380 tests, 6 intentional xfails on the original RuleRunner) |
| `experiments/` | `exp_uncertainty.py` (accuracy/calibration/risk–coverage vs noise), `exp7_richer_family.py` (soft divergence + state blowup) |
| `latex/` | Drafted probabilistic-verdict section + prose trimmed from the parent paper |
| `docs/` | Capability reframe, benchmark findings, decision-diagram design note |
| `results/` | Fork-date CSVs and figures (cpu/ and gpu/ runs) |
