# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on **Neuro-Symbolic Runtime Monitoring** combining LTLf (Linear Temporal Logic over finite traces) with differentiable automata. The immediate goal is **Paper A** for the NeSy conference (~1 month deadline): a three-way comparison of LTLf runtime monitoring paradigms, with a proof-of-concept gradient-based adaptation experiment.

This is an active research project — plans, experiments, and framing should be treated as working hypotheses, not fixed requirements. Expect iteration.

## Paper A: The Core Idea

Three paradigms for LTLf runtime monitoring, compared theoretically and experimentally:

| Paradigm | How it works | Key property |
|---|---|---|
| **Symbolic DFA** | Compile LTLf → minimal DFA; track state explicitly | Fastest single-trace; no learning |
| **RuleRunner** | Formula parse tree → extended truth tables → Horn clauses → CILP neural net | Learning constrained to syntactic neighbors |
| **DeepDFA** | Compile LTLf → minimal DFA → differentiable transition matrix | Native GPU batching; fully differentiable |

The argument: DeepDFA is not just faster in the batched setting — it is the right foundation for specification adaptation, because the DFA is a canonical semantic representation and the soft state is natively differentiable.

Paper A ends with a proof-of-concept adaptation experiment (Experiment 4) that motivates the follow-up Paper B (AAAI/AAMAS/ICLR).

## Planned Repository Structure

```
nesy_runtime_monitoring/
├── src/
│   ├── formula/
│   │   ├── parser.py          # LTLf formula parsing
│   │   └── compiler.py        # LTLf → minimal DFA (wraps ltlf2dfa)
│   ├── monitors/
│   │   ├── base.py            # Abstract Monitor interface
│   │   ├── symbolic_dfa.py    # Paradigm 1
│   │   ├── rulerunner.py      # Paradigm 2
│   │   └── deep_dfa.py        # Paradigm 3
│   ├── adaptation/
│   │   └── poc.py             # Proof-of-concept gradient adaptation
│   └── benchmarks/
│       ├── formulas.py        # Benchmark formula registry
│       └── runner.py          # Timing/metric harness
├── experiments/
│   ├── exp1_single_trace.py   # Single-trace speed
│   ├── exp2_batched.py        # Parallel trace scaling
│   ├── exp3_scalability.py    # Formula complexity scaling
│   └── exp4_adaptation.py     # PoC adaptation
├── tests/
├── results/                   # CSV/JSON outputs, plots
└── papers/                    # Reference papers and planning notes
```


## Abstract Monitor Interface

All three monitors implement the same interface (`monitors/base.py`):

```python
compile(formula: str) -> MonitorInstance
step(obs: dict[str, bool]) -> Verdict      # SATISFY / VIOLATE / UNDECIDED
reset()
batch_run(traces: list[list[dict]]) -> list[Verdict]
```

Three-valued semantics (`UNDECIDED`) applies online — a trace mid-execution may not yet have a determined verdict. Absorbing states (all successors accepting, or all rejecting) enable early termination.

## Key Technical Details

**LTLf → DFA compilation:** use `ltlf2dfa` (Python wrapper for MONA). Output is a minimal DFA with explicit states, alphabet, transition dict, initial state, and accepting states.

**DeepDFA transition tensor:** shape `(|Q|, |Σ|, |Q|)` where `T[q, σ, q']` = 1 iff state `q` transitions to `q'` on symbol `σ`. Crisp mode uses argmax; soft mode uses `softmax(einsum(T, obs, q))` and is differentiable.

**RuleRunner CILP encoding:** each Horn clause becomes a hidden unit; weights `+1`/`−1` per literal polarity; threshold set so the unit fires iff all positive body literals are true and no negative ones. Forward pass = one chaining iteration; repeat to fixpoint (bounded by formula depth). The convergence loop is the source of RuleRunner's sequential bottleneck — preserve it faithfully.

**Benchmark formulas** come from two sources: the original IJCNN 2014 tables (`◇a`, `□(a∨b∨c∨d)`, `◇((a∧Xb)∨(c∧Nd))`, scaled atom counts), and Declare/BPM constraint patterns (response `a→◇b`, chain response `a→Xb`, precedence, non-co-existence).

## Key Papers in `papers/`

- `Preliminary chat with Claude.txt` — full research planning document, primary reference for motivation and framing
- `IJCNN 2014.PDF` / `IJCNN 2015.pdf` — RuleRunner: the system being compared against and modernized
- `IS__NeSyPPM.pdf` — NeSy PPM paper: source of the existing DeepDFA implementation (Elena Umili collaboration)
- `RuleRunner.pdf` — earlier RuleRunner work
- `TOSEMv4.pdf` — Borges et al.: model-level adaptation (background, less central to Paper A)

## Environment Setup

The conda environment `nesy-monitoring` is already created and ready. To reproduce from scratch (e.g. on a new machine):

```bash
conda env create -f environment.yml
conda activate nesy-monitoring
```

`environment.yml` pins all versions including `torch==2.6.0+cu124` from the PyTorch CUDA index.

## Commands

```bash
# Activate environment
conda activate nesy-monitoring

# Install/reinstall project in editable mode
# Note: use the full path — conda run resolves to system pip on this machine
/home/matteo/miniconda3/envs/nesy-monitoring/bin/pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_symbolic_dfa.py::test_eventually_formula

# Lint
ruff check .

# Run an experiment
python experiments/exp1_single_trace.py
```

## Dependencies

- `ltlf2dfa==1.0.2` — LTLf → minimal DFA (Python wrapper for MONA)
- `torch==2.6.0+cu124` — PyTorch with CUDA 12.4 (RTX 3050 Laptop GPU)
- `numpy`, `matplotlib`, `pandas`, `scipy`, `tqdm` — via conda
- `pytest`, `ruff`, `black` — dev tools via conda
- `lark` — NOT installed; RuleRunner parse tree is built programmatically, not parsed from a file