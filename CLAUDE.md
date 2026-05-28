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

## Repository Structure

Files marked ✅ exist and are tested. Files marked 🔲 are planned but not yet written.

```
nesy_runtime_monitoring/
├── src/
│   ├── __init__.py
│   ├── formula/
│   │   ├── __init__.py
│   │   └── compiler.py        ✅ LTLf → minimal DFA (wraps ltlf2dfa)
│   ├── monitors/
│   │   ├── __init__.py
│   │   ├── base.py            ✅ Abstract Monitor interface + Verdict enum
│   │   ├── symbolic_dfa.py    ✅ Paradigm 1 — crisp DFA walk
│   │   ├── rulerunner.py      🔲 Paradigm 2 — CILP Horn-clause encoding
│   │   └── deep_dfa.py        🔲 Paradigm 3 — differentiable transition matrix
│   ├── adaptation/
│   │   └── poc.py             🔲 Proof-of-concept gradient adaptation
│   └── benchmarks/
│       ├── formulas.py        🔲 Benchmark formula registry
│       └── runner.py          🔲 Timing/metric harness
├── experiments/
│   ├── exp1_single_trace.py   🔲 Single-trace speed
│   ├── exp2_batched.py        🔲 Parallel trace scaling
│   ├── exp3_scalability.py    🔲 Formula complexity scaling
│   └── exp4_adaptation.py     🔲 PoC adaptation
├── tests/
│   ├── test_compiler.py       ✅ DFA structure + guard evaluation (9 tests)
│   └── test_symbolic_dfa.py   ✅ Step semantics, early termination, reset (21 tests)
├── results/                   🔲 CSV/JSON outputs, plots
└── papers/                    Reference papers and planning notes
```


## Abstract Monitor Interface

All three monitors implement the same interface (`monitors/base.py`):

```python
compile(formula: str) -> MonitorInstance   # classmethod
step(obs: dict[str, bool]) -> Verdict      # SATISFY / VIOLATE / UNDECIDED
final_verdict() -> Verdict                 # binary end-of-trace check; never UNDECIDED
reset() -> None
run(trace) -> Verdict                      # default: calls reset, loops step, then final_verdict
batch_run(traces) -> list[Verdict]         # default: sequential run(); DeepDFA overrides for GPU
```

Three-valued semantics (`UNDECIDED`) applies online — a trace mid-execution may not yet have a determined verdict. Absorbing states (all successors accepting, or all rejecting) enable early termination.

`final_verdict()` is a required separate method because response-style formulas like `G(a → F b)` have neither a trap state nor an accepting sink, so `step()` always returns `UNDECIDED`. The verdict is only binary at end-of-trace.

## Key Technical Details

**LTLf → DFA compilation:** use `ltlf2dfa` (Python wrapper for MONA). `to_dfa()` returns a DOT string with transitions labeled by boolean expressions over atoms (`~a`, `a & ~b`, `b | ~a`, `true`). The compiler parses this DOT, converts MONA guard syntax (`~`/`&`/`|`) to Python (`not`/`and`/`or`), and compiles each guard to a bytecode object once at construction time (compile-once, eval-many pattern). The `DFA` dataclass exposes `states`, `atoms`, `initial`, `accepting`, `transitions`, `trap_states`, `accepting_sinks`, and a `step(state, obs) -> state` method.

**Trap states and accepting sinks** are precomputed by graph reachability at DFA construction time (not at runtime). A trap is any state from which no accepting state is reachable; an accepting sink is any state from which all reachable states are accepting. Per-step verdict checks cost a single `set` membership test.

**`ltlf2dfa` quirk:** `lark` (a transitive dependency) emits two `DeprecationWarning`s about `sre_parse`/`sre_constants`. These are harmless and suppressed automatically by pytest; ignore them.

**DeepDFA transition tensor:** shape `(|Q|, |Σ|, |Q|)` where `T[q, σ, q']` = 1 iff state `q` transitions to `q'` on symbol `σ`. Crisp mode uses argmax; soft mode uses `softmax(einsum(T, obs, q))` and is differentiable.

**RuleRunner CILP encoding:** each Horn clause becomes a hidden unit; weights `+1`/`−1` per literal polarity; threshold set so the unit fires iff all positive body literals are true and no negative ones. Forward pass = one chaining iteration; repeat to fixpoint (bounded by formula depth). The convergence loop is the source of RuleRunner's sequential bottleneck — preserve it faithfully.

**Benchmark formulas** come from two sources: the original IJCNN 2014 tables (`◇a`, `□(a∨b∨c∨d)`, `◇((a∧Xb)∨(c∧Nd))`, scaled atom counts), and Declare/BPM constraint patterns (response `a→◇b`, chain response `a→Xb`, precedence, non-co-existence).

## Benchmark Design

**Use synthetic traces for Exps 1–3.** The trace content is irrelevant to per-step monitoring cost — what matters is trace length and formula complexity. Using real data would conflate paradigm speed with early-termination frequency (which is data-dependent), making the comparison less clean. IJCNN 2014 also uses randomly generated traces; this is the right methodology, not a compromise.

**Reproduce and extend IJCNN 2014.** That paper compares only RuleRunner variants (base/sparse/gpu) — the symbolic DFA and DeepDFA are absent. Adding them is our direct contribution. Use the same formula family and same leaf counts so the paper is directly legible to anyone who knows IJCNN 2014.

**IJCNN 2014 formula family for scalability:** `◇ V_{i=1}^{n-1}(a_0 ∧ a_i)` with n = 2, 4, 8, 16, 32 leaves (atoms renamed alphabetically). A leaf = a single propositional atom (the terminal alphabet). This is the x-axis for Exp 2/3.

**Planned experiments:**

| Exp | X-axis | Formula | Expected story |
|---|---|---|---|
| 1: trace length | 1k–10k cells | `F a` (simple) | All paradigms flat — per-step cost is constant |
| 2: formula complexity | n=2,4,8,16,32 leaves | IJCNN 2014 family | Symbolic DFA flat; RuleRunner linear; DeepDFA flat with GPU overhead |
| 3: batch size | 1–1024 parallel traces | fixed formula | DeepDFA's GPU advantage emerges here |
| 4: adaptation PoC | — | Declare pattern | Learning experiment — real dataset appropriate here |

**Real dataset for Exp 4 only.** The adaptation PoC needs realistic trace distributions for the learning experiment to be meaningful. Use a **BPI Challenge log** (BPIC 2012 or BPIC 2017 — both standard in process mining, freely available). Exps 1–3 are purely synthetic.

## Key Papers in `papers/`

- `Preliminary chat with Claude.txt` — full research planning document, primary reference for motivation and framing
- `Claude 2.txt` — design discussion for the symbolic baseline: why `ltlf2dfa + custom runner` is the right architecture, three-valued LTL3 semantics, trap/sink precomputation rationale, and why Declare4Py and RV-Monitor were ruled out
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
pytest tests/test_symbolic_dfa.py::test_eventually

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
- `lark` — NOT a direct dependency; it is pulled in transitively by `ltlf2dfa`. RuleRunner's parse tree is built programmatically in Python, not parsed from a grammar file.