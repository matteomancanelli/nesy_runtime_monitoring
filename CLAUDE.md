# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on **Neuro-Symbolic Runtime Monitoring** combining LTLf (Linear Temporal Logic over finite traces) with differentiable automata. The goal is **Paper A**: a three-way comparison of LTLf runtime monitoring paradigms, anchored on the *capabilities* neuro-symbolic monitors afford (handling soft/probabilistic observations; gradient-based specification adaptation), with timing experiments establishing competitiveness.

**No hard deadline (as of 2026-06-24).** The original NeSy-conference deadline was dropped — we are taking the time to "do the best we can." Full scope and exploratory directions are on the table.

This is an active research project — plans, experiments, and framing should be treated as working hypotheses, not fixed requirements. Expect iteration, and be open to unexpected directions (a new/hybrid paradigm, a new benchmark family, a fresh pivot).

## Paper A: The Core Idea

Three paradigms for LTLf runtime monitoring, compared theoretically and experimentally:

| Paradigm | How it works | Key property |
|---|---|---|
| **Symbolic DFA** | Compile LTLf → minimal DFA; track state explicitly | Fastest crisp single-trace; no learning; crisp boolean inputs only |
| **RuleRunner** | Formula parse tree → extended truth tables → Horn clauses → CILP neural net | Learning localized to syntactic neighbors; nested-temporal representational limit; within-step sequential |
| **DeepDFA** | Compile LTLf → minimal DFA → differentiable transition matrix | Native GPU batching; fully differentiable; soft-input propagation; alphabet (2^|AP|) blowup |

### Framing (post first-pass results — 2026-06-24)

The first implementation pass produced timing results (`results/exp1-3`) that reshaped the framing. **Read this before touching experiments or the paper narrative.**

- **Do NOT anchor the paper on speed.** For crisp boolean monitoring the symbolic DFA is the theoretical optimum — a DFA walk is a dict lookup; nothing differentiable beats it. Symbolic dominating throughput is *expected*, not a threat. Trying to make a NeSy paradigm "win" on raw speed is a losing proposition and reviewers will see through it.
- **Anchor on capability** — the reasons to go neuro-symbolic that symbolic *fundamentally cannot* offer: (1) **soft/probabilistic observations** (a neural perceptor emits probabilities, not booleans; DeepDFA's `soft_matrix` propagates probability mass to a calibrated verdict; symbolic must threshold and discard information), and (2) **gradient-based specification adaptation** (the differentiable monitor can be corrected from data; symbolic is frozen). Timing demotes to *competitiveness* evidence.
- **Keep paradigms neutral.** Present a capability matrix and let it speak; do **not** pre-crown DeepDFA. Each paradigm has a distinct Achilles heel (symbolic = state blowup; RuleRunner = nested-temporal limit + within-step sequential cost; DeepDFA = 2^|AP| alphabet blowup) — that balance is the honest three-way story.
- **Still try for an honest speed win.** We do want to improve speed and produce a *fair* comparison (see Phase 0). If a genuine batched-throughput advantage survives fair measurement (esp. on the GPU server, not the 4 GB laptop), it's a bonus — reported neutrally.

Paper A ends with a proof-of-concept adaptation experiment that motivates the follow-up Paper B (AAAI/AAMAS/ICLR).

## Research Plan (Paper A) — phases and steps

Status legend: ✅ done · 🟡 in progress · 🔲 not started.

### Phase 0 — Honest timing comparison (committed) 🔲

Make the existing throughput experiments measure the same workload across paradigms, and try to genuinely improve speed. This is foundational — the current Exp 2/3 numbers are partly artifacts.

- **0.1 — Kill the early-termination confound (Exp 2 & 3). 🟡 (mechanism done; re-run pending).** The IJCNN `◇(⋁(a₀∧aᵢ))` family early-terminates almost instantly on random traces, so Symbolic's reported ~1e-10 s/cell measures "how fast it gives up" (impossible as real per-cell compute: one Python dict lookup is ~5e-8 s) while the batched neural monitors process *all* cells uniformly — not apples-to-apples. Fix: add an `early_termination=False` measurement mode (force processing all cells for the per-cell-cost figures), and/or use a non-terminating family. State the choice explicitly in the paper.
  - **Implemented:** `Monitor.run/batch_run` now take `early_termination: bool` ([base.py](src/monitors/base.py)); when False the crisp symbolic walk processes all cells (absorbing states are sticky, so verdicts are unchanged — verified in [tests/test_early_termination.py](tests/test_early_termination.py)). The batched neural monitors (DeepDFA, RuleRunner) already process all cells, so they accept the flag for parity but their compute is unchanged. `time_monitor(..., early_termination=)` threads it through and records it in `TimingResult`; `reset_if_stale()` drops CSVs measured in the other mode (the two are different workloads — don't resume/mix). Exp 2 & 3 set `EARLY_TERMINATION = False` and annotate the figure. Confirmed effect: symbolic per-cell cost on `ijcnn_n8` jumps from ~3e-9 s (give-up artifact) to ~3e-7 s (real dict lookup).
  - **Pending:** re-run exp2/exp3 to regenerate `results/exp2*`, `results/exp3*` under the new mode (exp2 factored-DeepDFA is slow — see Phase 0.2 vectorization).
- **0.2 — Fix DeepDFA in Exp 2 (dual finding). 🟡 (vectorization + dense panel done; re-run pending).** Factored-DeepDFA *was* growing 7e-5→7e-4 with n — a Python-closure-rebuild artifact, not the model. Produce two curves: (a) **dense-capped** that visibly hits the 2^n VRAM wall (= the alphabet-blowup *finding*), and (b) **vectorized** factored crisp path (precompute require-true/require-false integer masks; drop per-cell sympy closures) that stays flat. Separates *memory wall* from *compute cost*.
  - **Implemented (b):** the factored *monitoring* path is now vectorized. Each guard is decomposed **once** at construction into a disjoint (orthogonal) cube cover by Shannon expansion (`_guard_cubes`/`_shannon_cubes` in [deep_dfa.py](src/monitors/deep_dfa.py)), stored as require-true/require-false integer masks; `DeepDFATensor.crisp_matrix(p)` builds the per-cell transition matrix as a single vectorized mask reduction (`∏_a [1 − rt·(1−p) − rf·p]`) — no per-cell sympy closures. `_advance` and `batch_run` route through it, and trace encoding is vectorized too (`encode_presence`, numpy, no per-cell tensor alloc). Effect: factored per-cell cost on the IJCNN family drops from ~2e-4 s (n=32, closure rebuild) to ~7e-6 s, and the n=2→32 growth from ~24× to ~3.7× (residual = genuine O(n²) mask reduction, not Python overhead). The differentiable `soft_matrix` (recursive, read-once-exact fractional) is **kept separate and unchanged**, so its read-once semantics and the non-read-once finding (Phase 3.3) are preserved.
  - **Implemented (a):** [exp2_formula_complexity.py](experiments/exp2_formula_complexity.py) adds `DeepDFAMonitorDense` (capped at `DENSE_MAX_LEAVES=16`; skipped beyond — 2^32 ≈ 64 GB at |Q|=2) and a third **memory-wall panel** plotting dense `|Q|²·2^|AP|` vs factored mask bytes (log-y, 4 GB VRAM line), computed analytically so the infeasible tensors are never built. Verified: dense is *fastest where it fits* (~3e-6 s/cell) but walls out at n=32 (68.7 GB); factored stays flat and scales (47 KB at n=32).
  - **Pending:** re-run exp2 to regenerate `results/exp2*` under the new mode with the dense/factored/memory panels (slow on the laptop; ideally the Docker GPU server, Phase 0.4).
- **0.3 — Exp 3 measurement hygiene.** One `torch.cuda.synchronize()` around the whole timed region (not per cell); keep the batch on-device; read verdicts only at end-of-loop. Lead figures with the **absolute time-per-trace** panel; demote/annotate the speedup panel (its baseline is RuleRunner's catastrophic batch-1 number, which makes its "speedup" misleading).
- **0.4 — Re-run Exp 3 on the Docker GPU server.** The 4 GB / 70 W laptop GPU saturates at tiny batch and undersells every GPU path. Re-run on the server (Docker-only host, see memory) and optionally with a larger automaton to give batched DeepDFA room. Honest outcome either way: a modest real win, or "GPU advantage needs larger automata/hardware."
- **0.5 — (optional) Within-step depth micro-benchmark.** `X`-nest `ijcnn_n8` to depth ~10, batch=1, len=1; isolates RuleRunner's depth-linear cost vs DeepDFA's flat one-matmul, independent of breadth. (Design already sketched in § Benchmark Design.)

**Exit criterion:** fair Exp 1–3 with the early-termination confound removed, DeepDFA's curve reflecting the model not Python overhead, and a clear verdict on whether any speed advantage survives.

### Phase 1 — Capability Exp A: monitoring under perceptual uncertainty (committed, do before adaptation) 🔲

The first capability that justifies NeSy, and simple in principle. Prioritized **before** adaptation because it reuses already-tested code (`soft_matrix` is row-stochastic and verified) and needs no training loop.

- **1.1 — Noisy-observation generator.** Generate crisp ground-truth traces + true verdicts (from the symbolic monitor as oracle), then corrupt each atom into a probability: flip with probability ε, or replace with a noisy probability (e.g. Beta centered on the true value). Sweep ε.
- **1.2 — Soft consumption per paradigm.** Symbolic: threshold each atom at 0.5 → crisp walk (the brittle baseline). DeepDFA: factored soft path → acceptance probability. RuleRunner: tanh variant → soft propagation. (Keep the comparison neutral — measure what each affords.)
- **1.3 — Metrics.** Verdict **accuracy vs ε** *and* **calibration** of the soft acceptance probability (does predicted confidence track empirical correctness?). Possibly AUC / ECE. Expected: soft paradigms degrade gracefully and emit calibrated confidence; symbolic is brittle and binary.
- **1.4 — New code.** `src/benchmarks/noise.py` (corruption models), `experiments/exp_uncertainty.py`, tests. The soft paths exist; main work is the harness + metrics.

**Exit criterion:** a figure showing accuracy-vs-noise and a calibration plot demonstrating a capability symbolic cannot have.

### Phase 2 — Capability Exp B: specification adaptation PoC 🔲

The headline NeSy payoff and the bridge to Paper B. Higher risk (training loop + data).

- **2.1 — Synthetic PoC first.** Start from a *wrong* spec (wrong target atom / over-strict threshold); make DeepDFA's soft transition (or acceptance) matrix learnable; train on labeled traces generated from the correct formula; show accuracy recovers. Symbolic = the control that cannot adapt.
- **2.2 — RuleRunner adaptation (parallel data point).** Swap CILP `sign`→`tanh` (recompute biases via Garcez & Zaverucha `Amin`), adapt weights on misclassified traces. Neutral comparison of *what each differentiable paradigm affords*.
- **2.3 — (stretch) Real data.** A BPIC log (BPIC 2012/2017) with a known Declare constraint, for a realistic adaptation story. Likely Paper B; a toy synthetic version suffices for Paper A.
- **Code:** `src/adaptation/poc.py`, `experiments/exp4_adaptation.py` (both currently 🔲).

### Phase 3 — Exploratory branches (open; any could become the contribution) 🔲

Pursue opportunistically; these are the "unexpected directions."

- **3.1 — Probabilistic three-valued LTLf monitoring (theory).** Formalize what it *means* to monitor a probabilistic trace: marginal acceptance probability (DeepDFA soft state) vs most-likely-path verdict (Viterbi) vs distribution over verdicts. Each paradigm naturally computes a different quantity; pinning down which is "correct" for safety monitoring is a paradigm-neutral theoretical contribution and a candidate true novelty.
- **3.2 — A fourth / hybrid paradigm.** The empty matrix cell: exact+fast at runtime *and* differentiable for adaptation. Candidates: differentiable read-once guard circuits over the minimal DFA (sidesteps both the 2^|AP| blowup and the speed penalty — may be latent in the factored path), or straight-through symbolic (crisp at inference, relaxed only during adaptation).
- **3.3 — Richer benchmark family.** The IJCNN `◇` family is a poor instrument (early-terminates; read-once guards make the soft path *exact*, hiding real divergence). Add Declare/BPM patterns, non-read-once-guard formulas (where soft paradigms provably diverge — a finding), and a state-blowup family (exposes symbolic's *and* DeepDFA's shared weakness — good for neutrality).
- **3.4 — (Paper B seed) End-to-end backprop through the monitor into a perceptor.** A toy where a spec-violation loss trains a perception network — the real NeSy dream.

### Phase 4 — Writing 🟡 (trails experiments; LaTeX in `latex/`)

- Sec 1 intro: thesis = "characterize the affordances of each paradigm; the case for NeSy is capability, not speed." Resolve the three `\textcolor{red}` TODOs.
- Sec 4 theory comparison (stub): the **capability matrix** + three Achilles heels, presented neutrally.
- Sec 5 experiments (stub): fixed Exp 1–3 + Capability A (+ B); state early-termination handling and hardware explicitly.
- Sec 6/7 related/conclusion (stubs).

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
│   │   ├── rulerunner/        🟡 Paradigm 2 — package, partial (see § Paradigm 2 below)
│   │   │   ├── __init__.py    ✅
│   │   │   ├── parse_tree.py  ✅ Internal LTLf parse-tree DAG
│   │   │   ├── rules.py       ✅ Per-operator templates + RuleSystem builder
│   │   │   ├── engine.py      ✅ Symbolic executor (Algorithm 2 of IJCNN 2014)
│   │   │   ├── cilp.py        ✅ CILP encoding to torch network
│   │   │   └── monitor.py     ✅ Monitor-interface wrapper (RuleRunnerMonitor)
│   │   └── deep_dfa.py        ✅ Paradigm 3 — differentiable transition tensor (dense + factored)
│   ├── adaptation/
│   │   └── poc.py             🔲 Proof-of-concept gradient adaptation (Phase 2)
│   └── benchmarks/
│       ├── __init__.py
│       ├── formulas.py        ✅ Benchmark formula registry (IJCNN suite + trace-length suite)
│       ├── noise.py           🔲 Observation-corruption models for Capability Exp A (Phase 1)
│       └── runner.py          ✅ Timing harness (time_monitor, random_traces, results_to_df)
├── experiments/
│   ├── exp1_single_trace.py   ✅ Trace-length scaling (G(a→Fb), 1k–10k cells)
│   ├── exp2_formula_complexity.py ✅ Formula complexity / IJCNN 2014 reproduction (n leaves)
│   ├── exp3_batch_size.py     ✅ Batch-size scaling (1–1024 traces, ijcnn_n8)
│   ├── exp_uncertainty.py     🔲 Capability Exp A — accuracy/calibration vs noise (Phase 1)
│   └── exp4_adaptation.py     🔲 PoC adaptation (Phase 2)
├── tests/
│   ├── test_compiler.py                    ✅ DFA structure + guard evaluation (9 tests)
│   ├── test_symbolic_dfa.py                ✅ Step semantics, early termination, reset (21 tests)
│   ├── test_rulerunner_parse_tree.py       ✅ Parse-tree DAG (18 tests)
│   ├── test_rulerunner_rules.py            ✅ Rule templates + worked example (23 tests)
│   ├── test_rulerunner_engine.py           ✅ Engine + equivalence sweep (37 tests, 3 xfail)
│   ├── test_rulerunner_cilp.py             ✅ CILP + equivalence vs engine + vs DFA (45 tests, 3 xfail)
│   ├── test_rulerunner_monitor.py          ✅ Monitor-ABC plumbing (6 tests)
│   └── test_deep_dfa.py                    ✅ DeepDFA dense+factored, batch, soft matrix (115 tests)
├── results/                   ✅ CSV + PNG outputs from exp1/2/3 (symbolic DFA only so far)
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
batch_run(traces) -> list[Verdict]         # default: sequential run(); DeepDFA AND RuleRunner override for batched CPU/GPU
```

Three-valued semantics (`UNDECIDED`) applies online — a trace mid-execution may not yet have a determined verdict. Absorbing states (all successors accepting, or all rejecting) enable early termination.

`final_verdict()` is a required separate method because response-style formulas like `G(a → F b)` have neither a trap state nor an accepting sink, so `step()` always returns `UNDECIDED`. The verdict is only binary at end-of-trace.

## Key Technical Details

**LTLf → DFA compilation:** use `ltlf2dfa` (Python wrapper for MONA). `to_dfa()` returns a DOT string with transitions labeled by boolean expressions over atoms (`~a`, `a & ~b`, `b | ~a`, `true`). The compiler parses this DOT, converts MONA guard syntax (`~`/`&`/`|`) to Python (`not`/`and`/`or`), and compiles each guard to a bytecode object once at construction time (compile-once, eval-many pattern). The `DFA` dataclass exposes `states`, `atoms`, `initial`, `accepting`, `transitions`, `trap_states`, `accepting_sinks`, and a `step(state, obs) -> state` method.

**Trap states and accepting sinks** are precomputed by graph reachability at DFA construction time (not at runtime). A trap is any state from which no accepting state is reachable; an accepting sink is any state from which all reachable states are accepting. Per-step verdict checks cost a single `set` membership test.

**`ltlf2dfa` quirk:** `lark` (a transitive dependency) emits two `DeprecationWarning`s about `sre_parse`/`sre_constants`. These are harmless and suppressed automatically by pytest; ignore them.

**DeepDFA transition tensor:** shape `(|Q|, |Σ|, |Q|)` where `T[q, σ, q']` = 1 iff state `q` transitions to `q'` on symbol `σ`. In our (non-mutually-exclusive) propositional setting `|Σ| = 2^|atoms|`. See § Paradigm 3 for the dense vs factored representations and the alphabet-blowup finding.

**RuleRunner CILP encoding:** each Horn clause becomes a hidden unit; weights `+1`/`−1` per literal polarity; threshold set so the unit fires iff all positive body literals are true and no negative ones. Forward pass = one chaining iteration; repeat to fixpoint (bounded by formula depth). The convergence loop is the source of RuleRunner's sequential bottleneck — preserve it faithfully.

**Benchmark formulas** come from two sources: the original IJCNN 2014 tables (`◇a`, `□(a∨b∨c∨d)`, `◇((a∧Xb)∨(c∧Nd))`, scaled atom counts), and Declare/BPM constraint patterns (response `a→◇b`, chain response `a→Xb`, precedence, non-co-existence).

## Paradigm 2 (RuleRunner) — implementation notes

This section is the **review document** for paradigm 2: every design decision, bug, limitation, and assumption surfaced while implementing it. Read this before reviewing the code under [src/monitors/rulerunner/](src/monitors/rulerunner/). The work is split into five sequential steps; steps 1–2.5 are done.

### Pipeline at a glance

```
LTLf formula string
  └─> parse_tree.parse()         (step 1)
        └─> Node DAG (subformula sharing, depth precomputed)
              └─> rules.build_rules()  (step 2)
                    └─> RuleSystem (eval rules + react rules + initial state)
                          ├─> engine.RuleEngine  (step 2.5) — symbolic executor
                          └─> cilp.CILPNet       (step 3, TODO) — torch network
                                └─> monitor.RuleRunnerMonitor  (step 4, TODO) — Monitor wrapper
```

Each layer is independently testable. `engine.py` is the oracle: it executes the rule system in pure Python and is checked against `SymbolicDFAMonitor` on randomized formulas + traces. The CILP network in step 3 must produce the same per-cell verdicts as the engine — the equivalence test transfers.

### Step 1 — parse_tree.py (✅ done)

**Internal representation.** A frozen-dataclass `Node` DAG with `(op, children, key, depth, atom)`. `key` is the canonical syntactic identity (just `str(ast)` of ltlf2dfa's AST) used as the name suffix for every R[.] / [.]V literal downstream.

**Decisions:**
- **Reuse ltlf2dfa's parser** but walk its AST into our own `Node` type. CLAUDE.md's earlier directive ("RuleRunner's parse tree is built programmatically, not parsed from a grammar file") rules out a grammar-driven re-parser; we still need a parser, and ltlf2dfa's is fine.
- **Binarize n-ary operators.** ltlf2dfa keeps `a & b & c` as a flat 3-tuple. We left-fold `&` / `|`, right-fold `->` / `U` / `R`. The U/R right-association was verified empirically: `to_mona()` produces the same MONA formula for `a U b U c` and `a U (b U c)`, so right-associating matches ltlf2dfa's downstream interpretation.
- **Subformula sharing.** Repeated `str(ast)` keys hit the cache and yield the same `Node` instance. So `(F b) & (G b)` produces a DAG with one shared `b` node, not two. This means every downstream subformula (R[ψ], [ψ]V) is emitted once, even for formulas with shared subexpressions.
- **`depth` precomputed.** Atoms have depth 0; parent depth is `1 + max(child.depth)`. Used by the engine's convergence-loop cap.

**Atoms can never be undecided** is enforced here too (`atom` field only set when `op is Op.ATOM`), and downstream pruning relies on this invariant.

### Step 2 — rules.py (✅ done)

**Data model.** `Literal(name: str, negated: bool)`, `Rule(body: frozenset[Literal], head: Literal)`, `RuleSystem(eval_rules, react_rules, initial_state, atoms, root_key)`. All literal naming is **string-based** with modes baked into the name as suffixes (e.g. `R[(a | F(b))]^B`, `[(a | F(b))]?^R`). This matches IJCNN 2014's paper notation directly and means CILP downstream gets one neuron per mode-distinct literal — no extra plumbing.

**Per-operator templates** (one function each):

| Operator | Modes | Reactivation re-installs operand subtree? |
|---|---|---|
| `ATOM` | none | n/a (atoms have no react rules) |
| `NOT` | none | no (child's own `?` reactivation handles it) |
| `AND`, `OR`, `IMPLIES` | B / L / R | no (per-mode self-only) |
| `EVENTUALLY` (◇), `ALWAYS` (□) | none | **yes** (operand subtree, fresh initial mode) |
| `UNTIL` (U), `RELEASE` (R) | none | **yes** (both operands' subtrees) |
| `NEXT` (X), `WEAK_NEXT` (WX) | B / A | **yes** in mode-I → mode-A transition only |

**Why some operators re-install the subtree and others don't.** When a temporal operator is `?` because its operand was F (e.g. `[F(φ)]?` because `[φ]F` at this cell), the operand's *own* reactivation doesn't fire (no `[φ]?` to consume). But the temporal operator still needs the operand re-monitored fresh next cell. So ◇/□/U/R explicitly re-install. By contrast, binary propositionals only ever go `?` when *the relevant child(ren)* are also `?` — and a child's `?` triggers its own reactivation, which handles its subtree. No double-install.

**The mode-tracking pattern** (AND/OR/IMPLIES). When a binary operator's mode B sees one child settled definite and the other still `?`, it transitions to mode L (left settled, watch right) or mode R (right settled, watch left). This **drops the settled child from monitoring** at later cells, which is essential for correctness: without it, atoms like `a` under `a ∨ ◇b` would get re-evaluated at every cell instead of being resolved once at cell 1, producing wrong verdicts.

The L/R-mode truth tables are derived from B's column at the **pin value** — the definite value that triggered the L/R transition. For AND the pin is T (because AND B's `?^L` cell is `(?, T)`); for OR/IMPLIES the pin is F. ⚠ **Initial bug**: my first draft pulled L/R from B's column at ψ=? instead of at ψ=pin. Caught before tests ran; fixed in [rules.py:152](src/monitors/rulerunner/rules.py#L152) by computing `psi_pin = next(qv for (pv, qv), tag in table_B.items() if pv == "?" and tag == "?L")`.

**Atoms can never be `?`** is exploited via `_undecided_modes(atom) = ()`. Any rule that would have body `[a]?` produces an empty product (no rule). This prunes dead "rule with [atom]? in body" code at template-instantiation time, keeping the rule count manageable.

**Faithful reproduction of IJCNN 2014's worked example.** The test [tests/test_rulerunner_rules.py::test_ijcnn2014_*](tests/test_rulerunner_rules.py) reproduces the paper's §III initial state, evaluation rules, and reactivation rules for `a ∨ ◇b` exactly (modulo our `(a | F(b))` key naming for the canonicalized form).

**Per-rule deduplication.** Templates may produce duplicate `(body, head)` pairs in their literal-name product expansions; `build_rules` deduplicates by `(body, head)` keys at the end. The engine sees a clean rule set.

**Assumption to revisit** (one knob): `psi_pin`/`phi_pin` derivation assumes every binary operator's mode-B table has exactly one `(?, V) → ?L` and one `(V, ?) → ?R` cell. True for AND/OR/IMPLIES; would need revisiting if we ever add asymmetric or n-mode binary operators. Not a concern for standard LTLf.

### Step 2.5 — engine.py (✅ done)

**Algorithm.** Per cell:
1. Inject `obs:a` literals for every atom observed in the cell. Non-observed atoms have nothing added — `~obs:a` literals in rule bodies are checked by **negation-as-failure** against this state, not by presence of a negated literal.
2. **Evaluation phase**: fire every eval rule whose body is satisfied, repeat for `depth + 1` passes (truth values propagate one parse-tree level per pass: pass 1 lifts atom obs into [a]T/F, pass `depth+1` reaches the root). Break early if no new facts.
3. Read root verdict: `[<root_key>]T` → SATISFY, `[<root_key>]F` → VIOLATE, else UNDECIDED.
4. If decided, mark absorbing and freeze. Else **reactivation phase**: fire every react rule once in parallel, keep only the `R[.]` literals for the next cell.

**End-of-trace resolution.** When the trace ends UNDECIDED, walk the parse tree and recursively resolve undecided subformulae per operator's end semantics. The interesting cases:
- `F φ`, `G φ` end-of-trace = **recurse on child** (not unconditional F/T). The original implementation returned `True` for G and `False` for F unconditionally — wrong when the child itself is unresolved temporal. E.g. `G(F b)` on `[F, F]` should be F (no b ever seen), not T.
- `φ U ψ`, `φ R ψ` end-of-trace = recurse on ψ (the consequent — U needs ψ at the last cell; R requires ψ throughout).
- `X φ` end = F, `WX φ` end = T (the FLTL strong/weak distinction).
- Binary propositionals in mode L/R end = resolve only the active child, pin the settled child to the mode's pin value, apply the truth table.

### Step 3 — cilp.py (✅ done)

**Algorithm.** Standard Garcez & Zaverucha 1999 CILP translation. Each rule = one hidden unit. Body literals connect to the unit with `+W` (positive literal) or `−W` (negated literal); hidden bias = `-W*(n - 0.5)` where n is the body length, so the unit's net input is `+0.5W` iff every body literal is satisfied and `−1.5W` (or worse) otherwise. Output literals are an OR of their incoming hidden units: weight `+W` per hidden, output bias = `W*(k - 1)` where k counts incoming hidden units (k=0 outputs get a negative bias so they stay at −1). Sign activation throughout.

**Two phases, shared literal space.** Eval and react have separate weight matrices but use the same literal-index vector. A `step()` call: build x with R[.] from the carried state + obs:a clamped from the cell → eval phase = `depth+1` sign-activated forward passes, OR-accumulated (`x = max(x, y)`) → read root verdict from `x[T_idx]`/`x[F_idx]` → if undecided, single react forward pass, write R[.] positions to next state.

**End-of-trace resolution shared with the engine.** Recipe is the same — recurse on the relevant child per operator. Implemented in [cilp.py:_resolve_end](src/monitors/rulerunner/cilp.py) parameterised by an `in_state(name) -> bool` predicate, so it works against either a Python `set[Literal]` (engine) or a `torch.Tensor` + literal index (CILP). The two file copies are intentionally near-identical; refactor into a shared helper if a third consumer appears.

**Equivalence with the engine: 0/N mismatches across the entire flat-temporal sweep.** This is the formal verification that the CILP encoding is faithful — sign activation matches set-membership semantics, OR-accumulation matches the fixed-point of the eval loop. Tested in [tests/test_rulerunner_cilp.py::test_cilp_matches_engine](tests/test_rulerunner_cilp.py).

**Subtle test-design issue.** The CILP xfail sweep first used 40 random traces and tripped an XPASS on `F (a & X b)` because the 1/80 divergent trace happened to fall outside the smaller sample (CILP and engine agree perfectly; both diverge from DFA on the same trace). Fixed by giving the xfail sweep an 80-trace budget so the rare divergent case is deterministically inside it. **Lesson**: when an xfail tracks an expected mismatch, the sweep must use enough traces to reliably hit the failure or the strict mode misreports as XPASS.

**Knobs we may want to revisit:**
- `_W = 1.0`. Any positive value works with sign activation; tanh would need a larger W for the activation to saturate. If we ever switch to tanh (for differentiability in Paper B's adaptation experiment), recompute hidden/output biases following Garcez & Zaverucha's `Amin` formula.
- Sign activation. Step 3 is exact (sign matches set semantics). For gradient-based adaptation later we'd replace sign with tanh and keep the same weight structure.
- Batching (✅ done). `step()` is still single-trace, but `CILPRunner.batch_run` adds a leading batch dimension to the activation tensor and runs the eval/react matmuls over the whole batch (`device="cpu"`/`"cuda"` via `from_formula(..., device=)`). Verified bit-for-bit equal to sequential `run()` in [tests/test_rulerunner_batch.py](tests/test_rulerunner_batch.py) on CPU and CUDA.

### Step 4 — monitor.py (✅ done)

**Thin Monitor-ABC adapter.** `RuleRunnerMonitor` is a 25-line subclass of `Monitor` that holds a `CILPRunner` and forwards `step` / `final_verdict` / `reset` to it; `compile` builds the runner from a formula string; `run` and `batch_run` come for free from the base class. The package's `__init__.py` re-exports `RuleRunnerMonitor` so experiments can `from src.monitors.rulerunner import RuleRunnerMonitor`.

**Smoke-test scope.** [tests/test_rulerunner_monitor.py](tests/test_rulerunner_monitor.py) only verifies the ABC plumbing (`issubclass(RuleRunnerMonitor, Monitor)`, `compile` returns an instance, `run` and `batch_run` work, `reset` actually resets). The deep correctness work lives in the engine and CILP sweeps; duplicating it here adds nothing.

**Subtle non-determinism caught after step 4 landed.** With three test files in play (engine sweep + CILP sweep + monitor smoke), the xfail-strict test for `G (a → X b)` started flipping to XPASS depending on test execution order. Root cause: the sweeps were seeding `np.random.default_rng` with `hash(formula) & 0xFFFFFFFF`. **Python's built-in `hash()` is randomised per interpreter session** (the `PYTHONHASHSEED` mechanism, designed against hash-collision DoS attacks), so each pytest invocation produced a different trace sample. The 1–4 traces (out of 80) where the nested-temporal limitation manifests sometimes fell outside the sample, and strict-xfail flagged that as a regression. **Fixed** by switching both sweeps to a stable MD5-based seed function `_stable_seed(formula)`. Verified across three independent runs: 155 passed / 6 xfailed every time. **Lesson**: never use `hash()` for cross-process reproducibility — either use a fixed integer or hash with `hashlib`.

### Fundamental limitation — nested temporal under F/G/U/R

The IJCNN 2014 encoding uses **one literal per subformula**. For a formula like `F(a & X b)`, F's reactivation creates a fresh `(a & X b)` instance at each cell, while X-b instances from prior cells are still resolving via mode A. Both instances share the literal `[X b]`:
- Mode A at cell N produces the cell-(N−1) X-b instance's resolution (definite T/F because b is atom).
- Mode B at cell N produces the cell-N fresh defer (`?^I`).

The binary operator's mode-R rules cannot tell which instance each `[X b]` literal belongs to and fire on **both**, corrupting the carry-over. Specifically, mode R reads `[X b]?^I` (cell-N's fresh defer) and treats it as "cell-(N−1) instance's X-b is still pending", indefinitely extending a wait that should have resolved.

A correct fix needs **cell-scoped literals** (e.g. `[X b @ now]` vs `[X b @ prev]`), which is a structural redesign that goes beyond what IJCNN 2014 documents.

**Decision.** Accept the limitation. Document it. Three formulas in the equivalence sweep are marked `xfail(strict=True)`:
- `F (a & X b)` — X nested under propositional under F
- `G (a -> F b)` — F nested under → under G (BPM response pattern)
- `G (a -> X b)` — X nested under → under G (chain response)

**Why this is fine for Paper A's experiments:**
- Exp 2 (IJCNN scalability family `◇(V(a_0 ∧ a_i))`) — all flat. Works correctly.
- Exp 3 (batched `ijcnn_n8`) — flat. Works correctly.
- Exp 1 uses `G(a → F b)` which is nested. **However** that formula was chosen *because* it has no trap/sink (early termination never fires), so the timing measurement is dominated by per-cell cost — which is well-defined regardless of whether RuleRunner's verdict is right.

**Why this is a *finding* for Paper A's framing.** The DFA-based monitor has no such limitation because its single canonical state machine doesn't conflate concurrent instances. The BPM response pattern `G(a → F b)` is the canonical example: it is structurally simple, semantically meaningful, widely cited — and **the rule-based encoding cannot represent it correctly without extension**. This directly supports the paper's thesis that the automata-based representation is the more general foundation.

### Open questions for the final review

1. **Should we attempt the cell-scoped fix?** It would unlock nested temporal but is a structural redesign of the rule system. Significant work, unclear that it converges in the Paper A timeline. Currently scoped out.
2. **Should the response_pattern `G(a → F b)` test be retained anywhere?** Currently removed from hand tests because it would silently pass on lucky traces. It is covered by the xfail sweep. If we ever fix the limitation, that xfail flips to PASS automatically (strict mode).
3. **CILP encoding (step 3) parity.** The CILP network needs to produce the same per-cell verdicts as the engine. If it diverges on flat-temporal formulas, that's a CILP bug. If it diverges on nested-temporal, that's the rule-system limitation transferring (expected).

### Test counts (post step 4)

- `test_compiler.py`: 9 ✅
- `test_symbolic_dfa.py`: 21 ✅
- `test_rulerunner_parse_tree.py`: 18 ✅
- `test_rulerunner_rules.py`: 23 ✅
- `test_rulerunner_engine.py`: 37 (34 ✅, 3 xfail-strict)
- `test_rulerunner_cilp.py`: 45 (42 ✅, 3 xfail-strict)
- `test_rulerunner_monitor.py`: 6 ✅
- `test_deep_dfa.py`: 115 ✅ (paradigm 3)
- **Total (whole repo): 270 passed, 6 xfailed** after paradigm 3. Stable across independent runs.

### Step 5 — Experiment integration (✅ done)

Uncommented `RuleRunnerMonitor` and added the import in all three experiment files: [experiments/exp1_single_trace.py](experiments/exp1_single_trace.py), [experiments/exp2_formula_complexity.py](experiments/exp2_formula_complexity.py), [experiments/exp3_batch_size.py](experiments/exp3_batch_size.py). Each now runs both `SymbolicDFAMonitor` and `RuleRunnerMonitor` in their `MONITORS` list. Smoke-tested with tiny config (5 traces × 50 cells × 1 repeat) — no crashes, both monitors produce timing data.

**Caveat for Exp 1 timing interpretation.** Exp 1's formula `G(a → F b)` is nested temporal, so RuleRunner's *verdicts* are wrong on some traces (see the limitation section above). But the formula was chosen precisely because it has no trap/sink — **no early termination ever fires**, so the per-cell cost is independent of verdict correctness. The timing measurement (`total_wall_time / (n_traces × trace_length)`) is fair. For the paper this is the right framing: "we measure per-cell cost on a formula where the rule encoding's correctness limitation does not affect the timing methodology, and separately document where the encoding diverges semantically." Exps 2 and 3 use the IJCNN scalability family (flat temporal) where the encoding is correct.

**Cross-trace batching (✅ done — was an open gap).** `CILPRunner.batch_run` now vectorises the trace axis: per cell it runs the `depth+1` eval passes and the reactivation pass as batched matmuls over the whole batch (the weight matrices broadcast natively), faithful to IJCNN 2014's matrix-matrix formulation. It carries a `device` (`"cpu"`/`"cuda"`) threaded through `RuleRunnerMonitor.compile` and the `time_monitor` harness; the experiments auto-select `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`. Per-trace early termination and end-of-trace resolution are reconstructed after the uniform batched pass (first decided cell within a trace's length wins, else EOT on its final-cell state), so `batch_run == [run(t) …]` exactly — verified on CPU and CUDA (`tests/test_rulerunner_batch.py`, 0/1040 mismatches incl. the nested-temporal divergences, which it reproduces identically).

This makes Exp 3 a *fair* comparison: vectorised RuleRunner vs vectorised DeepDFA on the same device, so the gap reflects the genuine architecture — RuleRunner's `depth+1` **within-step sequential** matmuls per cell vs DeepDFA's single matmul — not Python per-trace-loop overhead. RuleRunner benefits from cross-trace batching but stays bottlenecked by that within-step dependency, which is the clean story (DeepDFA is natively single-matmul-per-cell). The on-the-shelf `structured.py` (Fig. 5 variant) is intentionally left CPU/sequential — it's not used by the experiments.

### Remaining work

All five steps of paradigm 2 are done. The CILP→torch translation in step 3 also positions us for **Paper B's adaptation experiment** (deferred): the same network can be reused with `tanh` activation instead of `sign` to make it differentiable, and a learning loss on misclassified traces can adapt the weights.

## Paradigm 3 (DeepDFA) — implementation notes

This section is the **review document** for paradigm 3, mirroring the Paradigm 2 notes. Read it before reviewing [src/monitors/deep_dfa.py](src/monitors/deep_dfa.py).

### Source and the decision NOT to vendor

DeepDFA originates in the Umili & Capobianco line (ECAI 2024) and is used in the NeSy PPM paper (Mezini et al., `papers/IS__NeSyPPM.pdf`), Eq. 18. The reference implementation is [github.com/axelmezini/nesy-suffix-prediction-dfa](https://github.com/axelmezini/nesy-suffix-prediction-dfa), file `src/common/dfa.py` (~120 lines: `DeepDFA(nn.Module)` with `trans_prob (|Σ|,|Q|,|Q|)`, `accepting/rejecting` matrices, `forward`/`step`/`step_pi`/`simulate`).

**Decision: reimplement, do not submodule.** Rationale (same philosophy as RuleRunner — faithful reimplementation, not importing their tool):
- It is flat research code (no `pyproject`/`requirements`), tightly coupled to their DOT parser, mutual-exclusivity token handling, and EOT preprocessing; a submodule drags `sympy`/`pydot`/`networkx` + the whole training pipeline in for ~120 useful lines.
- **Representation mismatch (the important one).** Their DeepDFA assumes the **BPM mutual-exclusivity assumption**: exactly one atom (activity) true per step, so the alphabet is the *atoms themselves* (`n_actions = len(labels)+1`, the `+1` is EOT), and `valid_tokens_for_guard` sets one atom true and the rest false. Our benchmark family `◇ V(a_0 ∧ a_i)` **requires conjunctions** of simultaneously-true atoms, which is unsatisfiable under their encoding. DeepDFA must be the *canonical, exactly-correct* monitor in our comparison (it must match `SymbolicDFAMonitor` on every trace, including the nested-temporal formulas where RuleRunner diverges), so we cannot adopt their input encoding. Our alphabet is the full `2^|atoms|`.
- We already have DFA compilation ([compiler.py](src/formula/compiler.py)) with `trap_states`/`accepting_sinks` precomputed; theirs re-derives all of that differently (and adds EOT terminal states we don't need — we have `final_verdict()`).

The reusable idea (Eq. 18 forward + the verdict-matrix) is ~100 lines and is reimplemented against our `DFA` and `Monitor`. **When we do Paper B's adaptation PoC**, the relevant reuse becomes their `loss/global_loss.py` (GLL) + Gumbel-Softmax sampling — revisit then.

### The alphabet-blowup finding (a *finding*, like RuleRunner's nested-temporal limit)

For non-mutually-exclusive propositional LTLf, the DeepDFA transition tensor is indexed by `2^|atoms|` truth assignments. The IJCNN family's guards (`a_0 ∧ (a_1 ∨ … ∨ a_{n-1})`) depend on **all n atoms**, so there is no cheap symbol set: dense is `2^n`, and even a factored-by-support scheme would enumerate `2^n`. The only sub-exponential evaluation, `p_{a0}·(1−∏(1−p_{a_i}))`, needs the guard's **read-once circuit structure**, which a flat DFA doesn't hand you.

This is DeepDFA's structural scaling weakness, **dual to RuleRunner's nested-temporal limit and the symbolic DFA's state blowup** — each paradigm has a different Achilles heel, which is a clean three-way story. (The NeSy PPM paper sidesteps this only because the BPM mutual-exclusivity assumption makes `|Σ| = #activities`. That assumption is false for our benchmark.)

### Two representations (both implemented; `mode=` on `DeepDFAMonitor.compile`)

| mode | tensor | per-step cost | use |
|---|---|---|---|
| `dense` (default) | `T (|Q|, 2^|AP|, |Q|)` one-hot | one matmul / `bmm` | small `|AP|`; the **batching showcase** (Exp 3, ijcnn_n8 → 256 symbols) |
| `factored` | none materialized | vectorized cube-mask reduction (crisp) / per-edge guard-prob closure (differentiable) | large `|AP|` (Exp 2, n up to 32); the **differentiable** soft path |

**Factored details.** The factored mode has two complementary views of each edge guard (a MONA label like `a & (b | c)`):

- **Crisp monitoring path (Phase 0.2 — the path Exp 1–3 time).** Each guard is decomposed *once* at construction into a **disjoint (orthogonal) cube cover** by Shannon expansion (`_guard_cubes`/`_shannon_cubes`), stored as `require-true`/`require-false` integer masks over the atoms. `crisp_matrix(p)` then builds the `(…,|Q|,|Q|)` transition matrix as a single vectorized reduction `cube = ∏_a [1 − rt·(1−p) − rf·p]` (= `p` for a require-true atom, `1−p` for require-false, `1` for don't-care) scattered into the matrix. No per-cell sympy closures → per-cell cost is a couple of batched tensor ops, **flat in |AP|** (vs the old per-cell closure walk that grew with formula size). Exact for crisp 0/1 inputs: each cube is 0/1 and the cubes are mutually exclusive, so they sum to a 0/1 transition; rows sum to 1 because a state's out-guards partition the assignment space.
- **Differentiable soft path (`soft_matrix`, kept separate for the deferred adaptation PoC).** Each guard is compiled to a torch closure computing satisfaction probability over the boolean tree assuming atom independence: `P(a)=p_a`, `P(¬φ)=1−P(φ)`, `P(φ∧ψ)=P(φ)P(ψ)`, `P(φ∨ψ)=1−(1−P(φ))(1−P(ψ))`. **Crisp 0/1 inputs → exact for *any* guard.** **Fractional inputs → exact only for read-once guards** (the IJCNN guard is read-once after MONA's factoring; non-read-once would be approximate — the Phase 3.3 finding). This path is differentiable in `p` and is deliberately left recursive so the read-once semantics are unchanged; the crisp monitor does not use it.

(For read-once guards the two views coincide on fractional inputs too — verified in the tests — since the orthogonal-cube sum equals the recursive read-once probability there.)

### Monitor mechanics

- `step(obs)`: `q' = q @ T[:,σ,:]` (dense) or `q @ crisp_matrix(prob_vector(obs))` (factored), then read the three-valued verdict off the precomputed `trap_idx` / `sink_idx` (SATISFY/VIOLATE absorbing) — same early-termination semantics as `SymbolicDFAMonitor`. `final_verdict` = accepting membership of `argmax(q)`.
- `batch_run` **overrides** the base: encodes the whole batch once (`encode_presence`, vectorized numpy) and does **one `bmm` per cell** across all traces (dense: gather `T[:,σ_b,:]`; factored: batched `crisp_matrix`). This is the GPU-batching path Exp 3 measures. Per-trace early termination / end-of-trace is replayed from the recorded state path so `batch_run` matches `[run(t) …]` exactly. `device="cuda"` is supported via `compile(..., device="cuda")`.

### Correctness

`tests/test_deep_dfa.py` (121 tests): DeepDFA matches `SymbolicDFAMonitor` on the **full sweep including nested temporal — no xfails** (DeepDFA is exact where RuleRunner is not); dense == factored on crisp traces; `batch_run == [run(t) …]` in both modes; `soft_matrix` is row-stochastic and exact on the read-once IJCNN guard; `crisp_matrix` is row-stochastic, 0/1 on crisp input, equals `soft_matrix` on read-once guards (crisp *and* fractional), and batched-matches-unbatched; factored handles `n=24` atoms with no `2^24` tensor.

### Performance caveat (resolved — Phase 0.2)

Factored monitoring *used to* rebuild a `soft_matrix` from per-cell Python closures (~1.8 ms/cell at n=32). The crisp path is now the vectorized `crisp_matrix` over precomputed orthogonal-cube require-true/require-false masks (built once at construction), so per-cell cost is flat in |AP| (~7e-6 s/cell at n=32, ~3.7× over n=2→32 = the genuine O(n²) mask reduction). The recursive `soft_matrix` remains only for the differentiable fractional path (Paper B); it is no longer on the monitoring hot path.

### Experiment integration

`DeepDFAMonitor` is in the `MONITORS` list of all three experiments. exp1 (`G(a→F b)`, 2 atoms) and exp3 (ijcnn_n8, 256 symbols) use **dense**; exp2 (atoms up to 32) runs **both** a `DeepDFAMonitorFactored` subclass (all n, vectorized crisp path — flat) and a `DeepDFAMonitorDense` subclass capped at `DENSE_MAX_LEAVES=16` (skipped beyond — 2^32 would OOM), plus an analytic **memory-wall panel** (dense `|Q|²·2^|AP|` vs factored mask bytes) showing the alphabet-blowup finding (Phase 0.2).

## Benchmark Design

**Use synthetic traces for Exps 1–3.** The trace content is irrelevant to per-step monitoring cost — what matters is trace length and formula complexity. Using real data would conflate paradigm speed with early-termination frequency (which is data-dependent), making the comparison less clean. IJCNN 2014 also uses randomly generated traces; this is the right methodology, not a compromise.

**Reproduce and extend IJCNN 2014.** That paper compares only RuleRunner variants (base/sparse/gpu) — the symbolic DFA and DeepDFA are absent. Adding them is our direct contribution. Use the same formula family and same leaf counts so the paper is directly legible to anyone who knows IJCNN 2014.

**IJCNN 2014 formula family for scalability:** `◇ V_{i=1}^{n-1}(a_0 ∧ a_i)` with n = 2, 4, 8, 16, 32 leaves (atoms renamed alphabetically). A leaf = a single propositional atom (the terminal alphabet). This is the x-axis for Exp 2/3.

**Two kinds of parallelism — important to keep distinct.**

| Kind | What runs in parallel | Where it shows up |
|---|---|---|
| **Within-step** | evaluation rules (RuleRunner) / matmul atoms (DeepDFA) fire simultaneously within a single cell, rather than sequentially | Exp 2 (formula complexity): per-cell cost grows with parse-tree depth for RuleRunner (convergence loop runs `depth+1` passes), stays flat for DeepDFA (always one matmul); IJCNN 2014's primary contribution claim |
| **Cross-trace** | multiple traces batched as matrix rows, turning matrix-vector into matrix-matrix products | Exp 3 (batch size): both DeepDFA and RuleRunner now batch this axis on CPU/CUDA (`batch_run`). DeepDFA's advantage is that it pays one matmul/cell; RuleRunner benefits from cross-trace batching but stays bottlenecked by its `depth+1` within-step sequential passes |

IJCNN 2014 uses both, but their headline claim is within-step rule parallelism. Our Exp 2 captures this implicitly (per-cell time vs. formula size), and Exp 3 isolates cross-trace batching. They are separate axes and should be framed distinctly in the paper.

**Possible additional experiment (noted for discussion — not currently scoped):**

A micro-benchmark that isolates within-step parallelism explicitly, independent of formula size:
- Fix formula complexity (n=8 leaves, ijcnn_n8), vary **parse-tree depth** by wrapping in nested `X`: `ijcnn_n8`, `X(ijcnn_n8)`, `X(X(ijcnn_n8))`, … up to depth ~10.
- Fix batch size = 1, trace length = 1 (single cell), report per-cell time vs. depth.
- Expected curves: Symbolic DFA flat (one state lookup, depth doesn't matter); RuleRunner grows linearly with depth (convergence loop runs `depth+1` passes); DeepDFA flat (one matmul regardless of depth, since depth is absorbed into the DFA state count, not into a loop).
- This directly separates the within-step depth-cost claim from Exp 2's breadth (leaves) scaling. It would be the cleanest empirical support for the "one matmul per cell" framing.
- Scope assessment: straightforward to implement (add `X` wrappers in the formula list, reuse `time_monitor` with n_traces=1, trace_length=1). Main question is whether a 10-page paper has room for a fourth timing figure. Could be one subplot in a compound figure alongside Exp 1/2.

**Experiments (see § Research Plan for phases/status):**

| Exp | X-axis | Formula | Kind | Expected story |
|---|---|---|---|---|
| 1: trace length | 1k–10k cells | `G(a → F b)` (no trap/sink) | timing | All paradigms flat — per-step cost is constant |
| 2: formula complexity | n=2,4,8,16,32 leaves | IJCNN family (early-term **off**, Phase 0.1) | timing / within-step | Symbolic flat; RuleRunner linear in depth; DeepDFA dense hits 2^n wall, factored flat once vectorized |
| 3: batch size | 1–1024 parallel traces | `ijcnn_n8` (early-term **off**) | timing / cross-trace | Lead with absolute time; whether batched DeepDFA wins is an *honest open question* (re-run on server GPU, Phase 0.4) |
| A: perceptual uncertainty | noise level ε | response/IJCNN | **capability** | Soft paradigms degrade gracefully + emit calibrated confidence; symbolic is brittle (Phase 1) |
| 4: adaptation PoC | training steps | wrong→correct spec | **capability** | Differentiable monitors recover from data; symbolic cannot (Phase 2) |

`G(a → F b)` is used for Exp 1 because it has no trap or accepting sink, so early termination never fires. This isolates pure per-step cost from early-termination frequency. `G a` does have a trap state (fires the moment `a` is false), so on random traces early termination would dominate and obscure the per-cell cost signal.

**Timing methodology:** `total_wall_time / (n_traces × trace_length)` — divides by total potential cells, not actual cells processed. Follows IJCNN 2014. Early termination advantage is captured naturally: a paradigm that terminates early does less work and earns a lower per-cell cost.

**⚠ Early-termination confound (first-pass finding — Phase 0.1 fixes this).** The methodology above conflates two things when used to compare *per-cell compute* across paradigms. On the IJCNN `◇(⋁(a₀∧aᵢ))` family, the formula early-terminates almost immediately on random traces, so the **crisp** monitors (Symbolic) process ~2 cells and divide by thousands of potential cells, reporting ~1e-10 s/cell — physically impossible as real per-cell compute (one Python dict lookup is ~5e-8 s). Meanwhile the **batched** neural monitors (`batch_run`) process *all* cells uniformly before replaying early termination. So Exp 2/3 as originally run do not compare the same workload — they compare "how fast a paradigm gives up" against "full batched pass." For the per-cell-cost and within-step figures, disable early termination (or use a non-terminating family) so all paradigms process all cells; reserve the early-termination reward for a *separate* data-dependent experiment if wanted. This does not affect Exp 1 (`G(a→Fb)` has no trap/sink, never early-terminates — which is exactly why it was chosen).

**Extending experiments to new paradigms:** each script has a `MONITORS` list at the top. Uncomment `RuleRunnerMonitor` / `DeepDFAMonitor` once those are implemented — no other changes needed.

**Real dataset for Exp 4 only.** The adaptation PoC needs realistic trace distributions for the learning experiment to be meaningful. Use a **BPI Challenge log** (BPIC 2012 or BPIC 2017 — both standard in process mining, freely available). Exps 1–3 are purely synthetic.

## Key Papers in `papers/`

- `Claude 1.txt` — full research planning document, primary reference for motivation and framing
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