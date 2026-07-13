# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on **Neuro-Symbolic Runtime Monitoring** combining LTLf (Linear Temporal Logic over finite traces) with differentiable automata. The goal is an **ICLR submission** (venue decided with the supervisors, 2026-07): a *foundation* paper for neuro-symbolic LTLf monitoring — **modernizing and fixing RuleRunner** (the nested-temporal limitation and its progression-based repair) and **connecting it with the automata-based paradigms** (symbolic DFA, DeepDFA), with a strong crisp empirical section characterizing the efficiency landscape honestly.

**Scope decision (2026-07-13, supervisors' call): adaptation and probabilistic monitoring are OUT of this paper** — they are future work. Everything belonging to those threads (the uncertainty/calibration harness and experiments, the probabilistic-verdict theory section, the adaptation plan) lives in **[artur_future_work/](artur_future_work/)**, a self-contained fork with its own CLAUDE.md, ready to be extracted into its own repository. Do not re-grow those threads here; the paper mentions soft inputs and differentiability only as *affordances* motivating the paradigms (one paragraph, `latex/4_deepdfa.tex` §4.2) and defers the rest.

This is an active research project — plans, experiments, and framing should be treated as working hypotheses, not fixed requirements. Expect iteration.

## The Core Idea

Three paradigms for LTLf runtime monitoring, compared theoretically and experimentally:

| Paradigm | How it works | Key property |
|---|---|---|
| **Symbolic DFA** | Compile LTLf → minimal DFA; track state explicitly | Fastest crisp single-trace; frozen; crisp boolean inputs only |
| **RuleRunner** | Formula parse tree → extended truth tables → Horn clauses → CILP neural net | Learning-friendly syntactic locality; nested-temporal representational limit (**wrong verdicts** — our counterexample + repair is a core contribution); within-step sequential |
| **DeepDFA** | Compile LTLf → minimal DFA → differentiable transition matrix | Native GPU batching; differentiable; alphabet (2^|AP|) blowup |

### Framing — read before touching experiments or the paper narrative

- **Do NOT anchor the paper on speed.** For crisp boolean monitoring the symbolic DFA is the theoretical optimum — a DFA walk is a dict lookup; nothing differentiable beats it. Symbolic dominating throughput is *expected*, not a threat. The experiments map **where each paradigm's cost grows and where it walls out** — an honest, complete efficiency landscape.
- **Keep paradigms neutral.** Present a capability matrix and let it speak; do **not** pre-crown DeepDFA. Each paradigm has a distinct Achilles heel (symbolic = state blowup; RuleRunner = nested-temporal limit + within-step sequential cost, repaired at an alphabet cost; DeepDFA = 2^|AP| alphabet blowup + |Q|² step) — that balance is the honest three-way story.
- **Still try for an honest speed win.** If a genuine batched-throughput advantage survives fair measurement (Colab GPU, not the 4 GB laptop), it's a bonus — reported neutrally. Exp 3/6 test this; either outcome is honest ("modest real win" or "GPU advantage needs larger automata/hardware").
- **The ICLR challenge:** timing alone is thin for a pure-ML venue. The roadmap (Phases 2–4) lists the candidate additions, easiest → hardest; each gets scoped in a dedicated session before any code is written.

## Research Plan — phases and steps

> **Navigation:** [docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md) is the chart of the experiment/comparison space — what each experiment measures, the dense/factored & CILP/structured variants, the hardware comparison, and the remaining free axes. Read it before adding any experiment.

Status legend: ✅ done · 🟡 in progress · 🔲 not started.

### Phase 0 — Restructure for the ICLR frame ✅ (2026-07-13)

Split out the future-work threads into `artur_future_work/` (self-contained: full `src`/`tests` copies, the uncertainty harness + experiments + results, latex 5b + trimmed prose, own deps/CLAUDE.md). Removed Docker entirely (there is none in this project — the sweeps run on **Google Colab**). Moved the demo into `demo/`. Trimmed the paper's soft/adaptation material to affordance paragraphs. Fork point: commit `57d74e3`.

### Phase 1 — Efficiency landscape 🟡 (all *code* done; Colab re-runs + figure polish pending)

Complete the fair timing comparison across all monitor variants and produce the paper's figure set. The measurement-hygiene mechanisms are implemented and verified; what remains is **re-running on Colab (CPU + T4 GPU runtimes)** under the early-termination-off + CUDA-sync mode and polishing walls/crossovers into lead figures.

Key mechanisms already in place (details preserved here because they explain *why the numbers look the way they do*):

- **Early-termination confound killed (exp2/3).** The IJCNN `◇(⋁(a₀∧aᵢ))` family early-terminates almost instantly on random traces, so symbolic's old ~1e-10 s/cell measured "how fast it gives up" while batched neural monitors process *all* cells — not apples-to-apples. `Monitor.run/batch_run` take `early_termination: bool` ([base.py](src/monitors/base.py)); when False the crisp walk processes all cells (absorbing states are sticky — verdicts unchanged, verified in [tests/test_early_termination.py](tests/test_early_termination.py)). `reset_if_stale()` drops CSVs measured in the other mode. Confirmed effect: symbolic per-cell on `ijcnn_n8` jumps ~3e-9 → ~3e-7 s (real dict lookup).
- **DeepDFA factored path vectorized (exp2 dual finding).** Each guard is decomposed **once** at construction into a disjoint cube cover by Shannon expansion (`_guard_cubes`/`_shannon_cubes` in [deep_dfa.py](src/monitors/deep_dfa.py)), stored as require-true/require-false integer masks; `crisp_matrix(p)` is a single vectorized mask reduction — no per-cell sympy closures. Factored per-cell at n=32: ~2e-4 → ~7e-6 s; growth n=2→32: ~24× → ~3.7× (residual = genuine O(n²) mask reduction). Dense (`DeepDFAMonitorDense`, capped at `DENSE_MAX_LEAVES=16`) is fastest where it fits but walls out (2^32 ≈ 64 GB at |Q|=2); exp2's analytic **memory-wall panel** shows it without building the tensors.
- **CUDA timing hygiene (exp3).** `time_monitor` syncs CUDA once per timed repeat (after warm-up and after each full `batch_run`), never inside the per-cell loop ([runner.py](src/benchmarks/runner.py)); DeepDFA's `batch_run` stays on-device and reads verdicts once at the end. Exp3 leads with **absolute time-per-trace**; the speedup panel is annotated (each curve normalized to its own batch=1 — cross-monitor speedups are misleading).
- **Truthful device labeling (was a real data bug).** Every monitor exposes `effective_device` (Symbolic and the original structured RuleRunner are pure-Python CPU walks and stamp `cpu` even under `device="cuda"`); `time_monitor` stamps that and syncs CUDA only for monitors that truly use it. A CSV never claims a GPU run that did not happen. Resume does not key on device — keep one CSV per machine (`results/cpu/`, `results/gpu/`), merged by the plotters.
- **Overhead decomposition (why symbolic wins).** Exp 3 batch=1 vs 1024 shows **~83 µs fixed per-call overhead per cell vs ~1.5 µs actual compute** — overhead, not arithmetic, is what loses. Two implemented levers test whether anything survives:
  - **Larger automata — [exp6_state_scaling.py](experiments/exp6_state_scaling.py)** (`STATE_SCALING_SUITE`, bounded response `G(a → (b ∨ Xb ∨ … ∨ Xᵏb))`, |Q| linear in k, |AP|=2): symbolic per-cell is flat in |Q|; DeepDFA's O(|Q|²) step finally amortizes launch overhead. **A crossover is plausible but unproven — needs the Colab GPU run.**
  - **Parallel prefix scan — `DeepDFAMonitorScan`** ([deep_dfa.py](src/monitors/deep_dfa.py)): the crisp state path is a prefix product of per-cell matrices → Hillis–Steele scan, O(log L) big matmuls instead of L small ones. **Honest caveat (measured):** not a FLOP reduction (×|Q|·log L arithmetic) — wins only where launch overhead ≫ arithmetic (GPU + small |Q| + long traces), loses on CPU/large |Q|. Falls back past `SCAN_MEM_LIMIT_BYTES`. Verdict-identical to sequential ([test_deep_dfa_scan.py](tests/test_deep_dfa_scan.py)). In exp1 + exp3.
- **State blowup, exponential family — [exp7_state_blowup.py](experiments/exp7_state_blowup.py)** (`STATE_BLOWUP_SUITE`, `F(a & Xᵏb)`, |Q| = 2ᵏ+1, |AP|=2): the **shared-weakness finding** — symbolic per-cell flat (~0.3 µs) while DeepDFA rises O(|Q|²) to ~40 µs at |Q|=1025; analytic memory wall crosses 4 GB at k≈14 for DeepDFA vs k≈28 for symbolic's linear table. Completes the honest three-heel table ([docs/richer_benchmark_findings.md](docs/richer_benchmark_findings.md)).
- **Within-step depth micro-benchmark — [exp5_depth_microbench.py](experiments/exp5_depth_microbench.py):** nested-X depth 0..10 over `ijcnn_n8`, batch=1, trace length 500 (a single cell is buried in per-call overhead — verified). RuleRunner rises ~140→168 µs with depth; Symbolic ~0.24 µs and DeepDFA ~15 µs stay flat.
- **Cost of correctness (the paradigm-2 paper number).** `plots.correctness_cost_table`/`plot_correctness_cost`: corrected(progression)/original per-cell-time ratio on exp2's flat IJCNN family (where the original RR is also correct, so the ratio isolates the encoding's throughput cost, not the verdict fix).

**Exit criterion:** regenerated `results/cpu|gpu` for exp1/2/3/5/6/7 under the new mode on Colab; a clear verdict on whether any speed advantage survives (exp3 batching, exp6 crossover, scan); the paper's lead figures chosen.

### Phase 2 — Real-log case study 🔲 (scope in a dedicated session first)

Declare constraints on a BPIC log (2012/2017 — standard, freely available), **crisp**, at scale: realism/legitimacy evidence that the paradigms and the timing story hold on real traces, not just synthetic ones. The `DECLARE_SUITE` (7 templates, diverse trap/sink structure) is the constraint vocabulary. Cheap-ish; reviewers at ICLR may value it less than Phase 3 — decide the investment after Phase 1's results.

### Phase 3 — Neural sequence baselines 🔲 (scope in a dedicated session first)

The most ICLR-shaped addition: train RNN/Transformer monitors on labeled traces and compare against the three exact paradigms on **verdict accuracy, length generalization, and sample efficiency**. Supports the paper's motivation directly (pure neural sequence models lack a mechanism to adhere to the logic; the NeSy encodings are exact *by construction*). The intro already gestures at RNNs/Transformers; `8_conclusion.tex` has a Transformers stub. New code: training loop + trace/label generation (the symbolic oracle labels for free).

### Phase 4 — Decision-diagram transition representation 🔲 (hardest; the ambition-raiser; scope first)

BDD/SDD-compiled guard circuits as the principled upgrade of the factored cube cover — **crisp/scalability side only** in this repo (representation size vs dense `2^|Σ|` vs cube count; batched compiled-circuit throughput). Design note: [docs/decision_diagram_transition_representation.md](docs/decision_diagram_transition_representation.md) (read its scope note: the calibration/WMC headline belongs to `artur_future_work/`). First steps if pursued: the NeSyA novelty check, and whether MONA's internal MTBDD can be extracted directly.

### Phase 5 — Writing 🟡 (trails experiments; LaTeX in `latex/`)

- Port to the **ICLR template** (currently plain `article`).
- Sec 1 intro: thesis = "a foundation for neuro-symbolic LTLf monitoring: fix RuleRunner, connect with automata-based approaches, characterize the landscape honestly." Resolve the red TODOs (incl. "other NeSy approaches" and the results summary).
- Sec 3 RuleRunner: architecture + the nested-temporal counterexample + progression repair with soundness/completeness proof — **drafted**; polish.
- Sec 4 DeepDFA: architecture, affordance paragraph (§4.2 — keep short, points to future work), alphabet blowup, factored representation — **drafted**; polish.
- Sec 5 theory comparison: three Achilles heels; **capability matrix TODO**; cite Bacchus–Kabanza + the LTLf 2EXP bound.
- Sec 6 experiments: rewrite around Phase 1's regenerated figures; state early-termination handling and hardware explicitly.
- Sec 7 related work: **empty stub** — needs writing (RuleRunner line, DeepDFA/NeSyA/T-ILR line, LTLf monitoring, RV tools).
- Sec 8 conclusion: future-work stubs (Transformers / Specification Adaptation / Process Model Repair) + the inert decision-diagram signpost (`\iffalse`-guarded).
- Appendix candidates: [docs/appendix_ideas.md](docs/appendix_ideas.md).

### Out of scope (→ `artur_future_work/`)

Probabilistic monitoring (uncertainty/calibration experiments, the three-verdicts theory, non-read-once findings), specification adaptation (synthetic PoC, RuleRunner tanh route, BPIC adaptation), end-to-end perceptor training, and the calibration side of the decision-diagram monitor. The fork's CLAUDE.md records the established findings and a continuation plan per thread.

## Repository Structure

```
nesy_runtime_monitoring/
├── src/
│   ├── formula/compiler.py    ✅ LTLf → minimal DFA (wraps ltlf2dfa; trap/sink precomputation)
│   ├── monitors/
│   │   ├── base.py            ✅ Abstract Monitor interface + Verdict enum (+ early_termination flag)
│   │   ├── symbolic_dfa.py    ✅ Paradigm 1 — crisp DFA walk
│   │   ├── rulerunner/        ✅ Paradigm 2 — original encoding (parse_tree, rules, engine, cilp, monitor, structured)
│   │   ├── progression/       ✅ Paradigm 2 CORRECTED — progression-based RuleRunner (formula, progression, engine, eager, flat, structured)
│   │   └── deep_dfa.py        ✅ Paradigm 3 — DeepDFA (dense + factored + scan; soft path kept as the affordance, unexercised here)
│   └── benchmarks/
│       ├── formulas.py        ✅ Formula registry (IJCNN, trace-length, Declare, non-read-once, state-scaling, state-blowup suites)
│       └── runner.py          ✅ Timing harness (time_monitor, random_traces, resumable CSVs, effective-device stamping)
├── experiments/
│   ├── exp1_single_trace.py   ✅ Per-cell cost vs trace length (G(a→Fb), no trap/sink)
│   ├── exp2_formula_complexity.py ✅ Per-cell cost vs formula breadth + memory-wall panel
│   ├── exp3_batch_size.py     ✅ Time per trace vs batch size (1–1024)
│   ├── exp5_depth_microbench.py ✅ Per-cell cost vs nested-X depth
│   ├── exp6_state_scaling.py  ✅ Per-cell cost vs |Q| (linear family — crossover probe)
│   ├── exp7_state_blowup.py   ✅ Per-cell cost + memory wall vs |Q| = 2ᵏ+1 (exponential family)
│   ├── plots.py               ✅ All plotting, decoupled from runs (CSV→PNG; device overlays)
│   └── make_all_plots.py      ✅ The merged / gpu_only / device figure sets from results/cpu + results/gpu
├── tests/                     ✅ Full suite; 6 xfail-strict document the ORIGINAL RuleRunner's nested-temporal limit (by design — do not "fix")
├── results/                   ✅ cpu/ + gpu/ CSVs, figures/ PNGs (see results/README.md)
├── latex/                     🟡 The paper (main.tex + sections 1–8; dfa_script.py + gen_dfa_figs.sh for DFA figures)
├── demo/                      ✅ demo_monitors.py + demo_output/ — presentation aid (DFA rendering + RuleRunner run tables)
├── docs/                      ✅ EXPERIMENT_MAP, nested_temporal_limitation, rulerunner_progression_analysis, richer_benchmark_findings, decision_diagram note, appendix_ideas
├── scripts/run_all.sh         ✅ Run every timing experiment in sequence (resumable)
├── NeSy_Runtime_Monitoring.ipynb ✅ Colab entry point (CPU + GPU runtimes)
├── artur_future_work/         ✅ Self-contained future-work fork (probabilistic monitoring + adaptation) — own CLAUDE.md
└── papers/                    Reference papers and planning notes (gitignored)
```

## Abstract Monitor Interface

All monitors implement the same interface (`monitors/base.py`):

```python
compile(formula: str) -> MonitorInstance   # classmethod
step(obs: dict[str, bool]) -> Verdict      # SATISFY / VIOLATE / UNDECIDED
final_verdict() -> Verdict                 # binary end-of-trace check; never UNDECIDED
reset() -> None
run(trace, early_termination=True) -> Verdict
batch_run(traces, early_termination=True) -> list[Verdict]   # DeepDFA AND RuleRunner override for batched CPU/GPU
```

Three-valued semantics (`UNDECIDED`) applies online — a trace mid-execution may not yet have a determined verdict. Absorbing states (all successors accepting, or all rejecting) enable early termination; with `early_termination=False` all cells are processed (verdicts unchanged — absorbing states are sticky).

`final_verdict()` is a required separate method because response-style formulas like `G(a → F b)` have neither a trap state nor an accepting sink, so `step()` always returns `UNDECIDED`. The verdict is only binary at end-of-trace.

## Key Technical Details

**LTLf → DFA compilation:** use `ltlf2dfa` (Python wrapper for MONA). `to_dfa()` returns a DOT string with transitions labeled by boolean expressions over atoms (`~a`, `a & ~b`, `b | ~a`, `true`). The compiler parses this DOT, converts MONA guard syntax to Python, and compiles each guard to a bytecode object once at construction time (compile-once, eval-many). The `DFA` dataclass exposes `states`, `atoms`, `initial`, `accepting`, `transitions`, `trap_states`, `accepting_sinks`, and a `step(state, obs) -> state` method.

**Trap states and accepting sinks** are precomputed by graph reachability at DFA construction time. A trap is any state from which no accepting state is reachable; an accepting sink is any state from which all reachable states are accepting. Per-step verdict checks cost a single `set` membership test.

**`ltlf2dfa` quirk:** `lark` (a transitive dependency) emits two `DeprecationWarning`s about `sre_parse`/`sre_constants`. Harmless; suppressed by pytest.

**DeepDFA transition tensor:** shape `(|Q|, |Σ|, |Q|)` where `T[q, σ, q']` = 1 iff state `q` transitions to `q'` on symbol `σ`. In our (non-mutually-exclusive) propositional setting `|Σ| = 2^|atoms|`. See § Paradigm 3 for dense vs factored and the alphabet-blowup finding.

**RuleRunner CILP encoding:** each Horn clause becomes a hidden unit; weights `+1`/`−1` per literal polarity; threshold set so the unit fires iff all positive body literals are true and no negative ones. Forward pass = one chaining iteration; repeat to fixpoint (bounded by formula depth). The convergence loop is the source of RuleRunner's sequential bottleneck — preserve it faithfully.

**Benchmark formulas** come from the IJCNN 2014 tables (`◇a`, `□(a∨b∨c∨d)`, `◇((a∧Xb)∨(c∧Nd))`, scaled atom counts) and Declare/BPM constraint patterns, plus our state-scaling (`bounded_response`) and state-blowup (`kth_from_last`) families.

## Paradigm 2 (RuleRunner) — implementation notes

This section is the **review document** for paradigm 2: every design decision, bug, limitation, and assumption surfaced while implementing it. Read this before reviewing the code under [src/monitors/rulerunner/](src/monitors/rulerunner/).

### Pipeline at a glance

```
LTLf formula string
  └─> parse_tree.parse()         (step 1)
        └─> Node DAG (subformula sharing, depth precomputed)
              └─> rules.build_rules()  (step 2)
                    └─> RuleSystem (eval rules + react rules + initial state)
                          ├─> engine.RuleEngine  (step 2.5) — symbolic executor
                          └─> cilp.CILPNet       (step 3) — torch network
                                └─> monitor.RuleRunnerMonitor  (step 4) — Monitor wrapper
```

Each layer is independently testable. `engine.py` is the oracle: it executes the rule system in pure Python and is checked against `SymbolicDFAMonitor` on randomized formulas + traces. The CILP network produces the same per-cell verdicts as the engine (equivalence tested).

### Step 1 — parse_tree.py

Frozen-dataclass `Node` DAG with `(op, children, key, depth, atom)`. `key` = canonical syntactic identity (`str(ast)` of ltlf2dfa's AST), the name suffix for every R[.] / [.]V literal downstream. **Reuses ltlf2dfa's parser** but walks its AST into our own type. **Binarizes n-ary operators** (left-fold `&`/`|`, right-fold `->`/`U`/`R` — U/R right-association verified against `to_mona()`). **Subformula sharing:** repeated keys hit a cache and yield the same `Node` instance. `depth` precomputed (atoms 0; parent = 1 + max child). Atoms can never be undecided (enforced here; downstream pruning relies on it).

### Step 2 — rules.py

`Literal(name, negated)`, `Rule(body, head)`, `RuleSystem(eval_rules, react_rules, initial_state, atoms, root_key)`. Literal naming is **string-based** with modes baked in as suffixes (e.g. `R[(a | F(b))]^B`) — matches IJCNN 2014's notation; CILP gets one neuron per mode-distinct literal.

Per-operator templates: ATOM (no react rules); NOT (no reinstall); AND/OR/IMPLIES (modes B/L/R, no reinstall — a child's own `?` reactivation handles its subtree); EVENTUALLY/ALWAYS/UNTIL/RELEASE (reinstall operand subtree(s) — their operand's own reactivation doesn't fire when the operator is `?` because the operand was definite); NEXT/WEAK_NEXT (modes B/A, reinstall on I→A only).

**Mode-tracking (AND/OR/IMPLIES):** mode B sees one child settled → transitions to L/R, dropping the settled child from monitoring (essential: without it `a ∨ ◇b` re-evaluates `a` every cell → wrong verdicts). L/R truth tables derive from B's column at the **pin value** (AND pins T; OR/IMPLIES pin F). ⚠ Initial bug: first draft pulled L/R from B's column at ψ=? instead of ψ=pin; fixed in [rules.py:152](src/monitors/rulerunner/rules.py#L152). **Assumption to revisit:** the pin derivation assumes each binary mode-B table has exactly one `(?, V) → ?L` and one `(V, ?) → ?R` cell (true for AND/OR/IMPLIES).

Atoms can never be `?` → any rule with `[a]?` in the body is pruned at template-instantiation time. Templates deduplicate `(body, head)` pairs. The IJCNN 2014 §III worked example (`a ∨ ◇b`) is reproduced exactly in [tests/test_rulerunner_rules.py](tests/test_rulerunner_rules.py).

### Step 2.5 — engine.py

Per cell: (1) inject `obs:a` literals (negation-as-failure for `~obs:a`); (2) evaluation phase — fire eval rules for `depth+1` passes (truth propagates one level per pass), break early on no new facts; (3) read root verdict; (4) if decided, freeze (absorbing); else fire react rules once in parallel, keep only `R[.]` literals.

**End-of-trace resolution:** recursively resolve undecided subformulae per operator's end semantics. `F φ`/`G φ` = **recurse on child** (not unconditional F/T — `G(F b)` on `[F, F]` must be F); `U`/`R` = recurse on ψ; `X` = F, `WX` = T (FLTL strong/weak); binary L/R modes = resolve the active child, pin the settled one.

### Step 3 — cilp.py

Standard Garcez & Zaverucha 1999 translation. Each rule = one hidden unit; body literals connect with `±W`; hidden bias `-W*(n-0.5)`; output = OR of incoming hiddens (bias `W*(k-1)`; k=0 outputs get negative bias). Sign activation. Eval and react phases have separate weight matrices over one shared literal-index space. `step()`: build x (carried R[.] + clamped obs) → `depth+1` sign-forward passes OR-accumulated (`x = max(x, y)`) → read root → react pass if undecided. End-of-trace resolution shared with the engine (parameterized by an `in_state` predicate).

**Equivalence: 0/N mismatches vs the engine across the whole flat-temporal sweep** ([tests/test_rulerunner_cilp.py](tests/test_rulerunner_cilp.py)). Knobs: `_W = 1.0` (any positive works with sign; tanh would need Garcez–Zaverucha `Amin` bias recomputation — that route is future work, see `artur_future_work/`). **Batching:** `CILPRunner.batch_run` vectorizes the trace axis (batched matmuls per cell, `device="cpu"/"cuda"`), bit-for-bit equal to sequential `run()` ([tests/test_rulerunner_batch.py](tests/test_rulerunner_batch.py)).

### Step 4 — monitor.py

Thin Monitor-ABC adapter (`RuleRunnerMonitor` holds a `CILPRunner`). Smoke tests only ([tests/test_rulerunner_monitor.py](tests/test_rulerunner_monitor.py)) — correctness lives in the engine/CILP sweeps.

**Test-design lessons (keep):** (1) an xfail-strict sweep must use enough traces to deterministically hit the expected failure, or it XPASSes flakily (the 80-trace budget); (2) **never seed with `hash()`** — it is randomized per process (`PYTHONHASHSEED`); the sweeps use a stable MD5-based `_stable_seed(formula)`.

### Fundamental limitation — nested temporal under F/G/U/R (a core paper contribution)

The IJCNN 2014 encoding uses **one literal per subformula**. For `F(a & X b)`, F's reactivation creates a fresh `(a & X b)` instance each cell while prior X-b instances are still resolving via mode A; both share the literal `[X b]`, and the binary operator's mode-R rules fire on **both**, corrupting the carry-over. A correct fix needs cell-scoped literals — a structural redesign beyond what IJCNN 2014 documents. The impossibility is sharp: two traces (`σ_A = ({a},∅,{b})` vs `σ_B = (∅,{a},{b})` for `F(a & Xb)`) reach the **identical** conflicted register state with opposite correct verdicts, so no output-layer repair exists. Full write-up: [docs/nested_temporal_limitation.md](docs/nested_temporal_limitation.md) and `latex/3_rulerunner.tex` §3.2.

Three formulas in the equivalence sweep are marked `xfail(strict=True)`: `F (a & X b)`, `G (a -> F b)`, `G (a -> X b)`. **They test the OLD encoding and stay** (the progression monitors pass all three).

Why this doesn't poison the timing experiments: Exp 2/3 use the flat IJCNN family (encoding correct); Exp 1 uses `G(a→Fb)` *because* it has no trap/sink — no early termination ever fires, so per-cell cost is well-defined regardless of verdict correctness (state this in the paper).

### Paradigm 2, CORRECTED — the progression-based RuleRunner (`src/monitors/progression/`)

The nested-temporal limitation is a ceiling of the *one-literal-per-subformula* encoding, **not** of the rule-based idea. The **progression-based reformulation** (`latex/3_rulerunner.tex` §3.3, [docs/rulerunner_progression_analysis.md](docs/rulerunner_progression_analysis.md)) carries the *residual formula* (a multi-hot set of active top-level conjuncts) obtained by Bacchus–Kabanza progression, freshly re-derived each cell, so concurrent instances never share a slot. **Sound and complete on all LTLf** (theorem + proof in the paper) — matches `SymbolicDFAMonitor` on the full sweep including the three xfail formulas. Implemented as: lazy oracle (`ProgressionEngine`), eager residual-DFA + table oracle (`build_progression_dfa` / `ProgressionRuleRunnerEagerMonitor`, with cost metrics `n_states`/`n_roots`/`n_closure`), and two neural monitors mirroring the original pair — **`ProgressionRuleRunnerMonitor`** (flat CILP, batched CPU/CUDA — the experiment monitor) and **`ProgressionRuleRunnerStructuredMonitor`** (one CILP subnet per closure node). All wired into every timing experiment. **The price:** the eager construction enumerates the `2^|AP|` alphabet — progression's own wall, dual to the representational limit it fixes (capped at `DENSE_MAX_LEAVES` in exp2, `PROGRESSION_MAX_Q = 14` in exp6).

> ⚠ **Measured 2026-07-10 (exp6, `bounded_response`) — `|cl(φ)|` is inflated by a non-canonical `nf`. An IMPLEMENTATION artifact, not a paradigm property; consistent with `latex/3_rulerunner.tex` §3.3 on every claim it makes.**
>
> What blows up is `n_states` = reachable whole residuals: at deadline k = 8/10/12/14 it is 38/711/3776/16064 against minimal |Q| of 10/12/14/16, with build time following (≈5× per +2 in k, all inside `compile()`, dominated by `simplify_logic`). The *carried registers* (multi-hot top-level conjuncts) stay linear (`n_roots` 4→38, `n_closure` 11→203 over k=2..12) — exactly §3.3's factored bound. The paper already prices the eager realization at `|cl(φ)|·2^|P|` and never claims `|cl(φ)| ≤ |Q|`.
>
> **But the measured number is not the real `|cl(φ)|`:** residuals are finite *up to logical equivalence*, and our `nf` (sympy `simplify_logic`, temporal subformulas opaque) cannot apply temporal subsumption (e.g. `(b ∨ Xb) ∧ (b ∨ Xb ∨ X²b) ≡ (b ∨ Xb)`). **Verified by behavioural signature: at k=10 the 711 residual states exhibit only 8 distinct behaviours.** With a canonicalizing `nf` (BDD over temporal literals, or Myhill–Nerode quotienting), `|cl(φ)|` collapses toward |Q|.
>
> **Consequence:** do **not** report 16064 as progression's closure size — it overstates paradigm 2's wall and a reviewer who spots the subsumption will say so. Report the wall qualitatively (`|cl(φ)|·2^|P|`) or fix `nf` first. Mitigations: `build_progression_dfa` is `lru_cache`d (flat + structured share one BFS); exp6 caps both progression monitors at `PROGRESSION_MAX_Q = 14`.

The original RuleRunner stays alongside in the experiments for the before/after comparison (the cost-of-correctness figure).

### Structured variant caveat

`StructuredRuleRunnerMonitor` (IJCNN 2015 Fig. 5: one CILP subnet per parse-tree node) is device-aware and cross-trace batched, but within a cell it sweeps parse-tree nodes **sequentially** (a parent reads its children) — many small matmuls per cell, likely *less* GPU-friendly than the flat encoding's `depth+1` whole-network passes unless same-level siblings are fused (the tree parallelism IJCNN 2015 intends, which this naive sweep does not do). So exp3 contrasts *two batched RuleRunner encodings*, not batched-vs-unbatched.

## Paradigm 3 (DeepDFA) — implementation notes

The **review document** for paradigm 3. Read before reviewing [src/monitors/deep_dfa.py](src/monitors/deep_dfa.py).

### Source and the decision NOT to vendor

DeepDFA originates in the Umili & Capobianco line (ECAI 2024) and is used in the NeSy PPM paper (`papers/IS__NeSyPPM.pdf`, Eq. 18); reference implementation: github.com/axelmezini/nesy-suffix-prediction-dfa (`src/common/dfa.py`, ~120 lines). **Reimplemented, not submoduled**: their code is flat research code tightly coupled to their DOT parser and training pipeline, and — the important part — assumes the **BPM mutual-exclusivity assumption** (exactly one atom true per step, alphabet = atoms). Our benchmark family requires conjunctions of simultaneously-true atoms, so our alphabet is the full `2^|atoms|`. DeepDFA must be the *canonical, exactly-correct* monitor here (it matches `SymbolicDFAMonitor` everywhere, including where RuleRunner diverges).

### The alphabet-blowup finding

For non-mutually-exclusive propositional LTLf the transition tensor is indexed by `2^|atoms|` truth assignments. The IJCNN family's guards depend on **all n atoms**: dense is `2^n`, and only a guard's read-once circuit structure (which a flat DFA doesn't expose) permits sub-exponential evaluation. This is DeepDFA's structural weakness, dual to RuleRunner's nested-temporal limit and symbolic's state blowup — the clean three-way story. (The NeSy PPM paper sidesteps it only via the mutual-exclusivity assumption, which is false for our benchmark.)

### Representations (all on `DeepDFAMonitor.compile(mode=)` or subclasses)

| mode | tensor | per-step cost | use |
|---|---|---|---|
| `dense` (default) | `T (|Q|, 2^|AP|, |Q|)` one-hot | one matmul / `bmm` | small `|AP|`; the **batching showcase** (exp3) |
| `factored` | none materialized | vectorized cube-mask reduction | large `|AP|` (exp2, n up to 32) |
| `scan` (`DeepDFAMonitorScan`) | per-cell matrices, prefix product | O(log L) launches | long traces, small `|Q|`, GPU (exp1/3) |

**Factored crisp path (the path the experiments time):** each MONA guard is Shannon-expanded **once at construction** into a disjoint (orthogonal) cube cover, stored as require-true/require-false integer masks; `crisp_matrix(p)` builds the per-cell transition matrix as one vectorized reduction `∏_a [1 − rt·(1−p) − rf·p]`. Exact for crisp 0/1 inputs (cubes are 0/1 and mutually exclusive; rows sum to 1 because out-guards partition the assignment space). Flat in |AP| per cell (~7e-6 s at n=32; ~3.7× growth n=2→32 = genuine O(n²) mask reduction). ⚠ Not an unconditional escape: Shannon expansion can produce `Θ(2^k)` cubes for adversarial guards — the blowup shifts from "always" to "structure-dependent".

**Differentiable soft path (`soft_matrix` + `acceptance_probability`/`soft_verdict` readouts):** kept in the source as the affordance the paper points to (recursive guard-probability closures under atom independence; exact on crisp inputs for any guard, and on fractional inputs for read-once guards). **No experiment in this repo exercises it** — the uncertainty harness, the read-once/calibration findings, and the "which probabilistic verdict is correct" question all live in `artur_future_work/`. Do not delete it (the paper's §4.2/§4.4 reference it, and `tests/test_deep_dfa.py` covers `soft_matrix`'s crisp/read-once semantics).

### Monitor mechanics

- `step(obs)`: `q' = q @ T[:,σ,:]` (dense) or `q @ crisp_matrix(prob_vector(obs))` (factored); three-valued verdict off precomputed `trap_idx`/`sink_idx`; `final_verdict` = accepting membership of `argmax(q)`.
- `batch_run` **overrides** the base: encodes the whole batch once (`encode_presence`, vectorized numpy), one `bmm` per cell across all traces, per-trace early termination replayed from the recorded state path so `batch_run == [run(t) …]` exactly. `device="cuda"` supported.

### Correctness

`tests/test_deep_dfa.py`: DeepDFA matches `SymbolicDFAMonitor` on the full sweep **including nested temporal — no xfails**; dense == factored on crisp traces; `batch_run == [run(t) …]` in both modes; `crisp_matrix` row-stochastic, 0/1 on crisp input, equals `soft_matrix` on read-once guards, batched == unbatched; factored handles n=24 atoms with no `2^24` tensor. Scan variant verdict-identical to sequential and symbolic (`tests/test_deep_dfa_scan.py`).

## Benchmark Design

**Use synthetic traces for the timing experiments.** Trace content is irrelevant to per-step cost — what matters is length and formula complexity; real data would conflate paradigm speed with early-termination frequency. IJCNN 2014 does the same. (Real logs enter only via Phase 2, as a separate case study.)

**Reproduce and extend IJCNN 2014.** That paper compares only RuleRunner variants; adding the symbolic DFA and DeepDFA is our direct contribution. Same formula family and leaf counts: `◇ V_{i=1}^{n-1}(a_0 ∧ a_i)`, n = 2, 4, 8, 16, 32.

**Two kinds of parallelism — keep distinct in the paper:**

| Kind | What runs in parallel | Where it shows up |
|---|---|---|
| **Within-step** | evaluation rules / matmul atoms within a single cell | Exp 2 (breadth: RR grows with depth via the `depth+1` convergence loop; DeepDFA flat) + Exp 5 (depth, isolated) |
| **Cross-trace** | traces batched as matrix rows | Exp 3 (both DeepDFA and RuleRunner batch this axis; DeepDFA pays one matmul/cell, RR stays bottlenecked by within-step passes) |

**Experiments (see § Research Plan for status):**

| Exp | X-axis | Formula | Expected story |
|---|---|---|---|
| 1: trace length | 1k–10k cells | `G(a→Fb)` (no trap/sink) | All flat — per-step cost is constant |
| 2: formula breadth | n=2..32 leaves | IJCNN family (early-term **off**) | Symbolic flat; RR linear in depth; DeepDFA dense walls at 2ⁿ, factored flat |
| 3: batch size | 1–1024 traces | `ijcnn_n8` (early-term **off**) | Lead with absolute time; whether batched DeepDFA wins is an honest open question (Colab) |
| 5: nested-X depth | 0–10 | wrapped `ijcnn_n8` | Symbolic/DeepDFA flat; RR linear in depth |
| 6: state scaling | \|Q\| linear (deadline k) | `bounded_response` | Symbolic flat in \|Q\|; DeepDFA O(\|Q\|²) → crossover probe |
| 7: state blowup | \|Q\| = 2ᵏ+1 | `kth_from_last` | Shared wall: symbolic flat per cell but stores 2ᵏ states; DeepDFA \|Q\|² rises + walls earlier |

**Timing methodology:** `total_wall_time / (n_traces × trace_length)`, following IJCNN 2014, with `EARLY_TERMINATION = False` for per-cell-cost figures (all paradigms process all cells — the early-termination confound is documented in § Phase 1). Exp 1's formula never early-terminates by construction, which is exactly why it was chosen.

**Extending experiments:** each script has a `MONITORS` list at the top — adding a monitor variant is one line.

## Key Papers in `papers/`

- `Claude 1.txt` / `Claude 2.txt` — planning documents (motivation/framing; symbolic-baseline design rationale: why `ltlf2dfa + custom runner`, LTL3 semantics, trap/sink precomputation, why Declare4Py and RV-Monitor were ruled out)
- `IJCNN 2014.PDF` / `IJCNN 2015.pdf` — RuleRunner: the system being modernized
- `DeepDFA.pdf` — DeepDFA (ECAI 2024)
- `IS__NeSyPPM.pdf` — NeSy PPM paper: source of the DeepDFA formulation we adopt (Eq. 18)
- `RuleRunner.pdf`, `cilp.pdf`, `TOSEMv4.pdf` — earlier RuleRunner work, the CILP translation, model-level adaptation background

## Environment Setup

The conda environment `nesy-monitoring` is already created and ready. To reproduce from scratch:

```bash
conda env create -f environment.yml
conda activate nesy-monitoring
```

`environment.yml` pins all versions including `torch==2.6.0+cu124`. **Hardware:** heavy sweeps run on **Google Colab** (a CPU runtime and a GPU runtime, usually a Tesla T4) via `NeSy_Runtime_Monitoring.ipynb`; local dev is the conda env. **There is no Docker in this project.**

## Commands

```bash
# Activate environment
conda activate nesy-monitoring

# Install/reinstall project in editable mode
# Note: use the full path — conda run resolves to system pip on this machine
/home/matteo/miniconda3/envs/nesy-monitoring/bin/pip install -e ".[dev]"

# Run all tests / a single test
pytest
pytest tests/test_symbolic_dfa.py::test_eventually

# Lint
ruff check .

# Run one experiment (writes results/*.csv, then plots it) / all of them
python experiments/exp1_single_trace.py
bash scripts/run_all.sh

# Re-plot WITHOUT re-running (plotting is decoupled — reads the CSVs)
python experiments/plots.py                 # every figure from results/*.csv
python experiments/plots.py exp3            # just one experiment
python experiments/make_all_plots.py        # merged/gpu_only/device sets from results/cpu + results/gpu

# Demo (DFA rendering + RuleRunner run tables)
python demo/demo_monitors.py
```

## Dependencies

- `ltlf2dfa==1.0.2` — LTLf → minimal DFA (Python wrapper for MONA; MONA must be on PATH)
- `torch==2.6.0+cu124` — PyTorch with CUDA 12.4 (RTX 3050 Laptop GPU locally; T4 on Colab)
- `numpy`, `matplotlib`, `pandas`, `scipy`, `tqdm` — via conda
- `pytest`, `ruff`, `black` — dev tools via conda
- `lark` — NOT a direct dependency; pulled in transitively by `ltlf2dfa`. RuleRunner's parse tree is built programmatically, not parsed from a grammar file.
