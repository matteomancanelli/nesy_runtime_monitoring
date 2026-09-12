# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on **Neuro-Symbolic Runtime Monitoring** combining LTLf (Linear Temporal Logic over finite traces) with differentiable automata. The goal is an **ICLR submission** (venue decided with the supervisors, 2026-07): a *foundation* paper for neuro-symbolic LTLf monitoring — **modernizing and fixing RuleRunner** (the published shared-register conflation defect, a bounded-event middle repair, and the complete progression repair) and **connecting it with the automata-based paradigms** (symbolic DFA, DeepDFA), with a strong crisp empirical section characterizing the efficiency landscape honestly.

> **RuleRunner handoff:** read [docs/rulerunner_status.md](docs/rulerunner_status.md)
> before older planning notes.  It is the authoritative cross-version status;
> benchmark execution and result claims are currently deferred.

> **DeepDFA handoff:** read
> [docs/deepdfa_status_and_future_work.md](docs/deepdfa_status_and_future_work.md)
> before reopening paradigm 3. Its theory and implementation are frozen for
> the current paper; the next major phase is experiments/results.

> **Evaluation handoff (2026-08-25):** the authoritative benchmark plan is
> [docs/EXPERIMENTAL_EVALUATION_PLAN.md](docs/EXPERIMENTAL_EVALUATION_PLAN.md).
> The pre-E0 numbered experiments, July results, plots, experiment map, launcher,
> and Colab notebook are preserved under [old/](old/) and must not be used as
> current evidence.

**Scope decision (2026-07-13, supervisors' call): adaptation and probabilistic monitoring are OUT of this paper** — they are future work. Everything belonging to those threads (the uncertainty/calibration harness and experiments, the probabilistic-verdict theory section, the adaptation plan) lives in **[artur_future_work/](artur_future_work/)**, a self-contained fork with its own CLAUDE.md, ready to be extracted into its own repository. Do not re-grow those threads here; the paper mentions soft inputs and differentiability only as *affordances* motivating the paradigms (one paragraph, `latex/4_deepdfa.tex` §4.2) and defers the rest.

This is an active research project — plans, experiments, and framing should be treated as working hypotheses, not fixed requirements. Expect iteration.

## The Core Idea

Three paradigms for LTLf runtime monitoring, compared theoretically and experimentally:

| Paradigm | How it works | Key property |
|---|---|---|
| **Symbolic DFA** | Compile LTLf → minimal DFA; track state explicitly | Fastest crisp single-trace; frozen; crisp boolean inputs only |
| **Original RuleRunner** | Formula parse tree → extended truth tables → Horn clauses → CILP neural net | Faithful published baseline; one-register temporal-instance conflation is exactly certifiable formula-by-formula |
| **Bounded-event RuleRunner** | Finite-horizon event pipeline → certified original-RuleRunner skeleton | Static middle repair; exact on admitted formulas; fixed offset-indexed event modules |
| **Progression RuleRunner** | Residual roots → flat or structured evaluation/progression CILP modules | Complete for supported LTLf; exact permanent labels; residual/alphabet compilation cost |
| **DeepDFA** | Compile LTLf → minimal DFA → differentiable transition matrix | Native GPU batching; differentiable; alphabet (2^|AP|) blowup |

### Framing — read before touching experiments or the paper narrative

- **Do NOT anchor the paper on speed.** For crisp boolean monitoring the symbolic DFA is the theoretical optimum — a DFA walk is a dict lookup; nothing differentiable beats it. Symbolic dominating throughput is *expected*, not a threat. The experiments map **where each paradigm's cost grows and where it walls out** — an honest, complete efficiency landscape.
- **Keep paradigms neutral.** Present a capability matrix and let it speak; do **not** pre-crown DeepDFA. Each paradigm has a distinct Achilles heel (symbolic = state blowup; published RuleRunner = shared-register conflation + within-step sequential cost; complete progression repair = residual/alphabet compilation cost; DeepDFA = 2^|AP| alphabet blowup + |Q|² step) — that balance is the honest story.
- **Still try for an honest speed win.** If a genuine batched-throughput advantage survives fair measurement (Colab GPU, not the 4 GB laptop), it's a bonus — reported neutrally. Exp 3/6 test this; either outcome is honest ("modest real win" or "GPU advantage needs larger automata/hardware").
- **The ICLR challenge:** timing alone is thin for a pure-ML venue. The roadmap (Phases 2–4) lists the candidate additions, easiest → hardest; each gets scoped in a dedicated session before any code is written.

## Research Plan — phases and steps

> **Historical navigation:** [old/docs/EXPERIMENT_MAP.md](old/docs/EXPERIMENT_MAP.md)
> documents the archived July suite only. Use the current evaluation plan above
> before adding any experiment.

Status legend: ✅ done · 🟡 in progress · 🔲 not started.

### Phase 0 — Restructure for the ICLR frame ✅ (2026-07-13)

Split out the future-work threads into `artur_future_work/` (self-contained: full `src`/`tests` copies, the uncertainty harness + experiments + results, latex 5b + trimmed prose, own deps/CLAUDE.md). Removed Docker entirely (there is none in this project — the sweeps run on **Google Colab**). Moved the demo into `demo/`. Trimmed the paper's soft/adaptation material to affordance paragraphs. Fork point: commit `57d74e3`.

### Phase 1 — Efficiency landscape 🟡 (all *code* done; Colab re-runs + figure polish pending)

Complete the fair timing comparison across all monitor variants and produce the paper's figure set. The measurement-hygiene mechanisms are implemented and verified; what remains is **re-running on Colab (CPU + T4 GPU runtimes)** under the early-termination-off + CUDA-sync mode and polishing walls/crossovers into lead figures.

> ⚠ **The measured numbers in this section predate the RuleRunner faithfulness repairs (2026-08) and are pre-repair.** The rule system grew where the published qualifier tables were restored: `a U b` went from 11 to 28 rules, `G(a→Fb)`'s reactivation set from 10 to 15. The IJCNN family is byte-identical (107 eval / 25 react before and after), so exp2/exp3's RuleRunner figures still describe the current code, but exp1, exp5 and exp6 do not. Treat every figure below as an explanation of *which effect each mechanism has*, not as a current measurement.

Key mechanisms already in place (details preserved here because they explain *why the numbers look the way they do*):

- **Early-termination confound killed (exp2/3).** The IJCNN `◇(⋁(a₀∧aᵢ))` family early-terminates almost instantly on random traces, so symbolic's old ~1e-10 s/cell measured "how fast it gives up" while batched neural monitors process *all* cells — not apples-to-apples. `Monitor.run/batch_run` take `early_termination: bool` ([base.py](src/monitors/base.py)); when False the crisp walk processes all cells (absorbing states are sticky — verdicts unchanged, verified in [tests/test_early_termination.py](tests/test_early_termination.py)). `reset_if_stale()` drops CSVs measured in the other mode. Confirmed effect: symbolic per-cell on `ijcnn_n8` jumps ~3e-9 → ~3e-7 s (real dict lookup).
- **DeepDFA factored path vectorized (exp2 dual finding).** Each guard is decomposed **once** at construction into a disjoint cube cover by Shannon expansion (`_guard_cubes`/`_shannon_cubes` in [deep_dfa.py](src/monitors/deep_dfa.py)), stored as require-true/require-false integer masks; `exact_matrix(p)` (historical alias: `crisp_matrix`) is a single vectorized mask reduction — no per-cell sympy closures. It is exact and differentiable for fractional independent-Bernoulli inputs as well as crisp inputs. Factored per-cell at n=32: ~2e-4 → ~7e-6 s; growth n=2→32: ~24× → ~3.7× (residual = genuine O(n²) mask reduction). Dense (`DeepDFAMonitorDense`, capped at `DENSE_MAX_LEAVES=16`) is fastest where it fits but walls out (2^32 ≈ 64 GB at |Q|=2); exp2's analytic **memory-wall panel** shows it without building the tensors.
- **CUDA timing hygiene (exp3).** `time_monitor` syncs CUDA once per timed repeat (after warm-up and after each full `batch_run`), never inside the per-cell loop ([runner.py](src/benchmarks/runner.py)); DeepDFA's `batch_run` stays on-device and reads verdicts once at the end. Exp3 leads with **absolute time-per-trace**; the speedup panel is annotated (each curve normalized to its own batch=1 — cross-monitor speedups are misleading).
- **Truthful device labeling (was a real data bug).** Every monitor exposes `effective_device` (Symbolic and the original structured RuleRunner are pure-Python CPU walks and stamp `cpu` even under `device="cuda"`); `time_monitor` stamps that and syncs CUDA only for monitors that truly use it. A CSV never claims a GPU run that did not happen. Resume does not key on device — keep one CSV per machine (`results/cpu/`, `results/gpu/`), merged by the plotters.
- **Overhead decomposition (why symbolic wins).** Exp 3 batch=1 vs 1024 shows **~83 µs fixed per-call overhead per cell vs ~1.5 µs actual compute** — overhead, not arithmetic, is what loses. Two implemented levers test whether anything survives:
  - **Larger automata — [exp6_state_scaling.py](old/experiments/exp6_state_scaling.py)** (`STATE_SCALING_SUITE`, bounded response `G(a → (b ∨ Xb ∨ … ∨ Xᵏb))`, |Q| linear in k, |AP|=2): symbolic per-cell is flat in |Q|; DeepDFA's O(|Q|²) step finally amortizes launch overhead. **A crossover is plausible but unproven — needs the Colab GPU run.**
  - **Parallel prefix scan — `DeepDFAMonitorScan`** ([deep_dfa.py](src/monitors/deep_dfa.py)): the crisp state path is a prefix product of per-cell matrices → Hillis–Steele scan, O(log L) big matmuls instead of L small ones. **Honest caveat (measured):** not a FLOP reduction (×|Q|·log L arithmetic) — wins only where launch overhead ≫ arithmetic (GPU + small |Q| + long traces), loses on CPU/large |Q|. Falls back past `SCAN_MEM_LIMIT_BYTES`. Verdict-identical to sequential ([test_deep_dfa_scan.py](tests/test_deep_dfa_scan.py)). In exp1 + exp3.
- **State blowup, exponential family — [exp7_state_blowup.py](old/experiments/exp7_state_blowup.py)** (`STATE_BLOWUP_SUITE`, `F(a & Xᵏb)`, |Q| = 2ᵏ+1, |AP|=2): the **shared-weakness finding** — symbolic per-cell flat (~0.3 µs) while DeepDFA rises O(|Q|²) to ~40 µs at |Q|=1025; analytic memory wall crosses 4 GB at k≈14 for DeepDFA vs k≈28 for symbolic's linear table. Completes the archived three-heel table ([old/docs/richer_benchmark_findings.md](old/docs/richer_benchmark_findings.md)).
- **Within-step depth micro-benchmark — [exp5_depth_microbench.py](old/experiments/exp5_depth_microbench.py):** nested-X depth 0..10 over `ijcnn_n8`, batch=1, trace length 500 (a single cell is buried in per-call overhead — verified). RuleRunner rises ~140→168 µs with depth; Symbolic ~0.24 µs and DeepDFA ~15 µs stay flat.
- **Cost of correctness (the paradigm-2 paper number).** `plots.correctness_cost_table`/`plot_correctness_cost`: corrected(progression)/original per-cell-time ratio on exp2's flat IJCNN family (where the original RR is also correct, so the ratio isolates the encoding's throughput cost, not the verdict fix).

**Exit criterion:** regenerated `results/cpu|gpu` for exp1/2/3/5/6/7 under the new mode on Colab; a clear verdict on whether any speed advantage survives (exp3 batching, exp6 crossover, scan); the paper's lead figures chosen.

### Phase 2 — Real-log case study 🔲 (scope in a dedicated session first)

Declare constraints on a BPIC log (2012/2017 — standard, freely available), **crisp**, at scale: realism/legitimacy evidence that the paradigms and the timing story hold on real traces, not just synthetic ones. The `DECLARE_SUITE` (7 templates, diverse trap/sink structure) is the constraint vocabulary. Cheap-ish; reviewers at ICLR may value it less than Phase 3 — decide the investment after Phase 1's results.

### Phase 3 — Neural sequence baselines 🔲 (scope in a dedicated session first)

The most ICLR-shaped addition: train RNN/Transformer monitors on labeled traces and compare against the three exact paradigms on **verdict accuracy, length generalization, and sample efficiency**. Supports the paper's motivation directly (pure neural sequence models lack a mechanism to adhere to the logic; the NeSy encodings are exact *by construction*). The intro already gestures at RNNs/Transformers; `8_conclusion.tex` has a Transformers stub. New code: training loop + trace/label generation (the symbolic oracle labels for free).

### Phase 4 — Decision-diagram transition representation 🔲 (hardest; the ambition-raiser; scope first)

BDD/SDD-compiled guard circuits as a possible compactness upgrade of the factored cube cover — **crisp/scalability side only** in this repo (representation size vs dense `2^|Σ|` vs cube count; batched compiled-circuit throughput). NeSyA already establishes exact differentiable WMC over compiled symbolic-automaton guards, so neither that semantics nor knowledge compilation is a novelty claim here; the remaining delta is an LTLf-runtime backend and systems comparison. Design note: [docs/decision_diagram_transition_representation.md](docs/decision_diagram_transition_representation.md). First implementation question if pursued: whether MONA's internal MTBDD can be extracted directly.

### Phase 5 — Writing 🟡 (trails experiments; LaTeX in `latex/`)

- Port to the **ICLR template** (currently plain `article`).
- Sec 1 intro: thesis = "a foundation for neuro-symbolic LTLf monitoring: fix RuleRunner, connect with automata-based approaches, characterize the landscape honestly." Resolve the red TODOs (incl. "other NeSy approaches" and the results summary).
- Sec 3 RuleRunner: published architecture + shared-register counterexample + exact semantic boundary + bounded-event middle construction + progression repair with soundness/completeness proof — **drafted**; polish.
- Sec 4 DeepDFA: architecture, crisp-equivalence/exact-WMC/trace-marginal propositions, NeSyA positioning, alphabet blowup, factored representation, and implementation-level complexity — **drafted**; polish.
- Sec 5 theory comparison: three Achilles heels; **capability matrix TODO**; cite Bacchus–Kabanza + the LTLf 2EXP bound.
- Sec 6 experiments: rewrite around Phase 1's regenerated figures; state early-termination handling and hardware explicitly.
- Sec 7 related work: RuleRunner and DeepDFA/NeSyA/T-ILR lines **drafted**; LTLf monitoring and RV-tool coverage still needed.
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
│   │   ├── rulerunner/        ✅ Paradigm 2 — original encoding + exact certifier (+ certificates.json artifact) + bounded-event flat/structured repair
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
├── tests/                     ✅ Full suite; 6 xfail-strict document the ORIGINAL RuleRunner's shared-register conflation (by design — do not "fix")
├── results/                   ✅ cpu/ + gpu/ CSVs, figures/ PNGs (see results/README.md)
├── latex/                     🟡 The paper (main.tex + sections 1–8; dfa_script.py + gen_dfa_figs.sh for DFA figures)
├── demo/                      ✅ demo_monitors.py + demo_output/ — presentation aid (DFA rendering + RuleRunner run tables)
├── docs/                      ✅ rulerunner_status (authoritative handoff), bounded_event_rulerunner, nested limitation, progression analysis, and experiment/design notes
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

**Operator association follows ltlf2dfa, not convention.** `&`, `|` and `->` are left-folded; `U` and `R` are right-folded — that is what ltlf2dfa's MONA translation does. `->` in particular is **left**-associative (`a -> b -> c` ≡ `(a -> b) -> c`); our parse tree used to right-fold it, which made the rule-based paradigms monitor a different formula than the DFA-based ones. The DFA-based monitors are compiled from the formula *string* by ltlf2dfa, so association-sensitive formulas in `tests/test_semantic_oracle.py` fail whenever the two front ends diverge — keep them there.

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

Per-operator templates: ATOM/constants (no react rules); NOT (direct observation rules for the published NNF case); AND/OR/IMPLIES (modes B/L/R); EVENTUALLY/ALWAYS (including `G`'s `K` qualifier); UNTIL (published A/B/L/R tables); RELEASE (an implementation extension); NEXT/WEAK_NEXT (unqualified initial mode and monitoring mode M). Fresh activation is computed recursively exactly as in Algorithm 2: in particular, `X`/`W` initially activate only their root and install their operand on the following cell.

**Mode-tracking (AND/OR/IMPLIES):** mode B sees one child settled → transitions to L/R, dropping the settled child from monitoring (essential: without it `a ∨ ◇b` re-evaluates `a` every cell → wrong verdicts). L/R truth tables derive from B's column at the **pin value** (AND pins T; OR/IMPLIES pin F). ⚠ Initial bug: first draft pulled L/R from B's column at ψ=? instead of ψ=pin; fixed in [rules.py:152](src/monitors/rulerunner/rules.py#L152). **Assumption to revisit:** the pin derivation assumes each binary mode-B table has exactly one `(?, V) → ?L` and one `(V, ?) → ?R` cell (true for AND/OR/IMPLIES).

Atoms can never be `?` → any rule with `[a]?` in the body is pruned at template-instantiation time. Templates deduplicate `(body, head)` pairs. The IJCNN 2014 §III worked example (`a ∨ ◇b`) is reproduced exactly in [tests/test_rulerunner_rules.py](tests/test_rulerunner_rules.py).

### Step 2.5 — engine.py

Per cell: (1) inject `obs:a` literals (negation-as-failure for `~obs:a`); (2) evaluation phase — fire eval rules for `depth+1` passes (truth propagates one level per pass), break early on no new facts; (3) read root verdict; (4) if decided, freeze (absorbing); else fire react rules once in parallel, keep only `R[.]` literals.

**End-of-trace resolution:** recursively realizes the published END tables. `F φ`/`G φ` recurse on the child (`G`'s `?K` records the successful case); `U`/`R` recurse on ψ; an initial `X` is F and an initial `W` is T, while monitoring mode M mirrors the child's END value; binary L/R modes resolve the active child and pin the settled one. Empty-input semantics are handled separately.

### Step 3 — cilp.py

Standard Garcez & Zaverucha 1999 translation. Each rule = one hidden unit; body literals connect with `±W`; hidden bias `-W*(n-0.5)`; output = OR of incoming hiddens (bias `W*(k-1)`; k=0 outputs get negative bias). Sign activation. Eval and react phases have separate weight matrices over one shared literal-index space. `step()`: build x (carried R[.] + clamped obs) → `depth+1` sign-forward passes OR-accumulated (`x = max(x, y)`) → read root → react pass if undecided. End-of-trace resolution shared with the engine (parameterized by an `in_state` predicate).

**Equivalence: 0/N mismatches vs the engine across the whole flat-temporal sweep** ([tests/test_rulerunner_cilp.py](tests/test_rulerunner_cilp.py)). Knobs: `_W = 1.0` (any positive works with sign; tanh would need Garcez–Zaverucha `Amin` bias recomputation — that route is future work, see `artur_future_work/`). **Batching:** `CILPRunner.batch_run` vectorizes the trace axis (batched matmuls per cell, `device="cpu"/"cuda"`), bit-for-bit equal to sequential `run()` ([tests/test_rulerunner_batch.py](tests/test_rulerunner_batch.py)).

### Step 4 — monitor.py

Thin Monitor-ABC adapter (`RuleRunnerMonitor` holds a `CILPRunner`). Smoke tests only ([tests/test_rulerunner_monitor.py](tests/test_rulerunner_monitor.py)) — correctness lives in the engine/CILP sweeps.

**Test-design lessons (keep):** (1) an xfail-strict sweep must use enough traces to deterministically hit the expected failure, or it XPASSes flakily (the 80-trace budget); (2) **never seed with `hash()`** — it is randomized per process (`PYTHONHASHSEED`); the sweeps use a stable MD5-based `_stable_seed(formula)`.

### Fundamental limitation — shared-register temporal-instance conflation

The IJCNN 2014 encoding uses **one literal per subformula**. For `F(a & X b)`, F's reactivation creates a fresh `(a & X b)` instance each cell while prior X-b instances are still resolving in monitoring mode M; both share the literal `[X b]`, and the binary operator's mode-R rules fire on **both**, corrupting the carry-over. Any correct repair must enrich the recurrent address space beyond the published one-slot choice. The impossibility is sharp: two traces (`σ_A = ({a},∅,{b})` vs `σ_B = (∅,{a},{b})` for `F(a & Xb)`) reach the **identical** conflicted register state with opposite correct verdicts, so no output-layer repair exists. This does not mean every fixed formula needs unbounded memory: the bounded-event and progression versions provide two finite enrichments. Full write-up: [docs/nested_temporal_limitation.md](docs/nested_temporal_limitation.md) and `latex/3_rulerunner.tex` §3.2.

Three formulas in the equivalence sweep are marked `xfail(strict=True)`: `F (a & X b)`, `G (a -> F b)`, `G (a -> X b)`. **They test the OLD encoding and stay** (the progression monitors pass all three).

**Evaluation status:** benchmark reruns and result claims are deferred.  When that
phase resumes, every formula used with the original RuleRunner must carry its
exact certificate, and timing-only use of an unsafe formula must be disclosed
rather than treated as correctness evidence.

### Paradigm 2, CORRECTED — the progression-based RuleRunner (`src/monitors/progression/`)

The shared-register conflation is a ceiling of the *one-literal-per-subformula* encoding, **not** of the rule-based idea. The **progression-based reformulation** (`latex/3_rulerunner.tex` §3.3, [docs/rulerunner_progression_analysis.md](docs/rulerunner_progression_analysis.md)) carries the *residual formula* (a multi-hot set of active top-level conjuncts) obtained by Bacchus–Kabanza progression, freshly re-derived each cell, so concurrent instances never share a slot. **Sound and complete on all LTLf** (theorem + proof in the paper) — matches `SymbolicDFAMonitor` on the full sweep including the three xfail formulas. Implemented as: lazy oracle (`ProgressionEngine`), eager residual-DFA + table oracle (`build_progression_dfa` / `ProgressionRuleRunnerEagerMonitor`, with cost metrics `n_states`/`n_roots`/`n_closure`), and two neural monitors mirroring the original pair — **`ProgressionRuleRunnerMonitor`** (flat whole-residual CILP, batched CPU/CUDA) and **`ProgressionRuleRunnerStructuredMonitor`** (bottom-up per-node evaluation plus root-local progression/reactivation CILP modules). The structured recurrence never identifies a complete root set: each module reads one active root and its local guard, and all emitted successor roots are unioned. A separately compiled aggregate-state head supplies exact sink/trap labels without driving recurrence. The lazy engine's literal-constant early test remains sound but incomplete. The variants are available to the benchmark harness, but evaluation and result claims are deferred. **The price:** eager compilation still enumerates relevant observation alphabets and, for exact early labels, reachable aggregate states.

**Exact old-RuleRunner boundary and bounded-event middle repair.**
`certify_rule_runner` (`src/monitors/rulerunner/equivalence.py`) decides
formula-by-formula language equivalence with the canonical DFA by exact BFS over
their product and returns a shortest counterexample.  It drives the engine
through the public `RuleEngine.state()`/`load_state()` round trip; a guard test
fails if `RuleEngine` grows a field that the round trip would silently drop.  It also separates sound
early verdicts from exact online-label timing.  This corrected an older generated
claim: `X(Xa)` is safe after the faithful Next repairs; `(Xa) & X(Xa)` is a real
two-offset shared-register counterexample.  The executable semantic reference in
`src/monitors/rulerunner/bounded.py` extracts maximal finite-horizon event
islands, evaluates them through a fixed observation pipeline, flushes the finite
suffix at end-of-trace, and feeds an exactly-certified old-RuleRunner skeleton.
It repairs `G(Xa)`, `F(a & Xb)`, `G(a -> Xb)`, and `a U (b & Xc)` without full
progression; `G(a -> Fb)` remains outside because eventization leaves its unsafe
unbounded skeleton unchanged.  `src/monitors/rulerunner/bounded_cilp.py`
implements the fixed pipeline neurally: a recurrent CILP observation shift
register plus either a pooled/fixpoint event circuit and flat skeleton
(`BoundedEventRuleRunnerMonitor`), or per-`(subformula, offset)` event modules
and a structured skeleton (`BoundedEventStructuredRuleRunnerMonitor`).  Both
wrappers require exact language and prefix-soundness certificates, **looked up
from the offline artifact** `src/monitors/rulerunner/certificates.json`
(regenerate with `python -m src.monitors.rulerunner.certificates --refresh`;
each record is fingerprinted against the rule system, and a stale one is treated
as absent).  Certifying inside `compile()` would charge the construction for a
MONA DFA build it never uses at run time, so evaluation must pass
`certificate="cached"`.  `batch_run` is fused: equal-length suffix windows are
evaluated as tensor batches and the derived traces use the existing batched
flat/structured skeleton runners.

**Two configurations, two guarantees — keep them separate in the paper.**
The default (`exact_online=False`) is the construction Theorem 3.x describes: a
fixed static pipeline, no residual closure, no automaton anywhere in compile.
Its final verdicts are exact; its online labels are sound but **can lag
arbitrarily — not by `H`** (`G(X a)` is unsatisfiable on every non-empty trace
and its skeleton `G e` never sees this, so it stays UNDECIDED to the boundary).
Opting in with `exact_online=True` builds `bounded_extrapolation.py`'s CILP head
— one hidden unit per reachable composite state, classified by reverse
reachability — which reproduces the original DFA's timing exactly but costs
roughly `2^(|P|·H)`: 613 states for `G(a -> (b|Xb|X²b|X³b|X⁴b))` against a
6-state DFA.  Report the two separately; never quote the cheap pipeline's cost
next to the exact head's timing.  No bounded-event benchmark has yet been run.
Exact statement: `docs/bounded_event_rulerunner.md`.

> ⚠ **Implementation caveat, not an empirical result:** the current `nf` uses
> Boolean simplification with temporal subformulas treated as opaque atoms.  It
> cannot apply temporal subsumption, so raw syntactic residual counts can include
> many language-equivalent states and must not be reported as an intrinsic
> progression closure size.  Either canonicalize/quotient the residual graph or
> keep the cost statement qualitative (`|cl(φ)|·2^|P|`).  Measurement and any
> comparative result claim are deferred to the evaluation phase.

The original RuleRunner remains available for a future certified before/after
comparison.  The comparison and any “cost of correctness” result are deferred.

### Structured variant caveat

`StructuredRuleRunnerMonitor` (IJCNN 2015 Fig. 5: one CILP subnet per parse-tree node) is device-aware and cross-trace batched, but within a cell it sweeps parse-tree nodes **sequentially** (a parent reads its children) — many small matmuls per cell, likely *less* GPU-friendly than the flat encoding's `depth+1` whole-network passes unless same-level siblings are fused (the tree parallelism IJCNN 2015 intends, which this naive sweep does not do). So exp3 contrasts *two batched RuleRunner encodings*, not batched-vs-unbatched.

`ProgressionRuleRunnerStructuredMonitor` follows the same execution pattern over the residual closure: a sequential bottom-up evaluation sweep, then root-local reactivation modules whose outputs are OR-ed. Its fixed aggregate-state label head adds one further global pass for exact early verdicts. Neither structured implementation is a literal horizontally concatenated Fig. 5 tensor; both are functionally equivalent explicit module schedules.

## Paradigm 3 (DeepDFA) — implementation notes

The authoritative status and future-work review is
[docs/deepdfa_status_and_future_work.md](docs/deepdfa_status_and_future_work.md).
Read it before reviewing or extending
[src/monitors/deep_dfa.py](src/monitors/deep_dfa.py).

### Source and the decision NOT to vendor

DeepDFA originates in the Umili & Capobianco line (ECAI 2024) and is used in the NeSy PPM paper (`papers/IS__NeSyPPM.pdf`, Eq. 18); reference implementation: github.com/axelmezini/nesy-suffix-prediction-dfa (`src/common/dfa.py`, ~120 lines). **Reimplemented, not submoduled**: their code is flat research code tightly coupled to their DOT parser and training pipeline, and — the important part — assumes the **BPM mutual-exclusivity assumption** (exactly one atom true per step, alphabet = atoms). Our benchmark family requires conjunctions of simultaneously-true atoms, so our alphabet is the full `2^|atoms|`. DeepDFA must be the *canonical, exactly-correct* monitor here (it matches `SymbolicDFAMonitor` everywhere, including where RuleRunner diverges).

### The alphabet-blowup finding

For non-mutually-exclusive propositional LTLf the transition tensor is indexed by `2^|atoms|` truth assignments. The IJCNN family's guards depend on **all n atoms**, so dense is `2^n`; factored evaluation is sub-exponential only when the guard admits a compact representation. Read-once structure is sufficient but not necessary (compact cube covers or decision diagrams can also help). This is DeepDFA's structural weakness, dual to the published RuleRunner's shared-register conflation and symbolic's state blowup — the clean three-way story. (The NeSy PPM paper sidesteps it only via the mutual-exclusivity assumption, which is false for our benchmark.)

### Representations (all on `DeepDFAMonitor.compile(mode=)` or subclasses)

| mode | tensor | per-step cost | use |
|---|---|---|---|
| `dense` (default) | `T (|Q|, 2^|AP|, |Q|)` one-hot | one matmul / `bmm` | small `|AP|`; the **batching showcase** (exp3) |
| `factored` | none materialized | vectorized cube-mask reduction | large `|AP|` (exp2, n up to 32) |
| `scan` (`DeepDFAMonitorScan`) | per-cell matrices, prefix product | O(log L) launches | long traces, small `|Q|`, GPU (exp1/3) |

**Factored exact path (the path the experiments time):** each MONA guard is Shannon-expanded **once at construction** into a disjoint (orthogonal) cube cover, stored as require-true/require-false integer masks; `exact_matrix(p)` builds the per-cell transition matrix as one vectorized reduction `∏_a [1 − rt·(1−p) − rf·p]`. It is exact for crisp 0/1 inputs and is also exact WMC for arbitrary guards under fractional independent-Bernoulli inputs; rows sum to 1 because out-guards partition the assignment space. Flat in |AP| per cell (~7e-6 s at n=32; ~3.7× growth n=2→32 = genuine O(n²) mask reduction). ⚠ Not an unconditional escape: Shannon expansion can produce `Θ(2^k)` cubes for adversarial guards — the blowup shifts from "always" to "structure-dependent".

**Artifact surface:** [docs/deepdfa_artifact.md](docs/deepdfa_artifact.md) is the standalone usage/semantics guide. `acceptance_probability_tensor` is the autograd-preserving exact soft entry point and validates finite `[0,1]` Bernoulli parameters once at the public boundary. Every compiled monitor exposes an immutable `artifact_stats` record with `|AP|`, alphabet size, `|Q|`, transition/cube counts, and actual persistent tensor element/byte counts; the inner transition kernel remains validation-free so benchmark timing is unchanged.

**Differentiable probabilistic path:** `acceptance_probability_tensor(P, lengths)` consumes `(L, |AP|)` or `(B, L, |AP|)` tensors and returns tensors without detaching, so losses backpropagate to an upstream perceptor. It defaults to `exact_matrix`. The historical recursive closure is now explicit as `recursive_matrix` (`soft_matrix` remains a compatibility alias) and can be selected with `method="recursive"` only for diagnostic comparisons. **No probabilistic experiment in this repo exercises it** — the uncertainty harness and verdict-semantics study live in `artur_future_work/`.

### Monitor mechanics

- `step(obs)`: `q' = q @ T[:,σ,:]` (dense) or `q @ exact_matrix(prob_vector(obs))` (factored); three-valued verdict off precomputed `trap_idx`/`sink_idx`; `final_verdict` = accepting membership of `argmax(q)`.
- `batch_run` **overrides** the base: encodes the whole batch once (`encode_presence`, vectorized numpy), one `bmm` per cell across all traces, per-trace early termination replayed from the recorded state path so `batch_run == [run(t) …]` exactly. `device="cuda"` supported.

### Correctness

`tests/test_deep_dfa.py`: DeepDFA matches `SymbolicDFAMonitor` on the full sweep **including nested temporal — no xfails**; dense == factored on crisp traces; `batch_run == [run(t) …]` in both modes; `exact_matrix` is row-stochastic and agrees with brute-force WMC on a non-read-once guard; the tensor-native acceptance API preserves autograd and batch-length semantics; `exact_matrix == recursive_matrix` on read-once guards; factored handles n=24 atoms with no `2^24` tensor. Scan is verdict-identical to sequential and symbolic (`tests/test_deep_dfa_scan.py`).

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
