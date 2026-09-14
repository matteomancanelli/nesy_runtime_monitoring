# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

Research project on **Neuro-Symbolic Runtime Monitoring** for LTLf (Linear Temporal Logic
over finite traces). As of **2026-09-13 the work is split into two papers**; this repo is
**Paper A**.

- **Paper A — this repo.** *What a rule-based neural monitor can compute, and what
  completeness costs.* RuleRunner is the protagonist: the published one-register-per-
  subformula encoding, its exact semantic boundary, a bounded-horizon middle repair, and a
  progression-based repair that is sound and complete. The symbolic DFA and a fixed DeepDFA
  tensorization are the exact automata-based references the argument is measured against.
- **Paper B — [deepdfa_paper/](deepdfa_paper/).** *How a compiled transition function is
  represented and executed.* Alphabet blow-up, dense vs disjoint-cube vs prefix-scan
  representations, decision diagrams, and (its intended growth) Golog program graphs from
  KR 2026. Staged inside this repo, extraction-ready, with its own CLAUDE.md.

**The split line is semantics vs representation — not RuleRunner vs DeepDFA.** Paper A keeps
DeepDFA as a first-class monitor; what left is the representation engineering. Useful test
for any paragraph: *does removing it change what the monitor computes, or only how fast it
computes it?* Semantics stays here; speed went to Paper B. Provenance of every moved or
duplicated file: [deepdfa_paper/NOTES.md](deepdfa_paper/NOTES.md).

> **Handoffs — read before older planning notes.**
> - RuleRunner: [docs/rulerunner_status.md](docs/rulerunner_status.md) (authoritative
>   cross-version status).
> - DeepDFA: [docs/deepdfa_status_and_future_work.md](docs/deepdfa_status_and_future_work.md)
>   (theory/implementation frozen; §5's unsafe-claims list binds **both** papers).
> - Evaluation: [docs/EXPERIMENTAL_EVALUATION_PLAN.md](docs/EXPERIMENTAL_EVALUATION_PLAN.md)
>   (the authoritative benchmark plan; RQ1–RQ4).

**Out of scope (→ [artur_future_work/](artur_future_work/)):** probabilistic/uncertain
monitoring, calibration, specification adaptation, end-to-end perceptor training. This was
the **supervisors' call (2026-07-13)** — Artur and Antonio asked for a *base* paper. Do not
re-grow those threads here, and do not re-import the uncertainty experiment: it lives in the
fork, not in this tree. This paper mentions soft inputs and differentiability only as
*affordances* (`latex/5_deepdfa.tex` §Soft Observations) and defers the rest.

This is active research — plans and framing are working hypotheses, not fixed requirements.

## Paper A — the argument

The draft's current title (*A Comparison of Symbolic and Neuro-Symbolic Strategies for LTLf
Runtime Monitoring*) sells a survey. The contents are a result. The four technical pieces,
all drafted in `latex/4_rulerunner.tex`:

1. **A counterexample.** `F(a ∧ Xb)` on `σ_A = ({a},∅,{b})` vs `σ_B = (∅,{a},{b})`: the
   published encoding conflates two live instances of `Xb` in one register. Both traces
   reach the *identical* conflicted register with opposite correct verdicts, so **no
   output-layer repair exists**.
2. **An exact, decidable characterization.** `C_RR = {φ : L(RR_φ) = L(A_φ)}` is decided by
   BFS over the synchronous product with the canonical DFA, returning a shortest
   counterexample (`certify_rule_runner`). A readable syntactic fragment `F_RR` (temporal
   operators with propositional arguments only) is proved correct, with `F_RR ⊆ C_RR`.
   **The sharp observation, currently under-sold:** the boundary is not
   syntactically compositional and *not invariant under logical equivalence* — `X(Xa)` is
   safe, `(Xa) ∧ X(Xa)` is not, and the equivalent rewriting `X(a ∧ Xa)` is safe again.
3. **A bounded-horizon middle repair.** Finite-horizon event islands through a fixed static
   pipeline feeding an exactly-certified original skeleton. No automaton at compile time.
4. **The progression repair, sound and complete on all LTLf**, carrying residual formulas
   (multi-hot over residual roots) instead of subformula registers, with an explicit cost
   account (eager vs lazy) and an exponential state gap between whole-residual and
   root-factored state.

Plus the deflationary conclusion in `latex/6_theory_cmp.tex`, which should be promoted
rather than buried: **as monitors of crisp traces, the completeness-restored RuleRunner and
the fixed DeepDFA are semantically interchangeable.** What the neuro-symbolic paradigms buy
for runtime monitoring is *differentiability*, not compactness and not speed.

### The reframing — and the trap to avoid

A September 2026 planning brief proposed anchoring Paper A on a **lower bound**: that any
monitor whose state is a bounded, syntax-indexed register vector updated locally cannot be
sound and complete for LTLf. **Do not pursue that formulation.** It is vacuous as stated:
nothing in "bounded and syntax-indexed" constrains the register domain, so the class
contains a DFA in disguise (give the root register domain `Q` and let its update ignore its
children and apply `δ`). RuleRunner's failure is not a cardinality obstruction — it is a
*misalignment* obstruction, caused by requiring each register to mean "the truth value of
this subformula now". A class narrow enough to exclude DFA-in-disguise while still covering
RuleRunner and excluding progression has not been defined, and the claim that fuzzy-LTL /
LTL-as-loss / T-ILR share the shape is weak: those evaluate a whole trace, they are not
bounded-state online monitors.

**Use the provable substitute instead**, which keeps the same headline:

- **Necessity is free (Myhill–Nerode).** Any sound-and-complete monitor for `φ` must
  distinguish at least as many prefixes as `φ` has residual classes. This applies to *every*
  monitor, neural included, and gives "the DFA-size cost of the repair is necessary, not an
  artifact of our construction" without needing an impossibility theorem.
- **The defect is misalignment, not size.** `3^|sub(φ)|` register states are a *fixed* budget
  that is not aligned with the residual partition. Piece 2 above — non-invariance under
  logical equivalence — is the quotable evidence, and it is already proved.
- **State the no-go as a proposition with explicit hypotheses**, honestly scoped to the
  published encoding, not as a general lower bound on neuro-symbolic AI.

Other brief items and their status: retitle (open, Matteo's call); promote RQ1/RQ2 to core
evidence (agreed); keep the symbolic DFA as a first-class paradigm (**do not demote it** — it
is the certifier's reference and the yardstick); keep one honest cross-architecture latency
experiment in A (RQ4), since "differentiability is what you buy" is an empirical claim.

## Three paradigms

| Paradigm | How it works | Key property |
|---|---|---|
| **Symbolic DFA** | Compile LTLf → minimal DFA; track state explicitly | Fastest crisp single-trace; frozen; crisp inputs only |
| **Original RuleRunner** | Parse tree → extended truth tables → Horn clauses → CILP net | Faithful published baseline; one-register conflation exactly certifiable formula-by-formula |
| **Bounded-event RuleRunner** | Finite-horizon event pipeline → certified original skeleton | Static middle repair; exact on admitted formulas |
| **Progression RuleRunner** | Residual roots → flat or structured evaluation/progression CILP modules | Complete for supported LTLf; exact permanent labels; residual/alphabet compilation cost |
| **DeepDFA** | Compile LTLf → minimal DFA → differentiable transition tensor | Native GPU batching; differentiable; alphabet (2^\|AP\|) blowup |

### Framing rules — read before touching experiments or the narrative

- **Do NOT anchor the paper on speed.** For crisp boolean monitoring the symbolic DFA is the
  theoretical optimum — a DFA walk is a dict lookup. Symbolic dominating throughput is
  *expected*, not a threat.
- **Keep paradigms neutral.** Present a capability matrix and let it speak; do not pre-crown
  DeepDFA. Each paradigm has a distinct Achilles heel (symbolic = state blowup; published
  RuleRunner = shared-register conflation; complete progression repair = residual/alphabet
  compilation cost; DeepDFA = 2^|AP| alphabet blowup + |Q|² step).
- **A negative crossover is a valid result** if the experiment had the power to reveal a
  crossover had one existed.

## Status

Legend: ✅ done · 🟡 in progress · 🔲 not started.

| Phase | What | Status |
|---|---|---|
| Theory — RuleRunner | counterexample, `C_RR` certifier + `F_RR` fragment, bounded-horizon, progression (soundness/completeness) | ✅ drafted, needs polish |
| Theory — DeepDFA | architecture, crisp equivalence, exact WMC, trace-marginal | ✅ frozen |
| Implementation | all monitors, certifier, certificates artifact, benchmark harness | ✅ |
| E0 protocol/schema freeze | `e0.v1`, balanced IJCNN, budgets, provenance | ✅ 2026-08-25 |
| E1 / RQ1 semantic characterization | 68 exact records, 17 formulas, 4 constructions | ✅ frozen `rq1.v1` |
| E2 instrumentation | compilation/memory/artifact-size measurement | ✅ |
| E3 / RQ2 cost of correctness | fresh-process, stratified, exhaustive decision lag | ✅ local-CPU candidate |
| E4 / RQ3 structural scaling | 5 panels; **panels 2–3 are now Paper B's** | ✅ local-CPU candidate |
| E5 / RQ4 cross-architecture | 7 configurations, 3 modes | 🟡 CPU done; **CUDA rerun pending** |
| E6 / RQ5 external validity (Declare/BPIC) | real-log case study | 🔲 |
| Writing | see below | 🟡 |

**Immediate next action (unchanged by the split):** rerun the frozen `rq4.v1` grid on a
controlled CUDA host. The local requested-CUDA audit is explicitly unsupported and cannot
answer the GPU crossover or memory questions. RQ1 remains the semantic gate for everything.

**Every number under `results/` is a controlled local-CPU candidate, not a final
measurement.** The July suite under [old/](old/) is archived for archaeology and **must not
be used as current evidence** — it predates the RuleRunner faithfulness repairs and the
current timing protocol.

### Writing status (`latex/`, sections 1–8)

- Port to the target venue's template (currently plain `article`). Venue is **open**: ICLR
  was the original target but that cycle is gone; under the current framing IJCAI / AAAI /
  KR / a formal-methods venue are all plausible. Check deadlines fresh.
- `1_introduction.tex` — **rewrite.** Lead with the characterization + the price of
  completeness, not with "we compare three paradigms". Red TODO on the results summary.
- `2_background.tex` — keep.
- `3_related_work.tex` — real content, with a large `\begin{comment}` block holding drafted
  RuleRunner / DeepDFA / NeSyA / T-ILR paragraphs that should be restored into the section.
  TODO: RVLTL/RLTL/TRLTL; LTLf monitoring and RV-tool coverage.
- `4_rulerunner.tex` — the core (40 KB). Drafted; polish. Promote the
  non-invariance-under-equivalence observation.
- `5_deepdfa.tex` — **trimmed by the split.** Keeps Architecture, *Crisp equivalence*, Soft
  Observations with *Exact guard WMC* and *Trace-marginal semantics*, plus one paragraph
  stating the alphabet axis and forward-referencing Paper B.
- `6_theory_cmp.tex` — three heels + the interchangeability result. The dual-axis compression
  passage is compressed here and developed in full in Paper B. **Capability matrix TODO**;
  cite Bacchus–Kabanza and the LTLf 2EXP bound.
- `7_experiments.tex` — stub. Rebuild around RQ1/RQ2 as the core, RQ3 (A's panels) and RQ4 as
  support. State early-termination handling and hardware explicitly. A's experiment section
  is an argument, not a benchmark table.
- `8_conclusion.tex` — stub. Write; point forward to Paper B.
- `proofs.tex` — appendix proofs. Appendix candidates: [docs/appendix_ideas.md](docs/appendix_ideas.md).
- Cross-paper citation: `\paperA{}` / `\paperB{}` macros in `preamble.tex` resolve to a
  placeholder; switch to *(under submission)* / arXiv / a real `\cite` in one edit once
  publication order is decided.

### Open decisions for Matteo — do not resolve these unilaterally

1. Paper A's title.
2. Venue and cycle for each paper.
3. Publication order, and therefore the direction of the cross-citation.
4. Whether Artur and Antonio hear about the split before or after the draft they are
   receiving. The reframing changes *their* paper's headline and a claim in Artur's own
   earlier work — it is a conversation, not a fait accompli.
5. Packaging: does Paper B duplicate `deep_dfa.py` (current state) or depend on Paper A?
6. Whether `artur_future_work/` stays where it is, given it holds a third copy of the
   decision-diagram note.

## Repository Structure

```
nesy_runtime_monitoring/
├── src/
│   ├── formula/compiler.py        ✅ LTLf → minimal DFA (wraps ltlf2dfa; trap/sink precomputation)
│   ├── monitors/
│   │   ├── base.py               ✅ Monitor ABC + Verdict enum (+ early_termination flag)
│   │   ├── symbolic_dfa.py       ✅ Paradigm 1 — crisp DFA walk
│   │   ├── rulerunner/           ✅ Paradigm 2 — original encoding, exact certifier
│   │   │                            (+ certificates.json), bounded-event flat/structured repair
│   │   ├── progression/          ✅ Paradigm 2 CORRECTED — formula, progression, engine,
│   │   │                            eager, flat, structured
│   │   └── deep_dfa.py           ✅ Paradigm 3 — dense + factored + scan (soft path = affordance)
│   └── benchmarks/
│       ├── formulas.py           ✅ Formula registry (IJCNN balanced/left-deep, Declare, state families)
│       ├── schema.py             ✅ e0.v1 raw/provenance schema, structural characterization
│       ├── runner.py             ✅ Timing harness (effective-device stamping, resumable)
│       ├── e2.py / e2_worker.py  ✅ Compilation/memory/artifact instrumentation
│       └── rq1–rq4 (+ workers)   ✅ The four research-question drivers
├── experiments/
│   ├── rq1_semantic_characterization.py  ✅ Semantic gate: exact product certification
│   ├── rq2_cost_of_correctness.py        ✅ Price of the repairs
│   ├── rq3_structural_scaling.py         ✅ 5 panels (2–3 are Paper B's)
│   ├── rq4_cross_architecture.py         🟡 CPU done, CUDA rerun pending
│   └── e2_instrumentation_smoke.py       ✅
├── tests/                        ✅ Full suite; xfail-strict cases document the ORIGINAL
│                                    RuleRunner's shared-register conflation (by design —
│                                    do NOT "fix" them)
├── results/{rq1,rq2,rq3,rq4,e2}/ ✅ Frozen artifacts + manifests (local-CPU candidates)
├── latex/                        🟡 main.tex + sections 1–8 + proofs + preamble + citations
├── docs/                         ✅ Handoffs, RQ write-ups, evaluation plan, design notes
├── demo/                         ✅ demo_monitors.py — presentation aid
├── deepdfa_paper/                ✅ **Paper B**, extraction-ready, own CLAUDE.md/README/NOTES
├── artur_future_work/            ✅ Deferred threads (probabilistic + adaptation), own CLAUDE.md
├── old/                          📦 Archived July suite (exp1–exp7, plots, results) — NOT evidence
└── papers/                       Reference papers and planning notes (gitignored)
```

## Abstract Monitor Interface

All monitors implement the same interface (`monitors/base.py`):

```python
compile(formula: str) -> MonitorInstance   # classmethod
step(obs: dict[str, bool]) -> Verdict      # SATISFY / VIOLATE / UNDECIDED
final_verdict() -> Verdict                 # binary end-of-trace check; never UNDECIDED
reset() -> None
run(trace, early_termination=True) -> Verdict
batch_run(traces, early_termination=True) -> list[Verdict]   # DeepDFA AND RuleRunner override
```

Three-valued semantics (`UNDECIDED`) applies online — a trace mid-execution may not yet have
a determined verdict. Absorbing states (all successors accepting, or all rejecting) enable
early termination; with `early_termination=False` all cells are processed (verdicts
unchanged — absorbing states are sticky, verified in `tests/test_early_termination.py`).

`final_verdict()` is a required separate method because response-style formulas like
`G(a → F b)` have neither a trap state nor an accepting sink, so `step()` always returns
`UNDECIDED`. The verdict is only binary at end-of-trace.

## Key Technical Details

**Operator association follows ltlf2dfa, not convention.** `&`, `|` and `->` are left-folded;
`U` and `R` are right-folded — that is what ltlf2dfa's MONA translation does. `->` in
particular is **left**-associative (`a -> b -> c` ≡ `(a -> b) -> c`); our parse tree used to
right-fold it, which made the rule-based paradigms monitor a different formula than the
DFA-based ones. The DFA-based monitors are compiled from the formula *string* by ltlf2dfa, so
association-sensitive formulas in `tests/test_semantic_oracle.py` fail whenever the two front
ends diverge — keep them there.

**LTLf → DFA compilation:** `ltlf2dfa`'s `to_dfa()` returns a DOT string with transitions
labeled by boolean expressions over atoms (`~a`, `a & ~b`, `true`). The compiler parses this
DOT, converts MONA guard syntax to Python, and compiles each guard to a bytecode object once
at construction (compile-once, eval-many). The `DFA` dataclass exposes `states`, `atoms`,
`initial`, `accepting`, `transitions`, `trap_states`, `accepting_sinks`, and
`step(state, obs) -> state`.

**Trap states and accepting sinks** are precomputed by graph reachability at DFA construction.
A trap is a state from which no accepting state is reachable; an accepting sink is one from
which all reachable states are accepting. Per-step verdict checks cost one `set` membership test.

**`ltlf2dfa` quirk:** `lark` (a transitive dependency) emits two `DeprecationWarning`s about
`sre_parse`/`sre_constants`. Harmless; suppressed by pytest.

**Exact three-valued timing is never free for anyone.** It is a property of the continuation
language, so every paradigm pays with a global reachability analysis (DFA traps/sinks, the
eager progression monitor's residual graph, the bounded-event monitor's composite states).
What differs is the size of the graph classified, not whether one is needed. Keep this
symmetry explicit — it is easy to mis-present as one paradigm's advantage.

## Paradigm 2 (RuleRunner) — implementation notes

This is the **review document** for paradigm 2. Read it before reviewing
[src/monitors/rulerunner/](src/monitors/rulerunner/).

### Pipeline at a glance

```
LTLf formula string
  └─> parse_tree.parse()              (step 1)
        └─> Node DAG (subformula sharing, depth precomputed)
              └─> rules.build_rules() (step 2)
                    └─> RuleSystem (eval rules + react rules + initial state)
                          ├─> engine.RuleEngine  (step 2.5) — symbolic executor / oracle
                          └─> cilp.CILPNet       (step 3) — torch network
                                └─> monitor.RuleRunnerMonitor (step 4) — Monitor wrapper
```

Each layer is independently testable. `engine.py` is the oracle: pure Python, checked against
`SymbolicDFAMonitor` on randomized formulas + traces. The CILP network produces the same
per-cell verdicts as the engine (equivalence tested).

### Step 1 — parse_tree.py

Frozen-dataclass `Node` DAG with `(op, children, key, depth, atom)`. `key` = canonical
syntactic identity (`str(ast)` of ltlf2dfa's AST), the name suffix for every R[.] / [.]V
literal downstream. **Reuses ltlf2dfa's parser** but walks its AST into our own type.
**Binarizes n-ary operators** (left-fold `&`/`|`, right-fold `->`/`U`/`R` — verified against
`to_mona()`). **Subformula sharing:** repeated keys hit a cache and yield the same `Node`.
`depth` precomputed (atoms 0; parent = 1 + max child). Atoms can never be undecided (enforced
here; downstream pruning relies on it).

### Step 2 — rules.py

`Literal(name, negated)`, `Rule(body, head)`, `RuleSystem(eval_rules, react_rules,
initial_state, atoms, root_key)`. Literal naming is **string-based** with modes baked in as
suffixes (e.g. `R[(a | F(b))]^B`) — matches IJCNN 2014's notation; CILP gets one neuron per
mode-distinct literal.

Per-operator templates: ATOM/constants (no react rules); NOT (direct observation rules for
the published NNF case); AND/OR/IMPLIES (modes B/L/R); EVENTUALLY/ALWAYS (including `G`'s `K`
qualifier); UNTIL (published A/B/L/R tables); RELEASE (an implementation extension);
NEXT/WEAK_NEXT (unqualified initial mode and monitoring mode M). Fresh activation follows
Algorithm 2: `X`/`W` initially activate only their root and install their operand on the
following cell.

**Mode-tracking (AND/OR/IMPLIES):** mode B sees one child settled → transitions to L/R,
dropping the settled child from monitoring (essential: without it `a ∨ ◇b` re-evaluates `a`
every cell → wrong verdicts). L/R truth tables derive from B's column at the **pin value**
(AND pins T; OR/IMPLIES pin F). ⚠ Initial bug: first draft pulled L/R from B's column at ψ=?
instead of ψ=pin; fixed in [rules.py:152](src/monitors/rulerunner/rules.py#L152).
**Assumption to revisit:** the pin derivation assumes each binary mode-B table has exactly one
`(?, V) → ?L` and one `(V, ?) → ?R` cell (true for AND/OR/IMPLIES).

Atoms can never be `?` → any rule with `[a]?` in the body is pruned at instantiation.
Templates deduplicate `(body, head)` pairs. The IJCNN 2014 §III worked example (`a ∨ ◇b`) is
reproduced exactly in [tests/test_rulerunner_rules.py](tests/test_rulerunner_rules.py).

### Step 2.5 — engine.py

Per cell: (1) inject `obs:a` literals (negation-as-failure for `~obs:a`); (2) evaluation phase
— fire eval rules for `depth+1` passes, break early on no new facts; (3) read root verdict;
(4) if decided, freeze (absorbing); else fire react rules once in parallel, keep only `R[.]`.

**End-of-trace resolution** recursively realizes the published END tables: `F φ`/`G φ` recurse
on the child (`G`'s `?K` records the successful case); `U`/`R` recurse on ψ; an initial `X` is
F and an initial `W` is T, while monitoring mode M mirrors the child's END value; binary L/R
modes resolve the active child and pin the settled one. Empty-input semantics handled separately.

### Step 3 — cilp.py

Standard Garcez & Zaverucha 1999 translation. Each rule = one hidden unit; body literals
connect with `±W`; hidden bias `-W*(n-0.5)`; output = OR of incoming hiddens (bias `W*(k-1)`;
k=0 outputs get negative bias). Sign activation. Eval and react phases have separate weight
matrices over one shared literal-index space. `step()`: build x (carried R[.] + clamped obs) →
`depth+1` sign-forward passes OR-accumulated (`x = max(x, y)`) → read root → react pass if
undecided. End-of-trace resolution shared with the engine (parameterized by an `in_state`
predicate).

**Equivalence: 0/N mismatches vs the engine across the whole flat-temporal sweep**
([tests/test_rulerunner_cilp.py](tests/test_rulerunner_cilp.py)). `_W = 1.0` (any positive
works with sign; tanh would need Garcez–Zaverucha `Amin` bias recomputation — that route is
future work, see `artur_future_work/`). **Batching:** `CILPRunner.batch_run` vectorizes the
trace axis, bit-for-bit equal to sequential `run()`
([tests/test_rulerunner_batch.py](tests/test_rulerunner_batch.py)).

### Step 4 — monitor.py

Thin Monitor-ABC adapter (`RuleRunnerMonitor` holds a `CILPRunner`). Smoke tests only —
correctness lives in the engine/CILP sweeps.

**Test-design lessons (keep):** (1) an xfail-strict sweep must use enough traces to
deterministically hit the expected failure, or it XPASSes flakily (the 80-trace budget);
(2) **never seed with `hash()`** — it is randomized per process (`PYTHONHASHSEED`); the sweeps
use a stable MD5-based `_stable_seed(formula)`.

### Fundamental limitation — shared-register temporal-instance conflation

The IJCNN 2014 encoding uses **one literal per subformula**. For `F(a & X b)`, F's reactivation
creates a fresh `(a & X b)` instance each cell while prior X-b instances are still resolving in
monitoring mode M; both share the literal `[X b]`, and the binary operator's mode-R rules fire
on **both**, corrupting the carry-over. Any correct repair must enrich the recurrent address
space beyond the published one-slot choice. The impossibility is sharp: `σ_A = ({a},∅,{b})` vs
`σ_B = (∅,{a},{b})` for `F(a & Xb)` reach the **identical** conflicted register state with
opposite correct verdicts, so no output-layer repair exists. This does not mean every fixed
formula needs unbounded memory: the bounded-event and progression versions are two finite
enrichments. Full write-up: [docs/nested_temporal_limitation.md](docs/nested_temporal_limitation.md)
and `latex/4_rulerunner.tex` §The Nested-Temporal Limitation.

Three formulas in the equivalence sweep are marked `xfail(strict=True)`: `F (a & X b)`,
`G (a -> F b)`, `G (a -> X b)`. **They test the OLD encoding and stay** (the progression
monitors pass all three).

**Semantic gate:** every formula used with the original RuleRunner must carry its exact
certificate; timing-only use of an unsafe formula must be disclosed, never treated as
correctness evidence. This is what RQ1 enforces.

### Paradigm 2, CORRECTED — the progression-based RuleRunner (`src/monitors/progression/`)

The conflation is a ceiling of the *one-literal-per-subformula* encoding, **not** of the
rule-based idea. The **progression-based reformulation** (`latex/4_rulerunner.tex`
§Progression-Based Reactivation, [docs/rulerunner_progression_analysis.md](docs/rulerunner_progression_analysis.md))
carries the *residual formula* (a multi-hot set of active top-level conjuncts) obtained by
Bacchus–Kabanza progression, freshly re-derived each cell, so concurrent instances never share
a slot. **Sound and complete on all LTLf** (theorem + proof in the paper) — matches
`SymbolicDFAMonitor` on the full sweep including the three xfail formulas. Implemented as:
lazy oracle (`ProgressionEngine`), eager residual-DFA + table oracle (`build_progression_dfa` /
`ProgressionRuleRunnerEagerMonitor`, with cost metrics `n_states`/`n_roots`/`n_closure`), and
two neural monitors mirroring the original pair — **`ProgressionRuleRunnerMonitor`** (flat
whole-residual CILP, batched CPU/CUDA) and **`ProgressionRuleRunnerStructuredMonitor`**
(bottom-up per-node evaluation plus root-local progression/reactivation CILP modules). The
structured recurrence never identifies a complete root set: each module reads one active root
and its local guard, and all emitted successor roots are unioned. A separately compiled
aggregate-state head supplies exact sink/trap labels without driving recurrence. The lazy
engine's literal-constant early test remains sound but incomplete. **The price:** eager
compilation still enumerates relevant observation alphabets and, for exact early labels,
reachable aggregate states.

> ⚠ **Implementation caveat, not an empirical result:** the current `nf` uses Boolean
> simplification with temporal subformulas treated as opaque atoms. It cannot apply temporal
> subsumption, so raw syntactic residual counts can include many language-equivalent states
> and **must not** be reported as an intrinsic progression closure size. Either
> canonicalize/quotient the residual graph or keep the cost statement qualitative
> (`|cl(φ)|·2^|P|`). Relatedly: a reachable residual automaton is **not** automatically
> minimal — do not claim minimality.

### Exact boundary and bounded-event middle repair

`certify_rule_runner` (`src/monitors/rulerunner/equivalence.py`) decides formula-by-formula
language equivalence with the canonical DFA by exact BFS over their product and returns a
shortest counterexample. It drives the engine through the public
`RuleEngine.state()`/`load_state()` round trip; a guard test fails if `RuleEngine` grows a
field the round trip would silently drop. It also separates sound early verdicts from exact
online-label timing. This corrected an older generated claim: `X(Xa)` is safe after the
faithful Next repairs; `(Xa) & X(Xa)` is a real two-offset counterexample.

The executable semantic reference in `src/monitors/rulerunner/bounded.py` extracts maximal
finite-horizon event islands, evaluates them through a fixed observation pipeline, flushes the
finite suffix at end-of-trace, and feeds an exactly-certified original skeleton. It repairs
`G(Xa)`, `F(a & Xb)`, `G(a -> Xb)`, and `a U (b & Xc)` without full progression; `G(a -> Fb)`
remains outside because eventization leaves its unsafe unbounded skeleton unchanged.
`bounded_cilp.py` implements the fixed pipeline neurally: a recurrent CILP observation shift
register plus either a pooled/fixpoint event circuit and flat skeleton
(`BoundedEventRuleRunnerMonitor`), or per-`(subformula, offset)` event modules and a structured
skeleton (`BoundedEventStructuredRuleRunnerMonitor`). Both require exact language and
prefix-soundness certificates, **looked up from the offline artifact**
`src/monitors/rulerunner/certificates.json` (regenerate with
`python -m src.monitors.rulerunner.certificates --refresh`; each record is fingerprinted
against the rule system, and a stale one is treated as absent). Certifying inside `compile()`
would charge construction for a MONA DFA build it never uses at run time, so evaluation must
pass `certificate="cached"`. `batch_run` is fused: equal-length suffix windows are evaluated as
tensor batches and the derived traces use the existing batched flat/structured runners.

**Two configurations, two guarantees — keep them separate in the paper.** The default
(`exact_online=False`) is the construction the bounded theorem describes: a fixed static
pipeline, no residual closure, no automaton anywhere in compile. Its final verdicts are exact;
its online labels are sound but **can lag arbitrarily — not by `H`** (`G(X a)` is unsatisfiable
on every non-empty trace and its skeleton `G e` never sees this, so it stays UNDECIDED to the
boundary). Opting in with `exact_online=True` builds `bounded_extrapolation.py`'s CILP head —
one hidden unit per reachable composite state, classified by reverse reachability — which
reproduces the original DFA's timing exactly but costs roughly `2^(|P|·H)`: 613 states for
`G(a -> (b|Xb|X²b|X³b|X⁴b))` against a 6-state DFA. Report the two separately; never quote the
cheap pipeline's cost next to the exact head's timing. Exact statement:
[docs/bounded_event_rulerunner.md](docs/bounded_event_rulerunner.md).

### Structured variant caveat

`StructuredRuleRunnerMonitor` (IJCNN 2015 Fig. 5: one CILP subnet per parse-tree node) is
device-aware and cross-trace batched, but within a cell it sweeps parse-tree nodes
**sequentially** (a parent reads its children) — many small matmuls per cell, likely *less*
GPU-friendly than the flat encoding's `depth+1` whole-network passes unless same-level siblings
are fused (the tree parallelism IJCNN 2015 intends, which this naive sweep does not do). So RQ3
panel 1 and RQ4 contrast *two batched RuleRunner encodings*, not batched-vs-unbatched.

`ProgressionRuleRunnerStructuredMonitor` follows the same pattern over the residual closure,
plus one further global pass for exact early verdicts. Neither structured implementation is a
literal horizontally concatenated Fig. 5 tensor; both are functionally equivalent explicit
module schedules.

## Paradigm 3 (DeepDFA) — implementation notes

**Paper A needs exactly two things from DeepDFA:** that it is an *exact* tensorization of the
same minimal DFA (so it is a valid reference for the interchangeability result), and that it is
*differentiable* (so "differentiability is what you buy" has a referent). The representations,
their complexity accounting, and their scaling measurements are **Paper B's** — see
[deepdfa_paper/](deepdfa_paper/). The authoritative status doc is
[docs/deepdfa_status_and_future_work.md](docs/deepdfa_status_and_future_work.md); read it
before touching [src/monitors/deep_dfa.py](src/monitors/deep_dfa.py), which is **frozen**.

### Source and the decision NOT to vendor

DeepDFA originates in the Umili & Capobianco line (ECAI 2024) and is used in the NeSy PPM
paper (`papers/IS__NeSyPPM.pdf`, Eq. 18); reference implementation:
github.com/axelmezini/nesy-suffix-prediction-dfa. **Reimplemented, not submoduled**: their
code is tightly coupled to their DOT parser and training pipeline, and — the important part —
assumes the **BPM mutual-exclusivity assumption** (exactly one atom true per step, alphabet =
atoms). Our benchmark family requires conjunctions of simultaneously-true atoms, so our
alphabet is the full `2^|atoms|`. DeepDFA must be the *canonical, exactly-correct* monitor here
(it matches `SymbolicDFAMonitor` everywhere, including where RuleRunner diverges).

### The alphabet-blowup finding

For non-mutually-exclusive propositional LTLf the transition tensor is indexed by `2^|atoms|`
truth assignments. The IJCNN family's guards depend on **all n atoms**, so dense is `2^n`;
factored evaluation is sub-exponential only when the guard admits a compact representation.
Read-once structure is sufficient but not necessary. This is DeepDFA's structural weakness,
dual to the published RuleRunner's conflation and symbolic's state blowup — the three-way
story. (The NeSy PPM paper sidesteps it only via mutual exclusivity, false for our benchmark.)
**Paper A states this axis and places it in the capability matrix; Paper B develops it.**

### Representations

| mode | tensor | per-step cost | use |
|---|---|---|---|
| `dense` (default) | `T (\|Q\|, 2^\|AP\|, \|Q\|)` one-hot | one matmul / `bmm` | small `\|AP\|` |
| `factored` | none materialized | vectorized cube-mask reduction | large `\|AP\|` |
| `scan` (`DeepDFAMonitorScan`) | per-cell matrices, prefix product | O(log L) launches | long traces, small `\|Q\|`, GPU |

Each guard is Shannon-expanded **once at construction** into a disjoint cube cover stored as
require-true/require-false integer masks; `exact_matrix(p)` builds the per-cell transition
matrix as one vectorized reduction. Exact for crisp 0/1 inputs, and exact WMC for arbitrary
guards under fractional independent-Bernoulli inputs; rows sum to 1 because out-guards
partition the assignment space.

### Monitor mechanics

- `step(obs)`: `q' = q @ T[:,σ,:]` (dense) or `q @ exact_matrix(prob_vector(obs))` (factored);
  three-valued verdict off precomputed `trap_idx`/`sink_idx`; `final_verdict` = accepting
  membership of `argmax(q)`.
- `batch_run` **overrides** the base: encodes the whole batch once, one `bmm` per cell across
  all traces, per-trace early termination replayed from the recorded state path so
  `batch_run == [run(t) …]` exactly. `device="cuda"` supported.
- `acceptance_probability_tensor(P, lengths)` is the autograd-preserving exact soft entry
  point, validated at the public boundary; the inner transition kernel stays validation-free so
  benchmark timing is unchanged. **No experiment in this repo exercises the soft path** — the
  uncertainty harness lives in `artur_future_work/`.
- Every compiled monitor exposes an immutable `artifact_stats` record (`|AP|`, alphabet size,
  `|Q|`, transition/cube counts, actual persistent tensor elements/bytes).

### Correctness

`tests/test_deep_dfa.py`: matches `SymbolicDFAMonitor` on the full sweep **including nested
temporal — no xfails**; dense == factored on crisp traces; `batch_run == [run(t) …]` in both
modes; `exact_matrix` is row-stochastic and agrees with brute-force WMC on a non-read-once
guard; the tensor-native acceptance API preserves autograd and batch-length semantics; factored
handles n=24 atoms with no `2^24` tensor. Scan is verdict-identical to sequential
(`tests/test_deep_dfa_scan.py`).

### Unsafe claims — binding on both papers

From the status doc §5. Do **not** claim: that this artifact learns or extracts a DFA; that it
performs specification adaptation; that exact symbolic-automaton WMC or knowledge compilation
is new (**NeSyA owns it**); that cubes guarantee sub-exponential storage; that the recursive
Boolean evaluator is exact for every fractional guard; that exact acceptance marginals imply a
calibrated perceptor; that thresholding a marginal defines the correct probabilistic
three-valued semantics; that GPU DeepDFA is faster than symbolic monitoring before fair
measurement; that scan reduces total FLOPs; that a BDD/d-DNNF replacement is itself a
contribution.

## Benchmark Design

**Use synthetic traces for the timing experiments.** Trace content is irrelevant to per-step
cost — what matters is length and formula complexity; real data would conflate paradigm speed
with early-termination frequency. IJCNN 2014 does the same. (Real logs enter only via RQ5.)

**Reproduce and extend IJCNN 2014** — `◇ V_{i=1}^{n-1}(a_0 ∧ a_i)`, n = 2..32. ⚠ **The tree
shape was a material confound:** a left fold introduces a growing cost that must not be
attributed to formula breadth (RQ3 measured a left-deep/balanced latency ratio rising to 2.31×
at n=32). **`ijcnn_balanced_n*` is the paper-faithful family; `ijcnn_left_deep_n*` is a named
ablation only.**

**Two kinds of parallelism — keep distinct in the paper:**

| Kind | What runs in parallel | Where |
|---|---|---|
| **Within-step** | evaluation rules / matmul atoms within a single cell | RQ3 panel 1 (tree shape/depth) |
| **Cross-trace** | traces batched as matrix rows | RQ4 (both DeepDFA and RuleRunner batch this axis) |

**Timing methodology:** fresh process per cell, native cold compilation measured separately from
warm runtime, `EARLY_TERMINATION = False` for capacity figures (all paradigms process all cells
— the early-termination confound is real: on the IJCNN family symbolic's old ~1e-10 s/cell
measured "how fast it gives up"). Five independent seeds, medians with bootstrap 95% intervals,
short calls repeated inside ≥50 ms blocks. Every monitor stamps `effective_device`, so a CSV
never claims a GPU run that did not happen.

**Extending experiments:** follow `docs/EXPERIMENTAL_EVALUATION_PLAN.md` §6 (protocol) and §7
(schema). Do not add an experiment outside the RQ structure without scoping it there first.

## Key Papers in `papers/`

- `Claude 1.txt` / `Claude 2.txt` — planning docs (motivation; why `ltlf2dfa + custom runner`,
  LTL3 semantics, trap/sink precomputation, why Declare4Py and RV-Monitor were ruled out)
- `IJCNN 2014.PDF` / `IJCNN 2015.pdf` — RuleRunner: the system being modernized
- `DeepDFA.pdf` — DeepDFA (ECAI 2024)
- `IS__NeSyPPM.pdf` — NeSy PPM: source of the DeepDFA formulation we adopt (Eq. 18)
- `RuleRunner.pdf`, `cilp.pdf`, `TOSEMv4.pdf` — earlier RuleRunner work, the CILP translation,
  model-level adaptation background

## Environment Setup

The conda environment `nesy-monitoring` is already created. To reproduce from scratch:

```bash
conda env create -f environment.yml
conda activate nesy-monitoring
```

`environment.yml` pins all versions including `torch==2.6.0+cu124`. **Hardware:** heavy sweeps
run on **Google Colab** (CPU runtime and a GPU runtime, usually a Tesla T4); local dev is the
conda env. **There is no Docker in this project.**

## Commands

```bash
conda activate nesy-monitoring

# Install/reinstall in editable mode
# Note: use the full path — conda run resolves to system pip on this machine
/home/matteo/miniconda3/envs/nesy-monitoring/bin/pip install -e ".[dev]"

# Tests / one test
pytest
pytest tests/test_symbolic_dfa.py::test_eventually

# Lint
ruff check .

# The four research questions (fresh-process, resumable, writes results/<rq>/)
python experiments/rq1_semantic_characterization.py    # semantic gate — run first
python experiments/rq2_cost_of_correctness.py
python experiments/rq3_structural_scaling.py
python experiments/rq4_cross_architecture.py
python experiments/e2_instrumentation_smoke.py

# Regenerate the original-RuleRunner certificate artifact
python -m src.monitors.rulerunner.certificates --refresh

# Demo (DFA rendering + RuleRunner run tables)
python demo/demo_monitors.py
```

## Dependencies

- `ltlf2dfa==1.0.2` — LTLf → minimal DFA (Python wrapper for MONA; MONA must be on PATH)
- `torch==2.6.0+cu124` — PyTorch with CUDA 12.4 (RTX 3050 Laptop GPU locally; T4 on Colab)
- `numpy`, `matplotlib`, `pandas`, `scipy`, `tqdm` — via conda
- `pytest`, `ruff`, `black` — dev tools via conda
- `lark` — NOT a direct dependency; pulled in transitively by `ltlf2dfa`. RuleRunner's parse
  tree is built programmatically, not parsed from a grammar file.
