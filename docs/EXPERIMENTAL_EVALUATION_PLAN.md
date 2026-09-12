# Experimental Evaluation Plan

**Project:** Neuro-Symbolic Runtime Monitoring for LTLf  
**Target:** ICLR  
**Status:** active design and execution document  
**Last updated:** 2026-08-25

This document is the authoritative plan for the benchmark, experiment, and
results phase of the current paper. It records the decisions behind the
experimental design, tracks progress, and prevents stale measurements or old
assumptions from silently returning to the paper.

For implementation status, consult
[`rulerunner_status.md`](rulerunner_status.md) and
[`deepdfa_status_and_future_work.md`](deepdfa_status_and_future_work.md).
The older [`EXPERIMENT_MAP.md`](../old/docs/EXPERIMENT_MAP.md) and its associated
scripts/results are archived under `old/` for archaeology. The research
questions and validity requirements in this document govern the final
evaluation.

---

## 1. Experimental thesis

This is a theory/architecture foundation paper. Its empirical objective is not
to force a neuro-symbolic speed win over symbolic monitoring. On crisp traces,
a compact deterministic automaton is an exceptionally strong baseline and may
remain the fastest implementation throughout the tested range.

The experiments should instead establish four pieces of new knowledge:

1. the exact semantic boundary of the published RuleRunner representation;
2. the coverage and cost of the bounded and progression repairs;
3. the structural variables that govern each implementation's computation,
   memory, and compilation costs;
4. the deployment regimes in which the different representations are useful.

The intended result is an evidence-backed efficiency and applicability
landscape, not a single winner. A negative crossover result is valid if the
experiment has enough power to reveal the crossover had it existed.

The current paper remains crisp and does not claim probabilistic monitoring,
specification adaptation, or end-to-end perceptor learning. Those threads live
under `artur_future_work/`.

---

## 2. Current project status

### 2.1 Technical status

- Original RuleRunner is a faithful implementation of the published rule
  representation, including its one-register-per-subformula limitation.
- The exact product certifier decides final-language equivalence, prefix
  soundness, and exact online-label equivalence formula by formula.
- Bounded-event RuleRunner implements a partial fixed-window repair in flat and
  structured CILP organizations, with sound default online labels and an
  optional exact-online extrapolation head.
- Progression RuleRunner implements a complete repair in lazy, eager, flat, and
  structured forms.
- Fixed DeepDFA implements dense, exact cube-factored, and prefix-scan
  representations and exposes representation statistics.
- The complete repository test suite passed on 2026-08-25 with **884 passed,
  29 skipped, and 6 expected failures**. The expected failures preserve only
  the original RuleRunner limitation.

### 2.2 Result status

The CSVs now archived under `old/results/cpu/` and `old/results/gpu/` were
generated on 2026-07-09 and 2026-07-10. They predate the August RuleRunner
repairs and the current DeepDFA implementation. They are historical diagnostics,
not submission evidence, and must not be quoted as final results.

The current experimental section of the manuscript is a placeholder. In
particular, its claim that all paradigms use a shared `ltlf2dfa` compilation
pipeline is false and must be replaced after the new measurements exist.

---

## 3. Audit findings that constrain the redesign

### 3.1 IJCNN formula tree shape

The IJCNN 2014 paper rewrites its large disjunction into an explicitly balanced
binary tree. The current formula generator emits an unparenthesized n-ary
disjunction, while the RuleRunner parser left-folds n-ary `&` and `|` to match
`ltlf2dfa` semantics. This changes RuleRunner depth from approximately
`O(log n)` to `O(n)`.

The parser must **not** be globally rebalanced: its current association matches
the formula compiler and is needed for semantic consistency. The benchmark
generator must instead produce two explicitly parenthesized variants:

- `ijcnn_balanced_n*`: paper-faithful replication;
- `ijcnn_leftdeep_n*`: semantically equivalent tree-shape ablation.

Both variants must be checked for DFA-language equivalence and must record
their actual AST depth, AST node count, rule count, and compiled network size.
This gives a controlled experiment in which semantics, alphabet, and minimal
DFA are held fixed while syntactic depth changes.

### 3.2 Nested-Next reinstallation

The earlier audit found an incorrect `X` reinstallation implementation. That
defect has since been repaired. Existing Experiment 5 results nevertheless
predate the repair and are stale.

The existing nested-`X` depth family remains a poor depth instrument because
it changes temporal horizon, DFA size, progression closure, and original
RuleRunner applicability at the same time. Balanced-versus-skewed Boolean
association is the preferred controlled depth benchmark. Nested `X` remains
useful for temporal-horizon and bounded-event experiments.

### 3.3 Compilation is paradigm-specific

The paradigms do not share one compilation cost:

- symbolic DFA: LTLf/MONA construction, guard parsing, and sink/trap analysis;
- fixed DeepDFA: the same DFA followed by dense or cube lowering and device
  transfer;
- original RuleRunner: parse tree, rule generation, and CILP lowering, without
  MONA in ordinary compilation;
- bounded RuleRunner: event extraction, certificate lookup, pipeline lowering,
  and optionally composite-state extrapolation;
- progression RuleRunner: residual discovery, relevant-observation
  enumeration, label analysis, and CILP lowering.

`compile_ltlf` is globally cached. Cold-compilation measurements must therefore
use fresh subprocesses or explicitly cleared caches. Monitor order must never
determine whether a compilation measurement is a cache hit.

Two different quantities will be reported:

1. **native cold-start cost**: the complete formula-to-runnable-monitor path;
2. **incremental backend cost**: monitor-specific lowering from an already
   available shared artifact, where such sharing is meaningful.

### 3.4 Existing measurements are incomplete

The old harness records only mean and standard deviation of warm wall-clock
runtime over repeated execution of one compiled monitor on one fixed trace
sample. It does not measure:

- compilation stages;
- persistent or peak memory;
- input encoding separately from monitor computation;
- latency percentiles;
- variability over trace seeds or formulas;
- actual processed cells under early termination;
- timeouts, OOMs, unsupported configurations, or scan fallbacks;
- enough structural metadata to explain a scaling curve.

The final harness must retain raw repetition records and distinguish timing
jitter from variation over workloads.

### 3.5 Baseline descriptions must match their implementations

- The current symbolic `DFA.step` iterates over outgoing guard closures; it is
  not literally one dictionary lookup. An optional materialized symbol-indexed
  transition table is a useful intra-symbolic reference where the alphabet
  fits.
- The original IJCNN study compared vanilla, sparse, and GPU RuleRunner
  implementations. The current flat CILP matrices are dense. We may reuse the
  IJCNN formula family without claiming a complete reproduction, or implement
  a sparse flat backend if direct reproduction becomes a paper claim.
- The current structured RuleRunner schedule evaluates modules sequentially
  within a cell. It is functionally correct but not performance-equivalent to
  an ideal same-level-fused schedule. Results must use an accurate label such
  as "module-scheduled structured RuleRunner" unless level fusion is
  implemented.
- The evaluated DeepDFA is a fixed tensor realization of a compiled DFA, not
  the trainable automaton-induction architecture of the original DeepDFA work.

---

## 4. Research questions

### RQ1 — Semantic boundary and repair coverage

**Question.** Where exactly does the published RuleRunner representation fail,
and which portions of that boundary are recovered by bounded-event and
progression RuleRunner?

**Comparisons.** Original RuleRunner, bounded default, bounded exact-online,
and progression RuleRunner against the canonical finite-trace semantics.

**Measures.**

- final-language equivalence;
- prefix soundness;
- exact-online-label equivalence;
- shortest counterexample and its length;
- bounded construction admitted/rejected status;
- extracted bounded horizon;
- progression semantic agreement;
- counts by explicitly declared formula category.

An aggregate support percentage is valid only for a declared corpus or formula
sampling distribution. Curated suites should report raw category counts.

### RQ2 — Cost of correctness

**Question.** What compilation, representation, runtime, and online-latency
cost is paid when moving from original to bounded to complete progression
RuleRunner?

**Comparisons.**

- original flat versus original structured;
- bounded flat versus bounded structured;
- bounded default versus bounded exact-online;
- progression flat versus progression structured;
- paired original/bounded/progression comparisons on formulas where every
  displayed monitor is applicable.

**Measures.**

- native cold compilation and compilation-stage time;
- certificate generation time, reported separately from normal bounded
  compilation;
- peak compilation memory;
- persistent representation bytes;
- rule, literal, module, nonzero-weight, root, closure, and state counts;
- batch-1 latency and batched throughput;
- semantic earliest-decision index;
- reported decision index;
- decision lag in cells and wall-clock time.

For bounded default mode,

```text
decision_lag = reported_decision_index - semantic_earliest_decision_index
```

is a primary result rather than a footnote.

### RQ3 — Structural bottlenecks within each architecture

**Question.** Which structural variables govern each implementation, and do
the measured trends match the stated cost models?

**Intra-RuleRunner comparisons.**

- balanced versus left-deep equivalent formulas;
- flat versus module-scheduled structured CILP;
- optional dense versus sparse flat CILP;
- original versus bounded versus progression representation growth.

**Intra-DeepDFA comparisons.**

- dense versus exact cube-factored transitions;
- sequential dense versus prefix scan;
- guard families with different cube complexity at fixed or controlled
  `|AP|` and `|Q|`;
- a `(B, L, |Q|)` scan speedup phase diagram, with fallbacks recorded.

**Intra-symbolic comparisons.**

- current compact guarded transitions;
- optional materialized symbol-indexed table where `2^|AP|` is feasible.

**Measures.** Actual structural statistics, static bytes, peak temporary
memory, compilation time, latency, throughput, and observed scaling exponent or
paired growth ratio. Source-level formula size alone is not an adequate x-axis.

### RQ4 — Cross-architecture efficiency landscape

**Question.** Under equal semantic workloads, in which deployment regimes is
each exact architecture efficient?

**Headline comparisons.**

- symbolic DFA;
- original RuleRunner only where exactly certified;
- bounded RuleRunner only where admitted, with default and exact-online modes
  kept distinct;
- progression RuleRunner;
- fixed DeepDFA dense where feasible and exact cube-factored elsewhere.

**Runtime modes.**

1. **Capacity mode:** early termination disabled and all offered cells
   processed.
2. **Deployment mode:** early termination enabled on a declared trace
   distribution.

**Timing boundaries.**

1. **End-to-end:** raw observations through verdict production, including
   ordinary encoding and device transfer.
2. **Core/native:** pre-encoded inputs, device-resident when appropriate, used
   to explain architectural computation separately from interface overhead.

**Measures.**

- batch-1 median and p95 latency;
- absolute time per trace;
- cells per second and traces per second;
- processed-cell fraction under early termination;
- persistent and peak runtime memory;
- native cold-start cost;
- cold-start break-even workload;
- OOM/timeout boundary.

Absolute measurements precede speedups. Speedups always name their baseline.

### RQ5 — External validity on realistic specifications and traces

**Question.** Do the controlled findings survive realistic constraint
topologies, trace lengths, event frequencies, and early-verdict behavior?

**Benchmarks.** The existing Declare template suite followed by a crisp BPIC
case study if the controlled experiments are stable.

**Measures.**

- constraint and event-to-atom encoding;
- trace-count and length distribution;
- verdict balance and trigger frequency;
- original/bounded applicability;
- semantic earliest-decision distribution;
- end-to-end events/s and cases/s;
- p50 and p95 case latency;
- peak memory.

Uniform Bernoulli traces remain useful for controlled capacity experiments but
must not be the only deployment workload. Early-termination experiments require
stratification by verdict, trigger rate, and response delay.

### RQ6 — Optional learning question

**Question.** What does exact-by-construction monitoring buy relative to learned
sequence models in sample efficiency and length/formula generalization?

This RQ is **not currently in scope**. A formula-specific LSTM or Transformer
imitation baseline would not automatically strengthen the paper because it
solves a different, easily criticized problem. If the scope expands, the task
must be designed as a meaningful learning/generalization question rather than
as a token neural baseline.

For the current foundation-paper decision, unsupported claims about RNNs and
Transformers should be removed or softened. The introduction currently mentions
them, and the conclusion contains an empty `Transformers` subsection; these are
draft remnants, not evaluated contributions.

---

## 5. Benchmark families

| Family | Main variable | Controlled variables | Primary RQ |
|---|---|---|---|
| Balanced/left-deep IJCNN | AST depth and `|AP|` | equivalent language and minimal DFA for paired shapes | RQ3, RQ4 |
| Guard-complexity `F(g_n)` | `|AP|` and disjoint-cube count | usually `|Q| = 2` | RQ3 |
| Bounded response | deadline, horizon, roughly linear `|Q|` | `|AP| = 2` | RQ2–RQ4 |
| `F(a & X^k b)` | exponential `|Q| = 2^k + 1` | `|AP| = 2` | RQ3, RQ4 |
| RuleRunner boundary suite | overlapping obligations and unsafe skeletons | small interpretable formulas | RQ1, RQ2 |
| Declare suite | realistic temporal topology | common template vocabulary | RQ1, RQ5 |
| BPIC case study | empirical traces and length distribution | fixed declared constraints | RQ5 |

### 5.1 Guard-complexity strata

The `F(g_n)` family should include Boolean guards with different compiled
representations at comparable atom counts:

- compact read-once guards;
- threshold/majority guards;
- parity or another deliberately high cube-count family.

The x-axis must be the **actual compiled cube count** and mask bytes, not an
assumption based on surface syntax. MONA may rewrite guards.

### 5.2 State-growth strata

Linear and exponential state growth answer different questions and must remain
separate:

- bounded response is a controlled, small-alphabet state/horizon knob;
- `F(a & X^k b)` is a genuine exponential-state family and exposes a shared
  symbolic/DeepDFA wall.

Analytic extrapolations must be visibly separated from measured points.

---

## 6. Measurement protocol

### 6.1 Repetitions and statistics

- Retain one record per raw timing repetition.
- Use at least 5 independent trace seeds per workload; increase this if
  between-seed variability is material.
- Use identical traces for paired monitor comparisons.
- Randomize monitor execution order within a block to reduce order and thermal
  bias.
- Repeat until a minimum accumulated timed duration rather than relying only
  on a fixed number of very short measurements.
- Report median and bootstrap 95% confidence intervals as primary summaries.
- Report p95 latency for deployment-facing results.
- Use paired log-ratios with confidence intervals for relative comparisons.
- Treat repeated runs on one compiled monitor separately from fresh-process
  compilation repetitions.

### 6.2 Hardware and execution control

- Record CPU model, core count, RAM, GPU model, GPU memory, OS, Python, Torch,
  CUDA, driver, MONA, and `ltlf2dfa` versions.
- Record effective rather than requested device.
- Fix and record Torch/BLAS thread counts.
- Use explicit CUDA synchronization around timed regions.
- Reset CUDA peak-memory statistics separately for compilation and execution.
- Prefer CPU and GPU measurements from the same host when drawing a direct
  device comparison.
- Keep CPU-only baselines visibly labeled as CPU even when run during a GPU
  session.

### 6.3 Memory and representation size

Report both logical/allocated representation size and process-level peak
memory:

- DeepDFA: tensor bytes, cube count, mask elements, temporary matrix peak;
- RuleRunner: rules, literals, nonzero weights, dense/sparse tensor bytes,
  modules;
- progression: roots, syntactic closure, aggregate states, label-head size;
- bounded: horizon, event modules, pipeline state, composite exact-head states;
- symbolic: states, transitions, guard size, and process RSS.

Python object size estimates alone are insufficient. Peak host memory should be
measured in an isolated subprocess. Analytic idealized sizes may be included as
secondary curves if labeled clearly.

### 6.4 Failure and censoring records

Every attempted cell must yield a row, including unsuccessful cells:

```text
status = success | timeout | oom | unsupported | fallback | error
```

The record must include the resource budget and the reason. A missing point is
not an adequate representation of a scalability wall. Prefix-scan fallback must
never be reported as a scan measurement.

### 6.5 Correctness gate

Before timing a monitor/formula pair:

- the formula must compile successfully;
- complete monitors must agree on final semantics;
- original RuleRunner must have an exact stored certificate;
- bounded RuleRunner must have a valid cached certificate and explicit mode;
- unsafe or unsupported pairs must be rejected or placed only in RQ1, not in a
  headline efficiency comparison.

---

## 7. Result schema and provenance

The replacement result schema should not overload one generic `n_leaves`
column. At minimum, every record should identify:

### Run identity

- run ID and timestamp;
- git commit and dirty-worktree fingerprint;
- experiment/RQ version;
- host and software environment;
- random seeds.

### Formula identity

- stable formula ID and exact text;
- source and family;
- family parameters and tree shape;
- `n_atoms`, AST nodes, AST depth, and temporal depth;
- minimal-DFA states and transitions where available;
- compiled cube/rule/root/closure statistics as applicable.

### Monitor identity

- architecture, construction, and backend;
- requested and effective device;
- dtype and monitor-specific options;
- early-termination and online-label mode.

### Outcome

- status and failure reason;
- raw compilation-stage times;
- raw runtime duration;
- offered and processed traces/cells;
- persistent bytes, peak host RSS, and peak GPU memory;
- semantic and reported decision indices where applicable.

CSV can remain the analysis interchange format, but a JSON manifest should
record nested configuration/provenance cleanly.

---

## 8. Intended paper outputs

The main paper should aim for a small number of information-dense outputs:

1. **Correctness/applicability table** across formula categories and the three
   RuleRunner constructions.
2. **Cost-of-correctness figure** showing compilation, memory, runtime, and
   bounded-default decision lag.
3. **Structural phase figure** with tree depth, guard/cube complexity, and
   state-growth panels.
4. **Batch/hardware phase diagram** showing absolute latency and throughput
   regimes rather than many overlapping speedup curves.
5. **Realistic-workload table or figure** for Declare/BPIC end-to-end behavior.

Trace-length flatness is a harness sanity check and can move to the appendix.
Correctness of exact monitors is established by exhaustive/certifier evidence,
not by an accuracy bar chart.

---

## 9. Execution order and progress tracker

### Phase E0 — Freeze protocol and benchmark schema

- [x] Replace overloaded benchmark metadata with explicit structural fields.
- [x] Implement paper-faithful balanced IJCNN generation.
- [x] Retain left-deep IJCNN as a named ablation.
- [x] Add tests for shape, depth, and semantic equivalence.
- [x] Define the raw result and provenance schema.
- [x] Define timeout and memory budgets.

**Exit criterion:** the benchmark catalog and result schema are versioned, and
every later experiment can refer to stable formula IDs and structural fields.

**Completed 2026-08-25.** The schema version is `e0.v1`. The default per-cell
policy allows 300 seconds for native cold compilation and 300 seconds for
execution, with host and accelerator memory capped at 80% of the resolved
machine capacity. A run manifest must record the corresponding byte limits.
The archived July CSVs remain loadable through an explicit legacy migration,
but they remain stale diagnostics and do not satisfy the new raw schema.

### Phase E1 — RQ1 semantic characterization

- [x] Assemble original-safe controls.
- [x] Assemble targeted overlapping-obligation counterexamples.
- [x] Assemble bounded-event admitted and rejected formulas.
- [x] Add the Declare suite.
- [x] Run the exact original-RuleRunner certifier for every formula.
- [x] Record shortest witnesses and certificate dimensions.
- [x] Record bounded extraction/admission and exact-head state counts.
- [x] Verify progression and DFA agreement.
- [x] Produce the versioned RQ1 CSV/JSON artifact.
- [x] Draft the correctness/applicability table.

**Exit criterion:** no later timing experiment can silently include an unsafe
original RuleRunner or unsupported bounded RuleRunner pair.

**Completed 2026-08-25.** Schema `rq1.v1` contains 68 exact records for 17
formulas and four constructions. Original RuleRunner is applicable to all 9
declared safe controls and none of the 8 declared collision cases. Bounded
RuleRunner admits all 6 finite-horizon repair targets and explicitly rejects
the 2 unsafe unbounded skeletons; its default is exact-online on 12 of 15
admitted formulas and sound but late on 3. Its exact head is exact-online on all
15 admitted formulas. Progression agrees with the canonical DFA on all 17.
See [`RQ1_SEMANTIC_CHARACTERIZATION.md`](RQ1_SEMANTIC_CHARACTERIZATION.md).

### Phase E2 — Instrument compilation, representation, and memory

- [x] Add fresh-process native cold-compilation measurement.
- [x] Add compilation-stage instrumentation.
- [x] Add artifact statistics to all monitor families.
- [x] Add peak host RSS measurement.
- [x] Add peak CUDA memory measurement.
- [x] Record unsuccessful/OOM/timeout cells explicitly.

**Completed 2026-08-25.** Schema `e2.v1` defines twelve configurations across
symbolic DFA, original/bounded/progression RuleRunner, and dense/factored/scan
DeepDFA. Every cell runs in a fresh process behind the frozen RQ1 gate, records
native compilation stages and persistent artifact statistics, and emits an
explicit success, fallback, timeout, OOM, unsupported, or error row. Host RSS
is delimited to the compile phase and includes MONA descendants; CUDA allocated
and reserved peaks are synchronized and recorded when CUDA is available. The
checked-in CPU smoke covers all twelve adapters, but its numbers are
infrastructure checks rather than paper evidence. Real accelerator values still
require a controlled GPU run. See
[`E2_INSTRUMENTATION.md`](E2_INSTRUMENTATION.md).

### Phase E3 — RQ2 cost of correctness

- [x] Select paired formulas from the frozen RQ1 corpus.
- [x] Run original/bounded/progression flat comparisons.
- [x] Run encoding ablations separately from construction comparisons.
- [x] Measure bounded-default decision lag.
- [x] Produce cost-of-correctness figure/table.

**Completed 2026-08-25.** Schema `rq2.v1` contains 210 successful fresh-process
compilation rows, 420 successful paired runtime rows, 30 separately measured
offline certificate rows, and 8,800 exhaustive decision-lag rows. Invalid
original and unsupported bounded configurations are excluded by `rq1.v1`, not
timed as baselines. The exact-online head has zero lag on all 4,400 enumerated
traces; bounded default exposes the three predicted late-label regimes. The
current absolute timings are a controlled local-CPU candidate and require a
final-machine rerun before paper integration. See
[`RQ2_COST_OF_CORRECTNESS.md`](RQ2_COST_OF_CORRECTNESS.md).

### Phase E4 — RQ3 structural scaling

- [x] Balanced-versus-left-deep RuleRunner experiment.
- [x] DeepDFA dense/cube guard-complexity experiment.
- [x] DeepDFA scan `(B, L, |Q|)` phase diagram.
- [x] Linear state/horizon experiment.
- [x] Exponential state-blowup experiment.
- [x] Validate measured trends against recorded artifact statistics.

**Completed 2026-08-25.** Schema `rq3.v1` contains 335 successful isolated
compilations and 505 successful runtime rows across five non-conflated panels.
The left-deep IJCNN penalty grows to 2.31× at `n=32`; compiled guard cubes and
bytes distinguish read-once/threshold/parity representations; CPU scan wins at
batch 1 but usually loses at batch 32; and the state panels separate linear
growth from the `2^k+1` wall. State counts, dense bytes, and absence of scan
fallback are executable artifact gates. Absolute timings remain a controlled
local-CPU candidate. See
[`RQ3_STRUCTURAL_SCALING.md`](RQ3_STRUCTURAL_SCALING.md).

### Phase E5 — RQ4 cross-architecture evaluation

- [x] Capacity-mode end-to-end benchmark.
- [x] Core/native benchmark for overhead attribution.
- [x] Deployment-mode early-termination benchmark.
- [ ] CPU and GPU runs with controlled provenance.
- [x] Produce absolute latency/throughput and Pareto/phase outputs.

**CPU block completed 2026-08-25.** Schema `rq4.v1` contains 315 successful
fresh-process CPU compilations and 840 successful CPU runtime rows across three
semantically shared formulas, seven headline configurations, and the three
declared timing modes. Symbolic is the steady-state CPU runtime winner in every
measured workload; batching narrows but does not invert the gap. Progression
and original RuleRunner retain a short-lived total-cost regime through faster
MONA-free compilation. Deployment rows show that only the symbolic sequential
path currently converts early semantic decisions into skipped cells; all
vectorized tensor paths process the full rectangular batch. Absolute summaries,
symbolic-relative break-even rows, and three-dimensional Pareto rows are
checked in. The requested-CUDA audit produced explicit unsupported rows because
this host has no CUDA device, so the combined CPU/GPU item remains incomplete.
See [`RQ4_CROSS_ARCHITECTURE.md`](RQ4_CROSS_ARCHITECTURE.md).

### Phase E6 — RQ5 external validity

- [ ] Freeze Declare workload generation.
- [ ] Scope BPIC dataset and event encoding.
- [ ] Run certified applicability analysis.
- [ ] Measure realistic end-to-end throughput and decision timing.

### Phase E7 — Paper integration

- [ ] Replace the placeholder experimental section.
- [ ] Remove the false shared-compilation statement.
- [ ] Replace expected-result prose with measured claims.
- [ ] Trim unsupported RNN/Transformer framing.
- [ ] Add a reproducibility statement and artifact command table.
- [ ] Move secondary sanity checks and full grids to the appendix.

---

## 10. Immediate next action

Phases E0–E4 and the controlled CPU portion of E5 are complete. The immediate
next action is to rerun the frozen `rq4.v1` grid on a controlled CUDA host. The
local requested-CUDA audit is explicitly unsupported and cannot answer the GPU
crossover or memory questions. After that accelerator block is merged without
mixing requested/effective devices, proceed to Phase E6 / RQ5 external validity.
RQ1 remains the semantic gate.

---

## 11. Decision log

### 2026-08-25

- Keep the current paper as a crisp theory/architecture foundation paper.
- Do not chase a forced DeepDFA speed win; report the measured landscape.
- Treat all July timing results as stale diagnostics.
- Make RQ1 the first research question and semantic gate.
- Reproduce the IJCNN benchmark with an explicit balanced tree; retain the
  left-deep form only as an ablation.
- Measure compilation and compiled memory separately from warm runtime.
- Keep bounded default and bounded exact-online configurations separate.
- Trim unsupported RNN/Transformer claims rather than adding a token learning
  baseline.
- Freeze explicit source-family metadata (`family`, `source`, `parameters`,
  `tree_shape`, and semantic roles) separately from compiler-derived structure.
- Adopt `ijcnn_balanced_n*` as the canonical IJCNN family and
  `ijcnn_left_deep_n*` as its controlled tree-shape ablation.
- Adopt raw/provenance schema `e0.v1`, explicit failure statuses, and portable
  timeout/memory budgets resolved by the run manifest.
- Freeze the `rq1.v1` repair-ladder artifact: 17 formulas, 4 constructions, and
  exact products against canonical finite-trace semantics.
- Use semantic applicability rather than compilability: original RuleRunner is
  admitted only with exact final-language and sound-prefix certificates, while
  bounded RuleRunner is admitted only when its eventized skeleton satisfies the
  same gate.
- Preserve bounded default and exact-online as distinct rows: the default is
  final-exact and prefix-sound on all admitted formulas but is late on 3 of 15;
  the priced exact head removes those online-label differences.
- Define compile memory as whole-process-tree RSS during an explicit compile
  phase, separately from persistent tensor storage and the absolute worker
  peak; include MONA descendants in that compile peak.
- Launch every measured cell from a fresh process with an explicit Python
  environment snapshot so hidden native-runtime environment mutations cannot
  create test-order or monitor-order effects.
- Treat the twelve-cell E2 CPU artifact only as instrumentation validation;
  generate repeated, controlled CPU and GPU measurements for paper claims.
- Archive the superseded numbered experiment suite and July CPU/GPU results
  under `old/`; active artifacts begin at E0 and remain under top-level
  `results/`.
- Stratify RQ2 by semantic applicability: compare all three constructions only
  on safe formulas, bounded versus progression on bounded repairs, and only
  progression on unbounded cases rejected by the bounded construction.
- Treat cell lag as the primary synthetic online-latency metric. Monitor compute
  after semantic decisiveness is recorded separately and must not be presented
  as event-time latency without an arrival cadence.
- Separate RQ3 into tree, guard/cube, scan, linear-state, and exponential-state
  panels. A generic formula-size axis is not an acceptable substitute.
- Use compiled cube counts and tensor bytes as the guard-complexity axis; the
  small CPU grid does not support a universal latency law from source family
  names alone.
- Freeze the RQ4 headline matrix to three formulas supported by every displayed
  construction and seven configurations: symbolic, certified original,
  bounded default/exact, progression, and dense/factored DeepDFA.
- Treat predecoded native `Observation` mappings as the comparable core
  interface boundary. Backend-owned tensorization and transfer remain included;
  private device-kernel microbenchmarks are explanatory intra-architecture
  measurements rather than headline cross-architecture results.
- Record semantic early-decision indices separately from physical processed
  cells. The current tensor batch implementations receive no early-termination
  compute credit because they advance the full rectangular batch.
- Report the negative local-CPU crossover result: symbolic wins every measured
  steady-state workload, while MONA-free RuleRunner compilation creates a
  distinct short-lived total-cost regime.
- Keep E5's hardware item open until a real CUDA block exists. Requested CUDA
  on a CPU-only host yields explicit unsupported records, not GPU evidence.
