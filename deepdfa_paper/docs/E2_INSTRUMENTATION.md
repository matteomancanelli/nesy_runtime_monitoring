# E2 Compilation, Representation, and Memory Instrumentation

**Schema:** `e2.v1` over benchmark schema `e0.v1`  
**Semantic gate:** `rq1.v1`  
**Smoke command:** `python experiments/e2_instrumentation_smoke.py`

Phase E2 supplies the measurement substrate for the later cost and scaling
experiments. The checked-in smoke artifact verifies the substrate locally; its
numbers are not paper results and must not be quoted as architecture rankings.

## Isolation and cache hygiene

Every cell is launched as a distinct Python process. A worker receives exactly
one formula, monitor specification, device, repetition, and optional validation
workload. Therefore:

- `compile_ltlf`, progression, and certificate Python caches begin empty;
- monitor order cannot turn another construction's cold compile into a cache
  hit;
- a timeout or host-memory violation kills the entire process group, including
  MONA descendants;
- every attempted cell yields one explicit terminal row.

The parent passes an explicit snapshot of Python's environment to each worker.
This prevents hidden C-level environment mutations made by native numerical
runtimes (for example, an MKL threading-layer change after a particular import
order) from making later cells depend on parent-process test or import order.

RuleRunner cells first consult the frozen RQ1 applicability matrix. An unsafe
original construction or rejected bounded skeleton yields `unsupported`
without compilation. Symbolic DFA and fixed DeepDFA inherit canonical DFA
semantics and do not need an RQ1 construction row.

## Monitor registry and native stages

The registry contains twelve independently identified configurations:

- symbolic guarded DFA;
- original RuleRunner, flat and module-scheduled structured;
- bounded RuleRunner, flat/structured × default/exact-online;
- progression RuleRunner, flat and structured;
- fixed DeepDFA, dense, factored-cube, and dense prefix-scan.

| Family | Native stages |
|---|---|
| Symbolic DFA | LTLf→minimal DFA; monitor initialization |
| Original RuleRunner | formula parse; rules plus flat/structured CILP lowering; initialization |
| Bounded RuleRunner | event extraction; certificate lookup; pipeline lowering; skeleton lowering; optional exact-online head; initialization |
| Progression RuleRunner | residual/factorized graph discovery; flat/structured CILP lowering; initialization |
| Fixed DeepDFA | LTLf→minimal DFA; dense/cube tensor lowering; initialization |

`compile_total_s` spans the complete adapter entry to a runnable monitor after
the required device runtime is available. Named stage durations are explicitly
synchronized on CUDA. The small difference between total and the sum of stages
is module import, dispatch, and bookkeeping overhead and is intentionally
retained rather than assigned to an arbitrary stage.

## Memory definitions

The worker and parent record several non-interchangeable quantities:

- `startup_host_rss_bytes`: the fresh interpreter before project imports;
- `baseline_host_rss_bytes`: after common instrumentation and the required
  device runtime, immediately before monitor construction;
- `peak_compile_host_rss_bytes`: maximum whole-process-tree RSS while the
  worker phase is exactly `compile`, including MONA children;
- `peak_observed_host_rss_bytes`: maximum over the entire worker, including
  post-compilation artifact inspection;
- baseline, peak allocated, and peak reserved CUDA bytes for a CUDA compile;
- persistent unique tensor-storage bytes and logical tensor/NumPy sizes in the
  compiled monitor.

The compilation increment is the compile peak relative to its baseline. The
absolute worker peak is retained for capacity planning but must not be mistaken
for representation size. Python object graphs are not estimated with
`sys.getsizeof`; process-tree RSS and actual tensor storage are reported
separately.

## Representation statistics

Every successful row contains common storage counts plus native logical
statistics:

- symbolic: DFA states, guarded transitions, accepting/trap/sink states;
- original RuleRunner: evaluation/reactivation rules, literals, modules, AST
  nodes/depth, tensor bytes, and nonzero tensor elements;
- bounded RuleRunner: event horizon/islands/modules, skeleton rule statistics,
  certificate product states/source, and optional exact-head states;
- progression: residual/aggregate states, roots, closure, input subformulae,
  modules or transition clauses;
- DeepDFA: states, guarded transitions, alphabet size, dense tensor elements
  and bytes, or cube counts/masks/bytes.

These are allocated implementation sizes. They are not minimal semantic state
counts, and logical tensor bytes should not be substituted for process peak
memory.

## Terminal statuses and resource control

The schema uses the E0 status vocabulary:

```text
success | timeout | oom | unsupported | fallback | error
```

The parent polls whole-process-tree RSS and wall time against the resolved E0
budgets. The worker catches Python/CUDA OOM, construction guards, missing
certificates, and other errors. Optional validation execution captures warnings;
a DeepDFA scan warning containing an actual sequential fallback changes the row
status to `fallback`, so it cannot be reported as a scan measurement.

CUDA fields are populated only when CUDA is available and actually requested.
On a CPU-only machine a tensor monitor requested on CUDA produces an explicit
`unsupported` row. The checked-in smoke is CPU-only; final GPU measurements
must be generated on the controlled accelerator host.

## Smoke artifact

`results/e2/e2_instrumentation_smoke.{csv,json}` contains one successful CPU
cell for each of the twelve registry entries. It checks:

- distinct worker PIDs;
- nonempty native stage maps;
- monitor-specific representation statistics;
- compilation-delimited and whole-worker peak RSS;
- validation execution without silent fallback.

The artifact is labeled `instrumentation_smoke_not_paper_results`. Later E3–E5
scripts should reuse the executor but write separate versioned result sets with
multiple repetitions and controlled hardware provenance.
