# RQ2 Cost of Correctness

**Schema:** `rq2.v1`  
**Semantic gate:** `rq1.v1`  
**Generator:** `python experiments/rq2_cost_of_correctness.py`

## Experimental question

RQ2 measures the cost paid when replacing the published RuleRunner
representation with the bounded or complete progression repair. It deliberately
does not time a semantically invalid original monitor as if it were a valid
baseline.

The seven-formula corpus has three explicit strata:

- `all_applicable`: `nested_next_control` and `ijcnn_balanced_n4`, on which
  original, bounded, and progression constructions can be compared;
- `bounded_repair`: `eventual_next_alias` plus the three late-default cases
  `next_offset_alias`, `globally_next_alias`, and `until_next_alias`, on which
  bounded and progression are valid but original is not;
- `progression_only`: unbounded `response`, rejected by both original and the
  bounded construction.

Flat construction comparisons and flat-versus-structured encoding ablations
are separate axes. Bounded default and bounded exact-online are also separate
configurations.

## Protocol

Each formula/configuration has five independent fresh-process repetitions. A
worker performs native cold compilation, records the E2 stage/memory/artifact
measurements, warms the monitor, and times paired batch sizes 1 and 64 on the
same seeded length-32 traces. Early termination is disabled. Very short calls
are repeated inside a timed block lasting at least 50 ms; the raw record stores
the inner-iteration count and normalized duration.

Offline exact certificate generation is measured in thirty additional fresh
processes and is not charged to cached bounded compilation. Every attempted
cell has an explicit status. Median and bootstrap 95% intervals are computed
from the five process-level repetitions.

Decision lag is evaluated exhaustively, rather than estimated from the timing
traces. All fixed-length traces are enumerated for:

| Formula | Length | Traces per label mode |
|---|---:|---:|
| `eventual_next_alias` | 4 | 256 |
| `next_offset_alias` | 4 | 16 |
| `globally_next_alias` | 5 | 32 |
| `until_next_alias` | 4 | 4,096 |

The semantic decision index is the first permanent canonical-DFA label, or the
finite-trace boundary if no permanent label occurs earlier. The reported index
is defined identically for bounded RuleRunner. `additional_monitor_compute_s`
measures only monitor computation after semantic decisiveness; it is not an
event-time latency because the synthetic traces have no arrival cadence.

## Current controlled-CPU result

All 210 compilations, 420 runtime workloads, and 30 offline certificate runs
succeeded without fallback. The current artifact is a controlled CPU-run
candidate from the local host; absolute timings should be regenerated on the
final submission machine before entering the paper.

On the two safe flat controls, cached bounded-default compilation is about
2.1–2.2× original compilation, while batch-1 latency is about 1.07–1.14×.
The exact-online head adds compilation/storage cost but changes warm runtime
little at these sizes. Flat progression is compact and, in this implementation
and workload, faster than the other valid RuleRunner constructions; this is a
measured implementation result, not an asymptotic claim.

Structured scheduling is not a free semantic change. Across the applicable
formulas, its median batch-1 slowdown relative to the corresponding flat
encoding is approximately 3.8× for original, 1.4–1.5× for bounded, and 5.2×
for progression. These are encoding ablations and should not be conflated with
repair-construction comparisons.

Offline exact certificate generation has a roughly 50–53 ms median for these
small formulas on this host. Normal bounded compilation uses the frozen cache,
so this cost remains separately visible.

### Exhaustive decision lag

| Formula | Mode | Late traces | Late fraction | Median lag | p95 lag | Max lag |
|---|---|---:|---:|---:|---:|---:|
| `eventual_next_alias` | default | 0 / 256 | 0% | 0 | 0 | 0 |
| `next_offset_alias` | default | 8 / 16 | 50% | 0.5 | 1 | 1 |
| `globally_next_alias` | default | 32 / 32 | 100% | 1.5 | 4.45 | 5 |
| `until_next_alias` | default | 1,606 / 4,096 | 39.2% | 0 | 1 | 1 |
| all four formulas | exact-online | 0 / 4,400 | 0% | 0 | 0 | 0 |

The fractional median/p95 values summarize integer per-trace lags using the
standard quantile interpolation. Raw integer rows are retained.

## Artifacts

- `rq2_compilation.csv`: one native cold-compilation row per fresh worker;
- `rq2_runtime.csv`: paired batch-1/batch-64 runtime rows;
- `rq2_certificate_generation.csv`: offline certificate timings;
- `rq2_decision_lag.csv`: exhaustive per-trace lag evidence;
- `rq2_summary.csv` and `rq2_decision_summary.csv`: derived summaries;
- `rq2_manifest.json`: protocol, provenance, status counts, and artifact hash;
- `rq2_cost_of_correctness.png`: four-panel valid-flat-construction overview.

The CSV/JSON records are the evidence. The figure is a first information-dense
overview and may be restyled when the final hardware rerun is integrated into
the paper.
