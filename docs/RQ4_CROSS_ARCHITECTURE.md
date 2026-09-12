# RQ4 Cross-Architecture Efficiency Landscape

**Schema:** `rq4.v1`  
**Generator:** `python experiments/rq4_cross_architecture.py`  
**Current artifact:** controlled local-CPU candidate; CUDA block unsupported

RQ4 compares seven semantically gated headline configurations on three formulas
that every construction supports: an atomic fixed-overhead floor, a nested-Next
finite-horizon control, and the paper-faithful balanced IJCNN formula at four
atoms. Original RuleRunner is included only because all three formulas have an
exact `rq1.v1` certificate. Bounded default and bounded exact-online remain
separate configurations.

## Protocol

The experiment has three non-conflated modes:

1. **Capacity/end-to-end:** early termination is disabled; a NumPy Boolean cube
   is decoded into observations and passed through the public monitor API.
2. **Capacity/core-native interface:** early termination is disabled and the
   already decoded native `Observation` mappings are reused. Tensor backends
   still own their mandatory internal tensorization and device transfer. This
   is a comparable monitor-interface boundary, not a private kernel-only
   microbenchmark.
3. **Deployment/end-to-end:** early termination is enabled on declared
   early-satisfy, late-satisfy, boundary-violate, or fixed-horizon strata.

Capacity workloads use `L=64` and `B={1,32,128}`. Deployment uses `L=64` and
`B={1,64}`. Every successful workload has five independent trace seeds and
seven within-cell timing blocks. Each formula/monitor/mode/repetition runs in a
fresh process, includes native cold compilation, and is checked against the
canonical DFA before timing. Summaries report medians, bootstrap 95% intervals,
and p95 over the 35 retained timing blocks.

The local run resolved and now freezes Torch intra-op/inter-op threads at
`10/10`. Its manifest records Python 3.10.13, Torch 2.6.0+cu124, the CPU model,
logical core count, RAM, CUDA runtime, MONA, commit, and dirty-worktree hash.
The final-machine rerun should use the intended project environment and retain
its own exact provenance rather than being pooled with this local artifact.

Processed cells describe the implementation's physical execution, not merely
the semantic decision point. The current vectorized RuleRunner, progression,
and DeepDFA batch paths advance the whole rectangular batch; only the symbolic
sequential path currently saves computation after an early verdict.

## Current CPU findings

The balanced IJCNN workload gives the clearest cross-architecture comparison:

| Monitor | Capacity B=1 median | B=1 p95 | Capacity B=128 per trace | Deployment B=64 per trace | Processed cells |
|---|---:|---:|---:|---:|---:|
| symbolic guarded DFA | 0.137 ms | 0.143 ms | 0.133 ms | 0.121 ms | 0.677 |
| DeepDFA dense | 1.312 ms | 1.401 ms | 0.156 ms | 0.221 ms | 1.000 |
| progression RuleRunner flat | 4.396 ms | 4.637 ms | 0.167 ms | 0.187 ms | 1.000 |
| DeepDFA factored | 2.401 ms | 2.549 ms | 0.170 ms | 0.240 ms | 1.000 |
| original RuleRunner flat | 8.347 ms | 9.027 ms | 0.430 ms | 0.882 ms | 1.000 |
| bounded RuleRunner default | 8.925 ms | 9.627 ms | 2.478 ms | 2.108 ms | 1.000 |
| bounded RuleRunner exact | 9.149 ms | 10.089 ms | 2.516 ms | 2.891 ms | 1.000 |

On this CPU, symbolic has the lowest steady-state latency in every one of the
168 successful CPU workload groups. There is therefore no measured runtime
crossover. Batching closes much of the gap: on IJCNN, dense DeepDFA goes from
9.6× symbolic latency at batch 1 to 1.17× at batch 128, and progression goes
from 32.2× to 1.25×. This is a negative crossover result, not evidence that the
tensor architectures cannot cross on a GPU.

The predecoded boundary shows that upstream Boolean-cube decoding is material
for the fastest batched paths. At IJCNN batch 128, end-to-end/core median ratios
are 2.24× for symbolic, 1.95× for dense DeepDFA, 1.89× for progression, 1.83×
for factored DeepDFA, and 1.38× for original RuleRunner. The bounded pipeline is
compute-dominated at approximately 1.02–1.03×. These ratios should be used for
overhead attribution, not as a second headline performance ranking.

### Cold start changes the short-lived regime

The symbolic and DeepDFA constructions compile through MONA; original and
progression RuleRunner do not. On IJCNN, median native compilation is about
114 ms for symbolic and 57 ms for original/progression RuleRunner. Consequently,
although progression is slower at steady state, its total compile-plus-runtime
cost is lower for short-lived monitors. At batch 128, the estimated equality
point is about 1,691 traces for progression and 191 traces for original
RuleRunner; symbolic wins after those points. At batch 1, progression's
equality point is only about 13 traces because its per-trace runtime gap is much
larger.

For IJCNN capacity/end-to-end at batch 128, symbolic and progression are the
only three-dimensional Pareto points under latency, cold compilation, and
post-compilation RSS delta. This does not make progression the runtime winner;
it preserves its cold-start advantage as a separate axis.

### Deployment early termination is an implementation result

For the mixed IJCNN deployment batch, the symbolic path processes a median
0.677 of offered cells. Every vectorized tensor path processes 1.000. The
semantic decision indices are stored for all traces, so later masked/compacted
batch implementations can be compared without changing the workload. The
current result supports a concrete optimization target: tensor backends need
active-trace compaction before they can benefit computationally from early
verdicts.

## Evidence, failures, and remaining blocker

The artifact contains 630 compilation attempts and 1,680 runtime rows. All 315
CPU compilation cells and all 840 CPU runtime rows succeeded. The requested
CUDA audit contains 270 unsupported tensor-compilation rows and 720 associated
unsupported runtime rows. The 45 symbolic requested-CUDA cells remain correctly
labeled with `effective_device=cpu`; no result is presented as accelerator
evidence.

No CPU point hit an OOM or timeout boundary in this deliberately feasible
cross-architecture grid. The harness records both outcomes explicitly, while
RQ3 supplies the structural state/alphabet walls. A controlled CUDA run is the
only outstanding E5 evidence block. Until it exists, GPU crossover, GPU memory,
and CPU/GPU phase claims remain unsupported.

Absolute timings belong to the current local host and must be rerun on the
final submission hardware before they are copied into the paper. Raw records,
absolute summaries, explicitly symbolic-relative comparisons, Pareto rows, the
overview figure, and the hashed manifest live in `results/rq4/`.
