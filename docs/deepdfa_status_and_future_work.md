# DeepDFA status, decisions, and future-work handoff

**Status date:** 2026-08-25  
**Decision:** the DeepDFA theory and implementation are complete for the scope
of the current paper. Freeze them while the project moves to benchmarks,
experiments, and results.

This document is the authoritative handoff for the DeepDFA work completed in
the August 2026 audit thread. It records what is implemented and proved, what
the paper may safely claim, the relationship to DeepDFA/NeSyPPM/NeSyA/T-ILR,
and which ideas belong to near or distant future work. For API details see
[deepdfa_artifact.md](deepdfa_artifact.md); for the longer decision-diagram
design analysis see
[decision_diagram_transition_representation.md](decision_diagram_transition_representation.md).

---

## 1. Executive conclusion

The original concern was that this repository implemented a fixed DeepDFA-like
forward pass rather than the learned DeepDFA architecture. That is **not a
problem for monitoring a known LTLf specification**. Learning transition and
acceptance parameters would solve a different problem: automaton induction or
specification adaptation.

For the present paper, DeepDFA is intentionally a fixed tensorization of the
same DFA used by the symbolic monitor. Its purpose is to provide:

- an exact automata-based neuro-symbolic monitor;
- a differentiable tensor substrate for upstream perception;
- dense and factored transition representations;
- native cross-trace batching and an experimental prefix-scan variant;
- an honest comparison point against symbolic monitoring and RuleRunner.

The current paper should **not** add trainable automaton parameters merely to
look more like the original DeepDFA. Doing so would introduce an adaptation
objective, training protocol, extraction procedure, and evaluation problem
that are outside its monitoring question.

The project-level status is therefore:

- RuleRunner theory and implementation: technically complete for this paper;
- DeepDFA theory and implementation: technically complete for this paper;
- remaining major scientific phase: benchmarks, experiments, and results;
- remaining later work: integrate results and finish ordinary submission
  writing/polish.

---

## 2. Exact scope of the implemented DeepDFA

Starting from an LTLf formula, the compiler produces a deterministic complete
DFA. The monitor fixes its transition and acceptance parameters to their exact
one-hot/Boolean values. It does **not** learn the automaton.

### Implemented representations

| Mode | Persistent representation | Intended use | Principal cost |
|---|---|---|---|
| Dense | `T[|Q|, 2^|AP|, |Q|]` | Small alphabets; batching showcase | Exponential alphabet storage |
| Factored | Disjoint require-true/require-false cube masks | Large or structured alphabets; exact soft WMC | Structure-dependent cube count |
| Scan | Per-cell matrices plus Hillis--Steele prefix products | Long batched traces on suitable GPUs | High temporary memory and extra arithmetic |

The factored path has two guard evaluators:

- `exact_matrix`: the production/default evaluator. Disjoint cubes compute
  exact WMC for arbitrary Boolean guards under independent Bernoulli atoms.
- `recursive_matrix`: a retained diagnostic approximation. It is exact for
  crisp inputs and for fractional inputs when the relevant subexpressions are
  independent, including read-once guards. `soft_matrix` remains only as its
  historical compatibility alias.

The public `acceptance_probability_tensor` API accepts `(L, |AP|)` or
`(B, L, |AP|)` floating tensors, supports padded-batch lengths, preserves
autograd to its inputs, and validates finite `[0,1]` Bernoulli parameters. The
monitor also exposes immutable `artifact_stats` containing the representation,
`|AP|`, alphabet size, `|Q|`, transition/cube counts, and actual persistent
tensor element/byte counts.

The Python float/list convenience wrappers intentionally detach results for
ordinary monitoring output. Training code must call the tensor-native method;
this resolves the original audit finding without pretending that Python
scalars can carry an autograd graph.

### Explicit non-goals

The current artifact does not implement:

- trainable transition logits;
- temperature annealing;
- a learned acceptance vector;
- DFA extraction or minimization after training;
- specification adaptation;
- calibrated perception or uncertainty learning;
- a probabilistic three-valued early-verdict semantics.

None is required for exact crisp monitoring of a known specification.

---

## 3. Theory now present in the paper

Section 4 contains three explicit results.

1. **Crisp equivalence.** With the compiled one-hot tensor, the state vector is
   exactly the indicator of the DFA state reached after every prefix. DeepDFA
   and the symbolic monitor therefore return identical early and final
   verdicts.
2. **Exact guard WMC.** A pairwise-disjoint cube cover sums to the exact guard
   probability under independent Bernoulli atoms. Determinism and completeness
   make every expected-transition row sum to one; each entry is a differentiable
   multilinear polynomial.
3. **Trace-marginal semantics.** Under the stated within-cell and across-time
   independence model, the propagated state vector is the distribution over DFA
   states reached by the random prefix. Final accepting mass is the probability
   that the random trace satisfies the LTLf formula.

These results serve different roles:

- crisp equivalence establishes correctness of the evaluated monitor;
- exact cube WMC establishes correctness of the concrete factored backend;
- trace-marginal semantics states precisely what the differentiable soft output
  means.

The last two are **specializations of the more general NeSyA semantics**, not
novel claims about symbolic-automaton WMC.

### Complexity statement

Let `n=|AP|`, `m=|Q|`, `B` be batch size, `C` the total stored cube count, and
`S` the total recursive-guard syntax size. The paper records the implemented
post-compilation bounds:

| Representation | Static storage | Per-cell time |
|---|---:|---:|
| Dense crisp | `Theta(m^2 2^n)` | `Theta(Bm^2)` |
| Exact cubes | `Theta(Cn)` | `Theta(B(Cn+m^2))` |
| Recursive guards | `Theta(S)` | `Theta(B(S+m^2))` |

The exact cube kernel temporarily uses `Theta(B(Cn+m^2))` storage. A guard can
require `Theta(2^n)` cubes and a complete DFA can require
`C=Theta(m 2^n)`, so factoring changes the blow-up from unconditional alphabet
materialization to structure-dependent compilation; it does not eliminate the
worst case.

For trace length `L`, sequential execution multiplies the per-cell bounds by
`L`. The Hillis--Steele scan stores `Theta(LBm^2)` matrices and performs
`Theta(BLm^3 log L)` total arithmetic. It reduces launch dependency depth, not
asymptotic work.

---

## 4. Evidence and artifact state

The implementation evidence currently establishes:

- dense tensorization exactly matches the compiled DFA;
- factored crisp execution exactly matches dense and symbolic execution;
- sequential, batched, and scan verdicts agree;
- exact matrices are row-stochastic on fractional inputs;
- exact cube WMC agrees with exhaustive valuation enumeration on a
  non-read-once guard;
- the recursive and cube evaluators agree where the recursive assumptions hold;
- the tensor-native acceptance API retains autograd and correct padded lengths;
- invalid probabilistic inputs are rejected at the public boundary;
- reported artifact statistics agree with the actual allocated tensors;
- factored mode handles alphabets for which a dense tensor would be infeasible.

Latest verification from this thread:

- focused DeepDFA suites: **156 passed**;
- complete repository suite: **832 passed, 29 skipped, 6 expected failures**;
- Python formatting/lint checks: passed;
- LaTeX build: successful, with only pre-existing unrelated citation/reference
  warnings.

The expected failures preserve the known original-RuleRunner limitation; they
are not DeepDFA failures.

---

## 5. Safe and unsafe claims

### Safe claims

- The evaluated DeepDFA is a **fixed, exact tensor realization** of an
  LTLf-compiled DFA.
- On crisp traces it is verdict-equivalent to the symbolic DFA.
- The dense representation exposes the `2^|AP|` alphabet wall and the
  `|Q|^2` matrix-update cost.
- The cube representation avoids unconditional alphabet materialization when
  guards admit compact disjoint covers.
- Disjoint cubes compute exact differentiable WMC for arbitrary guards under
  the stated independent-Bernoulli model.
- The public soft tensor API can propagate gradients into an upstream
  perceptor without changing the specification.
- Scan is a hardware-dependent launch-depth trade-off, not an arithmetic
  complexity improvement.

### Claims not supported by the current paper

- that this artifact learns or extracts a DFA;
- that it performs specification adaptation;
- that exact symbolic-automaton WMC or knowledge compilation is new;
- that cubes guarantee sub-exponential storage;
- that the recursive Boolean evaluator is exact for every fractional guard;
- that exact acceptance marginals imply a calibrated neural perceptor;
- that thresholding a marginal defines the correct probabilistic
  three-valued runtime semantics;
- that GPU DeepDFA is faster than symbolic monitoring before the final fair
  measurements are run;
- that scan reduces total FLOPs;
- that a BDD/d-DNNF replacement would itself be a new paper contribution.

---

## 6. Positioning against neighboring work

### Original DeepDFA and NeSyPPM

Original DeepDFA learns a probabilistic relaxation of a finite automaton with
temperature-controlled transition and acceptance parameters. NeSyPPM uses a
fixed automaton as a differentiable knowledge evaluator for generated process
suffixes. The present monitor adopts the fixed forward algebra for online LTLf
monitoring over a non-mutually-exclusive propositional alphabet.

The distinction must remain explicit in the paper and artifact labels:
**fixed DeepDFA monitor**, not learned DeepDFA architecture.

### NeSyA

NeSyA already provides the general construction closest to the factored path:
compile symbolic-automaton guards to tractable circuits such as d-DNNF,
evaluate exact differentiable WMC, assemble expected transition matrices, and
propagate state mass. It also proves the general state-distribution semantics.

Therefore:

- the paper cites NeSyA as prior art for the semantics;
- the cube backend is presented as a concrete LTLf runtime specialization and
  representation choice;
- exact WMC and knowledge compilation are not claimed as contributions.

The [public NeSyA repository](https://github.com/nmanginas/nesya) was verified
during the thread at commit
`aa5830e12b81b9a618e52739de6de629afebd10e` (2025-12-16). It uses `dsharp` and
the Python `nnf` package, constructs a batch/time transition stack, and returns
final accepting mass. It does not expose this project's exact online
sink/trap-based three-valued interface. Its determinism checker tests mutual
exclusion but does not enforce exhaustive outgoing mass.

Consequently, NeSyA should **not** be inserted as an ordinary line in the
current crisp runtime benchmark. If a future paper promotes soft transition
evaluation or compiled circuits as a contribution, NeSyA becomes a mandatory
controlled backend baseline. Because the upstream artifact is GPL-3.0, run a
pinned copy in a separate environment rather than copying it into this codebase.

### T-ILR

T-ILR follows a formula-level route: fuzzy LTLf satisfaction under Zadeh
semantics plus iterative local refinement, without constructing an external
DFA. Its fuzzy satisfaction/refinement signal is not the same object as a
probabilistic automaton acceptance marginal. It belongs in related work, not as
an interchangeable DeepDFA baseline.

---

## 7. Immediate next phase: experiments and results

Experiment integration was deliberately deferred. The next session should
start from [EXPERIMENTAL_EVALUATION_PLAN.md](EXPERIMENTAL_EVALUATION_PLAN.md) and should not reopen the
settled DeepDFA semantics unless a measurement or test reveals a real defect.

The immediate goals are:

1. Reconfirm the exact monitor set and formulas for each experiment.
2. Regenerate CPU and GPU data after all RuleRunner/DeepDFA repairs.
3. Keep early termination disabled for per-cell cost comparisons unless a
   separate data-dependent experiment is explicitly designed.
4. Preserve truthful effective-device labels and CUDA synchronization.
5. Report absolute costs before normalized speedups.
6. Record `|AP|`, `|Q|`, cube count, representation bytes, batch size, length,
   hardware, and compile/run boundaries so every scaling claim is auditable.
7. Test the intended structural predictions rather than presupposing a winner:
   alphabet wall, state wall, batch scaling, within-step depth, and scan's
   hardware-dependent trade-off.
8. Select the smallest set of figures that supports the three-paradigm
   landscape, then rewrite the experiments section around measured outcomes.

Do not reuse old timing numbers as final evidence. Some recorded results
predate the RuleRunner faithfulness repairs and the final timing protocol. A
failed DeepDFA crossover or a symbolic throughput win is not a failed paper;
the contribution is an honest map of where each representation grows and
walls out.

NeSyA is not required in this immediate crisp benchmark. Add it only if the
scope is explicitly expanded to a soft/WMC backend experiment, in which case
compare identical DFAs and probability tensors and measure:

- output agreement;
- guard compilation time;
- circuit nodes/edges versus cube count and bytes;
- persistent and peak memory;
- CPU/GPU forward and backward time;
- batch and sequence scaling.

---

## 8. Near-term work after experiments

Once results are stable, the remaining work for this paper is integration and
submission polish rather than new DeepDFA research:

- write the abstract and final results summary;
- finish the introduction and conclusion;
- replace provisional experiment prose with measured claims and figures;
- add the capability matrix in the theoretical comparison;
- complete broader LTLf/runtime-verification related work;
- resolve the remaining bibliography/reference warnings;
- port the draft to the target conference template;
- prepare artifact commands, environment information, result manifests, and
  figure-generation instructions.

Candidate empirical additions already listed in the project roadmap—such as a
real-log case study or neural sequence baselines—must be scoped separately.
They are not missing parts of DeepDFA.

---

## 9. Medium-/far-term direction A: shared compiled circuits

The cube backend has no subfunction sharing. BDDs, SDDs, and d-DNNFs can share
repeated Boolean structure and may therefore reduce representation size, while
remaining exponential in the worst case.

### GPU-oriented execution scheme

The promising design is **not** to traverse a conventional pointer-based BDD
on the GPU. Instead:

1. Compile guards once to BDD/d-DNNF.
2. Deduplicate subgraphs, preferably across all transition guards.
3. Lower the DAG to a static integer-indexed tensor IR.
4. Topologically group nodes by level, operation, and possibly fan-in.
5. Evaluate every node group over `(batch x time)` observations with batched
   gathers and reductions.
6. Gather guard-output nodes into the transition matrices.
7. Apply the same automaton recurrence or prefix scan.

For a smoothed d-DNNF, literals evaluate to `p_a` or `1-p_a`, decomposable AND
nodes multiply child values, and deterministic OR nodes sum child values. For
a BDD node `u` testing atom `a`:

```text
value[u] = (1-p_a) value[low(u)] + p_a value[high(u)].
```

Both are exact WMC recurrences. For a circuit with `N` nodes, `E` wires, depth
`D`, batch `B`, and length `L`, a direct levelized evaluator performs
`Theta(BLE)` arithmetic with `O(D)` dependency stages and naive
`Theta(BLN)` workspace. Online or chunked evaluation can reduce workspace to
approximately `Theta(BN)`.

The actual systems research would concern packing and scheduling:

- cross-guard subgraph sharing;
- segmented reductions for irregular fan-in;
- level fusion with CUDA/Triton kernels;
- avoiding one kernel launch per gate or tiny level;
- memory-aware time chunking and autograd checkpointing;
- numerical stability on deep product circuits;
- balancing compact irregular circuits against regular cube tensors.

### What would make it publishable

Replacing cubes with d-DNNF is not enough: NeSyA already establishes exact
compiled-circuit WMC. A separate contribution needs a genuine delta, such as:

- direct reuse of MONA's internal MTBDD rather than decompression/recompilation;
- a globally packed GPU circuit evaluator demonstrably better than cubes and
  NeSyA's reference evaluator;
- a formal representation bound for an important LTLf guard family;
- or an end-to-end learning/adaptation result enabled by the backend.

Required baselines would include dense DeepDFA, exact cubes, upstream NeSyA or
an equivalently faithful d-DNNF evaluator, and symbolic monitoring where crisp
throughput is relevant. The headline must be representation size and/or
systems performance, not rediscovery of WMC correctness.

**Recommendation:** leave this out of the current paper. Revisit it after the
current experiments. It is a possible future paper only if preliminary
representation and throughput measurements reveal a substantive result.

---

## 10. Far-term direction B: learned adaptation

Specification adaptation is a separate problem from monitoring. A serious
learned DeepDFA extension would require at least:

- parameterized transition and acceptance logits;
- temperature scheduling or another discrete relaxation;
- an adaptation loss and data-generation protocol;
- constraints or regularizers preserving useful automaton structure;
- a policy for extracting a discrete DFA;
- minimization and equivalence/behavioral evaluation after extraction;
- comparison with syntactically localized RuleRunner adaptation;
- metrics for semantic drift, not only prediction accuracy.

The fixed current monitor is valuable groundwork because gradients already
flow through observations, the transition semantics are exact, and the
compiled artifact is auditable. It does not by itself solve adaptation.

The uncertainty/calibration experiments, probabilistic-verdict questions, and
adaptation planning have already been separated into `artur_future_work/`.
Keep them there unless a later project deliberately reunifies them.

---

## 11. Freeze conditions and reasons to reopen DeepDFA

Do not change the DeepDFA theory or core implementation during the experiment
phase merely to add features. Reopen it only if one of the following occurs:

- a correctness test fails;
- an experiment reveals a mismatch between the stated and actual cost model;
- artifact statistics expose an unaccounted persistent allocation;
- GPU execution violates the documented synchronization/device assumptions;
- a reviewer or supervisor requires a specific missing baseline or theorem;
- the paper's scope is explicitly expanded to probabilistic monitoring,
  adaptation, or compiled-circuit backends.

Otherwise, treat the section as frozen and focus on measuring it fairly.

---

## 12. Main pointers

- Paper theory: [`latex/4_deepdfa.tex`](../latex/4_deepdfa.tex)
- Cross-paradigm comparison: [`latex/5_theory_cmp.tex`](../latex/5_theory_cmp.tex)
- Related work: [`latex/7_related_work.tex`](../latex/7_related_work.tex)
- Implementation: [`src/monitors/deep_dfa.py`](../src/monitors/deep_dfa.py)
- Tests: [`tests/test_deep_dfa.py`](../tests/test_deep_dfa.py),
  [`tests/test_deep_dfa_scan.py`](../tests/test_deep_dfa_scan.py)
- Artifact/API guide: [deepdfa_artifact.md](deepdfa_artifact.md)
- Current evaluation plan: [EXPERIMENTAL_EVALUATION_PLAN.md](EXPERIMENTAL_EVALUATION_PLAN.md)
- Archived experiment map: [EXPERIMENT_MAP.md](../old/docs/EXPERIMENT_MAP.md)
- Decision-diagram analysis:
  [decision_diagram_transition_representation.md](decision_diagram_transition_representation.md)
- RuleRunner authoritative handoff: [rulerunner_status.md](rulerunner_status.md)
- Deferred probabilistic/adaptation project: [`artur_future_work/`](../artur_future_work/)

Suggested opening instruction for the next session:

> Read `docs/deepdfa_status_and_future_work.md`,
> `docs/rulerunner_status.md`, and `docs/EXPERIMENTAL_EVALUATION_PLAN.md`. Treat RuleRunner
> and DeepDFA theory/implementation as frozen. Audit the existing experiment
> scripts and results, then design the final fair benchmark/result phase
> without reusing stale measurements or expanding into probabilistic
> monitoring/adaptation.
