# Audit notes (2026-08-20)

This file records the findings from the first pass over the papers, draft,
Claude-generated notes, and implementation, followed by the implementation
audit.  It is a chronological audit: early bullets describe problems as first
found, while the later checklists record their resolution.  For a concise
current handoff across all three RuleRunner versions, use
[`docs/rulerunner_status.md`](docs/rulerunner_status.md) as the authoritative
status document.

## What the project currently contributes

The strongest contribution is not merely another implementation comparison.
The project identifies a correctness limitation in the published RuleRunner
representation: one truth register per *syntactic subformula* can conflate
several live temporal obligations that originate at different trace positions.
The pair of traces for `F(a & X b)` makes this observable.  The proposed repair
carries residual formulae produced by LTLf progression, so distinct pending
obligations remain distinct, and realizes that state both as a flat CILP
network and as a structured network.  The implementation then compares these
monitors with a symbolic DFA and a fixed DeepDFA tensorization.

## Draft audit

- The RuleRunner discussion should distinguish the published algorithm from
  the implementation and from the proposed progression repair.  Its central
  state consists of active rule names plus three-valued formula evaluations;
  qualifiers encode which operands remain relevant.  The important defect is
  *instance conflation during reinstallation*, not temporal nesting in general.
- The counterexample and repair are substantial, but the scope of the claim
  must be exact.  Statements such as “all temporal-under-temporal formulae
  fail” are too broad.  The failure occurs where reactivation creates
  overlapping instances that share the same subformula-indexed registers.
- The progression definition, prose, implementation, and correctness argument
  were inconsistent at `X`.  The paper-preferred total definition is
  `prog(X phi,s) = phi & F true`; dually, weak next needs the empty-suffix case,
  `prog(W phi,s) = phi | G false`.  This removes the old side condition that
  progression is only interpreted over a non-empty suffix.
- A reachable residual automaton is not automatically a *minimal* DFA.
  Boolean normalization is not full LTLf equivalence or DFA minimization.  The
  current normalizer can produce many syntactic residual states with the same
  right language.  Claims of minimality must be removed or backed by an
  explicit minimization/equivalence procedure.
- Lazy progression's early verdict test is sound but incomplete when temporal
  subformulae are treated as independent Boolean variables.  Syntactic
  temporal dualities are now recognized, but implications such as
  `(a U b) -> F b` can still remain nonliteral.  The draft should say “sound
  early termination, potentially delayed,” not complete earliest detection.
- **Resolved by repair point 6 below.**  At audit time the structured
  progression implementation had local evaluation modules but a global
  residual-state transition.  Its recurrence is now root-local: independently
  fired progression/reactivation modules emit successor roots, which are
  unioned.  Whole aggregate states are used only by the fixed exact label head.
- “DeepDFA” in this repository is a fixed tensor realization of a DFA.  It
  does not implement the original DeepDFA training, temperature annealing, or
  automaton extraction procedure.  The draft and plot labels must keep this
  distinction explicit.
- Exact soft weighted model counting by summing cubes is valid under the stated
  independent-atom model.  The recursive/factored route is only exact when it
  does not double-count repeated variables; otherwise it is an approximation.
  Claims that exactness follows merely from a read-once expression should be
  narrowed, and BDD/d-DNNF/SDD terminology should not be used interchangeably.
- The factored DeepDFA path is not literally a flat transition tensor, and the
  symbolic monitor is not literally one constant-time dictionary lookup per
  event once guard evaluation and observation encoding are counted.
- The present measurements support “the symbolic monitor is fastest for crisp
  monitoring in the tested regime.”  They do not yet establish a DeepDFA over
  symbolic crossover.  Any scalability or GPU conclusion must be tied to the
  actual measured range and hardware.
- Several paper sections still contain TODOs or claims awaiting experimental
  evidence.  Those should be resolved before the narrative is treated as
  submission-ready.

## Implementation audit

- **Initial activation was not paper-faithful.**  `build_rules` activated every
  subformula.  The published construction activates recursively for Boolean,
  `F`, `G`, and `U`, but initially activates only the root for `X`/`W`; their
  operand is installed on the next cell.
- **Reinstallation was not paper-faithful.**  `_subtree_reinstall` blindly
  activated every descendant.  It must use the same recursive construction as
  the published initial-state algorithm, so a nested next operator does not
  expose its operand one cell too early.
- **Next modes and end semantics were conflated.**  The code used `B -> ?I` and
  `A` as its monitoring mode, while the tables use an unqualified initial mode
  and `M` after reactivation.  More importantly, end-of-trace resolution always
  returned false for `X` and true for `W`, even in monitoring mode, where the
  result must follow the child.  This breaks `X X a`, `WX X a`, and `X WX a`.
- **The published qualifier tables were simplified away.**  `U` was represented
  by a single `?` state rather than the `A/B/L/R` tables.  `G` omitted its `?K`
  state.  A faithful baseline must preserve those modes (with any extensions,
  such as implication and release, identified as extensions).
- **Boolean constants were treated as ordinary observations.**  Consequently
  the original RuleRunner returned the wrong verdict for `true`.  They need
  unconditional evaluation rules and must not be listed as observed atoms.
- The symbolic, flat CILP, and structured implementations agreed with one
  another because they were generated from the same flawed rule system and
  shared the same hand-written end resolver.  Internal equivalence therefore
  did not establish semantic correctness.
- The current CILP implementation is two hard-threshold networks for evaluation
  and reactivation, rather than a literal reconstruction of the paper's single
  recurrent UPDATE/persistence network.  The structured implementation also
  executes per-node modules sequentially rather than as one flattened network.
  They can be useful equivalent realizations, but the paper must describe them
  accurately or the implementation must be brought closer to the published
  architecture.
- Sparse observations should mean “missing atom = false” consistently.  The
  progression and RuleRunner paths do this; the symbolic path was observed to
  reject sparse mappings and should be normalized at its API boundary.
- DeepDFA's crisp transition computation matched the DFA oracle in the audit.
  Its soft API currently detaches/copies inputs and has no learned parameters,
  so differentiability/training claims require either an implementation change
  or narrower wording.
- N-ary formula association differs from the paper's balanced construction:
  conjunction/disjunction are left-folded.  This preserves semantics but
  changes depth, network size, and therefore benchmark costs.
- Compilation-cost accounting and wording about shared compilation need care:
  the evaluated methods do not all share the same compilation pipeline or pay
  the same preprocessing cost.
- `ltlf2dfa` writes MONA input into its installed package directory.  Tests need
  an isolated writable package directory (and concurrency protection) rather
  than relying on a writable environment installation.
- GPU tests were skipped in the audited environment, so GPU-specific claims
  were not verified there.

## Evidence from the audit

- Baseline suite: 545 passed, 27 skipped, and 6 expected failures under the
  intended Conda environment, using a temporary writable directory for MONA.
- An independent direct LTLf evaluator was used on 35 formulae and every trace
  of lengths 1--4 over `{a,b}`.  The symbolic DFA, fixed DeepDFA paths, and
  progression monitor matched it.  All three original RuleRunner realizations
  matched each other but shared additional failures, including constants,
  nested next, and several reinstallation cases.
- A randomized progression check covered 500 generated formulae over 84 traces
  each (42,000 comparisons) without a semantic mismatch.
- A prototype of the total progression clauses using `F true` and `G false`
  matched the direct evaluator on the representative suite.

## Recommended repair order

1. Make the original RuleRunner baseline faithful: activation sets,
   reinstallation, next modes, end rules, constants, and published qualifiers.
2. Add an independent exhaustive semantic oracle suite, especially nested next,
   constants, short traces, and sparse observations.
3. Adopt the total progression definition (`F true` / `G false`) consistently
   in code, tests, and the draft.
4. Then repair theoretical wording: correctness scope, non-minimal residual
   automata, incomplete early detection, and structured/global update claims.
5. Finally align experiment labels and claims with the exact implemented
   architectures and rerun the full benchmark suite.  **Deferred by explicit
   project decision; no new benchmark or result claim follows from this audit.**

## Status after this repair pass

- [x] **1 — faithful original RuleRunner baseline.** Initial activation and
  temporal reinstallation now follow Algorithm 2; `X`/`W` use initial and `M`
  modes with mode-sensitive boundary resolution; `G` retains `?K`; `U` retains
  `A/B/L/R`; constants and empty traces are handled explicitly.  These changes
  feed the symbolic, flat CILP, and structured realizations from the same
  corrected rule system.  The published one-register instance-conflation
  limitation remains intentionally visible through the existing strict xfails.
- [x] **2 — independent exhaustive oracle.** `tests/test_semantic_oracle.py`
  directly implements finite-trace semantics and enumerates every sparse trace
  of lengths 0--3 over `{a,b}`.  It checks all three original RuleRunner
  realizations on their non-conflating domain and all complete architectures on
  a larger domain containing the known counterexamples.  DFA observations now
  consistently interpret omitted atoms as false, and MONA compilation uses a
  locked, isolated writable temporary directory.
- [x] **3 — total progression.** The implementation now uses
  `prog(X phi,s) = phi & F true` and `prog(W phi,s) = phi | G false`.
  End-of-trace verdicts evaluate the successor residual on the empty suffix.
  Unit tests establish the empty-continuation identity, and the proof and
  counterexample derivation in `latex/3_rulerunner.tex` have been aligned.
- [x] **4 — exact eager sink/trap labels.** After constructing the complete
  residual graph, the eager compiler now computes accepting sinks and traps by
  reverse reachability from empty-suffix accepting and rejecting states.  The
  eager, flat, and structured monitors therefore return exact permanent online
  verdicts even when normalization does not expose literal `true`/`false`.
  The lazy engine deliberately retains its sound-but-incomplete syntactic
  early test.  Regression cases use the nonliteral temporal tautology
  `(a U b) -> F b` and contradiction `(a U b) & G !b`.
- [x] **6 — comparable structured realizations.** The old and repaired
  RuleRunner now follow the same explicit organization: a bottom-up collection
  of syntactically-owned evaluation CILP modules followed by independently
  fired reactivation modules whose outputs are unioned into the next recurrent
  state.  In the repaired monitor, registers range over the progression-root
  closure and each reactivation module reads only its own source root and local
  observation guard.  Reachable whole root sets are used only by a fixed exact
  sink/trap label head, never to compute recurrence.  This replaces the prior
  structured implementation, whose recurrence reused the opaque whole-state
  flat transition network.  Direct per-cell tests also establish that the old
  flat and structured RuleRunner realizations remain behaviorally identical;
  their difference is only the grouping of the same rules into modules.
- [x] **Mixed batches containing empty traces.**  The repaired structured
  monitor now evaluates every zero-length member with the formula's empty-trace
  semantics instead of using the result of its padded batch cell.
- Verification checkpoint after these changes: **652 passed, 27 skipped, and 6
  expected failures**.  The six strict xfails are the deliberately retained
  original-RuleRunner instance-conflation cases.  The LaTeX draft also builds
  successfully; its pre-existing unresolved citation/reference warnings remain.

## Bounded-event RuleRunner follow-up

- The exploratory `bounded_horizon_RR.md` incorrectly identified `X(X a)` as a
  counterexample.  After the paper-faithful initial activation and Next-mode
  repairs, exact product checking certifies `X^k a` (`k=1..6`).  A genuine small
  shared-register counterexample is `(X a) & X(X a)`.
- `certify_rule_runner` now decides the exact semantic correctness class of the
  old monitor formula-by-formula by BFS over its product with the canonical DFA,
  returning shortest witnesses.  Language equivalence, sound early decisions,
  and exact online-label equivalence are reported separately.
- The bounded repair is formalized as maximal finite-horizon event islands, a
  fixed $H+1$ observation pipeline, finite-boundary flushing, and an old
  RuleRunner skeleton that must pass the exact certifier.  The semantic reference
  is now compiled into two neural organizations: a flat pooled/fixpoint CILP
  event evaluator plus flat skeleton, and addressable per-`(subformula, offset)`
  CILP modules plus structured skeleton.  Both share a recurrent CILP observation
  shift register and exact finite-suffix circuits.  See
  `docs/bounded_event_rulerunner.md` and
  `src/monitors/rulerunner/bounded_cilp.py`.
- The wrappers reject skeletons lacking exact language equivalence or prefix
  soundness.  A fixed CILP readout now classifies the full delayed-pipeline state
  by exact reverse reachability, so online labels match the original formula's
  DFA rather than being delayed by $H$.  This adapts Finkbeiner--Kuhtz's
  extrapolation principle to permanent LTL3 labels; their native truncated-path
  Boolean semantics is not silently equated with ours.
- `batch_run` now fuses all equal-length bounded-event windows into batched
  tensor evaluations and delegates the resulting trace batch to the existing
  flat/structured skeleton batch runner.  It no longer calls sequential `run()`.
- Verification after exact extrapolation and fused batching: **738 passed, 29
  skipped, and 6 expected failures**.  The added CUDA cases account for two of
  the skips on this machine.
- The consolidated current status, limitations, and next-session reading order
  are in [`docs/rulerunner_status.md`](docs/rulerunner_status.md).  Benchmark
  execution and empirical result claims remain deliberately deferred.
