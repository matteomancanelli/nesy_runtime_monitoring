# RuleRunner implementation status

This is the authoritative handoff for the RuleRunner part of the project as of
2026-08-23.  Read it before the older exploratory notes.  It records semantic
and implementation status only: benchmark execution, performance comparisons,
and result claims are deliberately deferred.

## The three versions

| Version | Current status | Remaining limitations |
|---|---|---|
| **Original RuleRunner** | Faithful to the three RuleRunner papers, including their one-register-per-subformula limitation.  The symbolic engine, flat CILP realization, and structured CILP realization implement the same evaluation/reactivation semantics.  `certify_rule_runner` decides correctness formula-by-formula against the canonical DFA and returns shortest witnesses.  The draft now proves correctness and prefix soundness for a readable flat-temporal fragment. | It is not complete for LTLf by design.  The flat grammar is sufficient rather than maximal; the exact maximal class is given semantically by the certifier. |
| **Bounded-event RuleRunner** | A fixed-window middle construction extracts maximal syntactically bounded islands, evaluates them as derived events, and feeds an exactly certified original-RuleRunner skeleton.  Flat and structured CILP versions implement the same pipeline.  Finite-boundary flushing, sound (possibly arbitrarily delayed) default online labels, opt-in exact permanent online labels, empty traces, and fused cross-trace batching are implemented.  The draft proves a readable eventized fragment that strictly contains the original flat fragment.  Finkbeiner--Kuhtz's interval-indexed Until is deliberately left as an abbreviation, since every fixed finite bound can be unrolled into the Boolean/Next fragment already supported. | Opt-in exact extrapolation explicitly explores a composite graph over the observation alphabet and can be expensive.  The construction is intentionally partial and rejects an unsafe remaining skeleton. |
| **Progression RuleRunner** | The complete repair carries progressed residual roots.  Lazy, eager table, flat CILP, and structured root-local CILP versions are implemented.  The eager/neural versions use exact sink/trap labels; the structured recurrence does not use an opaque whole-state transition table.  The construction is sound and complete for the LTLf syntax supported by the project. | Normalization applies canonical Boolean-skeleton simplification and finite-trace-safe temporal rewrites, but is not canonical modulo full LTLf language equivalence; raw closure sizes may therefore overcount right languages.  Eager/static compilation explicitly enumerates relevant observations, and exact earliest labels require global analysis of reachable aggregate states.  These are scalability costs rather than correctness gaps.  The lazy oracle deliberately trades exact earliest labels for on-demand construction: every definite early verdict is sound and every final verdict is exact, but permanence may be detected late. |

## Architectural correspondence

The original and progression versions deliberately use the same structured
organization: syntactically owned bottom-up evaluation modules, followed by
independently fired reactivation/progression modules whose outputs are combined
into the next recurrent state.  The bounded version preserves the same pairing:
its flat form pools the unrolled event rules and uses a flat skeleton, while its
structured form owns one module per `(subformula, offset)` and uses a structured
skeleton.  Whole aggregate states are used only by fixed exact label heads where
permanent three-valued classification requires global reachability; they do not
drive recurrence.

This is functionally comparable to the papers' neural encoding without claiming
to be a literal drawing-level reconstruction of either IJCNN figure.  The two
hard-threshold evaluation/reactivation networks are an equivalent executable
realization; differentiable adaptation is outside the present paper.

## Correctness evidence

- The original symbolic, flat, and structured realizations agree, including on
  the deliberately retained shared-register failures.
- An independent direct finite-trace semantic oracle covers constants, sparse
  observations, empty and short traces, nested Next, and the known conflation
  counterexamples.
- The exact product certifier separates final-language equivalence, sound early
  decisions, and exact online-label equivalence.
- The bounded semantic pipeline, both bounded CILP organizations, exact
  extrapolation head, and fused batch path are checked against the direct/DFA
  semantics.  The default (head-free) configuration is checked separately for
  soundness of every definite online label plus exactness of final verdicts.
- Exhaustive analyses no longer reach into monitor internals: `RuleEngine`
  exposes `state()`/`load_state()`, the CILP runners expose `label_state()`,
  and a guard test fails if `RuleEngine` grows a field that the round trip
  would drop.
- The checked-in certificate artifact is re-derived from scratch by
  `tests/test_rulerunner_certificates.py`, and a fingerprint mismatch voids a
  stored claim instead of trusting it.
- Normal compilation never writes that artifact, and the benchmark harness
  requires `certificate="cached"` plus a verified cache hit for bounded-event
  monitors, so a timing run cannot silently invoke canonical-DFA certification.
- The progression lazy/eager/flat/structured paths are checked against the DFA
  semantics, including formulas that are strict expected failures for the
  original version.
- The latest complete verification checkpoint is **818 passed, 29 skipped, and
  6 expected strict failures**.  The six expected failures document only the
  original published encoding's instance-conflation cases.

## Deliberately deferred work

- Do not claim benchmark outcomes for the bounded-event version until the
  evaluation phase is explicitly resumed and rerun.
- Do not reuse stale timing or scalability prose as evidence for any of the
  three versions.  Compilation cost, recurrent cost, batching, and memory must
  be measured and reported separately.
- The proved flat grammar is deliberately sufficient rather than maximal.
  Finding a more permissive readable grammar is an optional refinement; it is
  not required for the exact semantic boundary, which is already decidable
  formula-by-formula.
- Better canonicalization or language quotienting of progression residuals is a
  real implementation improvement.  Current raw residual counts must not be
  presented as an intrinsic lower bound of the paradigm.
- Same-level module fusion, parameter learning/adaptation, and probabilistic
  inputs are extensions, not unfinished correctness fixes for the present
  implementations.
- Same-level module fusion and parameter learning stay out of scope, as above.
- A cheaper exact-timing head for the bounded version (for example, quotienting
  the composite graph, or deriving the realizability constraint symbolically
  instead of by enumeration) is an open implementation question.  Today the
  honest framing is the two-configuration table above.

## Navigation for a new session

0. Regenerate certificates after any change to `rules.py`:
   `python -m src.monitors.rulerunner.certificates --refresh` (the test suite
   fails loudly if the artifact is stale, so this is not silent).
1. Read this file.
2. Read [`bounded_event_rulerunner.md`](bounded_event_rulerunner.md) for the
   exact bounded construction and its relation to Finkbeiner--Kuhtz.
3. Read [`nested_temporal_limitation.md`](nested_temporal_limitation.md) for the
   original shared-register counterexample and proof defect.
4. Read [`rulerunner_progression_analysis.md`](rulerunner_progression_analysis.md)
   for the progression design and representation tradeoffs.
5. Treat [`bounded_horizon_RR.md`](bounded_horizon_RR.md) as a superseded
   historical handoff only; in particular, its claim that `X(X a)` fails is
   false for the faithful implementation.

The main implementation entry points are
`src/monitors/rulerunner/{engine,cilp,structured,equivalence,certificates}.py`,
`src/monitors/rulerunner/{bounded,bounded_cilp,bounded_extrapolation}.py`, and
`src/monitors/progression/`.

Two corrections made after the audit pass are worth carrying forward, because
both were stated the other way round in earlier notes:

- The bounded pipeline's online latency is **not** bounded by $H$ (see above).
- `->` is **left**-associative here, matching ltlf2dfa.  Our parse tree used to
  right-fold it, so `a -> b -> c` was monitored as a different formula by the
  rule-based paradigms than by the DFA-based ones.  Association-sensitive
  formulas are now pinned in `tests/test_semantic_oracle.py`, which compares our
  parse tree against MONA's reading of the same string.
