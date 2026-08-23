> **Status: exploratory handoff, not an authoritative analysis.**  The claim
> below that `X(X a)` fails is false for the paper-faithful implementation after
> its initialization and Next-mode repairs: exact product checking finds no
> mismatch for `X^k a` (`k=1..6`).  The bounded-event direction remains useful,
> but its verified formulation is now in
> [bounded_event_rulerunner.md](bounded_event_rulerunner.md), and the current
> cross-version handoff is [rulerunner_status.md](rulerunner_status.md).  Do not
> use any formula classification or implementation status from the body below.

Certo. Ti lascio un handoff pensato per Codex in VS Code, focalizzato solo sul problema che stiamo esplorando adesso.
Context / Goal
We are working on an A* AI conference paper on neuro-symbolic LTLf runtime monitoring. The current strongest contribution is a re-analysis and repair of the original RuleRunner monitor.
The original RuleRunner construction has a structural correctness bug: it keeps one shared truth/register state per syntactic subformula. When a temporal context reinstalls a subformula while a previous temporal instance of the same subformula is still active, the two instances can be conflated, causing wrong verdicts. A minimal counterexample is F(a & X b). The current paper already contains:
- the RuleRunner architecture;
- the nested-temporal counterexample;
- an argument that the error is structural and not an implementation/CILP artifact;
- a progression-based corrected version;
- a soundness/completeness theorem for the corrected version.
The next research goal is not to work further on progression yet. We want to characterize how far the original RuleRunner architecture can be pushed before progression is necessary.
Current research question
Identify a nontrivial fragment / structural condition under which RuleRunner can be made correct while largely preserving its original fixed-size, compile-time rule architecture.
Do not use the word “obligation” as technical terminology in the write-up, because it has a specific meaning in LTLf+.
We initially considered characterizing formulas for which RuleRunner’s original “one register per syntactic subformula” representation is already safe. However, we now want to explore a strictly stronger intermediate construction.
New hypothesis: bounded-horizon / timestamp-indexed RuleRunner
The key idea is that some failures of original RuleRunner do not require full formula progression.
Example:
X(X a)
Original RuleRunner can conflate different temporal instances of X a, because both are mapped to the same syntactic register. But the truth of X^k a depends only on a statically bounded number of future positions.
So instead of one truth register for a subformula ψ, introduce a bounded family of temporally indexed registers, conceptually:
[ψ,0], [ψ,1], ..., [ψ,h(ψ)]
where h(ψ) is a finite look-ahead / temporal horizon known at compile time.
Each nested temporal evaluation can therefore retain its own timestamp/age class instead of being merged into one syntactic register.
This should preserve:
- a fixed rule set;
- a fixed-size recurrent/CILP encoding;
- compile-time bounded memory;
while correcting more formulas than original RuleRunner.
Relevant literature connection
This idea is closely related to Finkbeiner & Kuhtz, Monitor Circuits for LTL with Bounded and Unbounded Future.
Their work uses the notion of a finite temporal horizon and maintains values indexed by offsets in a bounded pipeline. We should reuse/cite their notion where appropriate rather than claiming the bounded-horizon idea itself as novel.
However, their construction is not RuleRunner. Our potential contribution is:
adapt the bounded-horizon idea to RuleRunner’s evaluation/reactivation architecture, preserving its fixed rule system and CILP encoding, and characterize the formulas for which this bounded extension is sufficient.

Important conceptual distinction
We may end up with three levels:
1. Original RuleRunner
   - one shared register per syntactic subformula;
   - correct only on a restricted class.
2. Bounded-horizon RuleRunner
   - a finite bank of timestamp/offset-indexed registers for finite-horizon subformulas;
   - still fixed-size and compile-time static;
   - should repair cases such as:
     - X(X a)
     - G(X a)
     - F(a & X b)
     - G(a -> X b)
     - likely a U (b & X c)
3. Progression-based RuleRunner
   - carries residual formulas;
   - correct for full LTLf;
   - more general, but pays the representation/compilation cost already discussed in the paper.
A key separator is probably:
G(a -> F b)
Here the inner F b has unbounded horizon, so a fixed timestamp pipeline based only on finite look-ahead is not obviously sufficient.
Do NOT assume that every unbounded-horizon nested formula requires progression, however. Empirically, formulas like F(F a), F(G a), G(F a), G(G a) were observed to work under original RuleRunner. Semantic absorption/idempotence may allow safe merging.
Therefore:
- finite horizon may give a strong sufficient condition;
- it is probably not the exact maximal correctness boundary.
Existing empirical observations
Original RuleRunner vs exact DFA:
- flat formulas such as F(a & b), a U b, G(a -> b) → no mismatches observed;
- F(F a), F(G a), G(F a), G(G a) → no mismatches observed;
- F(X a) → no mismatches observed;
- G(X a) → mismatches;
- X(X a) → many mismatches;
- F(a & X b) → mismatches;
- G(a -> X b) → mismatches;
- a U (b & X c) → mismatches;
- G(a -> F b) → mismatches.
These observations show that “nested temporal operator” or “temporal depth > 1” is too coarse.
What to investigate now
Start from the actual RuleRunner paper/rule-generation semantics in the repository and determine whether the bounded-horizon intuition can be made formal.
Please work in this order:
1. Inspect the original RuleRunner evaluation and reactivation rules, especially the handling of X, W, F, G, U, R.
2. Formalize what exactly a timestamp-/age-indexed truth register would mean operationally.
3. Test the smallest case X(X a) first.
   Determine:
   - which conflicting RuleRunner registers exist in the old construction;
   - the minimal extra indexing required;
   - how evaluation/reactivation rules change;
   - whether the resulting rule system remains static and CILP-encodable.
4. Generalize from X^k a.
   Try to prove a bound on the number of required temporal slots as a function of a finite temporal horizon.
5. Then test an unbounded outer operator with bounded inner horizon:
   - G(X a)
   - F(a & X b)
   - G(a -> X b)
   Check whether timestamp-indexing of the finite-horizon inner part is enough.
6. Compare with:
   - G(a -> F b)
   Determine precisely why the same static bounded-index mechanism fails or is insufficient there.
7. Only after the operational mechanism is understood, attempt a formal fragment/condition.
   Prefer defining first a semantic/operational property such as:
   every potentially conflated subformula admits only a statically bounded number of temporally distinct live evaluations
   
   and only then derive a syntactic condition or grammar.
8. Check whether Finkbeiner–Kuhtz’s temporal-horizon definition can be reused directly or needs adaptation to finite-trace LTLf / RuleRunner semantics.
What NOT to do yet
- Do not rewrite the paper section yet.
- Do not replace the current progression-based repair.
- Do not assume the bounded-horizon construction is sufficient for all finite-horizon LTLf without proof.
- Do not assume unbounded horizon automatically implies progression is necessary.
- Do not introduce new terminology casually; in particular avoid “obligation”.
- Do not optimize implementation before the semantics are clear.
Desired output of this exploration
Ideally produce:
1. a precise operational definition of bounded-horizon RuleRunner;
2. a worked execution for X(X a);
3. a generalization to finite-horizon nested temporal formulas;
4. a proof sketch/theorem stating a sufficient correctness condition;
5. counterexamples showing where the condition stops applying;
6. a comparison with Finkbeiner–Kuhtz clarifying what is borrowed and what is RuleRunner-specific;
7. a small empirical test plan comparing:
   - original RuleRunner,
   - bounded-horizon RuleRunner,
   - progression RuleRunner,
   - DFA oracle.
The scientific target is a clean story:
bug → exact source of information loss → bounded repair where finite temporal context suffices → general progression repair otherwise.
