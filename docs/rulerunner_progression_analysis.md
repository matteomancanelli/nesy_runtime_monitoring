# Repairing RuleRunner via Formula Progression — Analysis & Paper Positioning

*A design-analysis note. Read this together with
[`docs/nested_temporal_limitation.md`](nested_temporal_limitation.md), which
documents the bug this note analyzes. This note answers: **can RuleRunner be made
correct for all of LTLf, what does the fix cost, and is the fixed thing distinct
enough from the DFA to be worth a paper?** It began as analysis, but its
constructions have since been implemented and tested.  The authoritative
implementation status is [`rulerunner_status.md`](rulerunner_status.md); this
file remains the longer design argument.*

---

## 0. TL;DR

- The nested-temporal bug ("Artur bug") documented in
  `nested_temporal_limitation.md` is really **two** distinct limitations that
  that note conflates:
  - **L1 — instance conflation (a *construction* defect).** RuleRunner's
    rule-generation procedure loses information that a fixed-size state machine
    would keep. Proof: every failing formula has a *tiny* minimal DFA, so the
    required computation **is** representable by bounded state; the local rule
    construction just fails to produce it.
  - **L2 — succinctness ceiling (a genuine *expressivity* limit).** A state with
    one bounded-domain slot per subformula has ≤ `c^|φ|` (singly-exponential)
    configurations, but worst-case LTLf monitoring needs to distinguish up to
    `2^(2^|φ|)` (doubly-exponential) prefix classes (LTLf→DFA is 2EXP). So **no**
    construction that keeps the one-slot-per-subformula representation can be
    complete on all of LTLf.
- **A bounded-event partial fix exists** (§2.5): maximal finite-horizon
  subformulas are evaluated by a fixed observation pipeline and supplied as
  derived atoms to an exactly-certified old-RuleRunner skeleton.  This repairs
  `G(Xa)`, `F(a∧Xb)`, and related bounded-future aliases while remaining static.
  `X(Xa)` itself is already correct after the paper-faithful Next repairs.
- **The complete fix for L1/L2 is formula progression**, and progression's reachable
  state set (quotiented by equivalence) **is** a DFA. So "repair RuleRunner to
  completeness" provably **converges to the automaton**. There is no
  complete-and-strictly-more-succinct stopping point, because the minimal DFA is
  the Myhill–Nerode floor.
- **CILP is orthogonal to all of this.** CILP requires a finite neuron
  vocabulary, but those neurons may encode a factored pipeline or residual roots
  rather than one neuron per complete DFA state.  Exact global permanence can
  still require aggregate-state analysis; recurrence itself need not.
- As exact **crisp monitors**, the repaired RuleRunner and the DFA-derived
  encodings are semantically interchangeable.  A possible future distinction is
  their learning parameterization: an automaton tensor is state-indexed, whereas
  CILP rules are syntactically indexed.  Neither learning route is implemented or
  evaluated in this repository, so adaptation remains a hypothesis for a
  separate paper rather than a contribution or result here.

---

## 1. Precise localization of the limitation

The bug does **not** live where it might seem. It is not in:

- the per-operator truth tables (`_and`, `_or`, `_until`, … in
  [`src/monitors/rulerunner/rules.py`](../src/monitors/rulerunner/rules.py)) —
  these are locally correct;
- the CILP encoding ([`cilp.py`](../src/monitors/rulerunner/cilp.py)) — it
  reproduces the engine bit-for-bit;
- the intra-cell evaluation fixpoint loop
  ([`engine.py`](../src/monitors/rulerunner/engine.py)) — the eval phase is sound.

It lives in **one place: the addressing scheme of the recurrent state carried
across cells.** Concretely, the conjunction of:

1. `parse_tree.subformulae()` deduplicates by `Node.key` → **one slot per
   distinct syntactic subformula**;
2. `rules._subtree_reinstall` (called by `_eventually`, `_always`,
   `_until`, `_until_release`, `_next_like`) reinstalls a **fresh** operand instance onto
   the **same** `Node.key`;
3. the reactivation write-back in `engine.step`
   (`self._state = {lit … startswith("R[")}`) overwrites that shared slot.

The slot is addressed by **subformula identity**, but correctness requires
addressing by **(subformula, temporal context / instance)**. The unqualified
initial mode and `M` mode in `_next_like` distinguish "freshly deferred" from
"monitoring the operand," but overlapping instances still share the `[X b]` slot, and the
CILP OR-merge (condition C2) fuses them the wrong way. The eval phase is
intra-cell and fine; the defect is purely in the **cross-cell carry**
(reactivation) collapsing distinct temporal contexts onto one address.

---

## 2. L1 vs L2 — the two limitations, kept separate

### L1 — Instance conflation: a construction defect

Every confirmed failing formula (`F(a∧Xb)`, `G(a→Xb)`, `(Xa)∧X(Xa)`, `G(a→Fb)`;
see the mismatch table in `nested_temporal_limitation.md` §5) has a **tiny**
minimal DFA — a handful of states monitors each exactly. Therefore the required
computation **is** representable by a fixed-size machine (the DFA proves it
constructively). What fails is the **local rule-generation procedure**: it
discards information a fixed state would keep. For `F(a∧Xb)` the discarded datum
is a single bit ("was `a` true last cell?").

The §8 impossibility witness in `nested_temporal_limitation.md` (two traces with
an identical conflicted slot `[X b] = {F,?}` but opposite correct verdicts)
proves only that **the slot at RuleRunner's chosen granularity** is not a
sufficient statistic — **not** that no fixed state could be. That is the precise
difference between "the substrate cannot" and "this construction does not." L1 is
the latter.

> Note: `nested_temporal_limitation.md` §9 ("the only fix needs *unbounded*
> slots → architectural ceiling") gives the **wrong reason** for the small
> counterexamples — their required state is bounded (tiny DFA), not unbounded.
> §8 is solid as a proof that the *chosen granularity* is lossy; §9's
> generalization to "unbounded slots required" is true only in the worst case
> (that's L2), not the explanation of the minimal failures (that's L1).

### L2 — Succinctness ceiling: a genuine expressivity limit

RuleRunner's carried state is a **cube**: one value in `{T, F, ?}` per
subformula, independently — a point in `{T,F,?}^n`, `n = #subformulae`. That is
≤ `3^n` = **singly**-exponential configurations. Crucially it **factorizes**: it
assumes the thing to remember is one value per subformula, with **no
cross-subformula correlation**.

But correct monitoring sometimes requires a **correlation** between obligations
that does not decompose into per-subformula values — canonically a *disjunction
of obligations* ("either obligation X is live or Y is, but I don't know which").
RuleRunner has no slot for `X ∨ Y` as a joint object.

The counting argument (this is L2 proper):

- Reachable RuleRunner states ≤ `c^|φ|` (bounded values × `O(|φ|)` slots) =
  **singly** exponential.
- Correct LTLf monitoring must distinguish up to `2^(2^Ω(|φ|))` Myhill–Nerode
  classes (LTLf→DFA is **2EXP**: the minimal DFA can be doubly exponential).
- For large formulas in a hard family, `c^|φ| < 2^(2^Ω(|φ|))` ⇒ by pigeonhole,
  **no** construction over one-bounded-slot-per-subformula can be correct.

L2 is independent of L1 and is **not mentioned** in the existing note. It is the
real reason no succinct compositional encoding can be complete.

**Bottom line for Q1 ("expressivity or construction?"):** the documented "Artur
bug" is **construction** (L1); full-LTLf completeness is separately blocked by an
**expressivity** ceiling on succinctness (L2). Keep them distinct in the paper.

---

## 2.5 The bounded-event partial fix — stays static

The verified intermediate construction is not a bank attached only to `[Xb]`.
That would lose the correlation between `a@t` and `Xb@t` in `F(a∧Xb)`.  Instead,
identify each maximal finite-horizon subtree, evaluate that complete subtree
from a fixed observation pipeline, and feed its delayed Boolean value as an
event atom to the remaining RuleRunner skeleton.

The skeleton is admitted only if exact product checking proves that old
RuleRunner and the canonical DFA accept the same finite words.  This separates
the exact semantic condition from any conservative grammar proposed for the
paper.  The pipeline then gives a simple composition theorem: exact bounded
events plus a correct skeleton imply a correct final verdict for the original
formula.  End-of-trace flushes the last horizon cells using strong/weak Next
boundary values; early permanent decisions remain sound but may be delayed.

This covers `G(Xa)`, `F(a∧Xb)`, `G(a→Xb)`, and `a U (b∧Xc)`.  It does not alter
`G(a→Fb)`, whose relevant nested formula has unbounded horizon and whose
unchanged skeleton fails certification.  This is a limitation of the bounded
construction, not a proof that every formula with an unbounded island requires
progression: semantic absorption still makes examples such as `F(Fa)` safe.

Crucially, `X(Xa)` needs no repair.  Exact checking certifies nested Next chains;
the paper-faithful initial activation installs each child once at the correct
cell.  A genuine small shared-register example is `(Xa)∧X(Xa)`, where the same
deduplicated `Xa` register is reached at two offsets.

The complete formalization and executable reference are in
`bounded_event_rulerunner.md` and `src/monitors/rulerunner/bounded.py`.

---

## 3. Formula progression — the fix, and why it *is* the DFA

### What progression is

Bacchus–Kabanza progression: given `φ` and a single observation `s`, compute
`progress(φ, s) = φ'` such that `s·σ ⊨ φ ⟺ σ ⊨ φ'`. Defined compositionally,
per operator, then boolean-simplified:

```
progress(a, s)     = ⊤ if a∈s else ⊥
progress(¬φ, s)    = ¬progress(φ, s)
progress(φ∧ψ, s)   = progress(φ,s) ∧ progress(ψ,s)
progress(Xφ, s)    = φ ∧ F⊤                        ← F⊤ requires a non-empty suffix
progress(Wφ, s)    = φ ∨ G⊥                        ← G⊥ admits the empty suffix
progress(◇φ, s)    = progress(φ,s) ∨ ◇φ
progress(□φ, s)    = progress(φ,s) ∧ □φ
progress(φUψ, s)   = progress(ψ,s) ∨ (progress(φ,s) ∧ φUψ)
```

(with end-of-trace handled by evaluating the residual on the empty suffix). The carried state is a
**formula** — a point in the free boolean algebra over obligations, **not** a
cube. That is exactly why it can hold the disjunctions RuleRunner cannot.

### Worked example — the §8 witness that RuleRunner cannot distinguish

Formula `F(a ∧ X b)`.

**Trace B = (∅, {a}, {b})**, ground truth **SATISFY**:

- cell 0 `∅`: `progress(◇(a∧Xb)) = (a∧Xb → ⊥) ∨ ◇(a∧Xb) = ◇(a∧Xb)`
- cell 1 `{a}`: `(a∧Xb → a=⊤, Xb→b) = b`, so state `= b ∨ ◇(a∧Xb)`
- cell 2 `{b}`: `b → ⊤`, state `= ⊤` → **SATISFY** ✓

**Trace A = ({a}, ∅, {b})**, ground truth **VIOLATE**:

- cell 0 `{a}`: `(a∧Xb → b) ∨ ◇(a∧Xb) = b ∨ ◇(a∧Xb)`
- cell 1 `∅`: `b→⊥`, `◇(a∧Xb)→◇(a∧Xb)`, state `= ◇(a∧Xb)`  ← the failed b-obligation correctly drops to ⊥
- cell 2 `{b}`: state `= ◇(a∧Xb)`; end-of-trace `◇` = ⊥ → **VIOLATE** ✓

The decisive moment is **after cell 1**: A carries `◇(a∧Xb)`, B carries
`b ∨ ◇(a∧Xb)` — **different** formulas, so progression distinguishes them.
RuleRunner collapses both to the same per-subformula slot assignment and loses.
B's state is a **disjunction** ("either the live b-obligation, or a fresh ◇") —
precisely the joint object RuleRunner has no slot for. That is L2, concretely.

### Why the fix converges to the DFA

The set of formulas reachable by progression from `φ`, quotiented by logical
equivalence, is **finite** (boolean combinations of the Fischer–Ladner closure).
It defines an automaton: states = progressed formulas, transition = `progress`.
That automaton **is** a DFA equivalent to `φ`; minimized, it is the minimal DFA.
So:

> **Progression is not an alternative substrate to the DFA — it is the DFA built
> lazily**, materializing the successor state on demand instead of precompiling
> all states.

Hence any complete repair converges to the automaton: the minimal DFA is, by
Myhill–Nerode, the smallest complete deterministic online representation. Every
complete deterministic monitor has ≥ `|Q_min|` states. There is no
complete-and-strictly-more-succinct stopping point in the worst case.

**Why RuleRunner cannot "just do the same thing":** it can use progression — but
then its state is a formula (up to DFA size), forfeiting the factorized
one-slot-per-subformula representation that *defines* RuleRunner. Factorization
buys succinctness (`n` slots vs up to `2^(2^n)` states) and costs the ability to
hold disjunctions of obligations. Same choice, two sides.

---

## 4. CILP is orthogonal — and requires a finite residual vocabulary

CILP is only a way to compile a **fixed** rule set into a net (one hidden unit
per rule; `sign`/`tanh`), differentiable when you swap `sign→tanh`. It does **not**
require the one-slot-per-subformula state — that is RuleRunner's choice.

The sharp consequence is more nuanced than the earlier version of this note
claimed:

> **CILP needs a static neuron set, but those neurons may denote residual roots
> rather than whole deterministic states.**

Lazy progression rewrites formula syntax on the fly, so it is *not* a fixed
propositional rule set and is not CILP-encodable as-is.  A fixed CILP monitor
must precompute a finite vocabulary, but it can precompute the roots occurring
in progressed formulae and carry a multi-hot conjunction of them.  Because
progression distributes over conjunction, each active root can transition
independently.  Whole root sets need be enumerated only for operations that
genuinely quantify over the complete future language, such as exact sink/trap
classification.

The implemented merged design, "RuleRunner-eval + progression + CILP":

- **intra-cell**: keep RuleRunner's factorized, correct evaluation (parallel rule
  firing over subformula truth values — never the bug; cheap, CILP-friendly);
- **cross-cell**: replace the lossy reactivation with one local transition
  module per residual root.  An active root independently emits the top-level
  conjuncts of its progressed successor; all module outputs are unioned into
  the next multi-hot residual state;
- compile those modules as CILP layers over one shared literal space.  A module
  reads only its own root register and local observation guard, rather than
  recognizing the complete aggregate state.

Once the reachable residual graph is materialized, exact online permanence is
also a graph property. Mark residuals accepting on the empty suffix; a trap is
a state from which no accepting residual is reachable, and an accepting sink
is one from which no rejecting residual is reachable. Reverse reachability
precomputes both sets.  The flat monitor labels its whole-residual transitions;
the structured monitor uses a separate fixed aggregate-state label head.  That
head does not participate in recurrence. A lazy simplifier that stops only on
literal `true`/`false` remains sound but can detect the same permanent verdict
later.  The implemented normalizer also recognizes negation-normal-form
temporal dualities, idempotence of `F`/`G`, and finite-boundary-safe constant
identities.  These improve syntactic sharing but do not decide full LTLf
equivalence; for example, the valid implication `(a U b) -> F b` need not
reduce to literal `true`.

The resulting monitor is deterministic and finite-state, but its recurrent
vector is a structured multi-hot residual formula rather than a one-hot opaque
DFA state.  In the worst case the root vocabulary or the reachable aggregate
sets can still be exponential; in structured cases the recurrent register set
can be exponentially smaller than the number of aggregate states.  This is the
implemented middle ground: eager/static compilation with factorized runtime
recurrence, alongside a global fixed readout only where exact online permanence
requires it.

---

## 5. Repaired-RuleRunner vs DeepDFA — where they genuinely differ

### As monitors: semantically interchangeable

Once complete, progression and the canonical DFA recognize the same language
and return the same permanent verdicts on every crisp trace.  Their executable
representations and costs differ.  No RuleRunner speed advantage is claimed in
this note: comparative throughput and compilation measurements are deferred.

The real differences are three, in increasing importance.

### Axis 1 — State encoding: factored/symbolic vs one-hot

Does **not** violate L2 (both hit the ~`2^|φ|`-bit Myhill–Nerode floor in the
worst case); changes what you pay in the typical/structured case.

- **DeepDFA dense**: one-hot over `Q`, `|Q|` up to `2^(2^|φ|)` dimensions —
  exponentially redundant (`2^|φ|` bits of info in `2^(2^|φ|)` dimensions).
- **Progression/CILP**: state factored over subformulae (plus auxiliary literals
  for the boolean combinations needed for completeness); states sharing an
  obligation share literals. This is the **symbolic/BDD-like vs explicit**
  automaton distinction. On formulas with a **large but structured** state space
  (many states sharing obligations), the factored form can be exponentially more
  compact than DeepDFA's one-hot table.

Caveat for honesty: DeepDFA's *factored* path already attacks the **alphabet**
blow-up (`2^|AP|`); what a progression-state form adds is attacking the **state**
blow-up (`|Q|`). Two different dimensions of the `(|Q|, |Σ|, |Q|)` tensor. This
extends the project's capability matrix naturally: not "one wins," but "each
compresses a different blow-up dimension."

### Axis 2 — Construction: eager (compile-time) vs lazy (on-the-fly)

- **DeepDFA** compiles the whole DFA up front (ltlf2dfa/MONA): pays the
  determinization blow-up at compile time even if runtime visits three states.
  Some formulas have a doubly-exp minimal DFA and **MONA OOMs at compile time** —
  DeepDFA does not even start.
- **The progression semantics admits a lazy realization.**  The implemented
  `ProgressionEngine` materializes only the residual reached by the current
  trace.  The eager table, flat CILP, and structured CILP monitors instead
  compile reachable residual/aggregate graphs in advance because they require a
  fixed neural vocabulary and exact permanent labels.

If a reviewer says "you could make DeepDFA lazy too" — that *reinforces* the
thesis: the lazy construction of DeepDFA **is** the progression rules. The two
paradigms meet exactly there. That is the boundary, not a hole.

### Axis 3 — possible future adaptation distinction (not implemented here)

The current DeepDFA monitor is a fixed tensor realization, and the current CILP
monitors are hard-threshold exact encodings.  Neither trains parameters.  A
future project could ask: when a differentiable parameterization is introduced,
*what* is adapted and *what does the gradient mean?*  The following are design
hypotheses, not established properties of this implementation:

- **A DeepDFA-style learner** could parameterize transition-tensor entries and
  accepting/rejecting vectors. The gradient would adapt **transition probabilities
  between opaque states** (`q7` has no semantic label). Consequences:
  `|Q|²·|Σ|` parameters, no structural inductive bias, hard to regularize toward
  "sensible specs," and the learned object drifts into an **arbitrary weighted
  automaton** — no longer guaranteed to correspond to an LTLf formula. You cannot
  re-extract a readable spec.
- **A future progression/CILP learner** could place parameters on the **rules**, indexed by
  **subformula** and **operator**, attached at the **symbolic construction level**
  (before determinization closure). You parameterize a guard, an atom threshold,
  which operator governs a node; the determinized automaton is then a
  differentiable *function* of those parameters and the gradient flows through.
  This gives:
  - **syntactic locality** — adapt only the suspect subtree (this is IJCNN 2015's
    "local learning," now with a reason to exist);
  - **interpretability** — you learn a *corrected specification*, re-extractable
    as a formula, not an opaque matrix;
  - **inductive bias / sample efficiency** — few structured parameters vs
    `|Q|²·|Σ|`;
  - **closure in spec space** — you can constrain the result to remain valid LTLf.

**Future hypothesis:** the two are semantically interchangeable as crisp
monitors but could expose different inductive biases as learnable objects.  That
hypothesis requires an actual parameterization, training objective, and
evaluation before it can support a paper claim.

The present work therefore establishes only the exact structured substrate.  It
does not establish a “structured learner” or any adaptation result.

Minor axis (mention, do not lean on it): on non-read-once guards the two soft
semantics can diverge, because the factored form tracks boolean structure instead
of marginalizing over `Q` — ties into Phases 3.1/3.3, but subtle.

---

## 6. Possible future lifecycle split

Frame it by **lifecycle**, not "which is better":

- **Deployment / monitoring**: symbolic DFA (crisp, optimal) or DeepDFA
  (soft-input, GPU batch). Fast, explicit, spec **frozen**.
- **Spec-engineering from data / repair / learning**: a future
  progression-CILP learner could test whether syntactically indexed parameters
  enable localized, interpretable repair.

Only the fixed-spec monitoring side exists in this repository.  The repair side
is deferred and must not be described as an implemented capability.

---

## 7. Current paper positioning

Three moves, in order:

1. **State semantic equivalence precisely.**  Progression residuals induce a
   deterministic finite-state monitor recognizing the same language as the
   canonical DFA; the reachable residual graph is not automatically minimal.
2. **Locate the contribution in the representation analysis.**  The project
   identifies the information lost by the published RuleRunner address space,
   supplies an exact formula-level boundary certifier, derives a static bounded
   middle construction, and gives a complete progression construction while
   preserving rule-local recurrence.
3. **Separate present results from future motivation.**  Syntactically indexed
   rules may be a useful substrate for later adaptation, but no learning or
   performance advantage is claimed here.  Evaluation and adaptation require
   their own experiments and arguments.

One-line intro framing: *we retain RuleRunner's compositional rule structure,
identify exactly where its published recurrent state loses information, and
recover correctness either for bounded event islands or for all LTLf through
progression; any adaptation advantage is future work, not a present result.*

---

## 8. Implementation status and deferred extensions

The construction work proposed by the original version of this note is now
complete:

1. **Original boundary:** `certify_rule_runner` performs exact product checking
   against the canonical DFA and returns shortest witnesses.  It distinguishes
   language equivalence, prefix soundness, and exact online-label timing.
2. **Bounded-event CILP:** both the pooled/fixpoint flat organization and the
   per-`(subformula, offset)` structured organization are implemented, including
   the recurrent observation window and finite-suffix circuits.
3. **Exact extrapolation and batching:** a fixed CILP readout classifies the
   reachable composite pipeline state by reverse reachability, and equal-length
   windows are evaluated in fused cross-trace batches.
4. **Progression:** lazy, eager, flat CILP, and structured root-local CILP
   versions are implemented.  The eager/neural variants use exact permanent
   labels; the lazy oracle deliberately keeps its sound but incomplete
   syntactic early test.

Remaining work is extension or evaluation, not a missing correctness step:

- optionally seek a more permissive readable grammar for the original
  RuleRunner; the draft now proves a conservative flat-temporal fragment and
  the exact semantic boundary is already supplied by the certifier;
- optionally improve progression residual quotienting before treating raw
  closure sizes as right-language counts.  The current form is deterministic
  for its Boolean-skeleton/temporal-rewrite theory, not canonical modulo full
  LTLf equivalence;
- run benchmarks and formulate empirical performance claims only in the
  explicitly deferred evaluation phase;
- study adaptation separately.  No differentiable-learning result is claimed
  by the current RuleRunner implementation work.

---

## 9. Provenance

This note began with a design conversation (2026-07-01). Its implementation
claims are now cross-checked against `rules.py`, `engine.py`, `equivalence.py`,
the bounded CILP/extrapolation modules, `src/monitors/progression/`, and the
independent semantic-oracle tests.  Hand-worked traces remain explanatory
examples rather than primary evidence.  The LTLf-to-DFA 2EXP fact and the
Bacchus--Kabanza progression rules still require primary citations wherever
they are used in the paper; this note is not a citation source.
