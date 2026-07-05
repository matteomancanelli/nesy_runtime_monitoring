# Repairing RuleRunner via Formula Progression — Analysis & Paper Positioning

*A self-contained hand-off note. Read this together with
[`docs/nested_temporal_limitation.md`](nested_temporal_limitation.md), which
documents the bug this note analyzes. This note answers: **can RuleRunner be made
correct for all of LTLf, what does the fix cost, and is the fixed thing distinct
enough from the DFA to be worth a paper?** It is analysis only — no code was
written or run to produce it.*

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
- **A bounded partial fix exists for the `X`/`W` slice** (§2.5): a static
  shift-register of slots (depth = compile-time X-nesting depth) instead of one
  shared `[X b]` slot. Repairs the worst empirical offenders (`X(Xa)`, `G(Xa)`,
  `F(a∧Xb)`, …) **while keeping RuleRunner's architecture** (fixed-size, static).
  It does **not** cover `F`/`G`/`U`/`R` temporal-under-temporal nesting, whose
  in-flight instance count is unbounded — that needs progression. The fix is also
  the clean empirical separator of L1 from L2.
- **The complete fix for L1/L2 is formula progression**, and progression's reachable
  state set (quotiented by equivalence) **is** a DFA. So "repair RuleRunner to
  completeness" provably **converges to the automaton**. There is no
  complete-and-strictly-more-succinct stopping point, because the minimal DFA is
  the Myhill–Nerode floor.
- **CILP is orthogonal to all of this.** CILP encodes a *fixed* rule set; its
  static-neuron requirement *forces* you to materialize the reachable state set
  = build the DFA. So "RuleRunner-eval + progression + CILP" is a differentiable
  DFA with a *compositional construction* — essentially DeepDFA reached from the
  rule side.
- As **monitors**, the repaired RuleRunner and DeepDFA are interchangeable (same
  verdicts; on crisp throughput the symbolic DFA is optimal anyway). They are
  **not** interchangeable as **learnable objects**: DeepDFA adapts an opaque
  transition tensor; the progression/CILP form adapts a **structured,
  subformula-indexed specification**. That difference — not speed — is the reason
  both can exist and the reason the fix is worth publishing.

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
   `_until_release`, `_next_like`) reinstalls a **fresh** operand instance onto
   the **same** `Node.key`;
3. the reactivation write-back in `engine.step`
   (`self._state = {lit … startswith("R[")}`) overwrites that shared slot.

The slot is addressed by **subformula identity**, but correctness requires
addressing by **(subformula, temporal context / instance)**. The `A`/`B` modes
in `_next_like` are a partial attempt to distinguish "the maturing instance"
from "a freshly deferred instance," but they share the `[X b]` slot, and the
CILP OR-merge (condition C2) fuses them the wrong way. The eval phase is
intra-cell and fine; the defect is purely in the **cross-cell carry**
(reactivation) collapsing distinct temporal contexts onto one address.

---

## 2. L1 vs L2 — the two limitations, kept separate

### L1 — Instance conflation: a construction defect

Every empirically-failing formula (`F(a∧Xb)`, `G(a→Xb)`, `X(Xa)`, `G(a→Fb)`;
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

## 2.5 The bounded partial fix (X/W family) — stays *inside* RuleRunner

Before jumping to progression (the complete fix that *leaves* the architecture),
there is a cheaper, **incomplete** fix that repairs the worst empirical offenders
while keeping RuleRunner's defining properties: fixed-size, compile-time-static,
no runtime state growth. It is the concrete construction-level repair for the
`X`/`W` slice of L1.

**The idea.** The `A`/`B` modes in `_next_like` are already *trying* to
distinguish "the instance maturing from the previous cell" (mode `A`, must read
`b` now) from "a freshly deferred instance" (mode `B`, must *not* read `b` yet) —
they just make both share the one `[X b]` slot, and the CILP OR-merge fuses them.
The fix is to **stop sharing the slot**: give the next-operator a small
**shift-register** of slots instead of one.

**Why it is bounded and static.** Under `k` nested `X`s, the number of instances
of an operand that can be simultaneously "in flight" (staggered across cells) is
at most `k+1`, and **`k` is known at compile time** from the parse tree. So
allocate `k+1` slots per next-chain (equivalently: the minimal DFA of `X^k b` is
a `(k+2)`-state shift register — that is literally what you are re-deriving).
Each cell, values shift by one position; the maturing value lands in the slot
mode `A` reads, the fresh defer goes to a *distinct* slot. No OR-merge, no
conflation. The rule set stays finite and compile-time-static; the state stays
fixed-size. **This does not use progression and does not touch the automaton.**

**What it covers.** The reliable `X`/`W` triggers — the worst rows of the
`nested_temporal_limitation.md` §5 table: `X(Xa)` (158/400), `G(Xa)` (51),
`F(a∧Xb)` (52), `a U (b∧Xc)` (48), `G(a→Xb)` (58). All have a *deferring next*
as the conflated inner operator, and a **statically bounded** in-flight count.

**What it does NOT cover, and why.** Nesting whose inner deferring operator is an
**unbounded-horizon** temporal (`F`/`G`/`U`/`R`) — e.g. `G(a→Fb)` (the BPM
response pattern, 29/400). There the number of concurrently-live operand
instances is **trace-dependent and unbounded in the naming**, so no static slot
count suffices. The obligations are still *mergeable in principle* (◇ is
idempotent: "does any pending ◇-obligation survive" is a bounded summary — which
is why the DFA stays finite), but the correct merge is operator-specific and, in
general, entangles into disjunctions of obligations. That is exactly where the
bounded shift-register stops and you need the full boolean-combination state of
progression (§3). So:

- **`X`/`W` nesting** → fixed with a static shift-register, architecture intact.
  This is the paper-defensible "we repair a slice the three original papers get
  silently wrong, at no architectural cost."
- **`F`/`G`/`U`/`R` temporal-under-temporal nesting** → needs progression → needs
  DFA-sized state (L2). No bounded fix.

**This is the empirical separator of L1 from L2.** After the shift-register fix,
`X(Xa)`/`G(Xa)`/`F(a∧Xb)` should go to **0** mismatches while `G(a→Fb)` stays
**> 0** — proof that the two limitations are distinct (bounded-construction
defect vs succinctness ceiling), not one phenomenon. (This is the experiment in
§8.1.)

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
progress(Xφ, s)    = φ                              ← key: X "falls through" to its operand
progress(◇φ, s)    = progress(φ,s) ∨ ◇φ
progress(□φ, s)    = progress(φ,s) ∧ □φ
progress(φUψ, s)   = progress(ψ,s) ∨ (progress(φ,s) ∧ φUψ)
```

(plus end-of-trace handling for the finite semantics). The carried state is a
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

## 4. CILP is orthogonal — and forces materialization to the DFA

CILP is only a way to compile a **fixed** rule set into a net (one hidden unit
per rule; `sign`/`tanh`), differentiable when you swap `sign→tanh`. It does **not**
require the one-slot-per-subformula state — that is RuleRunner's choice.

The sharp consequence:

> **CILP needs a static neuron set ⟹ a fixed, finite state vocabulary ⟹ you must
> materialize the reachable progression states ⟹ that set is the DFA.**

Lazy progression rewrites formula syntax on the fly, so it is *not* a fixed
propositional rule set and is not CILP-encodable as-is (you cannot assign neurons
to states you have not enumerated). The moment you want CILP, you must precompute
the reachable set and index states — and that is the DFA, with rules
"state × guard → state." **CILP does not prevent the merge; it forces it to the
DFA.** For the adaptation goal (Paper B) this is fine — desirable, even.

The honest merged design, "RuleRunner-eval + progression + CILP":

- **intra-cell**: keep RuleRunner's factorized, correct evaluation (parallel rule
  firing over subformula truth values — never the bug; cheap, CILP-friendly);
- **cross-cell**: replace the lossy reactivation (collapse to one R-literal per
  subformula) with a **progressed-state → progressed-state** transition over the
  materialized reachable set (an automaton-state carry, not a cube);
- encode the whole transition in CILP with `tanh` for differentiability.

The result **is** a differentiable DFA with a **compositional construction** —
essentially **DeepDFA reached from the rule side**. What survives of the
RuleRunner flavor: (a) the compositional per-operator construction, (b)
differentiability/adaptation. What is irrecoverably lost: the
one-slot-per-subformula succinctness (L2 forbids keeping it while complete).

**Practical middle ground worth building:** a *lazy* progression monitor with
boolean simplification that materializes **only reachable** states. Complete,
often far smaller than the dense product (simplification quotients; you visit
only the reachable portion), CILP-encodable over the reachable fragment,
differentiable. Worst-case it still collapses to DFA size — nothing beats
Myhill–Nerode — but the typical/structured case can be exponentially smaller.

---

## 5. Repaired-RuleRunner vs DeepDFA — where they genuinely differ

### As monitors: interchangeable (concede this loudly)

Once complete, the progression-RuleRunner **is** the DFA: same verdicts on every
crisp trace. On crisp throughput the symbolic DFA (dict lookup) is the optimum,
and the rule form pays `depth+1` within-step passes per cell — pure overhead.
**RuleRunner-form never wins on monitoring speed**; do not try to defend it there
(it is exactly the speed trap the project's `CLAUDE.md` warns against).

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
- **Progression** is intrinsically **lazy**: materializes only reachable states
  on the fly, never builds the full DFA. For formulas whose full DFA explodes but
  whose *actually-visited* portion is small, it monitors without ever paying the
  worst case (classic on-the-fly vs explicit-state advantage).

If a reviewer says "you could make DeepDFA lazy too" — that *reinforces* the
thesis: the lazy construction of DeepDFA **is** the progression rules. The two
paradigms meet exactly there. That is the boundary, not a hole.

### Axis 3 — Adaptation: the real reason to keep both (bridge to Paper B)

Here they are **not** interchangeable. The question: when you backprop, *what*
are you adapting and *what does the gradient mean?*

- **DeepDFA**: learnable = transition-tensor entries (`soft_matrix`) and
  accepting/rejecting vectors. The gradient adapts **transition probabilities
  between opaque states** (`q7` has no semantic label). Consequences:
  `|Q|²·|Σ|` parameters, no structural inductive bias, hard to regularize toward
  "sensible specs," and the learned object drifts into an **arbitrary weighted
  automaton** — no longer guaranteed to correspond to an LTLf formula. You cannot
  re-extract a readable spec.
- **Progression/CILP**: learnable parameters sit on the **rules**, indexed by
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

**Net thesis:** the two are interchangeable **as monitors** and **not as
learnable objects**. DeepDFA adapts an opaque automaton; the progression form
adapts a structured specification. The gradient means different things. For the
NeSy dream — correct a wrong spec from data and hand back a readable one — the
structured form is the right substrate; for fast crisp/soft batched monitoring,
DeepDFA (or symbolic) is.

This fills the project's **Phase 3.2 empty cell** ("exact + fast at runtime
*and* differentiable for adaptation," the hybrid paradigm): the progression/CILP
form is the "structured *and* differentiable" corner — not a faster monitor, a
**structured learner**.

Minor axis (mention, do not lean on it): on non-read-once guards the two soft
semantics can diverge, because the factored form tracks boolean structure instead
of marginalizing over `Q` — ties into Phases 3.1/3.3, but subtle.

---

## 6. Do both make sense? Yes — split by lifecycle stage

Frame it by **lifecycle**, not "which is better":

- **Deployment / monitoring**: symbolic DFA (crisp, optimal) or DeepDFA
  (soft-input, GPU batch). Fast, explicit, spec **frozen**.
- **Spec-engineering from data / repair / learning**: progression-CILP form.
  Structured, interpretable, localized adaptation that returns a specification.
  Spec **alive**.

One watches a fixed spec; the other **repairs** it when data contradicts it.
Neither makes the other redundant — like an optimizing compiler does not make an
IDE redundant.

---

## 7. How to present the fix in the paper without "there's the DFA, why bother?"

Three moves, in order:

1. **Concede the monitoring equivalence first, loudly.** "As a monitor the
   repaired RuleRunner *is* the DFA, and on crisp throughput the symbolic DFA is
   optimal — we do not propose this form for speed." Remove the objection before
   it is raised. Defending RuleRunner on its weakest ground invites the takedown.
2. **Relocate the contribution from the monitor to the adaptable specification.**
   The reason to keep the RuleRunner lineage is not runtime: its representation
   carries **subformula-/operator-indexed structure that survives as a learning
   parameterization**. DeepDFA gives a differentiable automaton with opaque
   states; the progression form gives a differentiable **specification** with
   interpretable, localizable parameters. For "adapt/repair a spec and return
   readable LTLf" they are not interchangeable. This is Paper B's reason to exist.
3. **Sell the fix as a theoretical result, not a product.** Even if nobody runs
   the repaired monitor, the **analysis** is the contribution: "the minimal
   repair that makes the compositional rule encoding complete provably converges
   to the automaton, and the L1 (construction) / L2 (expressivity) decomposition
   says exactly where and why." You are **mapping the paradigm boundary with a
   theorem**, and answering the reviewer's own question ("why not just the DFA?")
   with a result instead of an excuse. The DFA is not the rival that beats you —
   it is the fixed point every correct compositional repair provably tends to,
   which is exactly the argument that supports Paper A's thesis (the
   automata-based representation is the more general foundation).

One-line intro framing: *we do not save RuleRunner as a monitor — we show that
repairing it turns it into a differentiable DFA with a compositional
construction, and that it is that compositional structure, not speed, that makes
it an adaptation substrate the opaque DFA is not.*

---

## 8. Concrete next steps (when moving from analysis to code)

Analysis only so far. When implementing:

1. **Bounded X/W repair (validates L1 ≠ L2 empirically).** Give the next-operator
   a small **static shift-register** (depth = static X-nesting depth) instead of a
   single `[X b]` slot shared between the maturing and freshly-deferred instances.
   Re-run the mismatch sweep of `nested_temporal_limitation.md` §5: expect
   `X(Xa)`, `G(Xa)`, `F(a∧Xb)` → **0** mismatches while `G(a→Fb)` (nesting of
   `F`, not `X`) stays **> 0**. That is the clean empirical proof L1 and L2 are
   separate.
2. **State-space growth measurement (materializes L2).** On a family with known
   state blow-up (deep nested `X`, or a pattern forcing exponential
   determinization), measure the size of the progression **reachable set** vs
   `#subformulae`. Expect the `singly-exp → doubly-exp` gap to appear exactly
   where L2 predicts, and stay small (≈ `#subformulae`) on the flat fragment
   where RuleRunner was already correct.
3. **Adaptation A/B (materializes Axis 3).** Take a wrong spec (wrong
   atom/threshold), adapt via gradient in **both** forms, and compare **not**
   recovered accuracy (likely similar) but: **parameter count, interpretability of
   the result, and whether the DeepDFA form stays a valid LTLf spec or drifts into
   a weighted automaton.** That is where the difference becomes a figure rather
   than a claim.

---

## 9. Provenance

This note distills a design conversation (2026-07-01). It is analysis and
argument, cross-checked against the repository's implementation
(`rules.py`, `engine.py`, `parse_tree.py`, `deep_dfa.py`) and against
`nested_temporal_limitation.md`. The worked progression traces (§3) were computed
by hand and should be re-verified against `SymbolicDFAMonitor` before being
quoted in the paper. The LTLf→DFA 2EXP fact and the Bacchus–Kabanza progression
rules are standard; cite primary sources in the paper rather than this note.