# Decision-diagram (BDD/SDD) transition representations for DeepDFA

> **Scope note (2026-07-13, post ICLR-refocus).** This is the future-work copy.
> The **calibration/WMC-exactness headline** (§3.1, §8 — exact soft transitions,
> calibrated probabilistic verdicts on arbitrary guards) is THIS fork's thread
> (see CLAUDE.md, Thread 2). The **crisp/scalability side** (representation
> size, batched compiled-circuit throughput) is on the parent repo's roadmap
> (its Phase 4). Coordinate before building either half.

**Status:** exploratory planning note (2026-07-07). Captures a candidate direction
raised while reviewing §4 (DeepDFA). Records the technical case, the GPU tension and
its resolution, the relationship to the LydiaSyft / symbolic-synthesis line, and a
strategic recommendation about *when* this belongs in a paper. Not yet scoped into any
phase. A signpost paragraph is drafted (inert) in `latex/8_conclusion.tex`.

---

## 0. TL;DR

Represent the DeepDFA transition function with **decision diagrams** — reduced ordered
BDDs, or better **SDDs** — over the atom variables, instead of (a) the dense `2^|Σ|`
tensor or (b) the flat disjoint-cube cover of our current factored mode. This is the
principled version of the factored path: a cube cover *is* the paths-to-1 of a BDD but
without the subfunction **sharing** that makes a diagram compact. The payoff is not
primarily speed; it is **exactness + tractability + differentiability at once**:

1. **Fixes the non-read-once calibration finding by construction.** The soft transition
   `M(p)` is a weighted model count (WMC). On a deterministic, decomposable diagram, WMC
   is exact in a **single pass linear in the diagram size** for *any* guard — not only
   read-once ones. No overshoot, no renormalization hack; `M(p)` is genuinely
   row-stochastic and `acc_t` is a calibrated probability.
2. **Mitigates the alphabet blow-up when structure exists.** Sharing compresses the guard
   representation the cube cover cannot; the exponential moves from "always `2^|Σ|`" to
   "only for guards with no compact diagram under any ordering."
3. **Stays GPU-compatible if used at compile time.** Compile the diagram once into a
   *fixed* arithmetic circuit for the guard probabilities; evaluate that circuit densely
   and batched. The runtime hot loop remains a small regular `|Q|×|Q|` matmul.

This is a strong candidate for the Phase 3.2 "fourth / hybrid paradigm" cell and,
plausibly, a **standalone contribution** stronger than the current draft on its own.

---

## 1. Where this came from

The factored representation we wrote up in §4.4 decomposes each MONA edge guard, by
Shannon expansion, into a disjoint cube cover (require-true / require-false integer masks)
and assembles the transition matrix by a vectorized mask reduction. Two honest weaknesses
were flagged in the text:

- **The cube count can be `Θ(2^k)`** for a guard on `k` atoms with no compact orthogonal
  cover (§4.4). So factoring *shifts* the alphabet blow-up rather than removing it.
- **The soft path double-counts on non-read-once guards** (§4.2 / Phase 1.4). The
  recursive independence closure `P(∨)=1−∏(1−·)` over-counts shared atoms; the raw
  acceptance score can exceed 1 and needs renormalization that fixes range, not
  calibration.

Both are symptoms of using a representation *without sharing and without a canonical WMC
semantics*. Decision diagrams are exactly the data structure the knowledge-compilation
community built to solve both. The connection to LydiaSyft (below) is that this same
line of research already moved LTLf DFAs from explicit to symbolic (BDD) representations
— for *synthesis* scalability, but the representation is reusable here.

---

## 2. The LydiaSyft / symbolic-synthesis line (what "symbolic DFA" means there)

Note the terminology clash: this project already overloads "symbolic" (Paradigm 1 =
explicit-state DFA with *symbolic boolean guards*). In the LydiaSyft/Zhu line "symbolic"
means the DFA's **state space and transition relation are encoded as BDDs** over boolean
state and input variables — no explicit state enumeration at all.

- **LydiaSyft** (Zhu & Favorito, TACAS 2025, `lydiasyft2025`): compositional LTLf→DFA,
  then the *explicit-state DFA is transformed into a symbolic-state DFA whose state space
  and transitions are both represented in BDDs*, and synthesis is a symbolic backward
  fixpoint game. The BDD form is what makes the game tractable.
- **Symbolic LTLf synthesis** (Zhu et al., IJCAI 2017, `LTL2DFA1` in the bib): the
  origin of the symbolic (BDD) DFA game for LTLf.
- **Lydia** (De Giacomo & Favorito, ICAPS 2021, `LTL2DFA3`): the compositional
  LTLf/LDLf → DFA construction underneath.
- **MONA** (which `ltlf2dfa` wraps): *already* represents its automaton transition
  functions as shared multi-terminal BDDs internally. The explicit transition table we
  parse out of the DOT is a **decompression** of a BDD form we currently throw away.

To avoid the terminology collision in any writeup, call this **"decision-diagram
transition representation,"** never just "symbolic."

---

## 3. Two structural limitations, addressed separately

Keep the **state axis** and the **alphabet axis** distinct — decision diagrams interact
very differently with each, and our actual bottleneck is the alphabet.

### 3.1 Alphabet axis (`2^|Σ|`) — the strong case

This is where a diagram is the natural, direct fix.

- A BDD/ADD over the atom variables represents the guard (or the whole per-symbol
  transition) compactly, sharing common subfunctions the cube list re-enumerates.
- **Exact soft transitions via WMC.** `M(p)[q,q'] = P(guard_{q→q'} satisfied | independent
  atom probs p)`. On a deterministic+decomposable diagram this marginal is one bottom-up
  pass, linear in diagram size, **exact for any guard**. This is the Darwiche knowledge-
  compilation-map result (`darwiche2002knowledge`), the same theory behind Semantic Loss
  / pseudo-semantic loss (`pseudosemantic_loss`) and probabilistic circuits
  (`liu2024tractable`). It makes the read-once caveat disappear.
- **Row-stochasticity restored.** Because the marginal is exact and a state's out-guards
  partition the assignment space, the row sums to 1 by construction — the Phase 1.4
  non-stochasticity finding is *solved*, not normalized around.

### 3.2 State axis (`|Q|`) — the conditional, mostly-orthogonal case

- Symbolic state encoding (log₂|Q| boolean vars + transition **relation** as a BDD) is
  what lets LydiaSyft scale synthesis past explicit enumeration; it can compress a
  doubly-exponential `|Q|` **when the automaton has BDD-friendly structure**.
- But it is **not** our current pain: in the IJCNN family `◇⋁(a₀∧aᵢ)`, `|Q|` is tiny
  (~2–3 states) while `2^|Σ|` explodes. So the state-axis symbolic encoding is a separate
  scalability story to reach for only with a **state-blowup formula family** (Phase 3.3),
  not for the benchmarks we have.
- Caveat: BDD size is **variable-ordering dependent** and still exponential in the worst
  case (some functions — e.g. integer multiplication — have exponential BDDs under every
  ordering). Diagrams move the exponential from "always" to "structure-dependent"; they do
  not abolish it. Same *character* of win as the factored cube cover, but strictly more
  powerful because of sharing.

---

## 4. The GPU tension and its resolution (the crux)

**Naive framing (wrong): "replace the tensor with a BDD engine."** This destroys the one
property that makes DeepDFA worth having.

- DeepDFA's batching advantage = dense, fixed-shape, branch-free matmuls; embarrassingly
  parallel across traces, batchable over time.
- BDD *manipulation* (apply, restrict, dynamic reordering, unique-table allocation) =
  pointer-chasing over an irregular DAG, data-dependent branching, poor locality, dynamic
  allocation — the canonical anti-pattern for SIMD/GPU hardware. Decades of "parallel BDD"
  work show only modest, specialized speedups. Running diagram traversal in the per-cell
  hot loop would throw away the batching.

**Correct framing: diagram at compile time, fixed arithmetic circuit at runtime.**

- Compile each guard's ADD/BDD **once** into a static arithmetic circuit for `P(guard | p)`
  (sums and products over atom probabilities — the WMC circuit).
- That circuit is a **fixed DAG**: identical structure across all traces and all
  timesteps; only the leaf `p` values change. A fixed DAG is GPU-friendly the same way a
  fixed neural network is — topologically layer it, and each level is a batched elementwise
  op. Vectorize over the (batch × time) axis, **not** over the diagram.
- Feed the resulting guard probabilities into the **same** small `|Q|×|Q|` batched matmul
  we already do. Keep states explicit-and-small (regular, GPU-happy); go symbolic only on
  the alphabet/guard axis (where the exponential is). This split is exactly what our
  benchmark motivates.

**Accurate one-liner:** BDDs as a *runtime engine* hurt batching; BDDs as a *compile-time
compiler into a fixed differentiable circuit* are compatible with it, and buy sharing +
exact tractable soft transitions the cube cover cannot.

---

## 5. Caveats (do not oversell)

1. **Variable ordering / worst case.** Compactness is structure-dependent; exponential in
   the worst case under every ordering for some functions. Report it as "structure-dependent
   compression," not a guaranteed reduction.
2. **SDD > BDD here, probably.** Sentential Decision Diagrams (with a vtree) usually
   compress better than OBDDs and support the same linear-time WMC; if we build this, SDDs
   (e.g. the UCLA `PySDD`) are likely the right substrate rather than plain BDDs.
3. **Circuit depth vs GPU efficiency.** A deep, skinny circuit layers into many small
   batched ops — fine for correctness, but the per-level kernel-launch overhead can bite at
   small `|Q|` (same overhead story as the existing DeepDFA micro-timing). Measure, don't
   assume a speed win — the win to lead with is **exactness/calibration**, not throughput.
4. **We already get MONA's BDDs for free-ish.** Investigate extracting MONA's internal
   (MT)BDD transition rather than re-compiling from the parsed explicit table; may save the
   compile step entirely.

---

## 6. Relationship to existing work (novelty check)

- **NeSyA** (`NesyA`, IJCAI 2025) and **T-ILR** (`t_ilr_2025`) already push probabilistic /
  neural inputs into automata in the non-mutex setting and are the closest neighbors. A
  contribution here must be explicit about the delta: *exact, tractable, differentiable
  soft transitions via compiled decision diagrams for **runtime three-valued monitoring**,
  with the calibration guarantee on non-read-once guards*, and the honest three-paradigm
  comparison as the frame. Check carefully what NeSyA compiles to and whether it already
  gives the exact-WMC transition — if it does, our delta is the monitoring framing +
  three-way comparison + the calibration analysis, not the representation itself.
- **Knowledge compilation / WMC** (`darwiche2002knowledge`) is the theoretical backbone.
- **Probabilistic circuits / tractable inference** (`liu2024tractable`, `pseudosemantic_loss`)
  are the "compile logic → differentiable GPU circuit" template to imitate.
- **LydiaSyft / symbolic synthesis** (`lydiasyft2025`, `LTL2DFA1`, `LTL2DFA3`) is the
  source of the symbolic-DFA representation, in an adjacent (synthesis) task.

---

## 7. Strategic recommendation — include now or postpone?

**Recommendation: postpone as a full contribution; signpost in the current draft now.**

Rationale:

- **Scope.** Paper A already has a coherent thesis (three-way capability comparison) and
  unfinished committed work (Phase 0 GPU re-runs; Phase 2 adaptation). A full
  decision-diagram monitor = new implementation + new experiments + a small theory section
  = a fourth paradigm's worth of work. Bolting it on delays a near-finishable paper and
  under-develops a strong idea.
- **Paper A does not need to *solve* the alphabet/calibration heel** — characterizing it
  neutrally is the honest three-way story. Solving it is the next paper.
- **This idea can carry its own paper.** "An exact, compact, differentiable runtime monitor
  via knowledge compilation" is theorem-backed (WMC tractability) and addresses the A*
  "too simple" worry head-on (see §8).
- **Adaptation is likely too big** (user's own read, 2026-07-07). Good — this direction is
  a *representation/monitoring* contribution that does **not** require the full adaptation
  training story, so it is a more tractable way to raise the paper's ambition than Phase 2.

Concretely: finish Paper A on its current thesis, keep the inert signpost paragraph in
`8_conclusion.tex`, and develop the full construction next — as its own paper, or as the
technical core of a paper that *replaces* the adaptation PoC as the headline.

---

## 8. Could this be THE contribution? (addressing the "too simple for A*" worry)

The current Paper A risk is that "we reproduced three known monitors and timed them" reads
as engineering, not novelty. A decision-diagram soft monitor gives a **crisp technical
claim** to anchor an A* submission:

> A runtime LTLf monitor whose soft (probabilistic-input) transition is an **exact**
> weighted model count, computed in time **linear in a compiled decision diagram**,
> **differentiable** in the atom probabilities, and therefore emitting a **calibrated**
> three-valued verdict on **arbitrary** (non-read-once) guards — where the tensorized
> DeepDFA soft path is only an independence approximation and the symbolic monitor cannot
> emit a probability at all.

Why this is A*-shaped:

- **A theorem, not just a system:** exactness + tractability (linear in diagram size) +
  differentiability, with the non-read-once calibration guarantee as the headline
  correctness result. Contrast cleanly against DeepDFA (approx on non-read-once) and
  symbolic (no soft output).
- **A neutrality-preserving story:** it fills the empty "hybrid" cell (exact+compact at
  runtime *and* differentiable), completing the three-Achilles-heels narrative rather than
  crowning a winner.
- **Evaluation is feasible without a training loop:** calibration/ECE + reliability
  diagrams on non-read-once families (extends the existing `exp_uncertainty.py` harness),
  plus alphabet-scaling (diagram size vs dense `2^|Σ|` vs cube count) and a batched-circuit
  throughput panel. No BPIC log, no adaptation training, no new dataset.

### Minimal experiment sketch (if pursued)

1. **Calibration correctness.** On `majority3` and other non-read-once guards: recursive
   soft path (over-counts) vs cube path (exact marginal) vs diagram-WMC (exact) — show the
   diagram is exactly row-stochastic and calibrated; reuse `calibration.py` (ECE, reliability).
2. **Representation size.** Diagram size (SDD/BDD) vs dense `2^|Σ|` vs disjoint-cube count,
   across the IJCNN breadth family and a deliberately non-read-once / non-decomposable
   family — show where sharing wins and where nothing does (honest worst case).
3. **Batched-circuit throughput.** Fixed compiled circuit evaluated batched on GPU vs dense
   DeepDFA vs cube-factored — establish competitiveness (lead with exactness, not speed).
4. **(stretch) Differentiability smoke test.** Gradient of `acc_t` wrt `p` through the
   compiled circuit — demonstrates the adaptation substrate without committing to the full
   Phase 2 training story.

### Open questions to settle before committing

- Does **NeSyA** already deliver the exact-WMC automaton transition? If yes, re-scope the
  delta to the *monitoring + three-valued verdict + calibration analysis + three-way
  comparison*, and say so explicitly.
- SDD vs BDD vs d-DNNF in practice for MONA guards — which compiles smallest, which is
  easiest to turn into a batched tensor circuit (`PySDD` maturity, differentiability).
- Can we reuse MONA's internal MTBDD directly, skipping recompilation from the DOT table?
- Circuit-eval throughput at small `|Q|`: is kernel-launch overhead a problem, and does
  layering / fusing the circuit help?

---

## 9. Pointers

- Signpost paragraph (inert): `latex/8_conclusion.tex` (remove `\iffalse`/`\fi` to promote).
- Factored representation it upgrades: `latex/4_deepdfa.tex` §4.4, `src/monitors/deep_dfa.py`
  (`_guard_cubes` / `_shannon_cubes` / `crisp_matrix` / `soft_matrix`).
- Calibration harness to extend: `src/benchmarks/calibration.py`, `experiments/exp_uncertainty.py`.
- Bib keys added for this: `lydiasyft2025`, `darwiche2002knowledge`, `t_ilr_2025`
  (plus existing `LTL2DFA1`, `LTL2DFA3`, `NesyA`, `pseudosemantic_loss`, `liu2024tractable`).
