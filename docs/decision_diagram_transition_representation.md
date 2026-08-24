# Decision-diagram (BDD/SDD) transition representations for DeepDFA

> **Scope note (2026-07-13, post ICLR-refocus).** This direction straddles the
> repo split. The **crisp/scalability side** (compact exact transition
> representation, batched compiled circuits, representation-size and throughput
> experiments) is Phase 4 of THIS repo's roadmap (see CLAUDE.md). The
> **uncertainty-learning/evaluation thread** (§3.1, §8 — calibration experiments
> and probabilistic verdict interpretation) belongs to the probabilistic-monitoring
> thread in `artur_future_work/`, which holds a copy of this note. Coordinate
> before building either half.

**Status:** exploratory planning note (2026-07-07; novelty check updated 2026-08-23). Captures a candidate direction
raised while reviewing §4 (DeepDFA). Records the technical case, the GPU tension and
its resolution, the relationship to the LydiaSyft / symbolic-synthesis line, and a
strategic recommendation about *when* this belongs in a paper. Not yet scoped into any
phase. A signpost paragraph is drafted (inert) in `latex/8_conclusion.tex`.

---

## 0. TL;DR

Represent the DeepDFA transition function with **decision diagrams** — reduced ordered
BDDs, or better **SDDs** — over the atom variables, instead of (a) the dense `2^|Σ|`
tensor or (b) the flat disjoint-cube cover of our current factored mode. This is the
principled compactness upgrade of the factored path: a cube cover *is* the paths-to-1 of
a BDD but without the subfunction **sharing** that can make a diagram compact. The
current cube backend is already exact and differentiable; the possible payoff is a
smaller compiled representation without surrendering those properties:

1. **Preserves exact WMC with sharing.** The soft transition `M(p)` is already computed
   exactly by our disjoint cubes. A deterministic, decomposable compiled circuit computes
   the same WMC in time linear in its circuit size and may reuse subfunctions that the
   flat cube list repeats. NeSyA already establishes this general construction and its
   probabilistic state-distribution semantics; it is prior art, not our novelty.
2. **Mitigates the alphabet blow-up when structure exists.** Sharing compresses the guard
   representation the cube cover cannot; the exponential moves from "always `2^|Σ|`" to
   "only for guards with no compact diagram under any ordering."
3. **Stays GPU-compatible if used at compile time.** Compile the diagram once into a
   *fixed* arithmetic circuit for the guard probabilities; evaluate that circuit densely
   and batched. The runtime hot loop remains a small regular `|Q|×|Q|` matmul.

This is a candidate systems extension, not a new semantic paradigm. Its value depends on
showing a material representation-size or batched-runtime improvement for LTLf monitors.

---

## 1. Where this came from

The factored representation we wrote up in §4.4 decomposes each MONA edge guard, by
Shannon expansion, into a disjoint cube cover (require-true / require-false integer masks)
and assembles the transition matrix by a vectorized mask reduction. Two honest weaknesses
were flagged in the text:

- **The cube count can be `Θ(2^k)`** for a guard on `k` atoms with no compact orthogonal
  cover (§4.4). So factoring *shifts* the alphabet blow-up rather than removing it.
- **The historical recursive approximation double-counts on non-read-once guards.**
  The closure `P(∨)=1−∏(1−·)` over-counts shared atoms. The production
  probabilistic path now uses the disjoint cubes as exact WMC, while retaining
  the recursion explicitly as a diagnostic baseline.

The remaining cube-count problem is a symptom of using a representation *without
subfunction sharing*; it is not an exactness problem. Decision diagrams are the data
structure the knowledge-compilation community built to address that compression axis.
The connection to LydiaSyft (below) is that this same
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

### 3.1 Alphabet axis (`2^|Σ|`) — the relevant case

This is where a diagram is the natural, direct fix.

- A BDD/ADD over the atom variables represents the guard (or the whole per-symbol
  transition) compactly, sharing common subfunctions the cube list re-enumerates.
- **Exact soft transitions via WMC.** `M(p)[q,q'] = P(guard_{q→q'} satisfied | independent
  atom probs p)`. Our disjoint cubes and deterministic+decomposable circuits both compute
  this marginal exactly for arbitrary guards; the diagram may be smaller through sharing.
  This is the Darwiche knowledge-compilation result (`darwiche2002knowledge`) and is used
  directly for symbolic automata by NeSyA (`NesyA`).
- **Row-stochasticity is preserved, not newly restored.** Exact guard marginals sum to one
  because each state's outgoing guards partition the valuation space. Both the current
  cube backend and a future compiled-circuit backend have this property.

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
   assume a speed win. Since exactness is already available from cubes and NeSyA supplies
   the compiled-WMC semantics, the result must stand on compactness and systems evidence.
4. **We already get MONA's BDDs for free-ish.** Investigate extracting MONA's internal
   (MT)BDD transition rather than re-compiling from the parsed explicit table; may save the
   compile step entirely.

---

## 6. Relationship to existing work (novelty check: resolved)

- **NeSyA** (`NesyA`, IJCAI 2025) already gives exact differentiable guard WMC using
  compiled forms such as d-DNNF, the expected transition matrix, and the theorem that its
  forward state mass equals the probability mass of traces reaching each state. This
  resolves the novelty check: exact compiled-WMC automaton semantics are prior art. Our
  defensible delta is the LTLf runtime-monitoring specialization, three-valued/crisp
  comparison, and an empirical backend comparison (cubes versus shared circuits), if the
  latter shows a useful size or throughput gain.
- **T-ILR** (`t_ilr_2025`) is adjacent but semantically different: it evaluates LTLf
  directly under fuzzy Zadeh semantics and performs iterative local refinement, rather
  than computing a probabilistic automaton acceptance marginal.
- **Knowledge compilation / WMC** (`darwiche2002knowledge`) is the theoretical backbone.
- **Probabilistic circuits / tractable inference** (`liu2024tractable`, `pseudosemantic_loss`)
  are the "compile logic → differentiable GPU circuit" template to imitate.
- **LydiaSyft / symbolic synthesis** (`lydiasyft2025`, `LTL2DFA1`, `LTL2DFA3`) is the
  source of the symbolic-DFA representation, in an adjacent (synthesis) task.

### Public NeSyA artifact audit (2026-08-23)

The implementation linked by the NeSyA paper is publicly available at
<https://github.com/nmanginas/nesya>; the live repository was verified at commit
`aa5830e12b81b9a618e52739de6de629afebd10e` (2025-12-16). The relevant reusable
component is `deepfa/automaton.py`:

- guards are compiled with the external `dsharp` executable through the Python
  `nnf` package, then smoothed;
- WMC is evaluated with `nnf.amc.eval` into a full
  `(batch, sequence, |Q|, |Q|)` transition stack;
- state propagation is sequential over time and returns final accepting mass;
- the supplied determinism check tests mutual exclusion but explicitly does not
  enforce exhaustive outgoing guards, so a fair adapter must independently
  check row mass/completeness;
- the repository contains experiment scripts but no substantive unit-test suite.

The repository is GPL-3.0. For a comparison, pin and run the upstream artifact
in a separate reference environment instead of copying its implementation into
this repository. Use the same LTLf-generated DFA, probability tensors, dtype,
device, warm-up, and synchronization policy. Compare final acceptance values,
guard-compilation time, circuit/cube representation size, peak memory, and
batched forward time. NeSyA does not provide this project's exact three-valued
online sink/trap interface, so presenting its complete application as a drop-in
runtime-monitor baseline would be misleading; the controlled comparison unit
is the expected-transition/WMC backend.

---

## 7. Strategic recommendation — include now or postpone?

**Recommendation: postpone as a full contribution; signpost in the current draft now.**

Rationale:

- **Scope.** Paper A already has a coherent thesis (three-way capability comparison) and
  unfinished committed work (Phase 0 GPU re-runs; Phase 2 adaptation). A full
  decision-diagram monitor = new implementation + new experiments + a small theory section
  = a fourth paradigm's worth of work. Bolting it on delays a near-finishable paper and
  under-develops a strong idea.
- **Paper A does not need to eliminate the alphabet heel** — characterizing it neutrally
  is the honest three-way story. Cubes already remove unconditional alphabet enumeration
  on structured guards; shared circuits would target the remaining representation-size
  problem.
- **This idea is not automatically a separate-paper contribution.** Exact differentiable
  automaton WMC is already established by NeSyA. It becomes substantial only if the LTLf
  runtime setting exposes a new algorithmic or systems result, such as direct reuse of
  MONA's MTBDD plus a convincing representation-size/throughput advantage.
- **Adaptation is likely too big** (user's own read, 2026-07-07). Good — this direction is
  a *representation/monitoring* contribution that does **not** require the full adaptation
  training story, so it is a more tractable way to raise the paper's ambition than Phase 2.

Concretely: finish Paper A on its current thesis, keep the inert signpost paragraph in
`8_conclusion.tex`, and develop the full construction next — as its own paper, or as the
technical core of a paper that *replaces* the adaptation PoC as the headline.

---

## 8. What could make this a contribution?

The current Paper A risk is that "we reproduced three known monitors and timed them" reads
as engineering, not novelty. A decision-diagram backend does not by itself fix that risk,
because NeSyA already provides the central exact-WMC construction. A defensible claim
would have to be narrower and empirically demonstrated:

> For LTLf-generated monitor guards, a shared compiled-circuit backend can preserve the
> existing exact differentiable transition semantics while reducing compiled size and/or
> batched runtime relative to disjoint cubes, without materializing the dense alphabet.

What would be needed to support that claim:

- **A nontrivial delta beyond NeSyA:** for example, a direct MONA-MTBDD extraction
  algorithm, a runtime-specific circuit batching scheme, or a formal representation bound
  for an important LTLf guard family. Re-proving exactness and linear-time circuit WMC is
  useful exposition but not novelty.
- **A neutrality-preserving story:** it fills the empty "hybrid" cell (exact+compact at
  runtime *and* differentiable), completing the three-Achilles-heels narrative rather than
  crowning a winner.
- **Evaluation can remain focused:** representation size (diagram versus cubes versus
  dense alphabet) and batched-circuit throughput are the decisive panels. Calibration is
  only a correctness control because the existing cube path is already exact.

### Minimal experiment sketch (if pursued)

1. **WMC correctness control.** On `majority3` and other non-read-once guards: recursive
   approximation (over-counts) versus cube and diagram paths (both exact). This verifies
   the implementation but is not the headline result.
2. **Representation size.** Diagram size (SDD/BDD) vs dense `2^|Σ|` vs disjoint-cube count,
   across the IJCNN breadth family and a deliberately non-read-once / non-decomposable
   family — show where sharing wins and where nothing does (honest worst case).
3. **Batched-circuit throughput.** Fixed compiled circuit evaluated batched on GPU vs dense
   DeepDFA vs cube-factored — determine whether sharing survives GPU execution overhead.
4. **(stretch) Differentiability smoke test.** Gradient of `acc_t` wrt `p` through the
   compiled circuit — demonstrates the adaptation substrate without committing to the full
   Phase 2 training story.

### Open questions to settle before committing

- Which concrete result goes beyond **NeSyA**: direct MONA reuse, a runtime-specific
  batching algorithm, a representation theorem for an LTLf family, or only an empirical
  backend comparison? Do not begin implementation without choosing one.
- SDD vs BDD vs d-DNNF in practice for MONA guards — which compiles smallest, which is
  easiest to turn into a batched tensor circuit (`PySDD` maturity, differentiability).
- Can we reuse MONA's internal MTBDD directly, skipping recompilation from the DOT table?
- Circuit-eval throughput at small `|Q|`: is kernel-launch overhead a problem, and does
  layering / fusing the circuit help?

---

## 9. Pointers

- Signpost paragraph (inert): `latex/8_conclusion.tex` (remove `\iffalse`/`\fi` to promote).
- Factored representation it upgrades: `latex/4_deepdfa.tex` §4.4, `src/monitors/deep_dfa.py`
  (`_guard_cubes` / `_shannon_cubes` / `exact_matrix` / `recursive_matrix`).
- Calibration harness to extend (moved to the future-work fork): `artur_future_work/src/benchmarks/calibration.py`, `artur_future_work/experiments/exp_uncertainty.py`.
- Bib keys added for this: `lydiasyft2025`, `darwiche2002knowledge`, `t_ilr_2025`
  (plus existing `LTL2DFA1`, `LTL2DFA3`, `NesyA`, `pseudosemantic_loss`, `liu2024tractable`).
