# Richer benchmark family — findings (Phase 3.3)

The IJCNN `◇(⋁(a₀∧aᵢ))` family is a poor instrument: it early-terminates, and
its guards are **read-once** after MONA factoring, so DeepDFA's `soft_matrix` is
*exact* and the paradigm divergence that the capability story rests on is
invisible. Three new families ([src/benchmarks/formulas.py](../src/benchmarks/formulas.py))
each expose a gap. Two produce experimental findings (Exp 7,
[experiments/exp7_richer_family.py](../experiments/exp7_richer_family.py)); the
Declare suite is legitimacy infrastructure that feeds them.

The families' claimed properties are **computed, not asserted**:
`characterize.guard_read_once` counts atom occurrences in each MONA guard, and
`characterize.exact_marginal` / `exact_marginal_trace` brute-force the true
probabilistic acceptance. Verified in
[tests/test_richer_formulas.py](../tests/test_richer_formulas.py).

---

## Finding 1 — soft acceptance over-counts on non-read-once guards, and the gap grows

`soft_matrix` computes guard-satisfaction probability assuming **atom
independence**. That is exact iff each atom is read once. On a non-read-once guard
the shared atoms are double-counted, so the marginal acceptance probability is
**over-estimated**. `NON_READ_ONCE_SUITE` (at-least-k-of-n threshold functions,
atom multiplicity growing 2→3→4→6) turns this into a curve. Sweeping a shared
per-atom probability `p` and taking the max over-count `soft_raw − exact_marginal`:

| formula | guard multiplicity | max raw over-count | max \|normalized over-count\| |
|---|---|---|---|
| `majority3` (2-of-3) | 2 | **+0.090** | 0.005 |
| `atleast2of4` | 3 | **+0.135** | 0.135 |
| `atleast2of5` | 4 | **+0.162** | 0.162 |
| `atleast3of5` | 6 | **+0.245** | 0.035 |
| `alt_response` (Declare) | 2 | ≈ 0.000 | 0.012 |

**Reading it honestly:**

- The raw over-count is **monotone in atom multiplicity** — the more a guard
  re-reads its atoms, the more the independence assumption over-counts. This is a
  *result*, not an identity: on the read-once IJCNN/response references the gap is
  exactly zero (the soft path is provably exact there).
- The raw soft score is **not a valid probability** on these guards (it exceeds 1
  before the max-over-p is taken). `normalize=True` divides by the propagated
  mass; it restores [0,1] but **does not restore calibration** — it helps unevenly
  (dramatic for `majority3`/`atleast3of5`, no help for `atleast2of4/5`, where raw
  == normalized). This is why Exp 7 keeps both readouts.
- `alt_response` — a *real* Declare template (`G(a → X(¬a U b))`) that MONA also
  keeps non-read-once — barely diverges (≈0) over a constant-p trace: its shared
  atoms sit across an alternation/until, so the independence errors nearly cancel.
  An honest structure-dependence: non-read-once is *necessary but not sufficient*
  for large divergence; the pure-conjunction threshold family is the strong
  instrument, `alt_response` the realistic anchor showing the phenomenon is real
  but structure-sensitive.

Figures: `results/exp7_divergence_vs_p.png` (over-count vs p, per formula),
`results/exp7_divergence_vs_size.png` (headline: divergence grows with
multiplicity, threshold family as the curve + `alt_response` as a standalone
anchor).

**Why it matters.** This is the empirical grounding for the Phase 3.1 theory
question — *what is the correct probabilistic verdict on a soft trace?* The
marginal that `soft_matrix` approximates is only exact for read-once guards; the
over-count is the concrete symptom that "propagate independence-product mass" is
not the same quantity as the true marginal once guards share atoms.

---

## Finding 2 — state blowup is a *shared* weakness (symbolic and DeepDFA)

`STATE_BLOWUP_SUITE` = `F(a & Xᵏb)` has minimal DFA size **|Q| = 2ᵏ + 1** with a
**tiny alphabet** (|AP| = 2). Unlike the alphabet blowup (which only hits
DeepDFA), this hits *both* paradigms — good for the neutrality mandate — but
**differently**:

- **Per-cell compute** (measured, `results/exp7_stateblowup_time.png`, GPU run):
  Symbolic stays **flat** (~0.3 µs/cell) across |Q| 5→1025 — a DFA walk only
  touches the current state's out-edges. DeepDFA's step is O(|Q|²), so its
  per-cell cost is flat at small |Q| then **rises sharply** past |Q|≈256, reaching
  ~40 µs/cell at |Q|=1025 (dense and factored alike). So *at runtime* the state
  blowup is DeepDFA's problem, not symbolic's.

- **Representation size** (analytic, `results/exp7_stateblowup_memory.png`):
  DeepDFA-dense (`|Q|²·2^|AP|`) and factored (`|Q|²` masks) cross **4 GB at
  |Q|≈2¹³·⁵ (k≈14)** — a |Q|² wall driven purely by state count, despite the tiny
  alphabet. The symbolic transition table is linear in |Q|, so it walls out much
  later (k≈28). But symbolic *must still build and store 2ᵏ states*, so its wall is
  real, just further out.

**The honest three-way heel table**, now complete:

| paradigm | Achilles heel | which family exposes it |
|---|---|---|
| Symbolic | state blowup \|Q\| (storage/compile), crisp-only, frozen | `STATE_BLOWUP_SUITE` (later wall) |
| RuleRunner | nested-temporal representational limit; within-step depth cost | (original) — corrected by progression at an alphabet cost |
| DeepDFA | alphabet blowup 2^\|AP\| **and** state blowup \|Q\|² | `IJCNN` (alphabet) + `STATE_BLOWUP_SUITE` (state, earlier wall) |

---

## What feeds what

- `DECLARE_SUITE` — realistic BPM constraints with diverse trap/sink structure;
  legitimacy + the `alt_response` non-read-once anchor. Free to add to any
  timing/capability experiment via its `MONITORS`/suite membership.
- `NON_READ_ONCE_SUITE` — Finding 1; also the substrate for the Phase 3.1 theory.
- `STATE_BLOWUP_SUITE` — Finding 2; distinct from `STATE_SCALING_SUITE`
  (`bounded_response`, *linear* in k — a deadline knob, not a blowup).
