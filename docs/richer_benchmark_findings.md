# Richer benchmark family — findings

The IJCNN `◇(⋁(a₀∧aᵢ))` family is a poor instrument on its own: it
early-terminates on random traces and its state space is tiny. The richer
families in [src/benchmarks/formulas.py](../src/benchmarks/formulas.py) each
stress a different axis; their claimed properties are **computed, not
asserted** (verified in
[tests/test_richer_formulas.py](../tests/test_richer_formulas.py)).

> **Scope note (2026-07-13).** The former Finding 1 — the soft acceptance
> over-count on non-read-once guards — belongs to the probabilistic-monitoring
> thread and moved with it to
> `artur_future_work/docs/richer_benchmark_findings.md` (full version). The
> `NON_READ_ONCE_SUITE` definitions stay here as ordinary benchmark families.

---

## Finding — state blowup is a *shared* weakness (symbolic and DeepDFA)

`STATE_BLOWUP_SUITE` = `F(a & Xᵏb)` has minimal DFA size **|Q| = 2ᵏ + 1** with a
**tiny alphabet** (|AP| = 2). Unlike the alphabet blowup (which only hits
DeepDFA), this hits *both* paradigms — good for the neutrality mandate — but
**differently**:

- **Per-cell compute** (measured, `results/figures/exp7_stateblowup_time.png`,
  GPU run): Symbolic stays **flat** (~0.3 µs/cell) across |Q| 5→1025 — a DFA
  walk only touches the current state's out-edges. DeepDFA's step is O(|Q|²),
  so its per-cell cost is flat at small |Q| then **rises sharply** past
  |Q|≈256, reaching ~40 µs/cell at |Q|=1025 (dense and factored alike). So *at
  runtime* the state blowup is DeepDFA's problem, not symbolic's.

- **Representation size** (analytic,
  `results/figures/exp7_stateblowup_memory.png`): DeepDFA-dense (`|Q|²·2^|AP|`)
  and factored (`|Q|²` masks) cross **4 GB at |Q|≈2¹³·⁵ (k≈14)** — a |Q|² wall
  driven purely by state count, despite the tiny alphabet. The symbolic
  transition table is linear in |Q|, so it walls out much later (k≈28). But
  symbolic *must still build and store 2ᵏ states*, so its wall is real, just
  further out.

**The honest three-way heel table:**

| paradigm | Achilles heel | which family exposes it |
|---|---|---|
| Symbolic | state blowup \|Q\| (storage/compile), crisp-only, frozen | `STATE_BLOWUP_SUITE` (later wall) |
| RuleRunner | nested-temporal representational limit; within-step depth cost | (original) — corrected by progression at an alphabet cost |
| DeepDFA | alphabet blowup 2^\|AP\| **and** state blowup \|Q\|² | `IJCNN` (alphabet) + `STATE_BLOWUP_SUITE` (state, earlier wall) |

---

## The families

- `DECLARE_SUITE` — 7 realistic BPM constraints with diverse trap/sink
  structure; legitimacy anchors, free to add to any timing experiment.
- `NON_READ_ONCE_SUITE` — `at_least_k_of_n` threshold family + `alt_response`;
  guards with genuine atom re-reads (MONA keeps them un-factored — tested).
  Its probabilistic use lives in the future-work fork.
- `STATE_BLOWUP_SUITE` — `kth_from_last(k)` = `F(a & Xᵏb)`, the exponential
  instrument above. Distinct from `STATE_SCALING_SUITE` (`bounded_response`,
  only *linear* in k — a deadline knob, not a blowup; exp6).
