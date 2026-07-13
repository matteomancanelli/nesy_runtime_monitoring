# Phase 1 reframe: stop scoring NeSy on boards it can't win; measure the affordances symbolic lacks

**Status:** planning note (2026-07-01). Motivation + spec for reworking Capability Exp A.
Feed this to Claude Code as the brief for the rework. Supersedes the accuracy-vs-ε
framing of the current `experiments/exp_uncertainty.py`.

---

## 1. Why the shift (the motivation)

Two premises seeded this project: (i) NeSy monitors run on GPUs, so they should be
*faster* than symbolic; (ii) their probabilistic nature should make them *more accurate
under noise* than a brittle symbolic monitor. A code + paper audit shows **neither
premise is supported by the source papers, and both are contradicted by our own data —
without any implementation bug.** Symbolic genuinely wins on speed and ties-or-wins on
hard-verdict accuracy. That is a *result*, not a threat, but it means we were measuring
the wrong things.

**Speed was never a NeSy claim.** IJCNN 2014 (RuleRunner) reports its GPU variant is
"stably slower" in absolute wall-clock, with "remarkable overhead in the CPU–GPU
communication"; its *sparse CPU* form is the fastest. The GPU's only advantage is a
**flatter scaling slope** (32-leaf time is ~1.5× the 2-leaf time on GPU vs ~16× on
CPU-base), never a lower absolute time. It never compared against a symbolic DFA at all.
Our results reproduce this. The reasons are structural, not fixable:

- The compiled automata are tiny (`ijcnn_n8` → ~2 DFA states); a per-step 2×2 matmul is
  pure kernel-launch overhead with no arithmetic to amortize it.
- Monitoring is an inherently **sequential scan** over the trace; only the batch axis is
  parallel, and at |Q|=2 that can't beat a C-level dict lookup.
- Symbolic per-step is **O(1) regardless of |Q|**; DeepDFA is O(|Q|²). Bigger automata
  help symbolic *more* per cell.

**DeepDFA's noise-resilience claim is about *learning*, not inference.** Its robustness
results concern learning a DFA from *corrupted training data* (label noise; noisy symbols
during grammar induction) versus combinatorial DFA-induction baselines (L*, DFA-inductor);
"fast" means faster *training convergence* than RNNs. It makes **no claim** that, given an
already-correct DFA, propagating soft symbols yields higher inference-time verdict
accuracy than threshold-then-symbolic. We compile the correct DFA from LTLf, so we skipped
the regime where DeepDFA shines and tested a regime where it can't pull ahead.

**Hard-verdict accuracy is a structural tie, and we can prove why.** For a *deterministic*
spec under *independent, symmetric* per-atom noise, per-atom MAP (threshold at 0.5) + the
exact DFA is already near-Bayes. Thresholding the marginal acceptance probability at 0.5 is
a *different* decision rule that is generally no better. A controlled simulation (acceptance
= `a∧b`, our Beta model, concentration 10) confirms it:

| ε | symbolic acc | soft acc (raw sample) | soft acc (calibrated posterior) |
|---|---|---|---|
| 0.2 | 0.998 | 0.996 | 0.998 |
| 0.4 | 0.981 | 0.957 | 0.981 |
| 0.6 | 0.915 | 0.862 | 0.916 |

Feeding **raw Beta samples** as if they were probabilities (what the current experiment
does) makes soft *lose*; feeding the **calibrated posterior** closes the gap to an exact
tie. Either way, soft never dominates on the 0/1 verdict. So the calibrated-posterior fix
is not worth building for its own sake — it only converts a loss into a tie.

**Conclusion.** Symbolic is the theoretical optimum for crisp monitoring of a known,
small DFA. The honest contribution is not "a NeSy monitor wins," it is **characterizing
the affordances of each paradigm and demonstrating the capabilities symbolic
*fundamentally cannot provide* — on metrics that actually reward them.**

---

## 2. What to drop, keep, and add in Capability Exp A

**Drop (from the headline):**
- Accuracy-vs-ε as the lead result. It is a near-tie (calibrated) or a mild loss (raw
  samples), and it is *expected* — move to an appendix sentence at most, framed as "as
  theory predicts, soft neither helps nor hurts the hard verdict under symmetric noise."
- The **bitflip** arm as evidence of anything. Bitflip emits crisp 0/1, so the soft
  monitor sees *identical input* to symbolic ⇒ provably identical verdicts. Keep only as
  an explicitly-labelled "destroyed-information control," or cut it.
- The "soft is competitive, not dominant" phrasing. It concedes the wrong battle.

**Keep (promote to findings):**
- The soft acceptance score as a **native confidence** the symbolic monitor cannot emit.
- The **non-read-once defect** (raw `soft_matrix` rows sum >1 on `majority3`; normalization
  restores range but not calibration). It is a genuine, reproducible finding and the
  empirical hook into the Phase 3.1 theory question (what *is* the correct probabilistic
  verdict). Keep the reliability + ECE + defect panels.

**Add (the actual wins — see §3):** ranking metrics and a selective-prediction / abstention
analysis, plus the framing that the confidence enables an **uncertainty-aware verdict**
(defer/escalate) — which is exactly what safety monitoring wants and connects to
three-/four-valued LTL runtime semantics.

---

## 3. Actual wins (measurable now, with existing code)

These are capabilities symbolic *structurally lacks*, evaluated on metrics that reward
them. All reuse `DeepDFAMonitorFactored.batch_acceptance_probability` and the existing
`src/benchmarks/calibration.py` (which already has `roc_auc`, `brier_score`,
`reliability_curve`, `expected_calibration_error`).

### 3a. Confidence / ranking
The soft monitor emits a scalar in [0,1]; symbolic emits a bit. So the soft monitor can
**rank** traces by how likely they are to satisfy the spec — symbolic cannot produce a
ranking, an AUC, or a Brier score at all.

- **Already in our data:** under Beta noise the soft score's ROC-AUC is ~0.96–1.0 across
  the suite (`raw_auc` column of `results/exp_uncertainty.csv`; e.g. response ε=0.5 →
  0.974, ijcnn_n4 ε=0.5 → 0.964), degrading gracefully with ε. Under bitflip AUC collapses
  to ~0.5 (no information survives) — that contrast is itself the point.
- **Deliverable:** an AUC-vs-ε (and Brier-vs-ε) panel, Beta noise only, with the explicit
  note that the symbolic row is empty *by construction*.

### 3b. Selective prediction / abstention (the cleanest win)
Because the soft monitor has a confidence, it can **abstain** on low-confidence traces and
defer to review — trading coverage for accuracy. Symbolic has no confidence to threshold,
so it must commit on every trace.

- **Illustrative (toy simulation, `a∧b`, ε=0.5) — must be reproduced on the real monitor:**
  accuracy at 100% coverage 0.956 → 70% coverage 0.998 → 50% coverage 1.000, abstaining on
  the least-confident traces.
- **Confidence measure:** `conf = |score − 0.5|` (equivalently `max(p, 1−p)`), rank traces
  by it, sweep a coverage threshold. Ranking is invariant to monotone transforms, so raw
  vs normalized scores give the *same* risk-coverage curve on read-once formulas — note
  this so the non-read-once >1 issue is a non-issue for this analysis (still clip/normalize
  for display).
- **Deliverable:** a **risk-coverage curve** (accuracy-on-kept vs coverage) for the soft
  monitor, with symbolic as a single point at coverage = 1.0.
- **Fairness guardrail (important for reviewers):** a critic will say "symbolic could
  abstain too." It can't *natively* — but to be bulletproof, add a symbolic abstention
  baseline using a crude margin (e.g. the min over cells of `|obs − 0.5|` on the soft trace
  before thresholding) and show the soft monitor's native confidence still yields a better
  risk-coverage curve. State plainly: the soft monitor's confidence is *native*;
  manufacturing one for symbolic requires extra machinery (per-cell margin, or
  ensembling/sampling), and even then it's weaker.
- **Framing:** this is not a generic-classifier trick — it is an **uncertainty-aware
  runtime verdict** (SATISFY / VIOLATE / *DEFER*), the natural NeSy extension of
  three-valued LTL monitoring for safety contexts.

---

## 4. Potential win (needs Phase 2 — the strongest untapped card)

**Specification adaptation.** DeepDFA's *actual* thesis is differentiability: the monitor
can be **corrected from data** by gradient descent; symbolic is frozen. This is the one
place a NeSy monitor does something symbolic fundamentally cannot, and it is the bridge to
Paper B.

- **Synthetic PoC (Phase 2.1):** start from a *wrong* spec (wrong target atom, or an
  over-strict threshold), make DeepDFA's soft transition (or acceptance) matrix learnable,
  train on labels generated from the *correct* formula, show accuracy recovers. Symbolic is
  the control that cannot move. Reuse the `soft_matrix` path (already differentiable in `p`
  and in the edge parameters if we expose them).
- Keep it a toy for Paper A; the real-data (BPIC) version is Paper B.
- This is where the headline should point: **capability, not competitiveness.**

---

## 5. What stays as competitiveness evidence (not a win)

Phase 0 timing stays in the paper, framed honestly per IJCNN 2014: symbolic is the crisp
optimum; the differentiable monitor is *competitive within an order of magnitude* and has a
*flatter scaling slope*, not a lower wall-clock. Pair it with the three-Achilles-heel
matrix (symbolic = state blowup; RuleRunner = nested-temporal limit + within-step
sequential cost; DeepDFA = 2^|AP| alphabet blowup). Do not oversell; the honest three-way
balance is the point.

---

## 6. Concrete task list for Claude Code

1. **Add metrics** to `src/benchmarks/calibration.py`:
   - `selective_accuracy(scores, labels, coverage)` and
     `risk_coverage_curve(scores, labels, grid)` — rank by `|score − 0.5|`, return
     (coverage, accuracy-on-kept). Pure numpy; unit-test against a hand-computed example.
2. **Rework `experiments/exp_uncertainty.py`:**
   - Demote/remove the accuracy-vs-ε figure (appendix, one line).
   - Cut or clearly label the bitflip arm as a control.
   - Keep the reliability + ECE + non-read-once-defect figure (Beta only).
   - **Add Figure: AUC-vs-ε and Brier-vs-ε** (Beta), symbolic explicitly absent.
   - **Add Figure: risk-coverage curve** for the soft monitor at a mid ε (e.g. 0.4–0.5),
     with symbolic as the coverage=1 point and the crude-margin symbolic baseline for
     fairness. Reproduce the abstention numbers on the *real* DeepDFA soft scores over
     `CALIBRATION_SUITE` (do **not** hardcode the toy-sim numbers).
   - Verify AUC > 0.5 and that retention accuracy actually rises before claiming the win;
     if a formula's confidence isn't informative, say so.
3. **Do NOT build** the calibrated-posterior accuracy path — it only turns a loss into a
   tie and isn't needed for the capability framing (documented here so it isn't
   re-litigated).
4. **Paper wiring:** intro thesis = "we characterize the affordances of each paradigm; the
   case for NeSy is capability (native calibrated confidence → uncertainty-aware verdict
   and selective abstention; differentiability → adaptation), not speed and not hard-verdict
   accuracy." Section 5: lead capability results with 3a/3b, then Phase 2 PoC.

---

## 7. One-paragraph summary (for the paper's framing)

Symbolic DFA monitoring is the theoretical optimum for crisp Boolean monitoring of a known,
small automaton — fastest and, under symmetric perceptual noise, at least as accurate on the
binary verdict as any differentiable monitor (a fact we derive and confirm, consistent with
IJCNN 2014's own finding that its GPU variant is stably slower, and with DeepDFA's
noise-resilience being a *learning* property). The case for neuro-symbolic monitoring is
therefore not speed and not verdict accuracy, but the affordances symbolic *cannot* provide:
a **native calibrated confidence** that enables ranking and an **uncertainty-aware,
abstaining verdict** for safety-critical review, and **differentiability** that lets the
monitor be adapted from data when the specification is wrong. We measure these directly.
