# CLAUDE.md — NeSy LTLf Monitoring: Future Work

This file provides guidance to Claude Code (claude.ai/code) when working in this
folder. **Read it fully before touching anything** — it exists so that work can
resume cold, months after the split, without re-deriving what is already known.

## What this is

A **self-contained fork** of the `nesy_runtime_monitoring` project, created
**2026-07-13** at parent commit `57d74e3` ("Pre-restructure snapshot"). It is
ready to be extracted into its own repository (`cp -r` + `git init` — all source,
tests, experiments, results, latex material, and dependency files are here).

**Why the split.** The parent project was refocused into an ICLR submission
whose frame — decided with the supervisors — is a *foundation* paper:
modernizing and fixing RuleRunner (the nested-temporal limitation and its
progression-based repair) and connecting it with automata-based neuro-symbolic
approaches (DeepDFA), with a purely **crisp** empirical section. Two research
threads were explicitly **deferred to future work** and moved here so they are
neither lost nor half-presented:

1. **Probabilistic monitoring** — monitoring under perceptual uncertainty:
   soft/probabilistic observations, calibrated verdict confidence, the theory of
   what a verdict on a probabilistic trace even *is*.
2. **Specification adaptation** — gradient-based correction of a wrong
   specification from data (never implemented; fully planned below).

A third thread, the **decision-diagram (WMC) monitor**, straddles the two repos:
its *crisp/scalability* side is on the parent's roadmap; its *calibration/
exactness* headline (the stronger claim) belongs here. See
[docs/decision_diagram_transition_representation.md](docs/decision_diagram_transition_representation.md).

**Relationship to the parent.** The `src/` and `tests/` trees are full copies as
of the fork date and are expected to **diverge** — do not try to keep them in
sync. The parent's git history (and its `CLAUDE.md` at commit `57d74e3`) holds
the deep per-paradigm implementation review notes (RuleRunner step-by-step
design decisions, DeepDFA representation notes); consult it for archaeology, but
treat *this* folder as authoritative for future-work code.

## Repository structure

```
artur_future_work/
├── CLAUDE.md                  this file
├── README.md                  one-page orientation + setup
├── conftest.py                sys.path shim so `import src` resolves locally (see Commands)
├── pyproject.toml             project = nesy-monitoring-future
├── environment.yml            conda env = nesy-monitoring-future (torch+cu124, ltlf2dfa/MONA)
├── src/
│   ├── formula/compiler.py    LTLf → minimal DFA (ltlf2dfa/MONA wrapper; trap/sink precomputation)
│   ├── monitors/
│   │   ├── base.py            Monitor ABC + three-valued Verdict
│   │   ├── symbolic_dfa.py    Paradigm 1 — crisp DFA walk (the oracle + threshold baseline)
│   │   ├── rulerunner/        Paradigm 2 — original IJCNN 2014/2015 encodings (flat CILP + structured)
│   │   ├── progression/       Paradigm 2 FIXED — progression-based RuleRunner (sound+complete)
│   │   └── deep_dfa.py        Paradigm 3 — DeepDFA (dense + factored + scan); THE soft path lives here
│   └── benchmarks/
│       ├── formulas.py        formula registry incl. CALIBRATION_SUITE, NON_READ_ONCE_SUITE,
│       │                      DECLARE_SUITE, STATE_BLOWUP_SUITE (read_once flags)
│       ├── characterize.py    guard_read_once + exact_marginal(_trace) — brute-force ground truth
│       ├── noise.py           BitFlipNoise / BetaNoise corruption models + symbolic oracle
│       ├── calibration.py     ECE / reliability / Brier / AUC / risk–coverage metrics (numpy+scipy)
│       └── runner.py          timing harness (kept for completeness; not the harness used here)
├── experiments/
│   ├── exp_uncertainty.py     Capability Exp A — accuracy/calibration/sharpness/risk–coverage vs ε
│   ├── exp7_richer_family.py  soft-divergence curve (Panel 1) + state-blowup (Panel 2)
│   ├── plots.py               full plotting module (uncertainty + timing figures)
│   └── make_all_plots.py      CSV → PNG regeneration
├── tests/                     full suite as of the fork (~380 tests, 6 xfail-strict on the
│                              ORIGINAL RuleRunner encoding — expected, they document its limit)
├── docs/
│   ├── phase1_capability_reframe.md              why "affordances, not speed" (the pivotal audit)
│   ├── richer_benchmark_findings.md              Finding 1 (soft divergence) + Finding 2 (state blowup)
│   └── decision_diagram_transition_representation.md  the DD/WMC monitor design note (§8 = the paper-sized claim)
├── latex/
│   ├── 5b_probabilistic_verdict.tex   a full drafted section: the three-verdicts theory (see below)
│   ├── trimmed_from_paper.tex         prose trimmed from the parent paper's §4.2/§5.2 (verbatim)
│   └── citations.bib                  bibliography copy (keys referenced by the .tex files)
└── results/                   fork-date experiment outputs
    ├── cpu/, gpu/             exp_uncertainty*.csv, exp7_divergence.csv per device
    └── figures/               exp_uncertainty_*.png, exp7_divergence_*.png
```

Reference PDFs (IJCNN 2014/2015, DeepDFA ECAI 2024, NeSy PPM, CILP, TOSEM) are
**not** copied — they live in the parent repo's `papers/` folder (gitignored
there). Grab them from the parent before extracting this into a standalone repo
if you want them alongside.

## The capability thesis (the frame this work argues)

Symbolic DFA monitoring is the theoretical optimum for **crisp** verdicts on a
known spec — nothing differentiable beats a dict lookup, and per-atom
thresholding + exact DFA is near-Bayes for the hard verdict under unbiased
noise. The case for neuro-symbolic monitoring is therefore **capability, not
speed or raw accuracy**: quantities and behaviours the symbolic walk
*fundamentally cannot produce*. Concretely: a calibrated verdict confidence, a
distribution over three-valued verdicts at every time point (an alarm dial
trading latency against false alarms), selective prediction (abstain when
unsure), and gradient-based adaptation of the spec itself. The full argument —
including why the two premises "NeSy should be faster" and "NeSy should be more
accurate under noise" are unsupported by the source papers *and* contradicted by
our data — is [docs/phase1_capability_reframe.md](docs/phase1_capability_reframe.md).

## Established findings — do NOT re-derive these

All of the below is implemented, tested, and measured (CSVs/figures in
`results/`). Trust it; re-verify only if you change the underlying code.

### F1 — Soft monitoring is competitive, not dominant, on the hard verdict
For unbiased perceptual noise (BetaNoise), thresholding the per-atom mean is
per-cell near-Bayes-optimal and the DFA is then exact, so the symbolic baseline
is *strong* — it edges out the soft monitor on 0/1 accuracy at our settings.
Under BitFlipNoise (information destroyed before the monitor) soft and symbolic
are **identical** — the no-free-lunch control. The NeSy payoff is the
**calibrated confidence** and what it enables (risk–coverage / selective
prediction, `exp_uncertainty.py` 1.5), not higher accuracy.
Theory for *why*: thresholding is exactly MAP-trace reconstruction
(Prop. `threshold-is-map` in 5b), i.e. a principled estimator, not a lossy hack.

### F2 — The recursive soft score is NOT a probability on non-read-once guards
DeepDFA's `soft_matrix` evaluates guard probabilities with independence rules
(`P(φ∧ψ)=PφPψ`, `P(φ∨ψ)=1−(1−Pφ)(1−Pψ)`). Exact for crisp inputs on any guard,
and for fractional inputs on **read-once** guards only. On non-read-once guards
(shared atoms across disjuncts) the rows of `M(p)` sum to >1 and the propagated
acceptance score can exceed 1 (measured up to ~1.22 on `majority3`). The
over-count `soft_raw − exact_marginal` is **monotone in guard atom
multiplicity** (majority3 +0.09 → atleast3of5 +0.245; exactly 0 on read-once
references), and it can **flip the verdict** (`majority3` at p≈0.47: true
marginal 0.455 = VIOLATE, recursive score 0.527 = SATISFY).
`batch_acceptance_probability(..., normalize=True)` divides by propagated mass:
restores [0,1], **not calibration** (helps unevenly — dramatic for majority3,
nothing for atleast2of4/5). Both readouts are kept on purpose so the defect
stays observable. Direction of error = sign of disjunct correlation, which is
why the Declare template `alt_response` (also non-read-once) barely diverges:
non-read-once is **necessary, not sufficient**.
Data: `results/*/exp7_divergence.csv`, figures `exp7_divergence_vs_*.png`;
details [docs/richer_benchmark_findings.md](docs/richer_benchmark_findings.md) Finding 1.

### F3 — The three-verdicts theory (5b, drafted in full)
[latex/5b_probabilistic_verdict.tex](latex/5b_probabilistic_verdict.tex) is a
complete drafted section, with proofs, establishing:
- A soft trace induces a distribution over crisp traces (independent-bit model);
  three natural verdicts are pairwise distinct (worked counterexample table):
  **(V1)** marginal acceptance probability, **(V2)** MAP-trace verdict,
  **(V3)** most-likely-run (Viterbi) verdict.
- Each paradigm natively computes a different one: symbolic+threshold = exactly
  V2; DeepDFA's architecture (forward recursion) targets V1 and is exact iff the
  per-cell matrix is the true `M*` (the disjoint-cube cover gives `M*` for any
  guard; the recursive closure only for read-once); a tanh-RuleRunner computes a
  fuzzy truth degree that is none of the three.
- V1 is Bayes-optimal for the 0/1 verdict (with honest qualifications:
  model-misspecification robustness favours V2; asymmetric losses need the
  scalar, not the bit; V3 is the *diagnosis/witness*, not the verdict).
- The online three-valued refinement: `q_t · 1_trap` / `q_t · 1_sink` give an
  exact distribution over {SAT, VIOL, UNDECIDED} at every step — the alarm-dial
  capability. **Not yet evaluated experimentally** (see Continuation).
- The unifying lens is knowledge compilation: determinism + decomposability
  (d-DNNF) is exactly the condition for tractable exact propagation; the
  non-read-once anomaly is the standard approximation error, not a bug.
- **One open TODO in the file:** position against Stoller et al.'s RV with
  state estimation (RVSE, RV 2011) and statistical model checking — verify
  their exact claims before submitting anywhere.

### F4 — Sharpness and risk–coverage (capability made actionable)
`exp_uncertainty.py` also sweeps perceptor sharpness (Beta concentration at
fixed ε — the soft−symbolic gap should widen as the perceptor fuzzes; run on a
real GPU was still pending at fork) and traces **risk–coverage curves**
(abstain on least-confident verdicts; symbolic is a single no-abstention
point). The risk–coverage figure is the strongest single capability figure.

### F5 — Readout discipline
The soft run must **never argmax mid-trace** (that collapses the state
distribution and yields a greedy path that is neither V1 nor V3 — see 5b
Prop. "forward"): propagate the full distribution, read
`confidence = q_final @ accepting` at end-of-trace, threshold at 0.5.

## Continuation plan — the future papers

Ordered by readiness. Each is scoped to be picked up in dedicated sessions.

### Thread 1 — Probabilistic-verdict theory + calibrated monitoring (closest to a paper)
Most of the theory (5b) and the empirical harness already exist. Remaining:
1. Resolve the RVSE/Stoller positioning TODO in 5b (read the RV 2011 paper;
   our delta is the *paradigm-level map* — three paradigms natively compute
   three different functionals — not the identification of the marginal).
2. Evaluate the **online three-valued distribution** (Eq. `prob-three-valued`):
   detection-latency vs false-alarm curves indexed by the tolerance δ, symbolic
   as the single fixed point. This is the sharpest capability experiment and is
   *unbuilt* (the readout is two inner products on the existing soft run).
3. Re-run `exp_uncertainty.py` sharpness sweep on a real GPU (Colab, T4) —
   whether the accuracy gap widens at low concentration was still empirical.
4. Optional: a Viterbi (V3) implementation as the explanation/witness readout —
   max-product over the same trellis; trivially different recursion.

### Thread 2 — Decision-diagram (BDD/SDD) WMC monitor (the ambition-raiser)
See [docs/decision_diagram_transition_representation.md](docs/decision_diagram_transition_representation.md),
especially §8 ("could this be THE contribution?"). One-line claim: compile each
guard to a deterministic+decomposable diagram once, evaluate its WMC circuit
batched → `M(p)` is exactly `M*` for **any** guard, so the soft verdict is
exact and calibrated by construction, differentiable, and GPU-batchable.
First step before any code: the **NeSyA novelty check** (does NeSyA/T-ILR
already deliver the exact-WMC transition? if yes, re-scope the delta to
monitoring + three-valued verdicts + calibration analysis). The parent repo
plans the *crisp/scalability* side of this same idea — coordinate if both
proceed.

### Thread 3 — Specification adaptation PoC (the old Phase 2; unimplemented)
The headline NeSy payoff and the original bridge to "Paper B". Plan as it stood:
- **3a. Synthetic PoC:** start from a *wrong* spec (wrong atom / over-strict
  threshold); make DeepDFA's transition (or acceptance) parameters learnable
  (their original Gumbel-Softmax relaxation; the reference implementation's
  `loss/global_loss.py` GLL is the reuse target — see
  github.com/axelmezini/nesy-suffix-prediction-dfa); train on traces labeled by
  the correct formula; show accuracy recovers. Symbolic = the control that
  cannot adapt.
- **3b. RuleRunner adaptation:** swap CILP `sign`→`tanh` (recompute biases via
  Garcez & Zaverucha's `Amin`), adapt on misclassified traces. The structured/
  progression-structured encodings are the **local-learning** substrate:
  parameters attach to syntactic subnetworks, so adaptation localizes to the
  offending subtree and the learned object re-extracts as a corrected spec —
  vs DeepDFA's opaque automaton drift. (The trimmed §5.2 prose in
  [latex/trimmed_from_paper.tex](latex/trimmed_from_paper.tex) states this
  contrast; reuse it.)
- **3c. Real data (stretch):** BPIC 2012/2017 log + a known Declare constraint.
- Planned-but-never-created files: `src/adaptation/poc.py`,
  `experiments/exp4_adaptation.py`.

### Thread 4 — End-to-end perceptor training (the NeSy dream; furthest out)
A toy where a spec-violation loss backprops *through the monitor* into a
perception network. Depends on Thread 1's readout semantics (which verdict do
you differentiate?) and benefits from Thread 2's exact `M(p)`.

## Warnings and sharp edges

- **Formula choice matters enormously.** On read-once families (IJCNN, response)
  the soft path is *provably exact*, so any calibration "win" there is a hollow
  identity. Every soft-path claim must include non-read-once formulas
  (`majority3`, `at_least_k_of_n`, `alt_response`) — that is what
  `NON_READ_ONCE_SUITE` and `characterize.exact_marginal` exist for.
- **Trace length floor:** single-cell traces hide the non-read-once defect;
  `exp_uncertainty.py` balances length for ~0.5 positive rate with MIN_LEN=3.
- **The 6 xfail-strict tests are correct behaviour** — they pin the ORIGINAL
  RuleRunner encoding's nested-temporal divergence (`F(a&Xb)`, `G(a→Fb)`,
  `G(a→Xb)`). The progression monitors pass those same formulas. Don't "fix" them.
- **Never seed with `hash()`** (randomized per process); the sweeps use a stable
  MD5-based `_stable_seed`. Keep it that way.
- `ltlf2dfa` needs **MONA** on PATH (`apt-get install mona` on Colab/Debian).
  `lark`'s two `sre_parse` DeprecationWarnings are harmless.

## Environment setup & commands

```bash
# Fresh setup (standalone repo or new machine)
conda env create -f environment.yml     # env: nesy-monitoring-future
conda activate nesy-monitoring-future

# While this folder still lives INSIDE the parent repo, the parent's editable
# install also provides a package named `src`. conftest.py pins `import src`
# to THIS folder for pytest; for scripts, run from this folder so CWD wins:
cd artur_future_work
python -m pytest                        # full suite (~380 tests, 6 xfail)
python -m experiments.exp_uncertainty   # Capability Exp A (resumable CSVs)
python -m experiments.exp7_richer_family
python experiments/make_all_plots.py    # regenerate figures from CSVs

# Once extracted to its own repo: pip install -e ".[dev]" in a fresh env,
# after which the conftest shim is redundant (but harmless).
```

`experiments/plots.py` is a full copy of the parent's plotting module: only the
uncertainty/divergence plotters are expected to be exercised here; the timing
plotters remain for reference and will fail gracefully on missing CSVs.
