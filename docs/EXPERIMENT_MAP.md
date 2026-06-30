# Experiment Map — navigating the comparison space

A navigation chart for Paper A's experiments: what exists, what each measures,
what we expect, what to do if results disappoint, and the full orthogonal space
of comparisons we *could* run (marked by cost: free / small / new code). Read
this before adding any experiment — most "new ideas" are already a free point in
this space, or are deliberately out of scope.

Grounded in the code as of 2026-06-30. Keep it in sync when experiments change.

---

## 0. The spine: the narrative everything hangs off

**The thesis is capability, not speed.** Symbolic DFA *will* win raw crisp
throughput — it is the theoretical optimum (a DFA walk is a dict lookup),
expected, not a threat. Therefore **a disappointing timing result is low-risk to
the narrative**: it demotes to "competitiveness" evidence and the capability
story (soft/probabilistic observations; gradient-based spec adaptation) carries
the paper. We are *not* trying to make a NeSy paradigm win on speed.

The honest three-way balance — each paradigm's distinct Achilles heel — is the
real contribution:

| Paradigm | Strength | Achilles heel |
|---|---|---|
| Symbolic | fastest crisp; exact | state blowup `\|Q\|`; crisp-only, frozen |
| RuleRunner | learning localized to syntactic neighbors | **within-step** depth-linear cost; nested-temporal representational limit |
| DeepDFA | native GPU batching; differentiable; soft inputs | **alphabet blowup** `2^{\|AP\|}` |

---

## 1. The experiments today

All four are crisp-input timing experiments. `MONITORS` lists are at the top of
each script; the device auto-selects `cuda` when available; `EARLY_TERMINATION`
is OFF (Phase 0.1) for the per-cell-cost figures.

Every experiment now runs the **full monitor set**: Symbolic, RuleRunner-CILP,
RuleRunner-Structured, DeepDFA-dense, DeepDFA-factored (exp2 caps dense at
n≤16). The `modes` column below names the *primary* curve(s) each is designed
around; the extra variants are within-paradigm reference lines.

| Exp | Axis (x) | Formula | Primary modes | Expect | If it disappoints |
|---|---|---|---|---|---|
| **1** ([exp1](../experiments/exp1_single_trace.py)) | trace length 1k–10k | `G(a→Fb)` (no trap/sink) | DeepDFA-**dense** | all **flat** (constant per-cell) | RR not flat → per-call overhead; not a narrative risk (per-cell-constant is the claim) |
| **2** ([exp2](../experiments/exp2_formula_complexity.py)) | formula breadth n=2..32 | DeepDFA-**factored** (all n) + **dense** (n≤16) + analytic memory-wall panel | Sym flat; RR ~linear in depth; **dense walls out at 2ⁿ**, factored flat | factored not flat → residual O(n²) mask reduction (genuine — annotate); dense doesn't wall → raise `DENSE_MAX_LEAVES` |
| **3** ([exp3](../experiments/exp3_batch_size.py)) | batch size 1–1024 | DeepDFA-**dense** (batching showcase); Structured-RR is the no-batching contrast | lead = absolute time/trace; **does batched DeepDFA win? — honest open question** | DeepDFA doesn't win even on Colab → "GPU advantage needs larger automata/HW"; demote speedup panel |
| **5** ([exp5](../experiments/exp5_depth_microbench.py)) | nested-X depth 0–10 | DeepDFA-**dense** | Sym/DeepDFA flat; **RR (both) linear in depth** | RR flat → raise `TRACE_LENGTH` (overhead burying signal); DeepDFA grows → nested-X inflates `\|Q\|`, note it |

**Two kinds of parallelism — keep framed separately:**
- **Within-step** (rules / matmul atoms fire within one cell): Exp 2 (breadth) + Exp 5 (depth).
- **Cross-trace** (traces batched as matrix rows): Exp 3 (batch size).

---

## 2. Within-paradigm variants (which is used, why, why not both)

### DeepDFA — three modes, one tensor file ([deep_dfa.py](../src/monitors/deep_dfa.py))

Reusable subclasses `DeepDFAMonitorDense` / `DeepDFAMonitorFactored` live in
[deep_dfa.py](../src/monitors/deep_dfa.py) and are now in **every** experiment's
`MONITORS` (the default `DeepDFAMonitor` is dense).

| mode | representation | per-cell cost | used by | why |
|---|---|---|---|---|
| **dense** (default) | `T(\|Q\|, 2^{\|AP\|}, \|Q\|)` one-hot | one `bmm` | exp1, exp2 (n≤16), exp3, exp5 | fastest where alphabet fits — the batching showcase |
| **factored** | vectorized cube-mask reduction (no `2^{\|AP\|}` tensor) | ~2–5× dense | **all** (exp1/2/3/5) | scales past where dense fits (n up to 32); elsewhere a constant-overhead reference line |
| **soft** (`soft_matrix`) | recursive read-once guard-prob closures (differentiable) | n/a (not timed) | **no current experiment** | reserved for capability: Phase 1 (uncertainty) + Phase 2 (adaptation) |

- **exp2 is where dense vs factored is the headline** — the dual finding (dense
  fast-but-walls-out vs factored flat-but-scales). The alphabet-blowup story.
- **Both now run everywhere** (factored added as a reference line in exp1/3/5).
  Where the alphabet is small it just trails dense slightly — a free robustness
  check, no new information expected.
- **All four timing experiments are crisp.** The soft path is exercised only by
  the (not-yet-built) capability experiments.

### RuleRunner — two variants ([rulerunner/](../src/monitors/rulerunner/))

- `RuleRunnerMonitor` (CILP, batched CPU/CUDA) — used in **all** experiments.
- `StructuredRuleRunnerMonitor` ([structured.py](../src/monitors/rulerunner/structured.py),
  the Fig-5 variant) — now **also in all experiments** (exported from the
  package; its `compile` accepts and ignores `device`). It is **CPU/sequential**:
  cheap per cell (~3 µs) but no cross-trace batching, so in exp3 its
  time-per-trace stays *flat* in batch size — the deliberate "no batching"
  contrast. This gives the base-vs-structured comparison mirroring IJCNN 2014's
  base/sparse/gpu family (we have base + structured, not sparse).

---

## 3. The full comparison space (the "n-D map")

Every experiment is a point in this space. The axes are orthogonal; most are
free to sweep without new code.

- **A. Paradigm** — Symbolic / RuleRunner / DeepDFA.
- **B. Within-paradigm variant** — DeepDFA {dense, factored, soft}; RuleRunner {CILP, structured}.
- **C. Workload stressor** — trace length · formula breadth (`2^{\|AP\|}`) · parse-tree depth · batch size · state count `\|Q\|`.
- **D. Measurement mode** — `early_termination` on/off (flag exists) · crisp vs soft input (soft needs new harness).
- **E. Hardware** — cpu / local-cuda / Colab-cuda / Docker-server (same code).
- **F. Metric derivation** — per-cell vs per-trace vs speedup (post-processing only).

### What counts as "no new code"

- Adding an existing monitor subclass to a `MONITORS` list — **free**.
- Toggling `EARLY_TERMINATION` / `DEVICE` — **free**.
- Changing axis ranges (lengths, batch sizes, depths, leaf counts) — **free** (config).
- Running the same script on different hardware — **free**.
- Soft-input timing / accuracy / calibration — **new code** (Phase 1 harness).
- A state-blowup formula family — **small new code** (new formulas only).
- Stamping the GPU name into results for hardware attribution — **~3 lines**.

### Comparison matrix — addressed vs free vs costly

| Comparison | Status |
|---|---|
| 3 paradigms × {length, breadth, depth, batch} | ✅ addressed (exp1/2/3/5) |
| DeepDFA dense vs factored | ✅ addressed (exp2) |
| dense memory-wall (analytic) | ✅ addressed (exp2 panel) |
| within-step vs cross-trace parallelism | ✅ separated (exp2/5 vs exp3) |
| factored as reference line in exp1/3/5 | ✅ addressed (now in all `MONITORS`) |
| RuleRunner CILP vs structured | ✅ addressed (now in all `MONITORS`) |
| GPU-name stamping for hardware attribution | ✅ addressed (`gpu_name` column) |
| early-termination ON vs OFF (data-dependent) | 🟢 FREE (toggle `EARLY_TERMINATION`) |
| local vs Colab vs server hardware | 🟢 FREE (run same script; stamped by `gpu_name`) |
| cpu vs cuda for same monitor | 🟢 FREE (toggle `DEVICE`) |
| state-blowup family (Sym + DeepDFA shared weakness) | 🟠 small new code (new formulas) |
| soft-input accuracy / calibration (Capability A) | 🔴 new code (Phase 1) |
| adaptation (Capability B) | 🔴 new code (Phase 2) |

---

## 4. Hardware comparison (Colab vs local) — what's possible

The CSV now records both a `device` column and a **`gpu_name`** column
(`torch.cuda.get_device_name()`, empty on CPU), so rows from different machines
(laptop vs Colab vs server) are self-identifying and directly overlay-able. What
it explains:

- Local (4 GB / 70 W) **saturates at tiny batch** → undersells every GPU path.
  Colab (bigger GPU) shows whether batched DeepDFA has real headroom. The
  *difference itself is the finding*: the GPU advantage is hardware-dependent.
- `gpu_name` makes a mixed CSV legible after the fact, but the result **key**
  does *not* include it.
- ⚠ Therefore still keep distinct filenames per machine (`exp3_local.csv` vs
  `exp3_colab.csv`). If a laptop-generated CSV is present when you run on Colab,
  same-mode `reset_if_stale` keeps it and `load_completed` *resumes* it —
  silently skipping the keys the laptop already did, so you'd never get the Colab
  rows for them. Separate files avoid this.

---

## 5. How to navigate (you-are-here)

- **The four timing experiments fully cover the crisp-speed space.** Adding
  timing axes will not move the paper — they are competitiveness evidence, and
  the matrix above shows the remaining timing cells are either free reference
  lines or low-value.
- **The paper's weight is in the 🔴 capability cells (Phase 1/2), not yet built.**
  That is where to point energy after Phase 0's re-runs land.
- **All ✅/green-handled items are done:** every experiment runs the full
  5-monitor set (CILP + Structured RuleRunner, dense + factored DeepDFA), and
  results are GPU-stamped. The only remaining free axes are pure toggles
  (`EARLY_TERMINATION`, `DEVICE`) and per-machine re-runs.

---

## Cross-references

- `CLAUDE.md` § Research Plan (phases/status), § Benchmark Design (formula families, parallelism axes).
- [docs/nested_temporal_limitation.md](nested_temporal_limitation.md) — RuleRunner's representational limit.
- [docs/DOCKER.md](DOCKER.md) — GPU server / Docker host for the heavy re-runs.
