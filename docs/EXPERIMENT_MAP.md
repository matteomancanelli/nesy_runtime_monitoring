# Experiment Map — navigating the comparison space

A navigation chart for the paper's experiments: what exists, what each
measures, what we expect, what to do if results disappoint, and the axes we
*could* still sweep (marked by cost: free / small / new code). Read this before
adding any experiment — many "new ideas" are already a free point in this
space, or are deliberately out of scope.

Updated 2026-07-13 for the ICLR refocus. Keep it in sync when experiments
change. The uncertainty/calibration experiments and their map rows moved to
`artur_future_work/` with the probabilistic-monitoring thread.

---

## 0. The spine: the narrative everything hangs off

**This is a foundation paper: modernize + fix RuleRunner, connect it with the
automata-based paradigms, and characterize the efficiency landscape honestly.**
Symbolic DFA *will* win raw crisp throughput — it is the theoretical optimum (a
DFA walk is a dict lookup), expected, not a threat. The experiments therefore
do not chase a NeSy speed win; they map **where each paradigm's cost grows and
where it walls out**, so the paper ends with an honest, complete efficiency
landscape plus the correctness result (the progression repair) priced in.

The honest three-way balance — each paradigm's distinct Achilles heel — is the
core empirical story:

| Paradigm | Strength | Achilles heel |
|---|---|---|
| Symbolic | fastest crisp; exact | state blowup `\|Q\|` (storage/compile); frozen |
| RuleRunner (original) | learning-friendly syntactic locality | **within-step** depth-linear cost; nested-temporal representational limit (wrong verdicts) |
| RuleRunner (progression) | sound + complete on all LTLf | residual-closure/alphabet cost of the fix |
| DeepDFA | native GPU batching; differentiable | **alphabet blowup** `2^{\|AP\|}` + `\|Q\|²` per-step cost |

---

## 1. The experiments today

All are crisp-input timing experiments. `MONITORS` lists are at the top of each
script; the device auto-selects `cuda` when available; `EARLY_TERMINATION` is
OFF (Phase-0 fix) for the per-cell-cost figures.

Every experiment runs the **full monitor set**: Symbolic, RuleRunner-CILP,
RuleRunner-Structured, **Progression-RR-flat, Progression-RR-structured** (the
corrected paradigm 2), DeepDFA-dense, DeepDFA-factored (exp2 caps dense **and
both progression monitors** at n≤16 — they share the 2^|AP| alphabet wall; exp1
and exp3 also run DeepDFA-scan). The `modes` column names the *primary*
curve(s) each experiment is designed around; the extra variants are
within-paradigm reference lines.

| Exp | Axis (x) | Formula | Primary modes | Expect | If it disappoints |
|---|---|---|---|---|---|
| **1** ([exp1](../experiments/exp1_single_trace.py)) | trace length 1k–10k | `G(a→Fb)` (no trap/sink) | DeepDFA-**dense** | all **flat** (constant per-cell) | RR not flat → per-call overhead; not a narrative risk (per-cell-constant is the claim) |
| **2** ([exp2](../experiments/exp2_formula_complexity.py)) | formula breadth n=2..32 | IJCNN family | DeepDFA-**factored** (all n) + **dense** (n≤16) + analytic memory-wall panel | Sym flat; RR ~linear in depth; **dense walls out at 2ⁿ**, factored flat | factored not flat → residual O(n²) mask reduction (genuine — annotate); dense doesn't wall → raise `DENSE_MAX_LEAVES` |
| **3** ([exp3](../experiments/exp3_batch_size.py)) | batch size 1–1024 | `ijcnn_n8` | DeepDFA-**dense** (batching showcase); both RuleRunner variants batch cross-trace | lead = absolute time/trace; **does batched DeepDFA win? — honest open question** | DeepDFA doesn't win even on Colab → "GPU advantage needs larger automata/HW"; demote speedup panel |
| **5** ([exp5](../experiments/exp5_depth_microbench.py)) | nested-X depth 0–10 | `ijcnn_n8` wrapped | DeepDFA-**dense** | Sym/DeepDFA flat; **RR (both) linear in depth** | RR flat → raise `TRACE_LENGTH` (overhead burying signal); DeepDFA grows → nested-X inflates `\|Q\|`, note it |
| **6** ([exp6](../experiments/exp6_state_scaling.py)) | `\|Q\|` linear in deadline k | `bounded_response` | Symbolic vs DeepDFA | Sym flat in `\|Q\|`; DeepDFA O(`\|Q\|²`) → possible crossover at large `\|Q\|` | no crossover → "GPU advantage needs larger automata"; progression monitors capped at `PROGRESSION_MAX_Q` (closure explosion is an nf artifact — see CLAUDE.md) |
| **7** ([exp7](../experiments/exp7_state_blowup.py)) | `\|Q\| = 2ᵏ+1` exponential | `kth_from_last` | Symbolic vs DeepDFA dense/factored | Sym per-cell flat, DeepDFA O(`\|Q\|²`) rising; analytic memory wall: both wall, symbolic later | Sym not flat → check early-term off |

**Two kinds of parallelism — keep framed separately:**
- **Within-step** (rules / matmul atoms fire within one cell): Exp 2 (breadth) + Exp 5 (depth).
- **Cross-trace** (traces batched as matrix rows): Exp 3 (batch size).

**Two kinds of state growth — keep framed separately:**
- **Linear** (`bounded_response`, exp6): a controlled knob to look for a
  symbolic/DeepDFA crossover.
- **Exponential** (`kth_from_last`, exp7): the genuine blowup; the shared-wall
  finding ([docs/richer_benchmark_findings.md](richer_benchmark_findings.md)).

**Cost of correctness** (the paradigm-2 paper number):
`plots.correctness_cost_table` / `plot_correctness_cost` report the
corrected/original per-cell-time **ratio** on exp2's flat IJCNN family (where
the *original* RR is also correct, so the ratio isolates the encoding's
throughput cost, not the verdict fix). >1 = the fix is slower.

---

## 2. Within-paradigm variants (which is used, why)

### DeepDFA — three crisp modes ([deep_dfa.py](../src/monitors/deep_dfa.py))

| mode | representation | per-cell cost | used by | why |
|---|---|---|---|---|
| **dense** (default) | `T(\|Q\|, 2^{\|AP\|}, \|Q\|)` one-hot | one `bmm` | exp1, exp2 (n≤16), exp3, exp5, exp6, exp7 | fastest where alphabet fits — the batching showcase |
| **factored** | vectorized cube-mask reduction (no `2^{\|AP\|}` tensor) | ~2–5× dense | **all** | scales past where dense fits (n up to 32); elsewhere a constant-overhead reference line |
| **scan** (`DeepDFAMonitorScan`) | Hillis–Steele prefix product over per-cell matrices | O(log L) launches, ×`\|Q\|` FLOPs | exp1, exp3 | wins only where launch overhead ≫ arithmetic (GPU + small `\|Q\|` + long traces); loses on CPU / large `\|Q\|` — honest caveat stays |

The differentiable `soft_matrix` path still exists in the source (it is the
affordance the paper points to as future work) but is **not exercised by any
experiment in this repo** — its harness lives in `artur_future_work/`.

### RuleRunner — two encodings × two constructions

- `RuleRunnerMonitor` (flat CILP, batched CPU/CUDA) — original encoding, all experiments.
- `StructuredRuleRunnerMonitor` (IJCNN-2015 Fig. 5, per-node subnets, batched) — all experiments; contrasts two batched encodings, not batched-vs-unbatched. Within a cell it sweeps parse-tree nodes sequentially (no sibling fusion) — less GPU-friendly per cell than the flat `depth+1` whole-network passes.
- `ProgressionRuleRunnerMonitor` (flat, multi-hot residual roots) — the **corrected** paradigm 2, the honest throughput competitor. Flat per cell (depth/|Q| absorbed into residual states).
- `ProgressionRuleRunnerStructuredMonitor` (per-closure-node subnets) — the local-learning substrate; per-cell eval sweeps the closure.

⚠ **Progression's own wall:** the eager residual construction enumerates
`2^k` observations per residual, so both progression monitors share the
alphabet wall (capped at `DENSE_MAX_LEAVES` in exp2, `PROGRESSION_MAX_Q` in
exp6). The measured closure size is inflated by a non-canonical `nf` — an
implementation artifact, not a paradigm property (details in CLAUDE.md; do not
report the inflated number as the paradigm's cost).

---

## 3. Remaining free axes

- Adding an existing monitor subclass to a `MONITORS` list — **free**.
- Toggling `EARLY_TERMINATION` / `DEVICE` — **free** (an early-termination-ON
  run is a legitimate separate *data-dependent* experiment if ever wanted).
- Changing axis ranges (lengths, batch sizes, depths, leaf counts) — **free**.
- Running the same script on other hardware — **free** (rows stamp the
  *effective* `device`/`gpu_name`; keep one CSV per machine, see §4).
- Declare-suite timing rows — **free** (suite exists; add to `MONITORS`/suites).

## 4. Hardware comparison (Colab vs local)

Each timing row records `device` and `gpu_name` (`torch.cuda.get_device_name()`,
empty on CPU) — the **effective** device the monitor really ran on (Symbolic
and the original structured RR are pure-Python CPU walks and stamp `cpu` even
under `device="cuda"`; a CSV never claims a GPU run that did not happen).

- Local (4 GB / 70 W) **saturates at tiny batch** → undersells every GPU path.
  Colab (T4) shows whether batched DeepDFA has real headroom. The *difference
  itself is a finding*: the GPU advantage is hardware-dependent.
- ⚠ Keep distinct files per machine (`results/cpu/`, `results/gpu/`): resume
  (`load_completed`) does not key on device, so a mixed file silently skips
  cells. `experiments/make_all_plots.py` merges at plot time.

## 5. Where the paper's empirical weight goes next (the roadmap)

The four candidate directions, easiest → hardest (scope each in a dedicated
session before building — see CLAUDE.md phases):

1. **Efficiency landscape** — finish the Colab GPU/CPU re-runs of exp1/2/3/5/6/7
   under early-termination-off + CUDA-sync; polish walls/crossovers +
   cost-of-correctness into the paper's figure set.
2. **Real-log case study** — Declare constraints on a BPIC log, crisp, at scale
   (realism/legitimacy evidence).
3. **Neural sequence baselines** — RNN/Transformer monitors trained on traces
   vs the exact paradigms: verdict accuracy, length generalization, sample
   efficiency. The most ICLR-shaped addition.
4. **Decision-diagram transition representation** (crisp side) — see
   [decision_diagram_transition_representation.md](decision_diagram_transition_representation.md).

---

## Cross-references

- `CLAUDE.md` § Research Plan (phases/status), § Benchmark Design.
- [docs/nested_temporal_limitation.md](nested_temporal_limitation.md) — RuleRunner's representational limit.
- [docs/richer_benchmark_findings.md](richer_benchmark_findings.md) — the state-blowup shared-weakness finding.
- `artur_future_work/` — the probabilistic-monitoring and adaptation threads (own CLAUDE.md).
