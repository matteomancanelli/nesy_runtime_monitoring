# Provenance — what came from where, and which paper depends on it

**Created:** 2026-09-13, from `nesy_runtime_monitoring` at parent commit `26dbaa9`
("sun 13"), on branch `claude/paper-split-review-xmf6os`.

**Why this folder exists:** the draft carried two research questions. Paper A asks
*what a rule-based neural monitor can compute and what completeness costs it*. Paper B
asks *how a compiled transition function should be represented and executed*. They shared
a benchmark harness and a LaTeX file, not a question. See `README.md` for Paper B's scope
and the parent `CLAUDE.md` for Paper A's.

**The split line is semantics vs representation, not RuleRunner vs DeepDFA.** Paper A keeps
DeepDFA as a first-class monitor: it is the exact automata-based reference against which
RuleRunner's defect and repair are measured, and Proposition *Crisp equivalence* is
load-bearing for Paper A's punchline. What moved here is everything about how the
transition function is *stored and executed*.

Useful test for any paragraph or file: **does removing it change what the monitor
computes, or only how fast it computes it?** Semantics stays in A; speed moves here.

---

## LaTeX

| File here | Origin | Disposition | Which paper's claims depend on it |
|---|---|---|---|
| `latex/deepdfa_semantics.tex` | `latex/5_deepdfa.tex` lines 1–85 | **duplicated** | **Both.** A needs it for the interchangeability result; B needs it as the correctness statement every representation must preserve. Keep in sync or have B cite A. |
| `latex/deepdfa_representation.tex` | `latex/5_deepdfa.tex` lines 87–142 | **moved** | **B only.** Alphabet blow-up, disjoint-cube construction, recursive approximation, complexity table `tab:deepdfa-complexity`, implementation-level complexity, artifact interface, dense-vs-factored closing paragraph. A retains a one-paragraph statement of the alphabet axis (`sec:deepdfa-alphabet`) that forward-references B. |
| `latex/representation_duality.tex` | `latex/6_theory_cmp.tex`, the dual-axis compression passage | **moved (A keeps a compressed version)** | **B primarily.** A still needs two sentences of it, because the alphabet axis is one of A's three capability heels. |
| `latex/inherited_stubs.tex` | new | — | Placeholder anchors (`sec:ltlf`, `sec:symbolic`, `sec:conclusion`) so this skeleton compiles standalone. |
| `latex/main.tex` | new | — | Compilable skeleton. Not a submission draft. |
| `latex/preamble.tex`, `latex/citations.bib` | copies | **duplicated** | Both. `preamble.tex` gained `\paperA{}`/`\paperB{}` placeholder macros in the parent as well. |

**Not moved, deliberately:** `4_rulerunner.tex`, `proofs.tex`, `2_background.tex`,
`3_related_work.tex`, `6_theory_cmp.tex` (the heels and the interchangeability result),
`1_introduction.tex`. These are Paper A.

## Docs

| File here | Origin | Disposition | Notes |
|---|---|---|---|
| `docs/decision_diagram_transition_representation.md` | `docs/` | **moved** | Pure representation design note (BDD/SDD/d-DNNF/ShMTBDD). It is also the natural bridge to the KR 2026 program-graph representation. ⚠ `artur_future_work/` holds a third copy — Matteo's call whether to prune it. |
| `docs/deepdfa_artifact.md` | `docs/` | **moved** | API/semantics guide for the tensor artifact. |
| `docs/deepdfa_status_and_future_work.md` | `docs/` | **duplicated** | Both. Its §5 safe/unsafe claim list governs **both** papers — in particular: do not claim exact symbolic-automaton WMC or knowledge compilation as new (NeSyA owns it), and do not claim a BDD/d-DNNF swap is itself a contribution. Its §12 pointers use the pre-split file numbering; the parent copy has been corrected. |
| `docs/RQ3_STRUCTURAL_SCALING.md` | `docs/` | **duplicated, needs splitting** | Panels 2 (guard/cube complexity) and 3 (prefix scan) are **B's**. Panels 1 (RuleRunner tree shape), 4 and 5 (state growth) are **A's**. Neither copy has been trimmed yet — do that when each paper's experiment section is written. |
| `docs/E2_INSTRUMENTATION.md` | `docs/` | **duplicated** | Shared measurement infrastructure (compilation/memory/artifact-size instrumentation). |
| `docs/EXPERIMENTAL_EVALUATION_PLAN_inherited.md` | `docs/EXPERIMENTAL_EVALUATION_PLAN.md` | **duplicated, renamed** | Read-only inheritance. It is Paper A's plan; it is here so B's protocol, schema, and provenance discipline start from the same rules rather than being reinvented. B needs its own plan before it runs anything. |

## Code

`src/` is a **full copy** of the parent's `src/`, following the convention already
established by `artur_future_work/`. Rationale, and the caveat:

- The package uses absolute `src.` imports and the benchmark layer is entangled:
  `src/benchmarks/schema.py` imports `src.monitors.rulerunner.parse_tree` at module level,
  and `src/benchmarks/e2.py` imports every monitor backend. A DeepDFA-only subtree would
  not import.
- So `src/monitors/rulerunner/` and `src/monitors/progression/` are present here **only as
  comparison baselines and to keep the harness importable**. They are Paper A's
  contribution. Do not develop them here; do not let B make claims about them.
- B's own core is `src/monitors/deep_dfa.py`, with `src/formula/compiler.py`,
  `src/monitors/base.py`, and `src/monitors/symbolic_dfa.py` as its substrate.
- **Duplication is deliberate** and preferred over a premature shared package. The cost is
  drift: a fix to `deep_dfa.py` must be applied in both trees until Matteo decides the
  packaging question. Flagged, not solved.

`tests/` carries the DeepDFA tests (`test_deep_dfa.py`, `test_deep_dfa_scan.py`) plus the
shared substrate tests they depend on (`test_compiler.py`, `test_symbolic_dfa.py`,
`test_early_termination.py`, `test_semantic_oracle.py`, `test_benchmark_schema.py`,
`test_rq3_structural_scaling.py`, `test_e2_instrumentation.py`).

`experiments/` carries `rq3_structural_scaling.py` (whose guard-complexity and prefix-scan
panels are B's headline) and `e2_instrumentation_smoke.py`.

`results/rq3/` and `results/e2/` are copied artifacts. **They are controlled local-CPU
candidates, not final measurements**, and the CUDA block is explicitly unsupported in the
parent's plan. Do not quote them as results.

## Not moved, and why

- `artur_future_work/` — untouched, per instruction. It is a separate deferred thread
  (probabilistic verdicts, calibration, adaptation), not part of this split.
- `results/rq1`, `results/rq2`, `results/rq4` and their docs — Paper A. RQ1 (semantic
  characterization) and RQ2 (cost of correctness) are A's core evidence. RQ4
  (cross-architecture landscape) stays in A because A's deflationary claim is empirical
  and needs one honest latency comparison; B may reuse its protocol, not its role.
- `old/` — archived July suite, not valid evidence for either paper.

## Extraction checklist

`mv deepdfa_paper ../deepdfa_paper && cd ../deepdfa_paper && git init` should work as-is:
no path in this folder reaches into the parent (the `sys.path.insert(0, ROOT)` calls in
the worker scripts resolve `ROOT` relative to their own file). Before extracting:

1. Trim `src/monitors/rulerunner/` and `src/monitors/progression/` to whatever B still
   needs as a baseline, or replace them with a dependency on Paper A's package.
2. Trim `docs/RQ3_STRUCTURAL_SCALING.md` to panels 2–3.
3. Write B's own evaluation plan; delete the inherited one.
4. Resolve `\paperA{}`/`\paperB{}` once publication order is decided.
