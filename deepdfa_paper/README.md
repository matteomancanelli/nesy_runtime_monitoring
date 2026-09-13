# Paper B — Tensorized automaton representations

**Status:** staged inside `nesy_runtime_monitoring`, extraction-ready. Not started as a
paper. Created 2026-09-13 by the split described in `NOTES.md`.

## The question

Paper A (the parent repo) asks what a rule-based neural monitor can *compute*, and what
completeness costs it. This paper asks the orthogonal question: **how should a compiled
transition function be represented and executed?**

A DFA transition function is a table with `|Q| · 2^|P|` entries. Every representation
compresses one axis and pays on the other:

| Representation | Compresses | Cost driver |
|---|---|---|
| Symbolic walk (guards + explicit state) | the `2^|P|` symbol axis | `|Q|` |
| Dense DeepDFA tensor | the state axis (one uniform matmul) | `2^|P|` |
| Factored / disjoint-cube DeepDFA | attempts both | number of cubes `C` (structure-dependent) |
| Decision diagrams (BDD/SDD/d-DNNF/ShMTBDD) | both, with subfunction sharing | diagram size |

## The claim to develop

A **Golog program graph** is already an explicit-state, symbolically-guarded object — the
KR 2026 construction (De Giacomo, Lespérance, Mancanelli, Parretti) represents program
graphs semi-symbolically over ShMTBDDs. That is structurally the same point in the table
above as this repo's factored DeepDFA mode, reached from a different direction. Paper B's
thesis is that these are one design space, and that the program-graph route reaches
representations the LTLf route reaches only at doubly exponential cost.

⚠ **State the separation carefully.** A program graph is polynomial in the *program*; the
DFA is doubly exponential in the *formula*. Those are different inputs, so "poly vs
doubly-exp" is not yet a theorem. A genuine separation needs a family in which the *same*
specification is written both ways. Until that family exists, Paper B has an engineering
story, not a lower bound. Settle this before committing to a venue.

## Scope ladder — stop at Level 1

Monitoring evaluates the *specification alone*; the domain enters only when the reasoning
task requires it.

| Level | Object | Needs the domain? | Where |
|---|---|---|---|
| 0 | LTLf formula → DFA → tensor | no | done (this code) |
| 1 | **Golog program → program graph → tensor** | **no** | **Paper B — stop here** |
| 2 | + cross-product with the domain | yes | future work |
| 3 | + game solving / `PreAdv` | yes | future work, gated by whether `PreAdv` tensorizes |

Stopping at Level 1 keeps Paper B about representation and evaluation rather than
reasoning, and makes it independent of the open question of whether the adversarial
predecessor operator tensorizes. **Do not let this paper creep into synthesis or planning.**

## Hard constraints inherited from the parent

From `docs/deepdfa_status_and_future_work.md` §5 — these bind Paper B as much as Paper A:

- Exact differentiable WMC over compiled symbolic-automaton guards is **NeSyA's**, not new
  here. Neither the semantics nor knowledge compilation may be claimed as a contribution.
- Replacing disjoint cubes with BDD/d-DNNF is **not by itself** a contribution. The delta
  has to be an LTLf/program-graph runtime backend plus a systems comparison.
- Disjoint cubes do **not** guarantee sub-exponential storage. Shannon expansion of a guard
  on `k` atoms can emit `Θ(2^k)` cubes.
- The prefix scan reduces launch depth, **not** FLOPs. It is a hardware-dependent
  trade-off, not a complexity improvement.
- No GPU-beats-symbolic claim before fair measurement on a controlled CUDA host.

## What is here

See `NOTES.md` for the file-by-file provenance table. Summary: the moved LaTeX
representation material, the decision-diagram and artifact docs, a full copy of `src/`
(RuleRunner and progression present only as baselines), the DeepDFA tests, the RQ3
structural-scaling driver, and its local-CPU candidate artifacts.

**Nothing in `results/` is a final measurement.** They are controlled local-CPU
candidates; the CUDA block is unsupported in the inherited plan.
