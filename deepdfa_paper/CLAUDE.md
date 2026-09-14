# CLAUDE.md — Paper B: tensorized automaton representations

Guidance for Claude Code working **inside this folder**. Read `README.md` (scope) and
`NOTES.md` (provenance) first; this file adds the working rules.

## What this folder is

A self-contained, extraction-ready staging area for **Paper B**, split out of
`nesy_runtime_monitoring` on 2026-09-13 at parent commit `26dbaa9`. Paper A (the parent) is
about *monitor semantics and the price of completeness*. Paper B is about *how a compiled
transition function is represented and executed*, and it is the intended home for the
Golog program-graph work (KR 2026).

Paper B **has not been started**. There is a compilable LaTeX skeleton, the moved
representation material, working code, and one experiment driver. There is no draft.

## Working rules

1. **Stay at Level 1 of the ladder** (`README.md`). Program → program graph → tensor. No
   domain cross-product, no game solving, no synthesis.
2. **Do not develop RuleRunner or progression here.** They are in `src/` only because the
   benchmark layer will not import without them (`src/benchmarks/schema.py` imports
   `src.monitors.rulerunner.parse_tree` at module level). They are Paper A's contribution.
3. **Do not weaken the inherited unsafe-claims list** (`docs/deepdfa_status_and_future_work.md`
   §5). NeSyA owns exact symbolic-automaton WMC and knowledge compilation. Cubes do not
   guarantee sub-exponential storage. Scan does not reduce FLOPs.
4. **The poly-vs-doubly-exponential separation is not yet a theorem.** Program graphs are
   polynomial in the program; DFAs doubly exponential in the formula. Do not write it as a
   separation until a family exists where the same specification is expressed both ways.
5. **`results/` holds candidates, not results.** Controlled local-CPU only; the CUDA block
   is unsupported. Never quote a number from here as a measurement.
6. **Duplication is deliberate.** `src/` and `latex/deepdfa_semantics.tex` are copies of
   Paper A's. A fix in one must be applied in the other until the packaging question is
   decided. If drift is found, say so rather than silently reconciling.
7. **Keep the folder self-contained.** No path may reach into the parent repo — the
   extraction test is `mv deepdfa_paper ../ && git init`.

## Layout

```
deepdfa_paper/
├── README.md    Paper B's scope, the level ladder, the inherited constraints
├── NOTES.md     File-by-file provenance: moved / duplicated / which paper depends on it
├── latex/       Skeleton (main.tex) + moved representation sections + duplicated semantics
├── docs/        Decision-diagram design note, artifact guide, duplicated status/plan/RQ3
├── src/         Full copy of the parent's src; deep_dfa.py is the core
├── tests/       DeepDFA tests + shared substrate tests
├── experiments/ rq3_structural_scaling.py (guard-complexity + prefix-scan panels are B's)
└── results/     rq3/, e2/ — local-CPU candidates only
```

## Environment

Same as the parent: conda env `nesy-monitoring` (`environment.yml` is copied here),
`ltlf2dfa==1.0.2` (MONA on PATH), `torch==2.6.0+cu124`. Heavy sweeps run on Google Colab.
There is no Docker in this project.

## First tasks when Paper B is actually opened

1. Verify the KR 2026 program-graph representation against `deep_dfa.py`'s factored mode —
   are they the same object, concretely, or only by analogy?
2. Settle the separation question in `README.md`. This decides whether Paper B has a
   theorem or a systems result, and therefore its venue.
3. Write Paper B's own evaluation plan; delete `docs/EXPERIMENTAL_EVALUATION_PLAN_inherited.md`.
4. Trim `docs/RQ3_STRUCTURAL_SCALING.md` to panels 2–3 (guard complexity, prefix scan).
5. Decide the packaging question with Matteo: duplicate `deep_dfa.py` or depend on A.
