# RQ1 Semantic Characterization

**Schema:** `rq1.v1` over benchmark schema `e0.v1`  
**Frozen artifact:** `results/rq1/rq1_characterization.{csv,json}`  
**Generator:** `python experiments/rq1_semantic_characterization.py`

This document is the paper-facing interpretation of Phase E1. The JSON and CSV
artifacts are the executable evidence; this table is a compact draft for the
experimental section.

## Question and semantic gate

RQ1 asks where the published one-register-per-subformula RuleRunner is exact,
which failures the bounded-event construction repairs, and whether progression
recovers the complete supported LTLf language.

The corpus is a declared repair ladder, not a random sample:

- 9 original-safe controls, including the balanced IJCNN formula;
- 8 shared-register counterexamples;
- 6 finite-horizon bounded-repair targets;
- 2 unsafe unbounded-skeleton rejection controls;
- all 7 Declare templates currently shipped by the benchmark registry.

Categories overlap. In particular, the Declare templates include both controls
and counterexamples. Consequently, raw counts are meaningful for this declared
corpus, but percentages must not be presented as estimates of prevalence in an
unspecified formula population.

Every row is checked by exact reachable-product exploration, not sampled trace
testing. Original RuleRunner is compared directly with the canonical minimal
DFA. The bounded pipeline graph is compared with the canonical DFA for final,
sound-prefix, default-online, and exact-head labels. The eager progression graph
is likewise checked in an exact product with the canonical DFA.

For this gate, `applicable` means semantically safe for later evaluation:

- original: exact final language and sound definite prefixes;
- bounded: its eventized skeleton has a complete exact certificate for both;
- progression: its exact product agrees with canonical semantics.

## Applicability and coverage table

| Formula ID | Original | Shortest final witness | Bounded | Horizon | Default online | Exact-head states | Progression states |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atomic_control` | safe | — | admit | 0 | exact | 3 | 3 |
| `nested_next_control` | safe | — | admit | 2 | exact | 11 | 5 |
| `nested_eventually_control` | safe | — | admit | 0 | exact | 3 | 2 |
| `immediate_response_control` | safe | — | admit | 0 | exact | 3 | 2 |
| `flat_until_control` | safe | — | admit | 0 | exact | 5 | 3 |
| `ijcnn_balanced_n4` | safe | — | admit | 0 | exact | 3 | 2 |
| `next_offset_alias` | unsafe | 2 | admit | 2 | late | 11 | 5 |
| `globally_next_alias` | unsafe | 2 | admit | 1 | late | 5 | 3 |
| `eventual_next_alias` | unsafe | 2 | admit | 1 | exact | 11 | 3 |
| `until_next_alias` | unsafe | 2 | admit | 1 | late | 33 | 5 |
| `response` | unsafe | 2 | reject | 0 | — | — | 2 |
| `chain_response` | unsafe | 2 | admit | 1 | exact | 11 | 3 |
| `precedence` | safe | — | admit | 0 | exact | 5 | 3 |
| `alt_response` | unsafe | 2 | reject | 0 | — | — | 4 |
| `resp_existence` | safe | — | admit | 0 | exact | 9 | 3 |
| `not_coexistence` | safe | — | admit | 0 | exact | 13 | 4 |
| `chain_precedence` | unsafe | 2 | admit | 1 | exact | 11 | 3 |

“Default online: late” means final verdicts remain exact and every definite
online label remains sound, but at least one shortest prefix exists on which
the canonical DFA has a permanent verdict while the default bounded skeleton
is still undecided. The exact-online head removes all such differences for all
15 admitted cases. It is a separate, priced configuration.

## Aggregate findings for the declared corpus

- Original RuleRunner is applicable to all 9 controls and to none of the 8
  declared shared-register counterexamples.
- Bounded-event RuleRunner admits all 6 finite-horizon repair targets. It
  rejects `response` and `alt_response` because eventization leaves an unsafe
  unbounded skeleton; rejection is therefore a correctness result, not a
  missing benchmark point.
- The bounded default is final-language exact and prefix-sound on all 15
  admitted formulas. It is exact-online on 12 and detectably late on 3.
- The bounded exact-online configuration agrees on final and online semantics
  for all 15 admitted formulas. Its reachable composite head ranges from 3 to
  33 states in this corpus.
- Progression agrees with the canonical DFA on final language, sound prefixes,
  and exact online labels for all 17 formulas.

The progression and exact-head state counts are implementation representation
sizes, not minimal semantic state counts. Progression normalization is not a
full language quotient, and the bounded head enumerates composite pipeline
states. Later cost experiments must price these quantities without calling
them intrinsic lower bounds.

## Reproduction and integrity

Regenerate the artifact:

```bash
python experiments/rq1_semantic_characterization.py
```

Check that the frozen records still match the current implementations without
rewriting them:

```bash
python experiments/rq1_semantic_characterization.py --check
```

The JSON manifest records the corpus/category summaries, resource guards,
software provenance, and a SHA-256 identity of the canonical record list. Every
unsafe or late case retains its shortest exact-product witness as structured
JSON in the corresponding row.
