# RQ3 Structural Bottlenecks

> ⚠ **Paper split, 2026-09-13.** This document is duplicated across the two papers and
> has **not** been trimmed. Paper B keeps panels 2 and 3; the other panels belong to the companion paper.

**Schema:** `rq3.v1`  
**Generator:** `python experiments/rq3_structural_scaling.py`  
**Current artifact:** controlled local-CPU candidate

RQ3 separates five structural questions that the old experiments partially
conflated. Every compilation repetition runs in a fresh process and every
runtime point retains its raw process-level repetition, status, compiled
artifact statistics, and resource measurements. All runtime calls force the
full trace.

## Panels

1. **Tree shape.** Balanced and explicitly left-deep IJCNN formulas at
   `n={4,8,16,32}`, using original RuleRunner flat and structured encodings.
   Paired formulas have identical language, atoms, and occurrence count; only
   association and AST depth change.
2. **Guard complexity.** `F(g_n)` with read-once conjunction, majority
   threshold, and parity guards at matched `n={3,4,5}`. Dense and exact
   cube-factored DeepDFA are compared using actual compiled cubes and bytes.
3. **Prefix scan.** Sequential dense versus dense prefix-scan at
   `|Q|={4,8,12}`, `B={1,32}`, and `L={32,128}`. A scan fallback would be an
   explicit failed condition for this panel.
4. **Linear state growth.** Bounded response with `|Q|={4,8,12,16}` and fixed
   two-atom alphabet.
5. **Exponential state growth.** `F(a & X^k b)` with
   `|Q|={5,17,65,257,1025}` and the same two-atom alphabet.

The run uses five independent trace seeds, median summaries, and bootstrap 95%
intervals. Short calls are repeated inside blocks lasting at least 50 ms.

## Main findings

### The old IJCNN shape was a material confound

For flat RuleRunner at batch 64, the left-deep/balanced latency ratio grows from
approximately 1.00× at `n=4` to 1.16× at `n=8`, 1.69× at `n=16`, and 2.31× at
`n=32`. The corresponding depths are `(4,4)`, `(5,8)`, `(6,16)`, and `(7,32)`.
This validates the original audit: a left fold introduces a growing cost that
must not be attributed generically to formula breadth. The balanced form is the
paper-faithful benchmark; left-deep remains only an ablation.

### Compiled representation, not syntax labels, is the guard axis

At `n=5`, the exact factored artifacts contain 7, 21, and 33 cubes for the
read-once, threshold, and parity strata respectively, with cube tensors of 336,
1,008, and 1,584 bytes. These representation trends are exact and monotone.
Warm CPU latency is not strictly ordered by cube count at these small sizes:
fixed tensor and Python overhead remains visible. The paper should therefore
show actual cubes/bytes and avoid claiming a runtime scaling law from this
small CPU grid alone.

### Prefix scan has a real workload-dependent phase boundary

At batch 1, scan is 2.0–5.1× faster than the sequential dense loop over the
measured grid. At batch 32 it is usually slower (sequential/scan 0.75–0.93),
with only the largest measured `(L=128, |Q|=12)` point marginally above parity
(1.06×). No row fell back. This is a phase result, not a claim that scan is
uniformly superior; the final GPU run is essential because launch and matrix
throughput trade-offs are device-dependent.

### Linear and exponential state growth are qualitatively different

With linear `|Q|=4…16`, symbolic latency remains roughly flat while dense
DeepDFA rises from about 74 to 148 microseconds per trace and factored DeepDFA
from about 98 to 132 microseconds. Dense storage follows the validated
`4·2^|AP|·|Q|²` byte model.

With exponential `|Q|=5…1025`, symbolic latency remains approximately
14–18 microseconds per trace. Dense DeepDFA grows from about 63 microseconds to
9.0 milliseconds and 16.8 MB of persistent tensor storage. Factored storage is
much smaller at the endpoint (about 74 KB), but its runtime also reaches about
9.1 ms because it still constructs and applies `|Q|×|Q|` transition matrices.
Factoring removes the alphabet tensor wall; it does not remove the state-square
runtime wall.

## Evidence and limitations

All 335 compilation rows and 505 runtime rows succeeded. The generator rejects
the artifact if bounded-response state counts differ from `k+2`, exponential
state counts differ from `2^k+1`, dense tensor bytes disagree with the analytic
model, or scan silently falls back.

Absolute numbers belong to the current CPU host and should be regenerated on
the final submission CPU/GPU machine. The structural identities, representation
counts, exact state counts, and qualitative separation of the five axes are the
stable conclusions.
