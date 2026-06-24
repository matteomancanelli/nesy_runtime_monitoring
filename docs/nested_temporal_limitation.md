# The Nested-Temporal Conflation Limitation of RuleRunner

*(informally, "the Artur bug")*

A structural limitation of the RuleRunner monitoring approach (Perotti, Boella,
d'Avila Garcez — RuleRunner technical report / IJCNN 2014 / IJCNN 2015): for a
class of formulas with **a temporal operator nested inside the operand of an
operator that reinstalls it**, the monitor returns the **wrong verdict**. This
note documents the mechanism, a minimal counterexample (verified against the
exact-correct DFA monitor), the precise flaw in the technical report's
correctness proof, the empirical scope, and why it cannot be repaired without
abandoning the architecture that defines RuleRunner.

This is a limitation of the **published approach**, not of our reimplementation:
our symbolic engine, our flat-CILP network, and our structured (IJCNN 2015
Fig. 5) network all reproduce it identically, and it is the published rule
system that forces it.

---

## 1. Root cause: one truth slot per subformula

RuleRunner's defining design choice (stated repeatedly as its central
advantage over tableaux methods):

> *"RuleRunner is rooted in maintaining a single state composed of the **unique
> truth value of every subformula** of the encoded property."* — IJCNN 2014

The state holds, per distinct subformula `ψ`, **one** rule-name `R[ψ]` (possibly
mode-tagged) and **one** truth slot `[ψ] ∈ {T, F, ?}`. The rule set is fixed at
compile time and never grows. This is what gives RuleRunner a constant rule
count, a fixed-size network, and non-branching state.

The limitation is the dark side of that same choice: **the architecture assumes
each subformula has exactly one live truth value per cell.**

## 2. Mechanism: instance conflation

Several operators **reinstall an operand subtree** on reactivation:

| operator | reinstalls operand subtree… |
|---|---|
| `F`, `G`, `U`, `R` | every cell they remain undecided (a *loop*) |
| `X`, `W` | once, at the defer → monitor transition |

If a reinstalled subformula `ψ` **defers across cells** (i.e. `ψ` is `X`/`W`, or a
temporal that does not resolve within the cell), then a **fresh** instance of
`ψ` is created while a **prior** instance of the same `ψ` is still resolving.
Both instances map onto the **one** slot `[ψ]` and fire the same rules — the
monitor cannot tell them apart. The carried state is corrupted.

This is exactly dual to the other two paradigms' weaknesses, which makes a clean
three-way story:

| paradigm | structural Achilles heel |
|---|---|
| Symbolic DFA | state-space blow-up |
| **RuleRunner** | **instance conflation (this note)** |
| DeepDFA | alphabet blow-up (`2^|AP|`) |

## 3. Minimal counterexample

Formula `φ = F(a ∧ X b)` ("eventually: `a` now and `b` next"), trace
`σ = ({a}, ∅, {b})` (i.e. `a` at cell 0, nothing at cell 1, `b` at cell 2).

**Ground truth:** `φ` needs some cell `i` with `a ∈ σ_i` and `b ∈ σ_{i+1}`. The
only `a` is at cell 0, but `b ∉ σ_1`. So `σ ⊭ φ` → **VIOLATE**.

**RuleRunner's actual state evolution** (dumped from the reference engine;
abbreviating `N=[X(b)]`, `G=[(a∧X b)]`, `D=[F(a∧X b)]`):

| cell | obs | key literals after evaluation | carried `R[.]` → next cell |
|---|---|---|---|
| 0 | `{a}` | `[a]T, [b]F, N?^I, G?^R, D?` | `R[G]^B, R[G]^R, R[D], R[N]^A, R[N]^B, R[a], R[b]` |
| 1 | `∅` | `[a]F, [b]F,` **`N F`** `,` **`N?^I`** `,` **`G F`** `,` **`G?^R`** `, D?` | `R[G]^B, R[G]^R, R[D], R[N]^A, R[N]^B, R[a], R[b]` |
| 2 | `{b}` | `[b]T,` **`N T`** `,` **`G T`** `,` **`G F`** `, D T` → **SUCCESS** | — |

**Output: SATISFY — wrong** (ground truth VIOLATE).

The smoking guns, all present in the real state:

- **Cell 0 → 1 carry:** `R[X(b)]^A` **and** `R[X(b)]^B` — two live `X b`
  instances (one monitoring `b@1`, one freshly deferred to `b@2`) on one
  subformula.
- **Cell 1:** the slot `[X(b)]` holds `F` **and** `?^I` simultaneously; the slot
  `[a∧X b]` holds `F` **and** `?^R`. The single state asserts contradictory
  values.
- **Cell 2:** `[a∧X b]` holds `T` **and** `F` at once — a flat violation of the
  "unique truth value per subformula" invariant — and the spurious `T` drives
  `[F(a∧X b)]T` → premature `SATISFY`.

Verified in code: `SymbolicDFAMonitor` (exact) returns `VIOLATE`;
`RuleRunnerMonitor`/`CILPRunner`/`RuleEngine` all return `SATISFY`.

## 4. The flaw in the technical report's correctness proof (§3.3)

The RuleRunner technical report proves (Theorem 1) that RuleRunner's state always
maps onto a valid FLTL judgement, via a function `map(φ, State, index)` and an
induction on `φ`. We have a counterexample above, so the theorem is **false** —
here is precisely where the proof fails.

`map` reads the state for each subformula:

```
if   [φ]T ∈ State  → ⊤
elif [φ]F ∈ State  → ⊥
elif [φ]?S ∈ State → aux ← S
else  find R[φ]S ∈ State ; aux ← S          # ← assumes a UNIQUE S
...
φ = ψ1 · ψ2, aux=R → map(ψ2)
φ = X ψ, aux ≠ M   → [u, index ⊨ X ψ]_F     # ← one judgement, one index
φ = X ψ, aux = M   → map(ψ)
```

**The proof is not wrong line-by-line; it is unsound in two compounding ways.**

**(a) It omits the operators where the bug is most visible.** The report states:
*"Since □ and ◇ are derivate operators … we omit them."* So `F`/`G` cases are
never proved. But `◇φ ≡ ⊤ U φ` and the `U` case *is* claimed — so the omission
does not actually avoid the bug; it reappears through `U`.

**(b) It relies on an unstated invariant that reactivation destroys.** Every
inductive step uses the hypothesis *"a RuleRunner system monitoring `ψ`
corresponds to a rewriting of `[u, i ⊨ ψ]`"* — **one** judgement at **one**
index `i`. For this, `map` must be a *function*: the clauses `find R[φ]S` and
`[φ]?S ∈ State` and the precedence `T > F > ?` all presuppose **at most one live
instance per subformula**. The proof never shows the **reactivation phase
preserves** that invariant — and it does not.

Walking `map` on the counterexample pinpoints the break:

- **Cell 0** (`index 0`): `map(X b)` reads the single `N?^I` (`aux = I ≠ M`) →
  `[u, 0 ⊨ X b]_F`. Well-defined; the proof holds here. ✓
- **Cell 0 → 1 carry:** the state now contains `R[X(b)]^A` **and** `R[X(b)]^B`.
  The clause `find R[X b]S` has **two** matches, `S ∈ {A, B}`:
  - `aux = A (= M)` → `map(X b) = map(b)` (about `b@1`);
  - `aux = B (≠ M)` → `[u,1 ⊨ X b]_F` (about `b@2`).
  **`map(X b)` is no longer a function.** ← the exact failure point. The
  inductive hypothesis ("one judgement at one index") is now false: the state
  encodes two `X b` obligations at indices 0 and 1 on one slot.
- **Cell 1:** the state holds `[X(b)]F` **and** `N?^I`. `map` checks `[φ]F`
  before `[φ]?S`, so it returns `⊥` and **silently discards** the pending fresh
  instance — `map` is now *lossy* by precedence.
- **Cell 2:** `[a∧X b]` holds `T` and `F`; `map` (precedence `T > F`) returns
  `⊤`, so `map(D) = ⊤`, claiming `[u,0 ⊨ F(a∧X b)] = ⊤`. Ground truth is `⊥`.
  **The theorem's conclusion is false.**

**Verdict on the proof:** each per-operator step is locally valid *under its
premise*; the proof is unsound because it (i) omits `◇`/`□`, and (ii) never
discharges the well-formedness obligation that the reactivated state stays in
`map`'s single-instance domain — which is exactly the obligation the
reactivation it reasons about violates.

## 5. Empirical scope — which formulas actually flip

The conflation *mechanism* needs **(1)** an inner subformula that defers across
cells (`X`/`W`, or a non-in-cell temporal) and **(2)** an enclosing operator
that reinstalls it while a prior instance is pending. But whether it produces an
**observable wrong verdict** depends on the operators. Mismatches vs the exact
DFA over 400 random traces each:

| formula | structure | mismatches/400 |
|---|---|---|
| `F(a∧b)`, `a U b`, `G(a→b)` | flat (inner propositional) | **0** |
| `X(a U b)` | inner `U` over atoms (no defer) | **0** |
| `F(F a)`, `F(G a)`, `G(F a)`, `G(G a)` | monotone whole-temporal nesting | **0** |
| `F(X a)` | inner `X`, **existential** outer | **0** |
| `G(X a)` | inner `X`, **universal** outer | **51** |
| `X(X a)` | inner `X` under `X` (consecutive next) | **158** |
| `F(a∧X b)` | inner `X` under `∧` under `F` | 52 |
| `G(a→X b)` | inner `X` under `→` under `G` | 58 |
| `a U (b∧X c)` | inner `X` under `∧` under `U` | 48 |
| `G(a→F b)` | inner `F` under `→` under `G` (BPM response) | 29 |

Read-off:

- **Flat temporal is always correct** (`F(a∧b)`, `a U b`, the IJCNN scalability
  family `◇⋁(a₀∧aᵢ)`) — no inner deferral, so no second instance. This is why
  our Exps 2/3 (which use the flat IJCNN family) are unaffected.
- **The reliable trigger is `X`/`W` in the nested position.** A one-cell exact
  defer is maximally prone to the `A`-mode (resolve now) vs `B`-mode (defer
  again) collision.
- **Monotone `F`/`G`-only nestings tend to be immune**, and the
  existential/universal asymmetry is sharp: `F(X a) = 0` but `G(X a) = 51`.
  Existential `F` absorbs the spurious carry; universal `G`, `U`, and the
  binary-under-loop patterns expose it.
- **Consecutive next `X(X a)` is among the worst** — and IJCNN 2014 explicitly
  reports testing "consecutive next operators," but only for *timing*, never for
  verdict correctness.

So the limitation is **not** "any nested temporal regardless of operators." It is
specifically a deferring temporal (`X`/`W`, or a non-in-cell temporal)
reinstalled-while-pending under a non-existential operator.

## 6. Why it was never noticed in the papers

1. The correctness proof **omits `◇`/`□`** — the headline cases.
2. Every worked correctness example dodges the trigger: `a∨◇b` and `a∨X b`
   (report Table 2) put the temporal operator under `∨` (non-reinstalling) or
   over an atom (no inner deferral).
3. The papers **verdict-test only flat formulas.** Their own benchmark
   `◇((a∧X b)∨(c∧W d))` (IJCNN 2014/2015) and the "consecutive next" tests have
   the failing structure but appear **only in timing plots** — verdicts were
   never checked.
4. The formalism cannot even express the bug: `map` takes a single `index` and
   does `find R[φ]S`, structurally assuming one instance per subformula. The
   defect is invisible from inside the framework; it only shows against an
   independent oracle (the DFA), which the papers never used.

## 7. Present in all three papers; the structured network does not fix it

The monitoring computation is identical across the RuleRunner technical report,
IJCNN 2014, and IJCNN 2015. IJCNN 2015's Fig. 5 "flattened tree" is the **same
single-hidden-layer recurrent CILP network** as IJCNN 2014's Fig. 2, re-drawn
with one subnetwork per parse-tree node (and used for *local learning*, not a
different monitor). It has **one output slot `{[ψ]ᵀ, [ψ]ᶠ, [ψ]?}` per
subformula** — i.e. it *is* the single-slot architecture. By CILP condition
**C2** (an output fires iff ≥1 incoming hidden fires = OR), the network **OR-
merges** the two conflicting instances at the shared output, reproducing exactly
the corrupted slot. Our structured reimplementation
([src/monitors/rulerunner/structured.py](../src/monitors/rulerunner/structured.py))
matches the flat CILP runner on 1080/1080 random comparisons, diverging from the
DFA on precisely the same nested formulas — empirical confirmation that Fig. 5
inherits the limitation rather than fixing it.

## 8. Output-layer "fixes" cannot work

A natural idea is to add a **priority** over the output slot (e.g. "definite
beats `?`") or otherwise **relax C2** to break the tie. This cannot work, and the
following two traces are a hard impossibility witness:

| trace | ground truth | conflicted slot at cell 1 |
|---|---|---|
| `A = ({a}, ∅, {b})` | VIOLATE | `[X b] = {F, ?}` |
| `B = (∅, {a}, {b})` | SATISFY | `[X b] = {F, ?}` |

Both present the **identical** conflicted slot `[X b] = {F, ?}`, but the correct
resolutions are **opposite**: A needs `F` to win (→ VIOLATE), B needs `?` to win
(→ SATISFY). Any function of only the output values must return the same answer
for both, so **no priority order, and no C2 relaxation, can be correct on both.**
(Verified: the current OR keeps both → right on B, wrong on A; a "definite ⊐ ?"
priority → right on A, wrong on B.) The deciding information — *which instance
owns the value* — is precisely what the shared slot erased.

## 9. The only real fix, and why it breaks the architecture

Correctly separating the instances requires **instance- or cell-scoped slots**
(e.g. `[X b @ now]` vs `[X b @ prev]`), hence an **unbounded** number of slots
for `F`/`G`/`U`/`R`-horizon operands. That destroys the fixed-size,
compile-time-static rule set / network that *defines* RuleRunner in all three
papers. So within the approach as published, the limitation is a genuine
architectural ceiling, not an implementation bug.

## 10. Status in this project

- **Accepted and documented**, not worked around. Three formulas are pinned as
  `xfail(strict=True)` in the equivalence sweeps
  (`F(a∧X b)`, `G(a→F b)`, `G(a→X b)`).
- **Does not affect the experiments:** Exps 2/3 use the flat IJCNN family
  (correct); Exp 1 uses `G(a→F b)`, chosen because it has no trap/sink so *early
  termination never fires* and the per-cell **timing** is well-defined
  regardless of verdict correctness.
- **It is a finding, not just a caveat:** the canonical BPM response pattern
  `G(a→F b)` is structurally simple, semantically central, and **the published
  rule encoding cannot monitor it correctly** — direct support for the paper's
  thesis that the automata-based representation is the more general foundation.

---

### Reproduce

```python
from src.monitors.symbolic_dfa import SymbolicDFAMonitor   # exact oracle
from src.monitors.rulerunner import RuleRunnerMonitor       # the encoding

f = "F(a & X b)"
def cell(a=False, b=False): return {"a": a, "b": b}
A = [cell(a=True), cell(), cell(b=True)]   # VIOLATE, but RuleRunner says SATISFY
B = [cell(), cell(a=True), cell(b=True)]   # SATISFY (both agree)

print(SymbolicDFAMonitor.compile(f).run(A).name, RuleRunnerMonitor.compile(f).run(A).name)
print(SymbolicDFAMonitor.compile(f).run(B).name, RuleRunnerMonitor.compile(f).run(B).name)
```
