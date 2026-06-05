"""Benchmark formula registry.

Two suites:

IJCNN_SUITE — reproduces and extends the scalability experiment from
  Perotti et al. IJCNN 2014. Formula: F(V_{i=1}^{n-1} (a0 & ai)) for
  n = 2, 4, 8, 16, 32 leaves (distinct atoms). IJCNN 2014 compared
  only RuleRunner variants; we add the symbolic DFA and DeepDFA baselines.

TRACE_LENGTH_SUITE — fixed formulas for the trace-length scaling
  experiment. G(a -> F b) is preferred because it has no trap or
  accepting sink, so it always runs to the end of the trace and
  isolates per-step cost from early-termination frequency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkFormula:
    name: str
    formula: str
    atoms: tuple[str, ...]
    n_leaves: int


# ---------------------------------------------------------------------------
# Atom naming: a–z for the first 26, then aa–af for 26–31
# ---------------------------------------------------------------------------

def _atom(i: int) -> str:
    if i < 26:
        return chr(ord("a") + i)
    return "a" + chr(ord("a") + i - 26)


# ---------------------------------------------------------------------------
# IJCNN 2014 formula family
# ---------------------------------------------------------------------------

IJCNN_LEAF_COUNTS: tuple[int, ...] = (2, 4, 8, 16, 32)


def ijcnn_formula(n: int) -> BenchmarkFormula:
    """Return the IJCNN 2014 benchmark formula with n leaves (atoms).

    Formula: F( OR_{i=1}^{n-1} (a0 & ai) )
    Atoms are named a, b, c, ... alphabetically; for n > 26 the
    overflow atoms are named aa, ab, ...
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    atoms = tuple(_atom(i) for i in range(n))
    a0 = atoms[0]
    disjuncts = [f"({a0} & {ai})" for ai in atoms[1:]]
    formula = "F(" + " | ".join(disjuncts) + ")"
    return BenchmarkFormula(
        name=f"ijcnn_n{n}",
        formula=formula,
        atoms=atoms,
        n_leaves=n,
    )


IJCNN_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    ijcnn_formula(n) for n in IJCNN_LEAF_COUNTS
)


# ---------------------------------------------------------------------------
# Trace-length suite (fixed formula, vary trace length)
# ---------------------------------------------------------------------------

# G(a -> F b): no trap, no accepting sink — verdict only at trace end.
# This is the cleanest choice for measuring per-step cost independently
# of early-termination frequency.
_RESPONSE = BenchmarkFormula(
    name="response",
    formula="G(a -> F b)",
    atoms=("a", "b"),
    n_leaves=2,
)

# Also include the simplest formulas for sanity / comparison.
_EVENTUALLY = BenchmarkFormula(
    name="eventually",
    formula="F a",
    atoms=("a",),
    n_leaves=1,
)

_GLOBALLY = BenchmarkFormula(
    name="globally",
    formula="G a",
    atoms=("a",),
    n_leaves=1,
)

TRACE_LENGTH_SUITE: tuple[BenchmarkFormula, ...] = (
    _RESPONSE,
    _EVENTUALLY,
    _GLOBALLY,
)
