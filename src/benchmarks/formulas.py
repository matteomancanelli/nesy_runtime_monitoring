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
    # Whether every DFA edge guard is read-once (each atom appears at most
    # once). On a read-once guard DeepDFA's soft_matrix is *exact*, so the
    # acceptance probability is the true marginal by construction — perfect
    # calibration would then be a hollow identity. The calibration claim
    # (Phase 1.3) must be made on a non-read-once formula. Default True: the
    # IJCNN / response references are read-once.
    read_once: bool = True


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


# ---------------------------------------------------------------------------
# Calibration suite (Capability Exp A, Phase 1.3)
# ---------------------------------------------------------------------------

# The soft acceptance-probability readout is only a *non-trivial* calibration
# target when the DFA's edge guards are NOT read-once. On a read-once guard
# DeepDFA's soft_matrix is exact (P(accept) is the true marginal), so any
# reliability curve is a hollow identity. The 2-of-3 majority function
# (a&b)|(b&c)|(a&c) is the classic non-read-once boolean (each atom appears
# twice); MONA keeps it un-factored on the accepting edge (verified — the
# guard is literally "(a & b) | (a & c) | (b & c)"), so the independence-
# assuming soft product over-counts and the confidence must be calibrated
# *empirically*. This is the formula that makes the calibration a result.
_MAJORITY3 = BenchmarkFormula(
    name="majority3",
    formula="F((a & b) | (b & c) | (a & c))",
    atoms=("a", "b", "c"),
    n_leaves=3,
    read_once=False,
)

# Read-once references (soft path is exact ⇒ calibration is the hollow
# identity — included as the contrast to the majority formula).
#   * response  G(a -> F b): the canonical BPM pattern, simple guards.
#   * ijcnn_n4:  F(OR (a0 & ai)) — read-once after MONA's factoring.
CALIBRATION_SUITE: tuple[BenchmarkFormula, ...] = (
    _MAJORITY3,
    _RESPONSE,
    ijcnn_formula(4),
)


# ---------------------------------------------------------------------------
# State-scaling suite (Exp 6): large automata, small alphabet
# ---------------------------------------------------------------------------

# The IJCNN family scales the *alphabet* (|AP|); this family scales the *state
# space* (|Q|) while keeping the alphabet tiny (2 atoms). It is the instrument
# for the "do larger automata invert the symbolic-wins trend?" question: a big
# |Q| gives DeepDFA's batched matmul real O(|Q|^2) work per launch to amortize
# the fixed per-call overhead against, whereas the symbolic walk only ever
# touches the current state's out-edges and stays ~flat in |Q|.
#
# Bounded response "every a is followed by b within k steps",
#   G( a -> (b | X b | X^2 b | ... | X^k b) ),
# is a real BPM pattern whose minimal DFA tracks the tightest pending deadline,
# so |Q| grows ~linearly with the deadline k while |AP| = 2 is fixed (dense
# 2^|AP| = 4, so the dense tensor stays feasible even at large |Q|). The exact
# |Q| is recorded at run time (Exp 6 compiles each formula and stamps |Q|).

STATE_SCALING_DEADLINES: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


def bounded_response(k: int) -> BenchmarkFormula:
    """G(a -> (b | X b | ... | X^k b)) — response within deadline k."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    consequent = " | ".join("X(" * j + "b" + ")" * j for j in range(k + 1))
    return BenchmarkFormula(
        name=f"boundedresp_k{k}",
        formula=f"G(a -> ({consequent}))",
        atoms=("a", "b"),
        n_leaves=k,  # deadline; Exp 6 overwrites this with the measured |Q|
    )


STATE_SCALING_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    bounded_response(k) for k in STATE_SCALING_DEADLINES
)
