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
from itertools import combinations


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

# MONA is the binding constraint, not |Q|. The `X^k b` disjunction makes MONA's
# intermediate BDD blow up well before the (linear) answer does: k=18 compiles in
# ~7 s / 0.3 GB, k=20 exhausts memory (~6 GB) and returns a failure stub, and by
# k=32 it also races ltlf2dfa's 30 s subprocess timeout. Both now raise
# MonaFailure instead of yielding a degenerate |Q|=2 DFA. Stay at k<=18 and take
# more points along the way; |Q| = k+2, so this still spans |Q| = 4..20.
STATE_SCALING_DEADLINES: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 18)


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


# ---------------------------------------------------------------------------
# Richer benchmark family (Phase 3.3)
# ---------------------------------------------------------------------------
#
# The IJCNN family is a poor instrument: it early-terminates, and its guards are
# read-once after MONA factoring (so DeepDFA's soft_matrix is *exact*, hiding the
# paradigm divergence the capability story rests on). These three families each
# target a gap. All read_once flags below are the values MONA actually produces,
# verified against `characterize.guard_read_once` in tests/test_richer_formulas.py.


# --- (A) Declare / BPM constraint templates --------------------------------
#
# Standard process-mining patterns: realistic, and with diverse trap/sink
# structure (unlike the IJCNN family, which is uniformly sink-terminating).
# `alt_response` is notable — a *real* constraint that MONA keeps non-read-once
# (each of a, b appears twice on the alternation guard), so it doubles as the
# realistic anchor of NON_READ_ONCE_SUITE below.

DECLARE_SUITE: tuple[BenchmarkFormula, ...] = (
    BenchmarkFormula("response", "G(a -> F(b))", ("a", "b"), 2),
    BenchmarkFormula("chain_response", "G(a -> X(b))", ("a", "b"), 2),
    BenchmarkFormula("precedence", "(!b) U a | G(!b)", ("a", "b"), 2),
    BenchmarkFormula(
        "alt_response", "G(a -> X(!a U b))", ("a", "b"), 2, read_once=False
    ),
    BenchmarkFormula("resp_existence", "F(a) -> F(b)", ("a", "b"), 2),
    BenchmarkFormula("not_coexistence", "!(F(a) & F(b))", ("a", "b"), 2),
    BenchmarkFormula("chain_precedence", "G(X(b) -> a)", ("a", "b"), 2),
)


# --- (B) Non-read-once family: the divergence instrument -------------------
#
# "At least k of n atoms true" — F( OR over all k-subsets S of (AND_{i in S} a_i) ).
# Each atom recurs C(n-1, k-1) times in the disjunction, so MONA keeps the guard
# non-read-once and DeepDFA's independence-assuming soft product OVER-counts the
# true marginal by a margin that grows with the family. This turns the single
# `majority3` data point into a *curve* over formula size (the Exp 7 finding).


def at_least_k_of_n(k: int, n: int) -> BenchmarkFormula:
    """F( OR_{|S|=k} AND_{i in S} a_i ) — "at least k of n atoms true"."""
    if not 1 <= k <= n:
        raise ValueError(f"require 1 <= k <= n, got k={k}, n={n}")
    atoms = tuple(_atom(i) for i in range(n))
    disjuncts = [
        "(" + " & ".join(atoms[i] for i in subset) + ")"
        for subset in combinations(range(n), k)
    ]
    return BenchmarkFormula(
        name=f"atleast{k}of{n}",
        formula="F(" + " | ".join(disjuncts) + ")",
        atoms=atoms,
        n_leaves=n,
        read_once=False,
    )


# majority3 (2-of-3) is defined once as _MAJORITY3 (CALIBRATION_SUITE) and reused
# here so the two suites cannot drift. The larger threshold points and the
# realistic `alt_response` anchor complete the divergence sweep.
_ALT_RESPONSE = DECLARE_SUITE[3]

NON_READ_ONCE_SUITE: tuple[BenchmarkFormula, ...] = (
    _MAJORITY3,
    at_least_k_of_n(2, 4),
    at_least_k_of_n(2, 5),
    at_least_k_of_n(3, 5),
    _ALT_RESPONSE,
)


# --- (C) State-blowup family: exponential |Q|, tiny alphabet ---------------
#
# "a holds and b holds exactly k steps later" — F(a & X^k b). The minimal DFA
# must track a sliding window of the last k observations, so |Q| = 2^k + 1 while
# |AP| = 2 stays fixed (dense alphabet 2^|AP| = 4). This is a *genuine
# exponential* state blowup, distinct from STATE_SCALING_SUITE's bounded_response
# (which is only LINEAR in k — a deadline knob). It exposes symbolic's storage /
# compile wall AND DeepDFA-dense's |Q|^2 tensor wall simultaneously (a shared
# weakness — good for the neutrality mandate). n_leaves is overwritten with the
# measured |Q| by Exp 7, mirroring STATE_SCALING_SUITE.

STATE_BLOWUP_DEPTHS: tuple[int, ...] = (2, 4, 6, 8, 10)


def kth_from_last(k: int) -> BenchmarkFormula:
    """F(a & X^k b) — a now, b exactly k steps later; |Q| = 2^k + 1."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    consequent = "X(" * k + "b" + ")" * k
    return BenchmarkFormula(
        name=f"kthlast_k{k}",
        formula=f"F(a & {consequent})",
        atoms=("a", "b"),
        n_leaves=k,  # depth; Exp 7 overwrites this with the measured |Q|
    )


STATE_BLOWUP_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    kth_from_last(k) for k in STATE_BLOWUP_DEPTHS
)
