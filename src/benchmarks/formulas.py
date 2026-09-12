"""Versioned benchmark-formula registry.

The registry records declared formula-family parameters explicitly. Compiler-
derived quantities such as AST depth and minimal-DFA size belong to the
characterization schema in :mod:`src.benchmarks.schema`; they are never hidden
inside a generic ``n_leaves`` field.

``IJCNN_SUITE`` is the paper-faithful, explicitly balanced IJCNN 2014 family.
``IJCNN_LEFTDEEP_SUITE`` denotes the same languages with a left-deep Boolean
tree and exists only as a controlled syntax-shape ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import TypeAlias

ParameterValue: TypeAlias = bool | int | float | str


def _parameters(**values: ParameterValue) -> tuple[tuple[str, ParameterValue], ...]:
    """Return immutable, deterministically ordered family parameters."""
    return tuple(sorted(values.items()))


@dataclass(frozen=True)
class BenchmarkFormula:
    """One stable formula identity plus declared benchmark metadata.

    ``parameters`` describe how the source family produced the formula (for
    example ``n_atoms=16`` or ``deadline=8``). ``dfa_states`` is optional
    characterization data filled by scripts that already compile the DFA; it is
    never used to overwrite a source-family parameter.
    """

    name: str
    formula: str
    atoms: tuple[str, ...]
    family: str
    source: str
    parameters: tuple[tuple[str, ParameterValue], ...] = ()
    tree_shape: str = "canonical"
    roles: tuple[str, ...] = ()
    dfa_states: int | None = None
    # Whether every DFA edge guard is read-once (each atom appears at most
    # once). This no longer marks where DeepDFA is exact: `exact_matrix` (the
    # default probabilistic path) is exact WMC for *any* guard. It marks where
    # the retained `recursive_matrix` approximation coincides with the exact
    # marginal — so a calibration study run on the recursive path would be a
    # hollow identity on a read-once formula, and must use a non-read-once one.
    # That study lives in `artur_future_work/`, not here. Default True: the
    # IJCNN / response references are read-once.
    read_once: bool = True

    @property
    def formula_id(self) -> str:
        """Stable identifier used by result/provenance records."""
        return self.name

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def parameter(self, name: str) -> ParameterValue:
        """Return one declared family parameter, raising on schema mistakes."""
        for key, value in self.parameters:
            if key == name:
                return value
        raise KeyError(f"formula {self.name!r} has no parameter {name!r}")

    @property
    def sweep_parameter(self) -> str | None:
        """Primary family axis, when the formula declares exactly one."""
        if len(self.parameters) == 1:
            return self.parameters[0][0]
        return None

    @property
    def sweep_value(self) -> ParameterValue | None:
        if len(self.parameters) == 1:
            return self.parameters[0][1]
        return None


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

IJCNN_ATOM_COUNTS: tuple[int, ...] = (2, 4, 8, 16, 32)


def _balanced_binary(parts: list[str], operator: str) -> str:
    """Parenthesize ``parts`` as a deterministic near-balanced binary tree."""
    if not parts:
        raise ValueError("cannot associate an empty expression")
    if len(parts) == 1:
        return parts[0]
    midpoint = len(parts) // 2
    left = _balanced_binary(parts[:midpoint], operator)
    right = _balanced_binary(parts[midpoint:], operator)
    return f"({left} {operator} {right})"


def _left_deep_binary(parts: list[str], operator: str) -> str:
    """Parenthesize ``parts`` as ``(((p0 op p1) op p2) ...)``."""
    if not parts:
        raise ValueError("cannot associate an empty expression")
    expression = parts[0]
    for part in parts[1:]:
        expression = f"({expression} {operator} {part})"
    return expression


def ijcnn_formula(n: int, *, tree_shape: str = "balanced") -> BenchmarkFormula:
    """Return an explicitly associated IJCNN 2014 benchmark formula.

    Formula: F( OR_{i=1}^{n-1} (a0 & ai) )
    Atoms are named a, b, c, ... alphabetically; for n > 26 the
    overflow atoms are named aa, ab, ... . ``balanced`` is the paper-faithful
    shape; ``left_deep`` is a semantically equivalent syntax ablation.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if tree_shape not in {"balanced", "left_deep"}:
        raise ValueError(
            f"tree_shape must be 'balanced' or 'left_deep', got {tree_shape!r}"
        )
    atoms = tuple(_atom(i) for i in range(n))
    a0 = atoms[0]
    disjuncts = [f"({a0} & {ai})" for ai in atoms[1:]]
    associate = _balanced_binary if tree_shape == "balanced" else _left_deep_binary
    formula = f"F({associate(disjuncts, '|')})"
    return BenchmarkFormula(
        name=f"ijcnn_{tree_shape}_n{n}",
        formula=formula,
        atoms=atoms,
        family="ijcnn",
        source="Perotti et al., IJCNN 2014",
        parameters=_parameters(n_atoms=n),
        tree_shape=tree_shape,
        roles=("alphabet_scaling", "tree_shape"),
    )


IJCNN_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    ijcnn_formula(n, tree_shape="balanced") for n in IJCNN_ATOM_COUNTS
)

IJCNN_LEFTDEEP_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    ijcnn_formula(n, tree_shape="left_deep") for n in IJCNN_ATOM_COUNTS
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
    family="declare",
    source="Declare response template",
    roles=("trace_length", "realistic_template"),
)

# Also include the simplest formulas for sanity / comparison.
_EVENTUALLY = BenchmarkFormula(
    name="eventually",
    formula="F a",
    atoms=("a",),
    family="sanity",
    source="project control",
    roles=("trace_length",),
)

_GLOBALLY = BenchmarkFormula(
    name="globally",
    formula="G a",
    atoms=("a",),
    family="sanity",
    source="project control",
    roles=("trace_length",),
)

TRACE_LENGTH_SUITE: tuple[BenchmarkFormula, ...] = (
    _RESPONSE,
    _EVENTUALLY,
    _GLOBALLY,
)


# ---------------------------------------------------------------------------
# Calibration suite (Capability Exp A, Phase 1.3)
# ---------------------------------------------------------------------------

# Retained for the future-work fork's harness; nothing in this repo consumes it.
# The distinction it encodes is now between the two guard backends, not between
# exact and inexact monitoring: `exact_matrix` returns the true marginal for any
# guard, while the retained `recursive_matrix` approximation over-counts shared
# atoms. A reliability curve computed on the recursive path is therefore a hollow
# identity on read-once guards and a genuine measurement only on non-read-once
# ones. The 2-of-3 majority function (a&b)|(b&c)|(a&c) is the classic
# non-read-once boolean (each atom appears twice); MONA keeps it un-factored on
# the accepting edge (verified — the guard is literally
# "(a & b) | (a & c) | (b & c)").
_MAJORITY3 = BenchmarkFormula(
    name="majority3",
    formula="F((a & b) | (b & c) | (a & c))",
    atoms=("a", "b", "c"),
    family="threshold_guard",
    source="project guard-complexity family",
    parameters=_parameters(k=2, n_atoms=3),
    roles=("guard_complexity",),
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
        family="bounded_response",
        source="bounded-response template",
        parameters=_parameters(deadline=k),
        roles=("state_scaling", "bounded_horizon"),
    )


STATE_SCALING_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    bounded_response(k) for k in STATE_SCALING_DEADLINES
)


# ---------------------------------------------------------------------------
# Richer benchmark family (Phase 3.3)
# ---------------------------------------------------------------------------
#
# The IJCNN family is a poor instrument: it early-terminates, and its guards are
# read-once after MONA factoring, so the recursive approximation coincides with
# the exact marginal there and hides the backend divergence the fork's capability
# story rests on. These three families each target a gap. All read_once flags
# below are the values MONA actually produces; the structural checks are in
# tests/test_richer_formulas.py, and the probabilistic verification layer
# (`characterize.guard_read_once`) lives in the future-work fork.


# --- (A) Declare / BPM constraint templates --------------------------------
#
# Standard process-mining patterns: realistic, and with diverse trap/sink
# structure (unlike the IJCNN family, which is uniformly sink-terminating).
# `alt_response` is notable — a *real* constraint that MONA keeps non-read-once
# (each of a, b appears twice on the alternation guard), so it doubles as the
# realistic anchor of NON_READ_ONCE_SUITE below.

DECLARE_SUITE: tuple[BenchmarkFormula, ...] = (
    _RESPONSE,
    BenchmarkFormula(
        "chain_response",
        "G(a -> X(b))",
        ("a", "b"),
        "declare",
        "Declare template",
        roles=("realistic_template", "bounded_horizon"),
    ),
    BenchmarkFormula(
        "precedence",
        "(!b) U a | G(!b)",
        ("a", "b"),
        "declare",
        "Declare template",
        roles=("realistic_template",),
    ),
    BenchmarkFormula(
        "alt_response",
        "G(a -> X(!a U b))",
        ("a", "b"),
        "declare",
        "Declare template",
        roles=("realistic_template", "guard_complexity"),
        read_once=False,
    ),
    BenchmarkFormula(
        "resp_existence",
        "F(a) -> F(b)",
        ("a", "b"),
        "declare",
        "Declare template",
        roles=("realistic_template",),
    ),
    BenchmarkFormula(
        "not_coexistence",
        "!(F(a) & F(b))",
        ("a", "b"),
        "declare",
        "Declare template",
        roles=("realistic_template",),
    ),
    BenchmarkFormula(
        "chain_precedence",
        "G(X(b) -> a)",
        ("a", "b"),
        "declare",
        "Declare template",
        roles=("realistic_template", "bounded_horizon"),
    ),
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
        family="threshold_guard",
        source="project guard-complexity family",
        parameters=_parameters(k=k, n_atoms=n),
        roles=("guard_complexity",),
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


def guard_complexity_formula(kind: str, n: int) -> BenchmarkFormula:
    """Return ``F(g_n)`` for a named, semantically distinct guard stratum."""
    if n < 2:
        raise ValueError("guard-complexity formulas require at least two atoms")
    atoms = tuple(_atom(i) for i in range(n))
    if kind == "read_once":
        guard = " & ".join(atoms)
        read_once = True
    elif kind == "threshold":
        k = (n + 1) // 2
        guard = " | ".join(
            "(" + " & ".join(atoms[i] for i in subset) + ")"
            for subset in combinations(range(n), k)
        )
        read_once = False
    elif kind == "parity":
        terms = []
        for values in product((False, True), repeat=n):
            if sum(values) % 2:
                terms.append(
                    "("
                    + " & ".join(
                        atom if value else f"!{atom}"
                        for atom, value in zip(atoms, values, strict=True)
                    )
                    + ")"
                )
        guard = " | ".join(terms)
        read_once = False
    else:
        raise ValueError(f"unknown guard-complexity kind {kind!r}")
    return BenchmarkFormula(
        name=f"guard_{kind}_n{n}",
        formula=f"F({guard})",
        atoms=atoms,
        family="guard_complexity",
        source="project controlled guard-complexity family",
        parameters=_parameters(guard_kind=kind, n_atoms=n),
        roles=("guard_complexity", "alphabet_scaling"),
        read_once=read_once,
    )


GUARD_COMPLEXITY_ATOM_COUNTS: tuple[int, ...] = (3, 4, 5)
GUARD_COMPLEXITY_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    guard_complexity_formula(kind, n)
    for n in GUARD_COMPLEXITY_ATOM_COUNTS
    for kind in ("read_once", "threshold", "parity")
)


# --- (C) State-blowup family: exponential |Q|, tiny alphabet ---------------
#
# "a holds and b holds exactly k steps later" — F(a & X^k b). The minimal DFA
# must track a sliding window of the last k observations, so |Q| = 2^k + 1 while
# |AP| = 2 stays fixed (dense alphabet 2^|AP| = 4). This is a *genuine
# exponential* state blowup, distinct from STATE_SCALING_SUITE's bounded_response
# (which is only LINEAR in k — a deadline knob). It exposes symbolic's storage /
# compile wall AND DeepDFA-dense's |Q|^2 tensor wall simultaneously (a shared
# weakness — good for the neutrality mandate). The temporal depth is declared
# here; the measured `|Q|` is stored separately during characterization.

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
        family="kth_from_last",
        source="project exponential-state family",
        parameters=_parameters(temporal_depth=k),
        roles=("state_blowup", "bounded_horizon"),
    )


STATE_BLOWUP_SUITE: tuple[BenchmarkFormula, ...] = tuple(
    kth_from_last(k) for k in STATE_BLOWUP_DEPTHS
)
