"""Correctness of the progression-based RuleRunner (Part 1, lazy realization).

The headline: unlike the original RuleRunner (whose one-literal-per-subformula
encoding conflates concurrent instances — see test_rulerunner_engine.py's
xfail set), the progression-based monitor matches ``SymbolicDFAMonitor`` on
*every* formula in the sweep, **including the three nested-temporal formulas**
``F(a & X b)``, ``G(a -> F b)``, ``G(a -> X b)``. There are no xfails here.

Also covers: the paper's exact counterexample traces (§3.2/§3.3), the
progression / last identities, and short-trace end-of-trace resolution.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.benchmarks.runner import random_traces
from src.monitors.base import Verdict
from src.monitors.progression import ProgressionEngine
from src.monitors.progression.formula import FALSE, TRUE, Op, atom, from_node
from src.monitors.progression.progression import holds_empty, last, prog
from src.monitors.rulerunner.parse_tree import parse
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

# The full RuleRunner equivalence sweep — flat and the three formerly-xfail
# nested-temporal formulas together. The progression monitor must agree with
# the DFA on ALL of them.
_SWEEP_FLAT = [
    "a", "!a", "a & b", "a | b", "a -> b",
    "(a & b) | (!a & !b)",
    "F a", "G a", "X a", "WX a",
    "a U b", "a R b",
    "F (a & b)", "G (a | b)",
    "F ((a & b) | (a & c))",
    "F ((a & b) | (a & c) | (a & d))",
    "(F a) -> (F b)",
    "G (!a | !b)",
]

# Precisely the formulas the original RuleRunner gets wrong; the fix must
# handle them.
_NESTED_TEMPORAL = [
    "F (a & X b)",
    "G (a -> F b)",
    "G (a -> X b)",
]

# A few harder nested/temporal combinations to stress the fix beyond the
# original xfail trio.
_NESTED_EXTRA = [
    "F (a & X (b & X c))",   # deeper next-nesting under F
    "G (a -> X (F b))",       # X F under response
    "(a U (b & X c))",        # next nested under until
    "F G a",                  # F of G
    "G F a",                  # infinitely-often style (over finite traces)
    "a U (b U c)",            # nested until
]

_ALL = _SWEEP_FLAT + _NESTED_TEMPORAL + _NESTED_EXTRA


def _stable_seed(formula: str) -> int:
    return int(hashlib.md5(formula.encode()).hexdigest()[:8], 16)


def _sweep_one(formula: str, atoms=("a", "b", "c", "d"), L=12, n=80) -> None:
    rng = np.random.default_rng(seed=_stable_seed(formula))
    traces = random_traces(atoms, trace_length=L, n_traces=n, rng=rng)
    mon = ProgressionEngine.compile(formula)
    dfa = SymbolicDFAMonitor.compile(formula)
    for trace in traces:
        pv = mon.run(trace)
        dv = dfa.run(trace)
        assert pv is dv, (
            f"Mismatch on {formula!r}, trace {trace}: progression={pv} vs dfa={dv}"
        )


@pytest.mark.parametrize("formula", _ALL)
def test_progression_matches_symbolic_dfa(formula: str) -> None:
    """Progression monitor agrees with the DFA on every random trace —
    flat AND nested-temporal, no xfails."""
    _sweep_one(formula)


@pytest.mark.parametrize("formula", _NESTED_TEMPORAL)
def test_fixes_original_rulerunner_failures(formula: str) -> None:
    """Explicit, separately-named guard: the exact formulas the original
    RuleRunner marks xfail now pass verdict-for-verdict against the DFA."""
    _sweep_one(formula, L=15, n=120)


def test_progression_matches_on_short_traces() -> None:
    """Short traces (length 1-3) most stress end-of-trace resolution."""
    rng = np.random.default_rng(seed=7)
    atoms = ("a", "b")
    formulas = ["X a", "WX a", "F a", "G a", "a U b", "a R b", "a -> F b",
                "F (a & X b)", "G (a -> F b)"]
    for L in (1, 2, 3):
        traces = random_traces(atoms, trace_length=L, n_traces=40, rng=rng)
        for f in formulas:
            mon = ProgressionEngine.compile(f)
            dfa = SymbolicDFAMonitor.compile(f)
            for trace in traces:
                assert mon.run(trace) is dfa.run(trace), (
                    f"Mismatch on {f!r}, L={L}, trace={trace}"
                )


# --------------------------- the paper's counterexample ---------------------

def test_counterexample_traces() -> None:
    """latex/3_rulerunner.tex §3.3: phi_b = F(a & X b), the two traces the
    one-register RuleRunner cannot tell apart."""
    phi_b = "F (a & X b)"
    A = [{"a": True, "b": False}, {"a": False, "b": False}, {"a": False, "b": True}]
    B = [{"a": False, "b": False}, {"a": True, "b": False}, {"a": False, "b": True}]
    assert ProgressionEngine.compile(phi_b).run(A) is Verdict.VIOLATE
    assert ProgressionEngine.compile(phi_b).run(B) is Verdict.SATISFY


def test_counterexample_decisive_residual() -> None:
    """After the a-cell, phi_b's residual is the single disjunctive obligation
    b | F(a & X b) — the atomic object the one-register state cannot hold."""
    phi_b = from_node(parse("F (a & X b)"))
    rho1 = prog(phi_b, {"a": True, "b": False})
    from src.monitors.progression.formula import simplify
    rho1 = simplify(rho1)
    # It is a disjunction of `b` and the original eventuality (order-insensitive).
    assert rho1.op is Op.OR
    leaf_keys = {rho1.args[0].key, rho1.args[1].key}
    assert "b" in leaf_keys
    assert any(k.startswith("EVENTUALLY(") for k in leaf_keys)


# ------------------------------- prog / last unit ---------------------------

def test_prog_atom() -> None:
    assert prog(atom("a"), {"a": True}) is TRUE
    assert prog(atom("a"), {"a": False}) is FALSE
    assert prog(atom("a"), {}) is FALSE  # missing atom = false


def test_prog_next_falls_through() -> None:
    """prog(X phi, s) = phi, carrying no residue of the X."""
    from src.monitors.progression.formula import simplify
    f = from_node(parse("X (a & b)"))
    expected = from_node(parse("a & b"))
    assert simplify(prog(f, {"a": False, "b": False})).key == expected.key


def test_last_boundary_operators() -> None:
    assert last(from_node(parse("X a")), {"a": True}) is False   # strong next
    assert last(from_node(parse("WX a")), {"a": False}) is True  # weak next
    assert last(from_node(parse("F a")), {"a": True}) is True
    assert last(from_node(parse("F a")), {"a": False}) is False
    assert last(from_node(parse("G a")), {"a": True}) is True
    assert last(from_node(parse("a U b")), {"a": True, "b": False}) is False
    assert last(from_node(parse("a U b")), {"a": False, "b": True}) is True


def test_simplify_detects_constants() -> None:
    from src.monitors.progression.formula import disj, neg, simplify
    fa = from_node(parse("F a"))
    assert simplify(disj(fa, neg(fa))).op is Op.TRUE       # x | ~x
    from src.monitors.progression.formula import conj
    assert simplify(conj(fa, neg(fa))).op is Op.FALSE      # x & ~x


def test_holds_empty() -> None:
    assert holds_empty(from_node(parse("G a"))) is True    # vacuous
    assert holds_empty(from_node(parse("F a"))) is False
    assert holds_empty(from_node(parse("WX a"))) is True
    assert holds_empty(from_node(parse("X a"))) is False
