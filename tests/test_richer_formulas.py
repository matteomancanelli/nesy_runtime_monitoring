"""Structural verification tests for the richer benchmark families.

Guarantees that the benchmark families in :mod:`src.benchmarks.formulas` have
the structural properties the paper relies on — and that those properties are
what MONA *actually* produces, not what we hoped it would:

  * the state-blowup family blows up exactly as ``|Q| = 2^k + 1`` with a tiny
    alphabet (the blowup is in |Q|, not |AP|);
  * MONA keeps the threshold family's guards un-factored (atoms genuinely
    re-read within a guard) — the property the NON_READ_ONCE_SUITE encodes;
  * the family generators validate their inputs and produce the expected shape.

The probabilistic verification layer (``guard_read_once`` / ``exact_marginal``
against the soft readout) lives in the future-work fork:
artur_future_work/tests/test_richer_formulas.py.
"""

from __future__ import annotations

import math

import pytest

from src.benchmarks.formulas import (
    DECLARE_SUITE,
    NON_READ_ONCE_SUITE,
    STATE_BLOWUP_SUITE,
    at_least_k_of_n,
    kth_from_last,
)
from src.formula.compiler import compile_ltlf

# ---------------------------------------------------------------------------
# Declare suite: every template compiles to a well-formed DFA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula", DECLARE_SUITE, ids=[f.name for f in DECLARE_SUITE]
)
def test_declare_suite_compiles(formula):
    dfa = compile_ltlf(formula.formula)
    assert dfa.initial in dfa.states
    assert set(formula.atoms) == set(dfa.atoms)


# ---------------------------------------------------------------------------
# Non-read-once family: MONA keeps it un-factored
# ---------------------------------------------------------------------------


def test_threshold_guard_survives_mona_unfactored():
    # The suite's defining property is that atoms are genuinely re-read within a
    # single guard. If MONA factored the threshold into a read-once form, the
    # family would silently lose that property. Assert an edge guard literally
    # contains an atom 2+ times.
    dfa = compile_ltlf(at_least_k_of_n(2, 4).formula)
    assert any(
        max((t.label.count(a) for a in dfa.atoms), default=0) >= 2
        for t in dfa.transitions
    )


def test_majority3_is_shared_not_duplicated():
    # majority3 must be the *same* object as CALIBRATION_SUITE's, so the two
    # suites cannot drift.
    from src.benchmarks.formulas import CALIBRATION_SUITE

    assert NON_READ_ONCE_SUITE[0] is CALIBRATION_SUITE[0]


# ---------------------------------------------------------------------------
# State-blowup family: exponential |Q| = 2^k + 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula", STATE_BLOWUP_SUITE, ids=[f.name for f in STATE_BLOWUP_SUITE]
)
def test_state_blowup_is_exponential(formula):
    dfa = compile_ltlf(formula.formula)
    k = formula.n_leaves
    assert len(dfa.states) == 2**k + 1
    # tiny alphabet — the blowup is in |Q|, not |AP|
    assert len(dfa.atoms) == 2


def test_state_blowup_generator_validates_input():
    with pytest.raises(ValueError):
        kth_from_last(0)


def test_at_least_k_of_n_validates_input():
    with pytest.raises(ValueError):
        at_least_k_of_n(3, 2)


def test_threshold_disjunct_count_is_binomial():
    # n_leaves == n; disjunction has C(n, k) conjuncts.
    f = at_least_k_of_n(3, 5)
    assert f.formula.count("&") == math.comb(5, 3) * (3 - 1)
