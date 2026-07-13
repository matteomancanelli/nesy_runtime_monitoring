"""Verification tests for the richer benchmark families (Phase 3.3).

Guarantees that the three new families in :mod:`src.benchmarks.formulas` have the
properties the paper relies on — and, crucially, that those properties are what
MONA *actually* produces, not what we hoped it would:

  * every ``BenchmarkFormula.read_once`` flag equals the computed
    :func:`guard_read_once` value (the flag is verified, never trusted);
  * the non-read-once family really is non-read-once AND MONA keeps the threshold
    guard un-factored (so the soft-path divergence is a *result*, not an identity);
  * the state-blowup family blows up exactly as ``|Q| = 2^k + 1``.
"""

from __future__ import annotations

import math

import pytest

from src.benchmarks.characterize import (
    exact_marginal,
    exact_marginal_trace,
    guard_read_once,
)
from src.benchmarks.formulas import (
    DECLARE_SUITE,
    NON_READ_ONCE_SUITE,
    STATE_BLOWUP_SUITE,
    at_least_k_of_n,
    kth_from_last,
)
from src.formula.compiler import compile_ltlf
from src.monitors.deep_dfa import DeepDFAMonitorFactored

ALL_SUITES = (
    ("declare", DECLARE_SUITE),
    ("non_read_once", NON_READ_ONCE_SUITE),
    ("state_blowup", STATE_BLOWUP_SUITE),
)


# ---------------------------------------------------------------------------
# guard_read_once
# ---------------------------------------------------------------------------


def test_guard_read_once_true_on_read_once_guard():
    # IJCNN guard is read-once after MONA factoring.
    ok, worst = guard_read_once(compile_ltlf("F((a & b) | (a & c) | (a & d))"))
    assert ok is True
    assert worst == {}


def test_guard_read_once_false_on_majority():
    ok, worst = guard_read_once(compile_ltlf("F((a & b) | (b & c) | (a & c))"))
    assert ok is False
    assert worst == {"a": 2, "b": 2, "c": 2}


def test_guard_read_once_multichar_atom_names():
    # An atom name that is a substring of another must not be miscounted.
    ok, _ = guard_read_once(compile_ltlf("F(a & aa)"))
    assert ok is True


# ---------------------------------------------------------------------------
# exact_marginal (the brute-force probabilistic oracle)
# ---------------------------------------------------------------------------

_MAJ = "F((a & b) | (b & c) | (a & c))"


def test_exact_marginal_crisp_is_zero_or_one():
    assert exact_marginal(_MAJ, {"a": 1.0, "b": 1.0, "c": 0.0}) == 1.0  # 2 true
    assert exact_marginal(_MAJ, {"a": 1.0, "b": 0.0, "c": 0.0}) == 0.0  # 1 true


def test_exact_marginal_matches_closed_form_majority():
    pa, pb, pc = 0.6, 0.7, 0.4
    # P(>=2 of 3) by inclusion of the four accepting assignments.
    expected = (
        pa * pb * pc
        + pa * pb * (1 - pc)
        + pa * (1 - pb) * pc
        + (1 - pa) * pb * pc
    )
    got = exact_marginal(_MAJ, {"a": pa, "b": pb, "c": pc})
    assert got == pytest.approx(expected)


def test_exact_marginal_trace_is_single_cell_generalization():
    cell = {"a": 0.3, "b": 0.6, "c": 0.9}
    assert exact_marginal_trace(_MAJ, [cell]) == pytest.approx(
        exact_marginal(_MAJ, cell)
    )


def test_exact_marginal_trace_multicell_response():
    # F(a): accepted iff a is true in >=1 of the L cells.
    trace = [{"a": 0.5}, {"a": 0.5}, {"a": 0.5}]
    # P(at least one a) = 1 - 0.5^3
    assert exact_marginal_trace("F(a)", trace) == pytest.approx(1 - 0.5**3)


def test_soft_readout_overcounts_exact_marginal_on_majority():
    # The finding: on a non-read-once guard the independence-product soft score
    # exceeds the true marginal on fractional inputs.
    cell = {"a": 0.6, "b": 0.7, "c": 0.4}
    exact = exact_marginal(_MAJ, cell)
    soft = DeepDFAMonitorFactored.compile(_MAJ).acceptance_probability([cell])
    assert soft > exact
    assert soft - exact == pytest.approx(0.0786, abs=1e-3)


# ---------------------------------------------------------------------------
# read_once flags are verified against the compiled DFA (all suites)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula",
    [f for _, suite in ALL_SUITES for f in suite],
    ids=[f.name for _, suite in ALL_SUITES for f in suite],
)
def test_read_once_flag_matches_compiled_dfa(formula):
    computed, _ = guard_read_once(compile_ltlf(formula.formula))
    assert formula.read_once == computed


# ---------------------------------------------------------------------------
# Non-read-once family: really non-read-once + MONA keeps it un-factored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula", NON_READ_ONCE_SUITE, ids=[f.name for f in NON_READ_ONCE_SUITE]
)
def test_non_read_once_suite_is_non_read_once(formula):
    ok, worst = guard_read_once(compile_ltlf(formula.formula))
    assert ok is False
    assert all(m >= 2 for m in worst.values())


def test_threshold_guard_survives_mona_unfactored():
    # If MONA factored the threshold into a read-once form, the calibration claim
    # would collapse to an identity. Assert an accepting-edge guard literally
    # contains the disjunction (an atom appearing 2+ times on a single guard).
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
