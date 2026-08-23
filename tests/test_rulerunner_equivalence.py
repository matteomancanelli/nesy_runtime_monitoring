"""Exact product checking for the original RuleRunner correctness class."""

from __future__ import annotations

import pytest

from src.monitors.rulerunner.engine import RuleEngine, RuleEngineState
from src.monitors.rulerunner.equivalence import certify_rule_runner
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


def test_engine_state_covers_every_mutable_attribute() -> None:
    """Guard for the exhaustive analyses that save and restore engine state.

    ``equivalence.py`` and ``bounded_extrapolation.py`` are only sound if
    ``RuleEngineState`` is the *complete* mutable state.  A new attribute on
    RuleEngine would otherwise leak across a restore and silently weaken the
    certifier, so adding one must fail here and force a decision.
    """
    engine = RuleEngine.from_formula("G (a -> X b)")
    engine.step({"a": True})
    expected = set(RuleEngine._STATE_FIELDS) | set(RuleEngine._CONSTANT_FIELDS)
    assert set(vars(engine)) == expected
    assert set(RuleEngineState.__dataclass_fields__) == {
        "active",
        "last_cell",
        "seen_cell",
        "decided",
    }


@pytest.mark.parametrize(
    "formula", ["G (a -> X b)", "a U (b & X c)", "X X a", "F (a & X b)"]
)
def test_engine_state_round_trip_preserves_behaviour(formula: str) -> None:
    """Saving and restoring every cell must not change any verdict."""
    trace = [
        {"a": bool(i % 2), "b": bool(i % 3), "c": bool(i % 5)} for i in range(8)
    ]
    direct = RuleEngine.from_formula(formula)
    restored = RuleEngine.from_formula(formula)

    for cell in trace:
        expected = direct.step(cell)
        state = restored.state()
        restored.load_state(state)
        assert restored.step(cell) is expected
        assert restored.final_verdict() is direct.final_verdict()
    assert restored.final_verdict() is direct.final_verdict()


@pytest.mark.parametrize(
    "formula",
    [
        "X X a",
        "X X X X a",
        "F F a",
        "F G a",
        "G F a",
        "G G a",
        "F a",
        "G (a -> b)",
        "a U b",
    ],
)
def test_certifies_known_equivalent_formulas(formula: str) -> None:
    result = certify_rule_runner(formula)
    assert result.complete
    assert result.language_equivalent
    assert result.language_witness is None


@pytest.mark.parametrize(
    "formula",
    [
        "G X a",
        "F (a & X b)",
        "G (a -> X b)",
        "a U (b & X c)",
        "G (a -> F b)",
        "(X a) & X (X a)",
    ],
)
def test_returns_replayable_shortest_language_witness(formula: str) -> None:
    result = certify_rule_runner(formula)
    witness = result.language_witness
    assert result.complete
    assert not result.language_equivalent
    assert witness is not None

    trace = list(witness.observations)
    rulerunner = RuleEngine.from_formula(formula).run(trace)
    dfa = SymbolicDFAMonitor.compile(formula).run(trace)
    assert rulerunner is witness.rulerunner
    assert dfa is witness.dfa
    assert rulerunner is not dfa


def test_nested_next_is_not_the_aliasing_counterexample() -> None:
    assert certify_rule_runner("X X a").language_equivalent
    result = certify_rule_runner("(X a) & X (X a)")
    assert not result.language_equivalent
    assert result.language_witness is not None
    assert len(result.language_witness.trace) == 2


def test_distinguishes_language_from_exact_online_labels() -> None:
    result = certify_rule_runner("G X a")
    assert not result.language_equivalent
    assert result.prefix_sound
    assert not result.online_equivalent
    assert result.online_witness is not None
    assert result.online_witness.rulerunner.name == "UNDECIDED"
    assert result.online_witness.dfa.name == "VIOLATE"


def test_state_limit_is_explicitly_incomplete() -> None:
    result = certify_rule_runner("F (a & X b)", max_product_states=1)
    assert not result.complete
    assert not result.language_equivalent


def test_atom_enumeration_guard_is_not_an_approximation() -> None:
    with pytest.raises(ValueError, match="max_atoms=1"):
        certify_rule_runner("a & b", max_atoms=1)
