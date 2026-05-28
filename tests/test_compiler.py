"""Tests for src/formula/compiler.py — DFA structure and sink/trap labeling."""

import pytest
from src.formula.compiler import compile_ltlf


# ---------------------------------------------------------------------------
# DFA structure
# ---------------------------------------------------------------------------


def test_eventually_structure():
    # F a: two states, one accepting, one accepting sink, no traps
    dfa = compile_ltlf("F a")
    assert len(dfa.states) == 2
    assert dfa.atoms == ("a",)
    assert len(dfa.accepting) == 1
    assert len(dfa.trap_states) == 0
    assert dfa.accepting_sinks == dfa.accepting


def test_globally_structure():
    # G a: two states, one accepting (initial), one trap, no accepting sinks
    dfa = compile_ltlf("G a")
    assert len(dfa.states) == 2
    assert dfa.atoms == ("a",)
    assert len(dfa.accepting) == 1
    assert dfa.initial in dfa.accepting
    assert len(dfa.trap_states) == 1
    assert len(dfa.accepting_sinks) == 0


def test_until_structure():
    # a U b: three states — initial (undecided), trap, accepting sink
    dfa = compile_ltlf("a U b")
    assert len(dfa.states) == 3
    assert dfa.atoms == ("a", "b")
    assert len(dfa.accepting) == 1
    assert len(dfa.trap_states) == 1
    assert dfa.accepting_sinks == dfa.accepting


def test_response_structure():
    # G(a -> F b): two states, both can be initial or looping; no trap, no sink
    dfa = compile_ltlf("G(a -> F b)")
    assert dfa.atoms == ("a", "b")
    assert len(dfa.trap_states) == 0
    assert len(dfa.accepting_sinks) == 0


def test_atom_extraction_multi():
    dfa = compile_ltlf("F(a & X b)")
    assert set(dfa.atoms) == {"a", "b"}


def test_atom_extraction_complex():
    dfa = compile_ltlf("(a & b) | c")
    assert set(dfa.atoms) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Guard evaluation
# ---------------------------------------------------------------------------


def test_guard_true_constant():
    dfa = compile_ltlf("F a")
    # The accepting sink has a self-loop labeled "true" — should fire for any obs
    sink = next(iter(dfa.accepting_sinks))
    outgoing = dfa._outgoing[sink]
    assert any(t.guard({}) for t in outgoing)
    assert any(t.guard({"a": True}) for t in outgoing)


def test_guard_missing_atom_raises():
    dfa = compile_ltlf("a U b")
    with pytest.raises(ValueError, match="missing an atom"):
        dfa.step(dfa.initial, {})


def test_dfa_step_deterministic():
    # For every state and every concrete observation, exactly one guard fires
    dfa = compile_ltlf("G(a -> F b)")
    for obs in [{"a": True, "b": True}, {"a": True, "b": False},
                {"a": False, "b": True}, {"a": False, "b": False}]:
        for q in dfa.states:
            fired = [t for t in dfa._outgoing[q] if t.guard(obs)]
            assert len(fired) == 1, f"state {q}, obs {obs}: {len(fired)} guards fired"
