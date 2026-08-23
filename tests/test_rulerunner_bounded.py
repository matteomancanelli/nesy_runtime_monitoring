"""Executable semantics for the bounded-event RuleRunner front end."""

from __future__ import annotations

from itertools import product

import pytest

from src.monitors.base import Verdict
from src.monitors.rulerunner.bounded import (
    BoundedEventPipeline,
    bounded_horizon,
    derive_event_trace,
    evaluate_bounded,
    eventize_bounded_islands,
)
from src.monitors.rulerunner.engine import RuleEngine
from src.monitors.rulerunner.equivalence import certify_rule_runner
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("a", 0),
        ("a & !b", 0),
        ("X a", 1),
        ("WX (a | X b)", 2),
        ("X X X a", 3),
        ("F a", None),
        ("G X a", None),
        ("a U X b", None),
    ],
)
def test_bounded_horizon(formula: str, expected: int | None) -> None:
    assert bounded_horizon(formula) == expected


def test_eventization_selects_maximal_bounded_islands() -> None:
    eventized = eventize_bounded_islands("F (a & X b)")
    assert eventized.horizon == 1
    assert len(eventized.islands) == 1
    island = eventized.islands[0]
    assert island.formula.key == "(a & X(b))"
    assert eventized.skeleton == f"F ({island.event})"


def test_unbounded_response_is_not_hidden_by_eventization() -> None:
    eventized = eventize_bounded_islands("G (a -> F b)")
    assert eventized.islands == ()
    assert not certify_rule_runner(eventized.skeleton).language_equivalent


def test_strong_and_weak_next_boundary_values() -> None:
    one_cell = [{"a": False}]
    assert not evaluate_bounded("X a", one_cell)
    assert evaluate_bounded("WX a", one_cell)


def test_pipeline_delays_then_flushes_the_finite_suffix() -> None:
    eventized = eventize_bounded_islands("F (a & X b)")
    event = eventized.islands[0].event
    pipeline = BoundedEventPipeline(eventized)

    assert pipeline.push({"a": True, "b": False}) is None
    first = pipeline.push({"a": False, "b": True})
    assert first is not None and first[event]
    tail = pipeline.finish()
    assert len(tail) == 1
    assert not tail[0][event]  # strong X is false at the last position


def test_streaming_pipeline_equals_direct_event_semantics() -> None:
    eventized = eventize_bounded_islands("G (a -> X (b | WX c))")
    trace = [
        {"a": True, "b": False, "c": False},
        {"a": False, "b": True, "c": False},
        {"a": True, "b": False, "c": True},
    ]
    derived = derive_event_trace(eventized, trace)
    assert len(derived) == len(trace)
    for island in eventized.islands:
        for position, cell in enumerate(derived):
            assert cell[island.event] is evaluate_bounded(
                island.formula, trace, position
            )


@pytest.mark.parametrize(
    ("formula", "atoms", "max_length"),
    [
        ("WX a", ("a",), 4),
        ("X X a", ("a",), 4),
        ("G X a", ("a",), 5),
        ("F (a & X b)", ("a", "b"), 4),
        ("G (a -> X b)", ("a", "b"), 4),
        ("a U (b & X c)", ("a", "b", "c"), 3),
        ("F (a & X X b)", ("a", "b"), 4),
    ],
)
def test_event_pipeline_composition_matches_original_dfa(
    formula: str,
    atoms: tuple[str, ...],
    max_length: int,
) -> None:
    eventized = eventize_bounded_islands(formula)
    assert certify_rule_runner(eventized.skeleton).language_equivalent
    skeleton = RuleEngine.from_formula(eventized.skeleton)
    original = SymbolicDFAMonitor.compile(formula)
    valuations = [
        dict(zip(atoms, values)) for values in product((False, True), repeat=len(atoms))
    ]

    for length in range(max_length + 1):
        for cells in product(valuations, repeat=length):
            trace = list(cells)
            if not trace:
                actual = Verdict.SATISFY if eventized.empty_value else Verdict.VIOLATE
            else:
                actual = skeleton.run(derive_event_trace(eventized, trace))
            assert actual is original.run(trace), (formula, trace)


def test_empty_word_value_is_explicit_when_the_root_is_eventized() -> None:
    eventized = eventize_bounded_islands("WX a")
    assert eventized.skeleton == eventized.islands[0].event
    assert eventized.empty_value
    assert derive_event_trace(eventized, []) == ()
