"""Bounded-event pipeline semantics for a RuleRunner front end.

This module is the executable semantic reference independent of its neural
realization.  It separates maximal finite-horizon formula islands, evaluates
them from a fixed observation window, and emits a delayed derived trace for an
ordinary RuleRunner skeleton.  The corresponding CILP compilers live in
``bounded_cilp.py``.

For a non-empty finite trace ``w`` and an eventized formula ``alpha``, the
derived proposition for island ``beta`` is true at position ``t`` exactly when
``w,t |= beta``.  Consequently ``w,0 |= phi`` iff the derived word satisfies
``alpha``.  The empty word is handled separately because it has no position at
which a derived proposition could be emitted.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from src.formula.compiler import Observation
from src.monitors.rulerunner.parse_tree import Node, Op, parse


@dataclass(frozen=True)
class BoundedIsland:
    """One maximal syntactically bounded formula replaced by an event atom."""

    event: str
    formula: Node
    horizon: int


@dataclass(frozen=True)
class EventizedFormula:
    """A formula split into bounded event islands and a RuleRunner skeleton."""

    original: Node
    atoms: tuple[str, ...]
    skeleton: str
    islands: tuple[BoundedIsland, ...]
    horizon: int
    empty_value: bool


_BINARY_SYMBOL = {
    Op.AND: "&",
    Op.OR: "|",
    Op.IMPLIES: "->",
    Op.UNTIL: "U",
    Op.RELEASE: "R",
}

_UNARY_SYMBOL = {
    Op.EVENTUALLY: "F",
    Op.ALWAYS: "G",
    Op.NEXT: "X",
    Op.WEAK_NEXT: "WX",
}


def _horizon(node: Node, memo: dict[str, int | None]) -> int | None:
    cached = memo.get(node.key)
    if node.key in memo:
        return cached

    if node.op is Op.ATOM:
        result: int | None = 0
    elif node.op is Op.NOT:
        result = _horizon(node.children[0], memo)
    elif node.op in (Op.AND, Op.OR, Op.IMPLIES):
        children = [_horizon(child, memo) for child in node.children]
        result = None if any(value is None for value in children) else max(children)
    elif node.op in (Op.NEXT, Op.WEAK_NEXT):
        child_horizon = _horizon(node.children[0], memo)
        result = None if child_horizon is None else 1 + child_horizon
    else:
        # Standard F/G/U/R have no syntactic finite bound.  Semantic
        # simplification may prove special cases bounded, but that belongs in a
        # normalization pass rather than this deliberately transparent test.
        for child in node.children:
            _horizon(child, memo)
        result = None

    memo[node.key] = result
    return result


def bounded_horizon(formula: str | Node) -> int | None:
    """Return the syntactic future horizon, or ``None`` when it is unbounded."""
    node = parse(formula) if isinstance(formula, str) else formula
    return _horizon(node, {})


def _fresh_event(existing_atoms: set[str], index: int) -> str:
    while True:
        candidate = f"rr_bounded_event_{index}"
        if candidate not in existing_atoms:
            existing_atoms.add(candidate)
            return candidate
        index += 1


def _atoms(node: Node) -> set[str]:
    return {
        sub.atom
        for sub in node.subformulae()
        if sub.op is Op.ATOM and sub.atom not in ("true", "false")
    }


def _holds_empty(node: Node) -> bool:
    if node.op is Op.ATOM:
        return node.atom == "true"
    if node.op is Op.NOT:
        return not _holds_empty(node.children[0])
    if node.op is Op.AND:
        return _holds_empty(node.children[0]) and _holds_empty(node.children[1])
    if node.op is Op.OR:
        return _holds_empty(node.children[0]) or _holds_empty(node.children[1])
    if node.op is Op.IMPLIES:
        return not _holds_empty(node.children[0]) or _holds_empty(node.children[1])
    if node.op in (Op.NEXT, Op.EVENTUALLY, Op.UNTIL):
        return False
    if node.op in (Op.WEAK_NEXT, Op.ALWAYS, Op.RELEASE):
        return True
    raise ValueError(node.op)


def eventize_bounded_islands(formula: str) -> EventizedFormula:
    """Replace maximal non-atomic bounded islands with fresh propositions.

    Atomic propositions are already suitable event inputs and are left intact.
    Equal bounded subformulas share one derived event, matching their equal
    truth value at a fixed position without confusing different positions.
    """
    root = parse(formula)
    horizons: dict[str, int | None] = {}
    _horizon(root, horizons)
    existing_atoms = _atoms(root)
    islands_by_key: dict[str, BoundedIsland] = {}

    def rewrite(node: Node) -> str:
        horizon = horizons[node.key]
        if horizon is not None and node.op is not Op.ATOM:
            island = islands_by_key.get(node.key)
            if island is None:
                event = _fresh_event(existing_atoms, len(islands_by_key))
                island = BoundedIsland(event, node, horizon)
                islands_by_key[node.key] = island
            return island.event

        if node.op is Op.ATOM:
            assert node.atom is not None
            return node.atom
        if node.op is Op.NOT:
            return f"!({rewrite(node.children[0])})"
        if node.op in _UNARY_SYMBOL:
            return f"{_UNARY_SYMBOL[node.op]} ({rewrite(node.children[0])})"
        if node.op in _BINARY_SYMBOL:
            left, right = node.children
            return f"({rewrite(left)} {_BINARY_SYMBOL[node.op]} {rewrite(right)})"
        raise ValueError(node.op)

    skeleton = rewrite(root)
    islands = tuple(islands_by_key.values())
    return EventizedFormula(
        original=root,
        atoms=tuple(sorted(_atoms(root))),
        skeleton=skeleton,
        islands=islands,
        horizon=max((island.horizon for island in islands), default=0),
        empty_value=_holds_empty(root),
    )


def _eval_bounded(node: Node, word: list[Observation], position: int) -> bool:
    if node.op is Op.ATOM:
        if node.atom == "true":
            return True
        if node.atom == "false":
            return False
        assert node.atom is not None
        return word[position].get(node.atom, False)
    if node.op is Op.NOT:
        return not _eval_bounded(node.children[0], word, position)
    if node.op is Op.AND:
        return _eval_bounded(node.children[0], word, position) and _eval_bounded(
            node.children[1], word, position
        )
    if node.op is Op.OR:
        return _eval_bounded(node.children[0], word, position) or _eval_bounded(
            node.children[1], word, position
        )
    if node.op is Op.IMPLIES:
        return not _eval_bounded(node.children[0], word, position) or _eval_bounded(
            node.children[1], word, position
        )
    if node.op is Op.NEXT:
        return position + 1 < len(word) and _eval_bounded(
            node.children[0], word, position + 1
        )
    if node.op is Op.WEAK_NEXT:
        return position + 1 >= len(word) or _eval_bounded(
            node.children[0], word, position + 1
        )
    raise ValueError(f"Formula {node.key!r} is not a bounded event island.")


def evaluate_bounded(
    formula: str | Node,
    trace: Iterable[Observation],
    position: int = 0,
) -> bool:
    """Evaluate a syntactically bounded formula at one valid trace position."""
    node = parse(formula) if isinstance(formula, str) else formula
    if bounded_horizon(node) is None:
        raise ValueError(f"Formula {node.key!r} has unbounded future horizon.")
    word = list(trace)
    if not 0 <= position < len(word):
        raise IndexError("bounded evaluation requires a position inside the trace")
    return _eval_bounded(node, word, position)


def derive_event_cell(
    eventized: EventizedFormula,
    suffix: Iterable[Observation],
) -> Observation:
    """Evaluate one derived cell at the first position of a non-empty suffix."""
    word = list(suffix)
    if not word:
        raise ValueError("a derived event cell requires a non-empty suffix")
    cell = dict(word[0])
    for island in eventized.islands:
        cell[island.event] = _eval_bounded(island.formula, word, 0)
    return cell


class BoundedEventPipeline:
    """Fixed-window transducer producing the delayed RuleRunner input word.

    ``push`` emits either zero or one derived cells.  ``finish`` supplies the
    finite-trace boundary and flushes the remaining suffix cells.  The buffer
    never contains more than ``H + 1`` observations.
    """

    def __init__(self, eventized: EventizedFormula) -> None:
        self.eventized = eventized
        self._window: deque[Observation] = deque()
        self._finished = False

    def reset(self) -> None:
        self._window.clear()
        self._finished = False

    def push(self, obs: Observation) -> Observation | None:
        if self._finished:
            raise RuntimeError("cannot push after finish(); call reset() first")
        self._window.append(dict(obs))
        if len(self._window) <= self.eventized.horizon:
            return None
        word = list(self._window)
        cell = derive_event_cell(self.eventized, word)
        self._window.popleft()
        return cell

    def finish(self) -> tuple[Observation, ...]:
        if self._finished:
            return ()
        self._finished = True
        cells: list[Observation] = []
        while self._window:
            word = list(self._window)
            cells.append(derive_event_cell(self.eventized, word))
            self._window.popleft()
        return tuple(cells)


def derive_event_trace(
    eventized: EventizedFormula,
    trace: Iterable[Observation],
) -> tuple[Observation, ...]:
    """Produce the complete derived word through the streaming pipeline."""
    pipeline = BoundedEventPipeline(eventized)
    output: list[Observation] = []
    for obs in trace:
        cell = pipeline.push(obs)
        if cell is not None:
            output.append(cell)
    output.extend(pipeline.finish())
    return tuple(output)
