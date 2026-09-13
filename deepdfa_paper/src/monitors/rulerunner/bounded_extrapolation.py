"""Exact three-valued extrapolation for the bounded-event pipeline.

Finkbeiner--Kuhtz extrapolate across the part of a trace still resident in the
bounded-future pipeline.  This module adapts that principle to this project's
LTL3-style monitor contract: a prefix is SATISFY or VIOLATE only when every
finite continuation has that value.

The delayed RuleRunner carry, its last evaluated cell, and the bounded input
window form a finite deterministic machine.  We enumerate its reachable graph,
classify states by reverse reachability from accepting/rejecting end states,
and compile the exact labels into one fixed CILP cube readout.  Recurrence still
belongs to the bounded pipeline and certified RuleRunner skeleton; the global
graph is used only for exact online labels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product

import torch

from src.formula.compiler import Observation
from src.monitors.base import Verdict
from src.monitors.rulerunner.bounded import EventizedFormula, derive_event_cell
from src.monitors.rulerunner.cilp import (
    SkeletonLabelState,
    _layer_matrices,
    _step_activation,
)
from src.monitors.rulerunner.engine import RuleEngine, RuleEngineState
from src.monitors.rulerunner.rules import Literal, Rule

_Layer = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

#: Compilation guard for the reachable composite graph.  This enumeration is
#: exponential in ``H * |atoms|``; an unbounded default would let a harmless
#: looking formula hang the compiler instead of failing loudly.
DEFAULT_MAX_STATES = 20_000


class ExtrapolationLimitExceeded(ValueError):
    """Exact extrapolation exceeded an explicit compilation guard."""


@dataclass(frozen=True)
class _PipelineState:
    active: frozenset[Literal]
    last_cell: frozenset[Literal]
    history: tuple[frozenset[str], ...]
    original_seen: bool
    skeleton_seen: bool
    decided: Verdict | None


def _literal_names(engine: RuleEngine) -> tuple[str, ...]:
    rules = engine.rule_system
    names = {literal.name for literal in rules.initial_state}
    for rule in rules.eval_rules + rules.react_rules:
        names.update(literal.name for literal in rule.body)
        names.add(rule.head.name)
    names.update(f"obs:{atom}" for atom in rules.atoms)
    names.add(f"[{rules.root_key}]T")
    names.add(f"[{rules.root_key}]F")
    return tuple(sorted(names))


def _observations(atoms: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(atom for atom, value in zip(atoms, values) if value)
        for values in product((False, True), repeat=len(atoms))
    )


def _as_observation(cell: frozenset[str]) -> Observation:
    return {atom: True for atom in cell}


class BoundedExtrapolationCILP:
    """Exact sink/trap readout for one bounded-event construction."""

    def __init__(
        self,
        eventized: EventizedFormula,
        device: torch.device,
        *,
        max_atoms: int = 12,
        max_states: int | None = DEFAULT_MAX_STATES,
    ) -> None:
        self.eventized = eventized
        self.device = device
        self.max_atoms = max_atoms
        self.max_states = max_states
        if len(eventized.atoms) > max_atoms:
            raise ExtrapolationLimitExceeded(
                f"Exact bounded extrapolation for {eventized.original.key!r} "
                f"requires enumerating 2^{len(eventized.atoms)} observations; "
                f"max_atoms={max_atoms}."
            )

        self._engine = RuleEngine.from_formula(eventized.skeleton)
        self.literal_names = _literal_names(self._engine)
        self.states, self.transitions = self._reachable_graph()
        self.accepting = frozenset(
            state for state in self.states if self._accepts_at_end(state)
        )
        self.accepting_sinks, self.traps = self._classify()
        self._compile_readout()

    @property
    def n_states(self) -> int:
        return len(self.states)

    def _initial_state(self) -> _PipelineState:
        self._engine.reset()
        initial = self._engine.state().active
        return _PipelineState(
            active=initial,
            last_cell=initial,
            history=(),
            original_seen=False,
            skeleton_seen=False,
            decided=None,
        )

    def _load_engine(self, state: _PipelineState) -> None:
        self._engine.load_state(
            RuleEngineState(
                active=state.active,
                last_cell=state.last_cell,
                seen_cell=state.skeleton_seen,
                decided=state.decided,
            )
        )

    def _advance(
        self,
        state: _PipelineState,
        symbol: frozenset[str],
    ) -> _PipelineState:
        if state.decided is not None:
            return state

        self._load_engine(state)
        history = state.history + (symbol,)
        if len(history) > self.eventized.horizon:
            suffix = tuple(_as_observation(cell) for cell in history)
            self._engine.step(derive_event_cell(self.eventized, suffix))
            if self.eventized.horizon:
                history = history[-self.eventized.horizon :]
            else:
                history = ()

        engine_state = self._engine.state()
        return _PipelineState(
            active=engine_state.active,
            last_cell=engine_state.last_cell,
            history=history,
            original_seen=True,
            skeleton_seen=engine_state.seen_cell,
            decided=engine_state.decided,
        )

    def _accepts_at_end(self, state: _PipelineState) -> bool:
        if not state.original_seen:
            return self.eventized.empty_value

        self._load_engine(state)
        for start in range(len(state.history)):
            suffix = tuple(
                _as_observation(cell) for cell in state.history[start:]
            )
            verdict = self._engine.step(derive_event_cell(self.eventized, suffix))
            if verdict is not Verdict.UNDECIDED:
                break
        return self._engine.final_verdict() is Verdict.SATISFY

    def _reachable_graph(
        self,
    ) -> tuple[
        tuple[_PipelineState, ...],
        dict[_PipelineState, tuple[_PipelineState, ...]],
    ]:
        alphabet = _observations(self.eventized.atoms)
        initial = self._initial_state()
        queue = deque([initial])
        seen = {initial}
        edges: dict[_PipelineState, tuple[_PipelineState, ...]] = {}

        while queue:
            state = queue.popleft()
            successors = tuple(self._advance(state, symbol) for symbol in alphabet)
            edges[state] = successors
            for successor in successors:
                if successor in seen:
                    continue
                if self.max_states is not None and len(seen) >= self.max_states:
                    raise ExtrapolationLimitExceeded(
                        "Exact bounded extrapolation exceeded "
                        f"max_states={self.max_states}."
                    )
                seen.add(successor)
                queue.append(successor)
        return tuple(seen), edges

    def _backward_reachable(
        self,
        targets: set[_PipelineState],
    ) -> set[_PipelineState]:
        reverse: dict[_PipelineState, set[_PipelineState]] = {
            state: set() for state in self.states
        }
        for source, successors in self.transitions.items():
            for target in successors:
                reverse[target].add(source)

        reachable = set(targets)
        queue = deque(targets)
        while queue:
            state = queue.popleft()
            for predecessor in reverse[state]:
                if predecessor not in reachable:
                    reachable.add(predecessor)
                    queue.append(predecessor)
        return reachable

    def _classify(
        self,
    ) -> tuple[frozenset[_PipelineState], frozenset[_PipelineState]]:
        accepting = set(self.accepting)
        rejecting = set(self.states) - accepting
        can_accept = self._backward_reachable(accepting)
        can_reject = self._backward_reachable(rejecting)
        accepting_sinks = set(self.states) - can_reject
        traps = set(self.states) - can_accept
        return frozenset(accepting_sinks), frozenset(traps)

    def _features(self) -> tuple[str, ...]:
        names = ["origseen", "skseen", "dec:sat", "dec:vio"]
        names.extend(f"fill:{fill}" for fill in range(self.eventized.horizon + 1))
        names.extend(
            f"hist:{atom}@{age}"
            for age in range(self.eventized.horizon)
            for atom in self.eventized.atoms
        )
        names.extend(f"carry:{name}" for name in self.literal_names)
        names.extend(f"last:{name}" for name in self.literal_names)
        return tuple(names)

    def _true_features(self, state: _PipelineState) -> set[str]:
        true: set[str] = {f"fill:{len(state.history)}"}
        if state.original_seen:
            true.add("origseen")
        if state.skeleton_seen:
            true.add("skseen")
        if state.decided is Verdict.SATISFY:
            true.add("dec:sat")
        elif state.decided is Verdict.VIOLATE:
            true.add("dec:vio")

        for age, cell in enumerate(reversed(state.history)):
            true.update(f"hist:{atom}@{age}" for atom in cell)
        true.update(f"carry:{literal.name}" for literal in state.active)
        true.update(f"last:{literal.name}" for literal in state.last_cell)
        return true

    def _compile_readout(self) -> None:
        self.features = self._features()
        all_names = set(self.features) | {"label:T", "label:F"}
        self.index = {name: i for i, name in enumerate(sorted(all_names))}
        rules: list[Rule] = []
        feature_set = set(self.features)
        for state in self.states:
            if state in self.accepting_sinks:
                head = Literal("label:T")
            elif state in self.traps:
                head = Literal("label:F")
            else:
                continue
            true = self._true_features(state)
            body = frozenset(
                Literal(name, negated=name not in true) for name in feature_set
            )
            rules.append(Rule(body, head))

        self.layer: _Layer = tuple(
            tensor.to(self.device)
            for tensor in _layer_matrices(tuple(rules), self.index, len(self.index))
        )
        self._T_index = self.index["label:T"]
        self._F_index = self.index["label:F"]

    def evaluate(
        self,
        *,
        history: torch.Tensor,
        fill: int,
        original_seen: bool,
        skeleton: SkeletonLabelState,
    ) -> Verdict:
        """Classify one live neural pipeline state with the compiled CILP head."""
        x = torch.full((len(self.index),), -1.0, device=self.device)

        def enable(name: str) -> None:
            x[self.index[name]] = 1.0

        enable(f"fill:{fill}")
        if original_seen:
            enable("origseen")
        if skeleton.seen_cell:
            enable("skseen")
        if skeleton.decided is Verdict.SATISFY:
            enable("dec:sat")
        elif skeleton.decided is Verdict.VIOLATE:
            enable("dec:vio")

        for age in range(fill):
            for atom_index, atom in enumerate(self.eventized.atoms):
                if bool(history[age, atom_index] > 0):
                    enable(f"hist:{atom}@{age}")

        for name in skeleton.active:
            enable(f"carry:{name}")
        for name in skeleton.last_cell:
            enable(f"last:{name}")

        W_ih, b_h, W_ho, b_o = self.layer
        hidden = _step_activation(W_ih @ x + b_h)
        output = _step_activation(W_ho @ hidden + b_o)
        if bool(output[self._T_index] > 0):
            return Verdict.SATISFY
        if bool(output[self._F_index] > 0):
            return Verdict.VIOLATE
        return Verdict.UNDECIDED
