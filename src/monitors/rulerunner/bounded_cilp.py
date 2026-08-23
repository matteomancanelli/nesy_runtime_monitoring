"""CILP realizations of the bounded-event RuleRunner construction.

Both public monitors implement the semantics in :mod:`.bounded`:

* a recurrent CILP layer stores the last ``H`` observations;
* bounded islands are unrolled over the available window and evaluated by
  monotone two-valued CILP rules;
* the resulting event cell is consumed by an exactly-certified old RuleRunner
  skeleton, using either its flat (IJCNN 2014) or structured (IJCNN 2015)
  realization;
* *optionally* (``exact_online=True``) a fixed CILP extrapolation head labels
  the complete delayed-pipeline state with exact accepting-sink/trap verdicts.

The extrapolation head is opt-in because it is not free: it enumerates the
reachable composite graph over ``2^|P|`` and compiles one hidden unit per
reachable state, which on the bounded-response family is already larger than
the canonical DFA it avoids constructing (see
``docs/bounded_event_rulerunner.md``).  The default configuration is the
construction the theorem describes: online verdicts come from the certified
skeleton alone and are sound, but their lag relative to the exact label is
unbounded even for a fixed ``H``.
Final verdicts are exact either way.

Certification is a *lookup*, not a computation: ``certificates.py`` stores the
exact product-check results offline, so compiling a bounded monitor does not
secretly build a DFA.  ``certificate='cached'`` enforces that.

The flat event evaluator pools all unrolled rules and iterates to a fixpoint.
The structured evaluator owns one addressable layer per ``(subformula, offset)``
and executes them in postorder.  The two differ only in rule grouping.
``batch_run`` groups all equal-length suffix windows into batched tensor passes
and then invokes the corresponding batched skeleton runner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.rulerunner.bounded import (
    EventizedFormula,
    eventize_bounded_islands,
)
from src.monitors.rulerunner.bounded_extrapolation import (
    DEFAULT_MAX_STATES,
    BoundedExtrapolationCILP,
)
from src.monitors.rulerunner.certificates import (
    Certificate,
    ensure_certificate,
)
from src.monitors.rulerunner.cilp import (
    CILPRunner,
    _layer_matrices,
    _step_activation,
)
from src.monitors.rulerunner.parse_tree import Node, Op
from src.monitors.rulerunner.rules import Literal, Rule
from src.monitors.rulerunner.structured import StructuredCILPRunner

_Layer = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_Owner = tuple[str, int]


class UnsafeRuleRunnerSkeleton(ValueError):
    """The unbounded skeleton is outside old RuleRunner's certified class."""

    def __init__(self, result: Certificate) -> None:
        self.result = result
        witness = result.language_witness or result.unsound_prefix_witness
        detail = ""
        if witness is not None:
            detail = f" Shortest witness: {list(witness.observations)!r}."
        super().__init__(
            f"Bounded-event skeleton {result.formula!r} is not certified for "
            f"old RuleRunner.{detail}"
        )


def _t(node: Node, offset: int) -> Literal:
    return Literal(f"B[{node.key}]@{offset}T")


def _f(node: Node, offset: int) -> Literal:
    return Literal(f"B[{node.key}]@{offset}F")


def _rules_for(node: Node, offset: int, length: int) -> tuple[Rule, ...]:
    """Two-valued monotone rules for one unrolled bounded-formula node."""
    T = _t(node, offset)
    F = _f(node, offset)
    if node.op is Op.ATOM:
        return ()  # clamped from the observation window

    if node.op is Op.NOT:
        child = node.children[0]
        return (
            Rule(frozenset({_t(child, offset)}), F),
            Rule(frozenset({_f(child, offset)}), T),
        )

    if node.op is Op.AND:
        left, right = node.children
        return (
            Rule(frozenset({_t(left, offset), _t(right, offset)}), T),
            Rule(frozenset({_f(left, offset)}), F),
            Rule(frozenset({_f(right, offset)}), F),
        )

    if node.op is Op.OR:
        left, right = node.children
        return (
            Rule(frozenset({_t(left, offset)}), T),
            Rule(frozenset({_t(right, offset)}), T),
            Rule(frozenset({_f(left, offset), _f(right, offset)}), F),
        )

    if node.op is Op.IMPLIES:
        left, right = node.children
        return (
            Rule(frozenset({_f(left, offset)}), T),
            Rule(frozenset({_t(right, offset)}), T),
            Rule(frozenset({_t(left, offset), _f(right, offset)}), F),
        )

    if node.op in (Op.NEXT, Op.WEAK_NEXT):
        if offset + 1 < length:
            child = node.children[0]
            return (
                Rule(frozenset({_t(child, offset + 1)}), T),
                Rule(frozenset({_f(child, offset + 1)}), F),
            )
        boundary = F if node.op is Op.NEXT else T
        return (Rule(frozenset(), boundary),)

    raise ValueError(f"Unbounded node {node.key!r} reached the event evaluator.")


def _postorder(
    node: Node,
    offset: int,
    length: int,
    seen: set[_Owner],
    owners: list[tuple[Node, int]],
) -> None:
    owner = (node.key, offset)
    if owner in seen:
        return
    seen.add(owner)
    if node.op in (Op.NEXT, Op.WEAK_NEXT):
        if offset + 1 < length:
            _postorder(node.children[0], offset + 1, length, seen, owners)
    else:
        for child in node.children:
            _postorder(child, offset, length, seen, owners)
    owners.append((node, offset))


@dataclass(frozen=True)
class _CompiledWindow:
    length: int
    index: dict[str, int]
    owners: tuple[tuple[Node, int], ...]
    flat_layer: _Layer | None
    structured_layers: dict[_Owner, _Layer]
    atom_slots: tuple[tuple[int, str, int, int], ...]
    event_slots: tuple[int, ...]


class _BoundedEventEvaluator:
    """All fixed CILP event-evaluation circuits for window lengths 1..H+1."""

    def __init__(
        self,
        eventized: EventizedFormula,
        device: torch.device,
        *,
        structured: bool,
    ) -> None:
        self.eventized = eventized
        self.device = device
        self.structured = structured
        self.atom_index = {atom: i for i, atom in enumerate(eventized.atoms)}
        self.windows = {
            length: self._compile_window(length)
            for length in range(1, eventized.horizon + 2)
        }

    def _compile_window(self, length: int) -> _CompiledWindow:
        owners: list[tuple[Node, int]] = []
        seen: set[_Owner] = set()
        for island in self.eventized.islands:
            _postorder(island.formula, 0, length, seen, owners)

        rules_by_owner: dict[_Owner, tuple[Rule, ...]] = {}
        rules: list[Rule] = []
        literal_names: set[str] = set()
        for node, offset in owners:
            literal_names.update({_t(node, offset).name, _f(node, offset).name})
            owned = _rules_for(node, offset, length)
            rules_by_owner[(node.key, offset)] = owned
            rules.extend(owned)
            for rule in owned:
                literal_names.update(literal.name for literal in rule.body)
                literal_names.add(rule.head.name)

        index = {name: i for i, name in enumerate(sorted(literal_names))}
        width = len(index)
        flat: _Layer | None = None
        structured_layers: dict[_Owner, _Layer] = {}
        if self.structured:
            structured_layers = {
                owner: tuple(
                    tensor.to(self.device)
                    for tensor in _layer_matrices(owned, index, width)
                )
                for owner, owned in rules_by_owner.items()
            }
        else:
            flat = tuple(
                tensor.to(self.device)
                for tensor in _layer_matrices(tuple(rules), index, width)
            )

        atom_slots: list[tuple[int, str, int, int]] = []
        for node, offset in owners:
            if node.op is not Op.ATOM:
                continue
            atom_slots.append(
                (
                    offset,
                    node.atom or "",
                    index[_t(node, offset).name],
                    index[_f(node, offset).name],
                )
            )
        event_slots = tuple(
            index[_t(island.formula, 0).name]
            for island in self.eventized.islands
        )
        return _CompiledWindow(
            length=length,
            index=index,
            owners=tuple(owners),
            flat_layer=flat,
            structured_layers=structured_layers,
            atom_slots=tuple(atom_slots),
            event_slots=event_slots,
        )

    @staticmethod
    def _forward(layer: _Layer, x: torch.Tensor) -> torch.Tensor:
        W_ih, b_h, W_ho, b_o = layer
        hidden = _step_activation(x @ W_ih.t() + b_h)
        return _step_activation(hidden @ W_ho.t() + b_o)

    def evaluate(self, window: torch.Tensor) -> torch.Tensor:
        """Evaluate ``(length, n_atoms)`` bipolar windows; return event bits."""
        return self.evaluate_batch(window.unsqueeze(0))[0]

    def evaluate_batch(self, windows: torch.Tensor) -> torch.Tensor:
        """Evaluate ``(batch, length, n_atoms)`` windows in one tensor pass."""
        length = windows.shape[1]
        compiled = self.windows[length]
        batch = windows.shape[0]
        x = torch.full(
            (batch, len(compiled.index)),
            -1.0,
            device=self.device,
        )
        for offset, atom, t_index, f_index in compiled.atom_slots:
            if atom == "true":
                value = torch.ones(batch, device=self.device)
            elif atom == "false":
                value = torch.full((batch,), -1.0, device=self.device)
            else:
                value = windows[:, offset, self.atom_index[atom]]
            x[:, t_index] = value
            x[:, f_index] = -value

        if self.structured:
            for node, offset in compiled.owners:
                layer = compiled.structured_layers[(node.key, offset)]
                x = torch.maximum(x, self._forward(layer, x))
        else:
            assert compiled.flat_layer is not None
            for _ in range(len(compiled.owners) + 1):
                nxt = torch.maximum(x, self._forward(compiled.flat_layer, x))
                if torch.equal(nxt, x):
                    break
                x = nxt

        if not compiled.event_slots:
            return torch.empty((batch, 0), device=self.device)
        slots = torch.tensor(compiled.event_slots, dtype=torch.long, device=self.device)
        return x.index_select(1, slots)


class _ObservationShiftCILP:
    """One recurrent CILP layer implementing an ``H``-cell observation buffer."""

    def __init__(
        self,
        atoms: tuple[str, ...],
        horizon: int,
        device: torch.device,
    ) -> None:
        self.atoms = atoms
        self.horizon = horizon
        self.device = device
        names = [f"obs:{atom}" for atom in atoms]
        names.extend(
            f"hist:{atom}@{age}" for age in range(horizon) for atom in atoms
        )
        self.index = {name: i for i, name in enumerate(names)}
        rules: list[Rule] = []
        if horizon > 0:
            for atom in atoms:
                rules.append(
                    Rule(
                        frozenset({Literal(f"obs:{atom}")}),
                        Literal(f"hist:{atom}@0"),
                    )
                )
                for age in range(1, horizon):
                    rules.append(
                        Rule(
                            frozenset({Literal(f"hist:{atom}@{age - 1}")}),
                            Literal(f"hist:{atom}@{age}"),
                        )
                    )
        self.layer = tuple(
            tensor.to(device)
            for tensor in _layer_matrices(tuple(rules), self.index, len(self.index))
        )
        self.obs_slots = torch.tensor(
            [self.index[f"obs:{atom}"] for atom in atoms],
            dtype=torch.long,
            device=device,
        )
        self.history_slots = torch.tensor(
            [
                self.index[f"hist:{atom}@{age}"]
                for age in range(horizon)
                for atom in atoms
            ],
            dtype=torch.long,
            device=device,
        )

    def initial(self) -> torch.Tensor:
        return torch.full(
            (self.horizon, len(self.atoms)), -1.0, device=self.device
        )

    def advance(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if self.horizon == 0:
            return history
        x = torch.full((len(self.index),), -1.0, device=self.device)
        if len(self.atoms):
            x[self.obs_slots] = current
            x[self.history_slots] = history.flatten()
        W_ih, b_h, W_ho, b_o = self.layer
        hidden = _step_activation(W_ih @ x + b_h)
        output = _step_activation(W_ho @ hidden + b_o)
        return output.index_select(0, self.history_slots).reshape_as(history)


class _BoundedEventCILPMonitor(Monitor):
    """Shared execution schedule; subclasses choose the CILP organization."""

    _structured = False
    # The timing harness merges these defaults into compile().  A cache miss
    # must fail before a benchmark rather than silently constructing a DFA.
    benchmark_compile_kwargs = {"certificate": "cached"}
    benchmark_certificate_source = "cache"

    def __init__(
        self,
        formula: str,
        device: str | torch.device = "cpu",
        *,
        exact_online: bool = False,
        certificate: str = "auto",
        max_extrapolation_atoms: int = 12,
        max_extrapolation_states: int | None = DEFAULT_MAX_STATES,
    ) -> None:
        self.formula = formula
        self.exact_online = exact_online
        self.eventized = eventize_bounded_islands(formula)
        self.certificate, self.certificate_source = ensure_certificate(
            self.eventized.skeleton, certificate
        )
        if (
            not self.certificate.language_equivalent
            or not self.certificate.prefix_sound
        ):
            raise UnsafeRuleRunnerSkeleton(self.certificate)

        self._device = torch.device(device)
        self.event_net = _BoundedEventEvaluator(
            self.eventized,
            self._device,
            structured=self._structured,
        )
        self.shift_net = _ObservationShiftCILP(
            self.eventized.atoms,
            self.eventized.horizon,
            self._device,
        )
        if self._structured:
            self.skeleton = StructuredCILPRunner.from_formula(
                self.eventized.skeleton,
                device=self._device,
            )
        else:
            self.skeleton = CILPRunner.from_formula(
                self.eventized.skeleton,
                device=self._device,
            )
        self.extrapolation = (
            BoundedExtrapolationCILP(
                self.eventized,
                self._device,
                max_atoms=max_extrapolation_atoms,
                max_states=max_extrapolation_states,
            )
            if exact_online
            else None
        )
        self._event_index = {
            island.event: i for i, island in enumerate(self.eventized.islands)
        }
        self._atom_index = {
            atom: i for i, atom in enumerate(self.eventized.atoms)
        }
        self.reset()

    @classmethod
    def compile(
        cls,
        formula: str,
        device: str | torch.device = "cpu",
        *,
        exact_online: bool = False,
        certificate: str = "auto",
        max_extrapolation_atoms: int = 12,
        max_extrapolation_states: int | None = DEFAULT_MAX_STATES,
    ) -> "_BoundedEventCILPMonitor":
        return cls(
            formula,
            device=device,
            exact_online=exact_online,
            certificate=certificate,
            max_extrapolation_atoms=max_extrapolation_atoms,
            max_extrapolation_states=max_extrapolation_states,
        )

    @property
    def effective_device(self) -> str:
        return "cuda" if self._device.type == "cuda" else "cpu"

    def reset(self) -> None:
        self._history = self.shift_net.initial()
        self._fill = 0
        self._seen_input = False
        self._decided: Verdict | None = None
        self.skeleton.reset()

    def _observation_tensor(self, obs: Observation) -> torch.Tensor:
        values = torch.full(
            (len(self.eventized.atoms),), -1.0, device=self._device
        )
        for atom, index in self._atom_index.items():
            if obs.get(atom, False):
                values[index] = 1.0
        return values

    def _derived_cell(
        self,
        window: torch.Tensor,
        events: torch.Tensor,
    ) -> Observation:
        oldest = window[0]
        cell: Observation = {}
        for atom in self.skeleton._rs.atoms:
            event_index = self._event_index.get(atom)
            if event_index is not None:
                value = events[event_index]
            else:
                value = oldest[self._atom_index[atom]]
            if bool(value > 0):
                cell[atom] = True
        return cell

    def _consume_window(self, window: torch.Tensor) -> Verdict:
        events = self.event_net.evaluate(window)
        verdict = self.skeleton.step(self._derived_cell(window, events))
        if verdict is not Verdict.UNDECIDED:
            self._decided = verdict
        return verdict

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        self._seen_input = True
        current = self._observation_tensor(obs)
        verdict = Verdict.UNDECIDED
        if self.eventized.horizon == 0:
            verdict = self._consume_window(current.unsqueeze(0))
        elif self._fill == self.eventized.horizon:
            window = torch.cat((torch.flip(self._history, dims=(0,)), current[None]))
            verdict = self._consume_window(window)

        self._history = self.shift_net.advance(self._history, current)
        self._fill = min(self.eventized.horizon, self._fill + 1)
        if verdict is not Verdict.UNDECIDED or self.extrapolation is None:
            # Without the extrapolation head the label is the certified
            # skeleton's own, which is sound but may lag arbitrarily even
            # when this formula's horizon H is fixed.
            return verdict
        verdict = self.extrapolation.evaluate(
            history=self._history,
            fill=self._fill,
            original_seen=self._seen_input,
            skeleton=self.skeleton.label_state(self.extrapolation.literal_names),
        )
        if verdict is not Verdict.UNDECIDED:
            self._decided = verdict
        return verdict

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        if not self._seen_input:
            return (
                Verdict.SATISFY
                if self.eventized.empty_value
                else Verdict.VIOLATE
            )

        for length in range(self._fill, 0, -1):
            newest_first = self._history[:length]
            verdict = self._consume_window(torch.flip(newest_first, dims=(0,)))
            if verdict is not Verdict.UNDECIDED:
                return verdict
        return self.skeleton.final_verdict()

    def batch_run(
        self,
        traces: Iterable[Iterable[Observation]],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Fuse event windows, then use the skeleton's batched CILP runner."""
        trace_list = [list(trace) for trace in traces]
        if not trace_list:
            return []

        results: list[Verdict | None] = [None] * len(trace_list)
        nonempty_indices = [i for i, trace in enumerate(trace_list) if trace]
        for index, trace in enumerate(trace_list):
            if not trace:
                results[index] = (
                    Verdict.SATISFY
                    if self.eventized.empty_value
                    else Verdict.VIOLATE
                )
        if not nonempty_indices:
            return [result for result in results if result is not None]

        tensors = {
            index: torch.stack(
                [self._observation_tensor(obs) for obs in trace_list[index]]
            )
            for index in nonempty_indices
        }
        derived: dict[int, list[Observation]] = {}
        skeleton_atoms = self.skeleton._rs.atoms
        event_atoms = set(self._event_index)
        for index in nonempty_indices:
            cells: list[Observation] = []
            for obs in trace_list[index]:
                cells.append(
                    {
                        atom: True
                        for atom in skeleton_atoms
                        if atom not in event_atoms and obs.get(atom, False)
                    }
                )
            derived[index] = cells

        buckets: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
        if self.eventized.islands:
            for index in nonempty_indices:
                tensor = tensors[index]
                for position in range(len(trace_list[index])):
                    length = min(
                        self.eventized.horizon + 1,
                        len(trace_list[index]) - position,
                    )
                    buckets.setdefault(length, []).append(
                        (index, position, tensor[position : position + length])
                    )

            for records in buckets.values():
                windows = torch.stack([record[2] for record in records])
                event_values = self.event_net.evaluate_batch(windows)
                for row, (index, position, _) in enumerate(records):
                    for event, event_index in self._event_index.items():
                        if bool(event_values[row, event_index] > 0):
                            derived[index][position][event] = True

        derived_batch = [derived[index] for index in nonempty_indices]
        verdicts = self.skeleton.batch_run(
            derived_batch,
            early_termination=early_termination,
        )
        for index, verdict in zip(nonempty_indices, verdicts):
            results[index] = verdict
        return [result for result in results if result is not None]


class BoundedEventRuleRunnerMonitor(_BoundedEventCILPMonitor):
    """Flat CILP bounded-event evaluator plus flat RuleRunner skeleton."""


class BoundedEventStructuredRuleRunnerMonitor(_BoundedEventCILPMonitor):
    """Structured CILP event evaluator plus structured RuleRunner skeleton."""

    _structured = True
