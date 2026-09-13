"""Exact equivalence checking for the original RuleRunner construction.

The original monitor and the canonical LTLf DFA are both deterministic finite-
state machines.  Their language equivalence is therefore decidable by exploring
their synchronous product.  This module performs that exploration on demand and
returns shortest counterexamples (breadth-first-search order) when the old
RuleRunner is not correct for a formula.

This is deliberately a *formula certifier*, not a claim that the exact semantic
class has a convenient maximal grammar.  Syntactically different formulas can
collapse through LTLf identities, while an otherwise innocent repeated
subformula can alias one RuleRunner register at two trace positions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from src.formula.compiler import Observation, compile_ltlf
from src.monitors.base import Verdict
from src.monitors.rulerunner.engine import RuleEngine, RuleEngineState
from src.monitors.rulerunner.rules import Literal


@dataclass(frozen=True)
class EquivalenceWitness:
    """A shortest trace witnessing one kind of inequivalence."""

    trace: tuple[frozenset[str], ...]
    rulerunner: Verdict
    dfa: Verdict

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Return the sparse-observation representation used by monitors."""
        return tuple({atom: True for atom in cell} for cell in self.trace)


@dataclass(frozen=True)
class RuleRunnerEquivalenceResult:
    """Exact product-check result for one formula.

    ``language_witness`` compares final LTLf acceptance after every finite word.
    ``unsound_prefix_witness`` records an early definite RuleRunner verdict that
    disagrees with the DFA's exact sink/trap label.  ``online_witness`` is
    stricter: it also records cases where RuleRunner remains undecided after the
    DFA can already issue a permanent verdict.

    When ``complete`` is false, a user-supplied product-state limit interrupted
    exploration.  Witnesses already found remain valid, but their absence is not
    a certificate.
    """

    formula: str
    atoms: tuple[str, ...]
    explored_product_states: int
    complete: bool
    language_witness: EquivalenceWitness | None
    unsound_prefix_witness: EquivalenceWitness | None
    online_witness: EquivalenceWitness | None

    @property
    def language_equivalent(self) -> bool:
        return self.complete and self.language_witness is None

    @property
    def prefix_sound(self) -> bool:
        return self.complete and self.unsound_prefix_witness is None

    @property
    def online_equivalent(self) -> bool:
        return self.complete and self.online_witness is None


@dataclass(frozen=True)
class _RuleCarry:
    """The part of RuleEngine state that influences a future cell.

    ``RuleEngineState`` also carries ``last_cell`` and ``seen_cell``, which
    only ``final_verdict`` reads.  Because this exploration calls
    ``final_verdict`` immediately after every ``step`` — which rewrites both —
    they cannot distinguish two futures and are deliberately excluded from the
    product key.
    """

    active_rules: frozenset[Literal]
    decided: Verdict | None

    def as_engine_state(self) -> RuleEngineState:
        return RuleEngineState(
            active=self.active_rules,
            last_cell=frozenset(),  # invalid until the next step rewrites it
            seen_cell=self.decided is not None,
            decided=self.decided,
        )


_ProductState = tuple[_RuleCarry, int]


def _dfa_label(state: int, trap: frozenset[int], sink: frozenset[int]) -> Verdict:
    if state in trap:
        return Verdict.VIOLATE
    if state in sink:
        return Verdict.SATISFY
    return Verdict.UNDECIDED


def _symbols(atoms: tuple[str, ...]) -> tuple[tuple[frozenset[str], Observation], ...]:
    rows: list[tuple[frozenset[str], Observation]] = []
    for values in product((False, True), repeat=len(atoms)):
        true_atoms = frozenset(atom for atom, value in zip(atoms, values) if value)
        rows.append((true_atoms, {atom: True for atom in true_atoms}))
    return tuple(rows)


def _trace_to(
    state: _ProductState,
    parents: dict[_ProductState, tuple[_ProductState, frozenset[str]] | None],
) -> tuple[frozenset[str], ...]:
    reverse: list[frozenset[str]] = []
    cursor = state
    while parents[cursor] is not None:
        parent, symbol = parents[cursor]  # type: ignore[misc]
        reverse.append(symbol)
        cursor = parent
    reverse.reverse()
    return tuple(reverse)


def _advance_rule_engine(
    engine: RuleEngine, carry: _RuleCarry, obs: Observation
) -> tuple[_RuleCarry, Verdict, Verdict]:
    """Advance from an immutable carry while reusing RuleEngine's semantics."""
    engine.load_state(carry.as_engine_state())
    step_verdict = engine.step(obs)
    final_verdict = engine.final_verdict()
    state = engine.state()
    next_carry = _RuleCarry(state.active, state.decided)
    return next_carry, step_verdict, final_verdict


@lru_cache(maxsize=None)
def certify_rule_runner(
    formula: str,
    *,
    max_atoms: int = 12,
    max_product_states: int | None = None,
) -> RuleRunnerEquivalenceResult:
    """Decide whether old RuleRunner is correct for ``formula``.

    The default exploration has no state limit and is exact.  ``max_atoms`` is
    only a guard against accidentally enumerating an impractical alphabet; it
    does not approximate the result.  Set ``max_product_states`` to obtain an
    explicitly incomplete resource-bounded analysis.
    """
    engine = RuleEngine.from_formula(formula)
    dfa = compile_ltlf(formula)
    atoms = tuple(sorted(set(engine.atoms) | set(dfa.atoms)))
    if len(atoms) > max_atoms:
        raise ValueError(
            f"Equivalence checking {formula!r} requires enumerating "
            f"2^{len(atoms)} observations; max_atoms={max_atoms}."
        )

    initial_carry = _RuleCarry(engine.state().active, None)
    initial: _ProductState = (initial_carry, dfa.initial)
    queue = deque([initial])
    parents: dict[_ProductState, tuple[_ProductState, frozenset[str]] | None] = {
        initial: None
    }

    language_witness: EquivalenceWitness | None = None
    unsound_witness: EquivalenceWitness | None = None
    online_witness: EquivalenceWitness | None = None

    # The empty word has no transition edge, so compare it explicitly.
    rr_empty = engine.final_verdict()
    dfa_empty = Verdict.SATISFY if dfa.initial in dfa.accepting else Verdict.VIOLATE
    if rr_empty is not dfa_empty:
        language_witness = EquivalenceWitness((), rr_empty, dfa_empty)

    alphabet = _symbols(atoms)
    complete = True
    while queue:
        current = queue.popleft()
        carry, dfa_state = current
        prefix = _trace_to(current, parents)

        for symbol, obs in alphabet:
            next_carry, rr_step, rr_final = _advance_rule_engine(engine, carry, obs)
            next_dfa = dfa.step(dfa_state, obs)
            dfa_step = _dfa_label(
                next_dfa,
                dfa.trap_states,
                dfa.accepting_sinks,
            )
            dfa_final = (
                Verdict.SATISFY if next_dfa in dfa.accepting else Verdict.VIOLATE
            )
            trace = prefix + (symbol,)

            if language_witness is None and rr_final is not dfa_final:
                language_witness = EquivalenceWitness(trace, rr_final, dfa_final)
            if online_witness is None and rr_step is not dfa_step:
                online_witness = EquivalenceWitness(trace, rr_step, dfa_step)
            if (
                unsound_witness is None
                and rr_step is not Verdict.UNDECIDED
                and rr_step is not dfa_step
            ):
                unsound_witness = EquivalenceWitness(trace, rr_step, dfa_step)

            successor: _ProductState = (next_carry, next_dfa)
            if successor in parents:
                continue
            if max_product_states is not None and len(parents) >= max_product_states:
                complete = False
                queue.clear()
                break
            parents[successor] = (current, symbol)
            queue.append(successor)

    return RuleRunnerEquivalenceResult(
        formula=formula,
        atoms=atoms,
        explored_product_states=len(parents),
        complete=complete,
        language_witness=language_witness,
        unsound_prefix_witness=unsound_witness,
        online_witness=online_witness,
    )
