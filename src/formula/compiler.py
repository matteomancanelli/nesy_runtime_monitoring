"""LTLf -> minimal DFA compilation via ltlf2dfa.

Produces a `DFA` dataclass shared by all three monitoring paradigms:
the symbolic monitor steps through it directly, and the differentiable
monitor (DeepDFA) uses the same DFA as the structural backbone of its
transition tensor. Trap states and accepting sinks are precomputed
once at construction so that three-valued early-termination verdicts
cost a single set membership check per step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from ltlf2dfa.parser.ltlf import LTLfParser

Observation = dict[str, bool]
Guard = Callable[[Observation], bool]


@dataclass
class Transition:
    src: int
    label: str
    guard: Guard
    dst: int


@dataclass
class DFA:
    """Minimal DFA produced by ltlf2dfa/MONA for an LTLf formula.

    States are integers as numbered by MONA (1-indexed). The DFA is
    complete and deterministic: for every state and every observation
    over `atoms`, exactly one outgoing transition's guard fires.

    `trap_states` and `accepting_sinks` enable three-valued early
    termination: a trap means the trace can no longer be satisfied,
    an accepting sink means the trace is already satisfied regardless
    of any continuation.
    """

    states: frozenset[int]
    atoms: tuple[str, ...]
    initial: int
    accepting: frozenset[int]
    transitions: tuple[Transition, ...]
    trap_states: frozenset[int]
    accepting_sinks: frozenset[int]
    _outgoing: dict[int, tuple[Transition, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_src: dict[int, list[Transition]] = {q: [] for q in self.states}
        for t in self.transitions:
            by_src[t.src].append(t)
        self._outgoing = {q: tuple(ts) for q, ts in by_src.items()}

    def step(self, state: int, obs: Observation) -> int:
        for t in self._outgoing[state]:
            if t.guard(obs):
                return t.dst
        raise ValueError(f"No transition fires from state {state} on {obs!r}")


class MonaFailure(RuntimeError):
    """MONA did not produce a usable DFA for the formula."""


@lru_cache(maxsize=None)
def compile_ltlf(formula: str) -> DFA:
    """Compile an LTLf formula string to a minimal DFA.

    Cached: MONA is an external process that can take tens of seconds and
    several GB on large formulas, and every monitor paradigm compiles the same
    formula independently. The returned DFA is treated as read-only by all
    callers (monitors hold their current state separately).
    """
    parser = LTLfParser()
    dot = parser(formula).to_dfa()
    # ltlf2dfa runs MONA with a hardcoded 30 s subprocess timeout and returns
    # False on expiry; on other failures it returns a stub DOT with no edges
    # (a lone `init -> 1`). Both must raise rather than yield a silently
    # degenerate 2-state DFA — Exp 6 plots the measured |Q| as its x-axis.
    if not isinstance(dot, str):
        raise MonaFailure(
            f"MONA timed out (ltlf2dfa's 30 s limit) on: {formula!r}"
        )
    dfa = _parse_mona_dot(dot)
    if not dfa.transitions:
        raise MonaFailure(
            f"MONA produced no transitions for: {formula!r} — it exhausted "
            f"memory or failed to build the automaton."
        )
    return dfa


# MONA's DOT lists accepting states as `node [shape = doublecircle]; 1; 2; 4;`
# — the shape attribute applies to every subsequent bare-id declaration
# until the next `node [...]` or `init [...]` directive. The block regex
# captures everything between the shape directive and the next directive;
# then we extract the digit ids.
_DOUBLECIRCLE_BLOCK_RE = re.compile(
    r"node\s*\[\s*shape\s*=\s*doublecircle\s*\]\s*;([^\[]*?)(?=node\s*\[|init\s*\[)",
    re.DOTALL,
)
_DIGIT_RE = re.compile(r"\d+")
_INIT_RE = re.compile(r"init\s*->\s*(\d+)\s*;")
_EDGE_RE = re.compile(r"(\d+)\s*->\s*(\d+)\s*\[\s*label\s*=\s*\"([^\"]*)\"\s*\]\s*;")
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_RESERVED = {"true", "false"}


def _parse_mona_dot(dot: str) -> DFA:
    block = _DOUBLECIRCLE_BLOCK_RE.search(dot)
    accepting = (
        frozenset(int(s) for s in _DIGIT_RE.findall(block.group(1)))
        if block else frozenset()
    )

    m = _INIT_RE.search(dot)
    if m is None:
        raise ValueError("MONA DOT output is missing the 'init -> q' line.")
    initial = int(m.group(1))

    states: set[int] = {initial} | set(accepting)
    atoms: set[str] = set()
    transitions: list[Transition] = []
    for src_s, dst_s, label in _EDGE_RE.findall(dot):
        src, dst = int(src_s), int(dst_s)
        states.update({src, dst})
        for tok in _IDENT_RE.findall(label):
            if tok not in _RESERVED:
                atoms.add(tok)
        transitions.append(
            Transition(src=src, label=label, guard=_compile_guard(label), dst=dst)
        )

    states_fset = frozenset(states)
    trap_states, accepting_sinks = _compute_sink_labels(
        states_fset, transitions, accepting
    )

    return DFA(
        states=states_fset,
        atoms=tuple(sorted(atoms)),
        initial=initial,
        accepting=accepting,
        transitions=tuple(transitions),
        trap_states=trap_states,
        accepting_sinks=accepting_sinks,
    )


def _compile_guard(label: str) -> Guard:
    stripped = label.strip()
    if stripped == "true":
        return lambda obs: True
    if stripped == "false":
        return lambda obs: False
    
    py_src = label.replace("~", " not ").replace("&", " and ").replace("|", " or ")
    code = compile(py_src.strip(), f"<dfa-guard:{label}>", "eval")

    def guard(obs: Observation, _code: object = code) -> bool:
        try:
            return bool(eval(_code, {"__builtins__": {}}, obs))
        except NameError as e:
            raise ValueError(
                f"Observation {obs!r} is missing an atom required by guard {label!r}"
            ) from e

    return guard

def _reachable_from(q0: int, succ: dict[int, set[int]]) -> set[int]:
    seen = {q0}
    stack = [q0]
    while stack:
        q = stack.pop()
        for q2 in succ[q]:
            if q2 not in seen:
                seen.add(q2)
                stack.append(q2)
    return seen

def _compute_sink_labels(
    states: frozenset[int],
    transitions: list[Transition],
    accepting: frozenset[int],
) -> tuple[frozenset[int], frozenset[int]]:
    succ: dict[int, set[int]] = {q: set() for q in states}
    for t in transitions:
        succ[t.src].add(t.dst)

    traps: set[int] = set()
    sinks: set[int] = set()
    for q in states:
        r = _reachable_from(q, succ)
        if not (r & accepting):
            traps.add(q)
        if r <= accepting:
            sinks.add(q)
    return frozenset(traps), frozenset(sinks)
