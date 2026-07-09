"""Eager residual-DFA construction for the progression-based RuleRunner.

The lazy monitor (``monitor.py``) progresses-and-simplifies residuals on the
fly. The *eager* realization of §3.3 precomputes the whole reachable residual
transition system once — which, materialized and quotiented, *is* an automaton
equivalent to ``phi`` (progression being the standard route from LTLf to its
determinized automaton). This module builds it and exposes:

* ``ProgressionDFA`` — reachable residual states (canonical ``Formula``s),
  their per-state guard atoms, and the transition / online-verdict / last-cell
  tables the batched network (Part 2b) and the fast eager monitor consume;
* ``build_progression_dfa`` — the BFS that constructs it;
* ``ProgressionRuleRunnerEagerMonitor`` — a fast, table-driven *reference*
  monitor (verdict-for-verdict identical to ``ProgressionEngine`` and to
  ``SymbolicDFAMonitor``), used as a correctness oracle for the flat neural
  monitor; it is not a headline experimental paradigm.

The neural monitor the experiments run is ``ProgressionRuleRunnerMonitor``
(the flat CILP network, ``flat.py``), which is *built from* this construction.

**Alphabet.** ``prog(rho, obs)`` reads only the atoms occurring in ``rho``, so
each state enumerates observations over *its own* guard atoms — cheap for
formulas whose residual guards depend on few atoms, but exhibiting the honest
``2^k`` blowup for a guard over ``k`` atoms (the IJCNN family, whose guards
touch all ``n`` atoms, is exactly the ``2^n`` alphabet wall — dual to DeepDFA
dense). ``MAX_GUARD_ATOMS`` caps this so an infeasible build fails loudly
rather than hanging.

**Cost of correctness.** ``ProgressionDFA`` records ``n_states`` (reachable
residual states, ~ the minimal DFA size), ``n_roots`` (distinct ``∧``-conjunct
obligations), and ``n_closure`` (their subformula support) — the numbers that
quantify how far the residual closure grows beyond the original ``|sub(phi)|``,
i.e. the price the corrected monitor pays for handling nested temporal.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.progression.formula import (
    FALSE,
    TRUE,
    Formula,
    atoms_of,
    from_node,
    simplify,
    split_conj,
    subformulae,
)
from src.monitors.progression.progression import holds_empty, last, prog
from src.monitors.rulerunner.parse_tree import parse

MAX_GUARD_ATOMS = 20  # 2^20 ~ 1e6 guard rows per state; refuse beyond this.

# Online verdict codes (per state, per local guard symbol).
_UNDECIDED, _SATISFY, _VIOLATE = 0, 1, 2


@dataclass(frozen=True)
class ProgressionDFA:
    formula: Formula                       # simplify(phi); the initial residual
    atoms: tuple[str, ...]                 # all atoms of phi (global order)
    states: tuple[Formula, ...]            # index -> canonical residual state
    initial: int
    relevant: tuple[tuple[str, ...], ...]  # state -> guard atoms (local order)
    trans: tuple[dict[int, int], ...]      # state -> {local symbol -> next state}
    online: tuple[dict[int, int], ...]     # state -> {local symbol -> verdict code}
    last_accept: tuple[dict[int, bool], ...]  # state -> {local symbol -> last()}
    # --- residual closure C_phi / roots R_phi (structured encoding, Part 3) ---
    roots: tuple[Formula, ...]             # distinct top-level conjuncts (R_phi)
    closure: tuple[Formula, ...]           # C_phi in bottom-up (post) order
    # --- cost-of-correctness metrics ---
    n_states: int
    n_roots: int
    n_closure: int
    n_input_sub: int                       # |sub(phi)| of the ORIGINAL formula

    def symbol(self, state: int, obs: Observation) -> int:
        s = 0
        for j, a in enumerate(self.relevant[state]):
            if obs.get(a, False):
                s |= 1 << j
        return s


def _postorder(f: Formula, seen: dict[str, Formula], out: list[Formula]) -> None:
    """Append the distinct subformulae of ``f`` to ``out`` in post-order
    (children before parents, each key once). This is the bottom-up sweep
    order the structured monitor's per-node eval needs: a node is only visited
    after every child whose truth value it reads."""
    if f.key in seen:
        return
    seen[f.key] = f
    for a in f.args:
        _postorder(a, seen, out)
    out.append(f)


def build_progression_dfa(
    formula: str, max_guard_atoms: int = MAX_GUARD_ATOMS
) -> ProgressionDFA:
    phi = simplify(from_node(parse(formula)))
    atoms = tuple(sorted(atoms_of(phi)))

    states: list[Formula] = [phi]
    index: dict[str, int] = {phi.key: 0}
    relevant: list[tuple[str, ...]] = []
    trans: list[dict[int, int]] = []
    online: list[dict[int, int]] = []
    last_accept: list[dict[int, bool]] = []

    head = 0
    while head < len(states):
        rho = states[head]
        head += 1
        rel = tuple(sorted(atoms_of(rho)))
        if len(rel) > max_guard_atoms:
            raise ValueError(
                f"Residual {rho.key!r} has {len(rel)} guard atoms; enumerating "
                f"2^{len(rel)} observations exceeds MAX_GUARD_ATOMS="
                f"{max_guard_atoms}. This is the alphabet-blowup wall (dual to "
                f"DeepDFA dense) for {formula!r}."
            )
        tr: dict[int, int] = {}
        on: dict[int, int] = {}
        la: dict[int, bool] = {}
        for sym in range(1 << len(rel)):
            obs = {rel[j]: bool((sym >> j) & 1) for j in range(len(rel))}
            nxt = simplify(prog(rho, obs))
            v_now = last(rho, obs)
            if nxt.op is TRUE.op and v_now:
                on[sym] = _SATISFY
            elif nxt.op is FALSE.op and not v_now:
                on[sym] = _VIOLATE
            else:
                on[sym] = _UNDECIDED
            la[sym] = v_now
            key = nxt.key
            j = index.get(key)
            if j is None:
                j = len(states)
                index[key] = j
                states.append(nxt)
            tr[sym] = j
        relevant.append(rel)
        trans.append(tr)
        online.append(on)
        last_accept.append(la)

    # --- roots R_phi + subformula closure C_phi over all states ---
    # Roots are the distinct top-level conjuncts (first-seen order); the closure
    # is their subformula support, emitted bottom-up (children before parents)
    # so the structured monitor's per-node eval sweep can consume it directly.
    roots: dict[str, Formula] = {}
    closure: dict[str, Formula] = {}
    closure_order: list[Formula] = []
    for st in states:
        for r in split_conj(st):
            roots.setdefault(r.key, r)
            _postorder(r, closure, closure_order)
    input_sub: dict[str, Formula] = {}
    subformulae(phi, input_sub)

    return ProgressionDFA(
        formula=phi,
        atoms=atoms,
        states=tuple(states),
        initial=0,
        relevant=tuple(relevant),
        trans=tuple(trans),
        online=tuple(online),
        last_accept=tuple(last_accept),
        roots=tuple(roots.values()),
        closure=tuple(closure_order),
        n_states=len(states),
        n_roots=len(roots),
        n_closure=len(closure),
        n_input_sub=len(input_sub),
    )


class ProgressionRuleRunnerEagerMonitor(Monitor):
    """Table-driven eager realization of the progression-based RuleRunner.

    Verdict-for-verdict identical to ``ProgressionEngine`` (lazy) and to
    ``SymbolicDFAMonitor``: ``step`` is an O(1) table lookup, so early
    termination is exact (no lazy lag). Pure Python / CPU (no tensors →
    honestly ``effective_device`` "cpu"). Used as a test oracle for the flat
    neural monitor, not as an experimental paradigm."""

    def __init__(self, dfa: ProgressionDFA) -> None:
        self._dfa = dfa
        self._state = dfa.initial
        self._last_v: bool | None = None
        self._decided: Verdict | None = None

    @classmethod
    def compile(
        cls, formula: str, device: object = "cpu"
    ) -> "ProgressionRuleRunnerEagerMonitor":
        return cls(build_progression_dfa(formula))

    def reset(self) -> None:
        self._state = self._dfa.initial
        self._last_v = None
        self._decided = None

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        i = self._state
        sym = self._dfa.symbol(i, obs)
        self._last_v = self._dfa.last_accept[i][sym]
        code = self._dfa.online[i][sym]
        self._state = self._dfa.trans[i][sym]
        if code == _SATISFY:
            self._decided = Verdict.SATISFY
            return Verdict.SATISFY
        if code == _VIOLATE:
            self._decided = Verdict.VIOLATE
            return Verdict.VIOLATE
        return Verdict.UNDECIDED

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        if self._last_v is None:
            return (
                Verdict.SATISFY if holds_empty(self._dfa.formula) else Verdict.VIOLATE
            )
        return Verdict.SATISFY if self._last_v else Verdict.VIOLATE
