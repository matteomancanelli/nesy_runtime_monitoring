from __future__ import annotations

from src.formula.compiler import DFA, Observation, compile_ltlf
from src.monitors.base import Monitor, Verdict


class SymbolicDFAMonitor(Monitor):
    """Paradigm 1: crisp DFA walk with precomputed three-valued early termination.

    Compile once; step() is a single dict lookup + two set membership tests.
    """

    def __init__(self, dfa: DFA) -> None:
        self._dfa = dfa
        self._state = dfa.initial

    @classmethod
    def compile(cls, formula: str, device: object = "cpu") -> "SymbolicDFAMonitor":
        # `device` is accepted for a uniform compile() signature across all
        # three paradigms but ignored: this is a pure-Python DFA walk.
        return cls(compile_ltlf(formula))

    def step(self, obs: Observation) -> Verdict:
        self._state = self._dfa.step(self._state, obs)
        if self._state in self._dfa.trap_states:
            return Verdict.VIOLATE
        if self._state in self._dfa.accepting_sinks:
            return Verdict.SATISFY
        return Verdict.UNDECIDED

    def final_verdict(self) -> Verdict:
        return (
            Verdict.SATISFY if self._state in self._dfa.accepting else Verdict.VIOLATE
        )

    def reset(self) -> None:
        self._state = self._dfa.initial
