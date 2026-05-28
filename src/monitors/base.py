"""Abstract Monitor interface shared by all three LTLf monitoring paradigms.

The interface enforces three-valued semantics during the trace
(SATISFY / VIOLATE / UNDECIDED) and a binary verdict at end-of-trace,
matching the LTL3 / LTLf runtime monitoring conventions. The split is
necessary because response-style formulas like G(a -> F b) have no
trap and no accepting sink, so step() can only ever return UNDECIDED:
the verdict only becomes binary once the trace is known to be complete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum

from src.formula.compiler import Observation


class Verdict(Enum):
    SATISFY = "SATISFY"
    VIOLATE = "VIOLATE"
    UNDECIDED = "UNDECIDED"


class Monitor(ABC):
    """Abstract LTLf runtime monitor.

    Subclasses implement the four primitives `compile`, `step`,
    `final_verdict`, and `reset`. The convenience methods `run` and
    `batch_run` are derived from these and may be overridden for
    paradigms that benefit from vectorization (DeepDFA on GPU).
    """

    @classmethod
    @abstractmethod
    def compile(cls, formula: str) -> "Monitor":
        """Compile an LTLf formula and return a ready-to-step monitor."""

    @abstractmethod
    def step(self, obs: Observation) -> Verdict:
        """Advance one step and return the running three-valued verdict.

        Returns SATISFY / VIOLATE only when an accepting sink / trap
        state has been reached and the verdict is therefore absorbing;
        UNDECIDED otherwise.
        """

    @abstractmethod
    def final_verdict(self) -> Verdict:
        """End-of-trace binary verdict. Never returns UNDECIDED in LTLf."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial configuration for monitoring a fresh trace."""

    def run(self, trace: Iterable[Observation]) -> Verdict:
        """Process a single trace, terminating early on a decided verdict."""
        self.reset()
        for obs in trace:
            v = self.step(obs)
            if v is not Verdict.UNDECIDED:
                return v
        return self.final_verdict()

    def batch_run(
        self, traces: Iterable[Iterable[Observation]]
    ) -> list[Verdict]:
        """Sequential default; subclasses override for vectorized execution."""
        return [self.run(t) for t in traces]
