"""Progression engine — the lazy, pure-Python reference realization.

``ProgressionEngine`` is the semantics oracle for the progression-based
RuleRunner (analogous to ``rulerunner/engine.py``'s ``RuleEngine``): it is the
reference the eager construction and the flat CILP neural monitor are tested
against, and the paper's "lazy" on-the-fly realization. The *neural* monitor
experiments use ``ProgressionRuleRunnerMonitor`` (the flat CILP network in
``flat.py``); this engine is not a headline experimental paradigm.

It is the corrected RuleRunner of latex/3_rulerunner.tex §3.3. It carries
a *residual* formula ``rho`` (a set of active roots, read conjunctively, kept
here as one Boolean-simplified formula) and updates it by progression:

    rho_{t+1} = simplify(prog(rho_t, sigma_t)).

Verdicts:

* **Online (absorbing).** ``step`` reports an absorbing ``SATISFY`` when the
  residual has become valid over *every* continuation — including the empty
  one, i.e. the accepting-sink condition — which is exactly
  ``prog(rho, obs) ≡ TRUE`` *and* ``last(rho, obs) = True``. Symmetrically for
  ``VIOLATE`` (the trap condition). Because ``simplify`` only returns ``TRUE``/
  ``FALSE`` on a genuine Boolean tautology / contradiction over the temporal
  leaves, these signals are *sound*; they may lag the true earliest verdict by
  a few cells (the lazy under-approximation of §3.3), but never fire wrongly.
* **End-of-trace.** ``final_verdict`` returns ``last(rho_before_last,
  last_obs)`` — exact, so ``run`` matches ``SymbolicDFAMonitor`` on *every*
  formula, including the nested-temporal ones (``F(a & X b)``, ``G(a -> F b)``,
  ``G(a -> X b)``) where the original RuleRunner diverges.

This "lazy" realization progresses-and-simplifies on the fly, materializing
only the residuals a given trace actually visits. It is pure Python / CPU
(no tensors — a GPU cannot accelerate it, exactly like the symbolic DFA walk;
``effective_device`` is therefore honestly "cpu").
"""

from __future__ import annotations

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.progression.formula import Formula, Op, from_node, simplify
from src.monitors.progression.progression import holds_empty, last, prog
from src.monitors.rulerunner.parse_tree import parse


class ProgressionEngine(Monitor):
    """Lazy, pure-Python reference — residual-carrying, progression-driven."""

    def __init__(self, phi: Formula) -> None:
        self._phi = simplify(phi)
        self._rho: Formula = self._phi
        # Verdict this cell would yield if the trace ended here; None until the
        # first `step`. `_decided` freezes an absorbing verdict once reached.
        self._last_v: bool | None = None
        self._decided: Verdict | None = None

    @classmethod
    def compile(
        cls, formula: str, device: object = "cpu"
    ) -> "ProgressionEngine":
        # `device` accepted for a uniform signature across paradigms; the lazy
        # realization is pure Python (no tensors), so it is ignored and
        # `effective_device` stays "cpu".
        return cls(from_node(parse(formula)))

    def reset(self) -> None:
        self._rho = self._phi
        self._last_v = None
        self._decided = None

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided

        # Verdict if this cell were the last (exact, finite-trace boundary).
        self._last_v = last(self._rho, obs)
        # Residual for a non-empty continuation.
        nxt = simplify(prog(self._rho, obs))
        self._rho = nxt

        # Accepting sink: valid on every continuation, empty included.
        if nxt.op is Op.TRUE and self._last_v:
            self._decided = Verdict.SATISFY
            return Verdict.SATISFY
        # Trap: unsatisfiable on every continuation, empty included.
        if nxt.op is Op.FALSE and not self._last_v:
            self._decided = Verdict.VIOLATE
            return Verdict.VIOLATE
        return Verdict.UNDECIDED

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        if self._last_v is None:
            # No cell was ever read (empty trace): fall back to empty-word
            # semantics on the original formula.
            return Verdict.SATISFY if holds_empty(self._phi) else Verdict.VIOLATE
        return Verdict.SATISFY if self._last_v else Verdict.VIOLATE
