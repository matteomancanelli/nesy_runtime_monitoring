"""Formula progression and finite-trace boundary evaluation.

Two operations drive the progression-based monitor (latex/3_rulerunner.tex
§3.3):

* ``prog(f, s)`` — the residual that must hold on the remaining suffix after
  reading cell ``s``. Characterized by
  ``s . tau |= f  <=>  tau |= prog(f, s)`` for every finite ``tau``, including
  the empty trace.
* ``last(f, s)`` — the truth value of ``f`` on the *one-cell* trace ``(s)``,
  i.e. treating ``s`` as the final cell. This is the finite-trace counterpart
  of RuleRunner's END-triggered rules and yields the binary end-of-trace
  verdict.

The next clauses make this definition total: ``X f`` progresses to
``f & F true`` (the suffix must be non-empty), while ``W f`` progresses to
``f | G false`` (the empty suffix satisfies weak next). Results are built with
the constant-folding smart constructors of ``formula.py`` and are expected to
be ``simplify``-ed by the caller.

``holds_empty`` gives the truth value on the empty trace.  With total
progression this is also the exact boundary test for the residual after the
last observed cell.
"""

from __future__ import annotations

from src.formula.compiler import Observation
from src.monitors.progression.formula import (
    FALSE,
    TRUE,
    Formula,
    Op,
    always,
    conj,
    disj,
    eventually,
    neg,
)


def prog(f: Formula, obs: Observation) -> Formula:
    """One-step progression of ``f`` through the observation ``obs``.

    Missing atoms are read as false (matching the codebase's guard
    convention). The result is *not* simplified here — callers wrap it in
    ``formula.simplify``.
    """
    op = f.op
    if op is Op.TRUE or op is Op.FALSE:
        return f
    if op is Op.ATOM:
        return TRUE if obs.get(f.atom, False) else FALSE
    if op is Op.NOT:
        return neg(prog(f.args[0], obs))
    if op is Op.AND:
        return conj(prog(f.args[0], obs), prog(f.args[1], obs))
    if op is Op.OR:
        return disj(prog(f.args[0], obs), prog(f.args[1], obs))
    if op is Op.NEXT:
        # F true is exactly the non-empty-suffix test.
        return conj(f.args[0], eventually(TRUE))
    if op is Op.WEAK_NEXT:
        # G false is true exactly on the empty suffix.
        return disj(f.args[0], always(FALSE))
    if op is Op.EVENTUALLY:
        # F x = x | X F x
        return disj(prog(f.args[0], obs), f)
    if op is Op.ALWAYS:
        # G x = x & X G x
        return conj(prog(f.args[0], obs), f)
    if op is Op.UNTIL:
        # x U y = y | (x & X(x U y))
        x, y = f.args
        return disj(prog(y, obs), conj(prog(x, obs), f))
    if op is Op.RELEASE:
        # x R y = y & (x | X(x R y))
        x, y = f.args
        return conj(prog(y, obs), disj(prog(x, obs), f))
    raise TypeError(f"Cannot progress op {op}")


def last(f: Formula, obs: Observation) -> bool:
    """Truth value of ``f`` on the single-cell trace ``(obs)``.

    Temporal boundary cases mirror engine._resolve / the END rules:
    strong next is false, weak next is true, and U/R/F/G reduce to their
    consequent (resp. operand) at the final cell.
    """
    op = f.op
    if op is Op.TRUE:
        return True
    if op is Op.FALSE:
        return False
    if op is Op.ATOM:
        return bool(obs.get(f.atom, False))
    if op is Op.NOT:
        return not last(f.args[0], obs)
    if op is Op.AND:
        return last(f.args[0], obs) and last(f.args[1], obs)
    if op is Op.OR:
        return last(f.args[0], obs) or last(f.args[1], obs)
    if op is Op.NEXT:
        return False  # strong next: no successor cell exists
    if op is Op.WEAK_NEXT:
        return True
    if op is Op.EVENTUALLY or op is Op.ALWAYS:
        return last(f.args[0], obs)
    if op is Op.UNTIL or op is Op.RELEASE:
        return last(f.args[1], obs)
    raise TypeError(f"Cannot evaluate op {op} at last cell")


def holds_empty(f: Formula) -> bool:
    """Truth value of ``f`` on the empty trace (no cells).

    Only used for the degenerate empty-input case. Atoms and strong
    eventualities are false (there is no cell to witness them); weak
    operators (``W``, ``R``, ``G``) are vacuously true.
    """
    op = f.op
    if op is Op.TRUE:
        return True
    if op is Op.FALSE:
        return False
    if op is Op.ATOM:
        return False
    if op is Op.NOT:
        return not holds_empty(f.args[0])
    if op is Op.AND:
        return holds_empty(f.args[0]) and holds_empty(f.args[1])
    if op is Op.OR:
        return holds_empty(f.args[0]) or holds_empty(f.args[1])
    if op is Op.NEXT:
        return False
    if op is Op.WEAK_NEXT:
        return True
    if op is Op.EVENTUALLY:
        return False
    if op is Op.ALWAYS:
        return True
    if op is Op.UNTIL:
        return False  # x U y needs a cell witnessing y
    if op is Op.RELEASE:
        return True  # x R y vacuously holds on the empty trace
    raise TypeError(f"Cannot evaluate op {op} on the empty trace")
