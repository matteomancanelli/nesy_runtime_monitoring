"""Internal formula representation for the progression-based RuleRunner.

The original RuleRunner (``src/monitors/rulerunner``) carries truth
registers addressed by the *syntactic subformulae of the input formula*,
which conflates two live instances of the same subformula spawned from
different temporal contexts (the shared-register limitation — see
CLAUDE.md / latex/3_rulerunner.tex §3.2). The progression-based
reformulation instead carries *residual formulae* produced by formula
progression, which may be syntactically new (e.g. ``F(a & X b)`` can
progress to ``b | F(a & X b)``). It therefore needs a formula type that
can be freely constructed, Boolean-simplified, and tested for validity /
unsatisfiability — none of which the parse-tree ``Node`` (tied to the
input formula's syntax, with no truth constants) supports.

This module provides that type:

* a small immutable ``Formula`` AST with the two Boolean constants
  ``TRUE``/``FALSE`` the residuals need;
* smart constructors that fold constants on the fly (so a residual never
  drags around ``a & true`` etc.);
* a converter ``from_node`` from the shared parse tree (desugaring
  ``->`` and constants); and
* ``simplify`` — recursive syntactic temporal normalization plus Boolean
  simplification via sympy, treating each atom and each *maximal temporal
  subformula* as an independent Boolean variable.
  This is sound for validity/unsatisfiability (a Boolean tautology over
  those variables holds under the one real assignment too), which is all
  the monitor's absorbing early-termination needs; it is deliberately an
  under-approximation of true LTLf validity (it does not know, e.g., that
  ``(a U b) & G ~b`` is unsatisfiable), so lazy early termination may lag
  arbitrarily but never fires a wrong verdict. See ``progression.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import reduce

import sympy

from src.monitors.rulerunner.parse_tree import Node
from src.monitors.rulerunner.parse_tree import Op as PTOp


class Op(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    ATOM = "ATOM"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    NEXT = "NEXT"
    WEAK_NEXT = "WEAK_NEXT"
    UNTIL = "UNTIL"
    RELEASE = "RELEASE"
    EVENTUALLY = "EVENTUALLY"
    ALWAYS = "ALWAYS"


_TEMPORAL: frozenset[Op] = frozenset(
    {Op.NEXT, Op.WEAK_NEXT, Op.UNTIL, Op.RELEASE, Op.EVENTUALLY, Op.ALWAYS}
)
_BINARY: frozenset[Op] = frozenset({Op.AND, Op.OR, Op.UNTIL, Op.RELEASE})


@dataclass(frozen=True)
class Formula:
    """Immutable LTLf formula node.

    ``args`` holds the operand subformulae (empty for ``TRUE``/``FALSE``/
    atoms); ``atom`` is set iff ``op is Op.ATOM``. ``key`` is a deterministic
    structural string, used for hashing/deduplication and (later, in the
    eager construction) for residual-state identity.  It is not a canonical
    identifier of the formula's full LTLf right language.
    """

    op: Op
    args: tuple["Formula", ...] = ()
    atom: str | None = None

    @property
    def key(self) -> str:
        if self.op is Op.ATOM:
            return self.atom  # type: ignore[return-value]
        if self.op is Op.TRUE:
            return "true"
        if self.op is Op.FALSE:
            return "false"
        inner = ",".join(a.key for a in self.args)
        return f"{self.op.value}({inner})"

    @property
    def is_temporal(self) -> bool:
        return self.op in _TEMPORAL

    def __str__(self) -> str:
        return self.key


# ---------------------------------------------------------------------------
# Constants and smart constructors (fold Boolean constants eagerly)
# ---------------------------------------------------------------------------

TRUE = Formula(Op.TRUE)
FALSE = Formula(Op.FALSE)


def atom(name: str) -> Formula:
    return Formula(Op.ATOM, (), name)


def neg(x: Formula) -> Formula:
    if x.op is Op.TRUE:
        return FALSE
    if x.op is Op.FALSE:
        return TRUE
    if x.op is Op.NOT:
        return x.args[0]  # double negation
    # Negation normal form, including the finite-trace temporal dualities.
    if x.op is Op.AND:
        return disj(neg(x.args[0]), neg(x.args[1]))
    if x.op is Op.OR:
        return conj(neg(x.args[0]), neg(x.args[1]))
    if x.op is Op.NEXT:
        return weak_next(neg(x.args[0]))
    if x.op is Op.WEAK_NEXT:
        return next_(neg(x.args[0]))
    if x.op is Op.EVENTUALLY:
        return always(neg(x.args[0]))
    if x.op is Op.ALWAYS:
        return eventually(neg(x.args[0]))
    if x.op is Op.UNTIL:
        return release(neg(x.args[0]), neg(x.args[1]))
    if x.op is Op.RELEASE:
        return until(neg(x.args[0]), neg(x.args[1]))
    return Formula(Op.NOT, (x,))


def _are_complements(x: Formula, y: Formula) -> bool:
    """Recognize structural NNF duals without constructing ``neg(x)``."""
    if x.op is Op.NOT:
        return x.args[0].key == y.key
    if y.op is Op.NOT:
        return y.args[0].key == x.key
    duals = {
        Op.NEXT: Op.WEAK_NEXT,
        Op.WEAK_NEXT: Op.NEXT,
        Op.EVENTUALLY: Op.ALWAYS,
        Op.ALWAYS: Op.EVENTUALLY,
        Op.UNTIL: Op.RELEASE,
        Op.RELEASE: Op.UNTIL,
    }
    if duals.get(x.op) is not y.op or len(x.args) != len(y.args):
        return False
    return all(_are_complements(a, b) for a, b in zip(x.args, y.args))


def conj(x: Formula, y: Formula) -> Formula:
    if x.op is Op.FALSE or y.op is Op.FALSE:
        return FALSE
    if x.op is Op.TRUE:
        return y
    if y.op is Op.TRUE:
        return x
    if x.key == y.key:
        return x
    if _are_complements(x, y):
        return FALSE
    return Formula(Op.AND, (x, y))


def disj(x: Formula, y: Formula) -> Formula:
    if x.op is Op.TRUE or y.op is Op.TRUE:
        return TRUE
    if x.op is Op.FALSE:
        return y
    if y.op is Op.FALSE:
        return x
    if x.key == y.key:
        return x
    if _are_complements(x, y):
        return TRUE
    return Formula(Op.OR, (x, y))


def next_(x: Formula) -> Formula:
    # Strong Next of false is false on non-empty and empty suffixes alike.
    if x.op is Op.FALSE:
        return FALSE
    return Formula(Op.NEXT, (x,))


def weak_next(x: Formula) -> Formula:
    # Weak Next of true is true, including at the finite boundary.
    if x.op is Op.TRUE:
        return TRUE
    return Formula(Op.WEAK_NEXT, (x,))


def until(x: Formula, y: Formula) -> Formula:
    # true U y is F y, while x U false is false.  Other familiar-looking
    # reductions (for example false U y = y) fail on the empty suffix under
    # the total residual semantics and are deliberately not used.
    if y.op is Op.FALSE:
        return FALSE
    if x.op is Op.TRUE:
        return eventually(y)
    return Formula(Op.UNTIL, (x, y))


def release(x: Formula, y: Formula) -> Formula:
    # false R y is G y, while x R true is true.  These identities preserve
    # the empty-suffix value as well as every non-empty finite trace.
    if y.op is Op.TRUE:
        return TRUE
    if x.op is Op.FALSE:
        return always(y)
    return Formula(Op.RELEASE, (x, y))


def eventually(x: Formula) -> Formula:
    if x.op is Op.FALSE:
        return FALSE
    if x.op is Op.EVENTUALLY:
        return x
    return Formula(Op.EVENTUALLY, (x,))


def always(x: Formula) -> Formula:
    if x.op is Op.TRUE:
        return TRUE
    if x.op is Op.ALWAYS:
        return x
    return Formula(Op.ALWAYS, (x,))


# ---------------------------------------------------------------------------
# Conversion from the shared parse tree
# ---------------------------------------------------------------------------

_PT_UNARY = {
    PTOp.NOT: neg,
    PTOp.NEXT: next_,
    PTOp.WEAK_NEXT: weak_next,
    PTOp.EVENTUALLY: eventually,
    PTOp.ALWAYS: always,
}
_PT_BINARY = {
    PTOp.AND: conj,
    PTOp.OR: disj,
    PTOp.UNTIL: until,
    PTOp.RELEASE: release,
}


def from_node(node: Node) -> Formula:
    """Convert a RuleRunner parse-tree ``Node`` into a ``Formula``.

    Desugars ``->`` into ``~a | b`` and maps the ``true``/``false`` atom
    leaves ltlf2dfa produces onto the Boolean constants.
    """
    if node.op is PTOp.ATOM:
        if node.atom == "true":
            return TRUE
        if node.atom == "false":
            return FALSE
        return atom(node.atom)  # type: ignore[arg-type]
    if node.op is PTOp.NOT:
        return neg(from_node(node.children[0]))
    if node.op in _PT_UNARY:
        return _PT_UNARY[node.op](from_node(node.children[0]))
    if node.op is PTOp.IMPLIES:
        a, b = node.children
        return disj(neg(from_node(a)), from_node(b))
    if node.op in _PT_BINARY:
        a, b = node.children
        return _PT_BINARY[node.op](from_node(a), from_node(b))
    raise TypeError(f"Unsupported parse-tree op: {node.op}")


# ---------------------------------------------------------------------------
# Boolean simplification via sympy
# ---------------------------------------------------------------------------
#
# Atoms and maximal temporal subformulae are the Boolean "variables"; the
# &/|/~ skeleton above them is what sympy simplifies.  Smart constructors also
# apply a deliberately small set of finite-trace-safe temporal identities as
# formulas are parsed and progressed.  This collapses structural duplicates and
# detects TRUE/FALSE, but is not a quotient by full LTLf language equivalence.


def atoms_of(f: Formula) -> frozenset[str]:
    """The set of atom names occurring anywhere in ``f`` (inside temporal
    operators too). ``prog(f, obs)`` reads only these atoms, so the eager
    construction enumerates observations over exactly this set — which is why
    a residual whose guards depend on few atoms stays cheap while the IJCNN
    family (guards over all n atoms) exhibits the 2^n alphabet blowup."""
    if f.op is Op.ATOM:
        return frozenset({f.atom})  # type: ignore[arg-type]
    if f.op is Op.TRUE or f.op is Op.FALSE:
        return frozenset()
    out: set[str] = set()
    for a in f.args:
        out |= atoms_of(a)
    return frozenset(out)


def _collect_leaves(f: Formula, acc: dict[str, Formula]) -> None:
    """Gather the Boolean leaves of ``f`` keyed by their canonical ``key``.

    The Boolean skeleton is AND/OR/NOT/TRUE/FALSE; every atom and every
    maximal temporal subformula is an opaque leaf. Keying by ``Formula.key``
    (a path-independent structural string) makes the symbol assignment below
    deterministic, so logically-equal residuals reached by different routes
    canonicalize identically — essential for the eager BFS to dedup states.
    """
    if f.op is Op.TRUE or f.op is Op.FALSE:
        return
    if f.op is Op.NOT or f.op is Op.AND or f.op is Op.OR:
        for a in f.args:
            _collect_leaves(a, acc)
        return
    acc[f.key] = f


def _to_sympy(f: Formula, sym_of: dict[str, sympy.Symbol]):
    if f.op is Op.TRUE:
        return sympy.true
    if f.op is Op.FALSE:
        return sympy.false
    if f.op is Op.NOT:
        return sympy.Not(_to_sympy(f.args[0], sym_of))
    if f.op is Op.AND:
        return sympy.And(_to_sympy(f.args[0], sym_of), _to_sympy(f.args[1], sym_of))
    if f.op is Op.OR:
        return sympy.Or(_to_sympy(f.args[0], sym_of), _to_sympy(f.args[1], sym_of))
    # Atom or temporal subformula: an opaque Boolean variable.
    return sym_of[f.key]


def _from_sympy(expr, name2f: dict[str, Formula]) -> Formula:
    if expr is sympy.true or expr is True:
        return TRUE
    if expr is sympy.false or expr is False:
        return FALSE
    if getattr(expr, "is_Symbol", False):
        return name2f[expr.name]
    if isinstance(expr, sympy.Not):
        return neg(_from_sympy(expr.args[0], name2f))
    if isinstance(expr, sympy.And):
        return reduce(conj, (_from_sympy(a, name2f) for a in expr.args))
    if isinstance(expr, sympy.Or):
        return reduce(disj, (_from_sympy(a, name2f) for a in expr.args))
    raise TypeError(f"Unexpected sympy node: {expr!r} ({type(expr)})")


def simplify(f: Formula) -> Formula:
    """Return a deterministic, semantics-preserving normalized residual.

    Smart constructors apply finite-trace dualities, idempotence of ``F``/``G``,
    and safe constant identities.
    Maximal temporal subformulae are then treated as independent Boolean
    variables so SymPy can collapse the current propositional skeleton
    (``a & true`` → ``a``, ``x | ~x`` → ``true``).

    The result is canonical for the syntactic identities implemented here only;
    it is not canonical modulo full LTLf equivalence.  In particular, temporal
    leaves related by a non-local language identity may remain distinct, and
    SymPy deliberately avoids exhaustive truth-table minimization above its
    variable threshold.  ``canonical_key`` is therefore a structural
    normalization key, not a right-language identifier.
    """
    leaves: dict[str, Formula] = {}
    _collect_leaves(f, leaves)
    keys = sorted(leaves)
    sym_of = {k: sympy.Symbol(f"v{i}") for i, k in enumerate(keys)}
    expr = sympy.simplify_logic(_to_sympy(f, sym_of))
    name2f = {sym_of[k].name: leaves[k] for k in keys}
    return _from_sympy(expr, name2f)


def canonical_key(f: Formula) -> str:
    """Structural dedup key of the syntactically normalized residual.

    Equal residuals have equal keys, but language-equivalent residuals need not.
    The historical function name is retained for API compatibility.
    """
    return simplify(f).key


def split_conj(f: Formula) -> tuple[Formula, ...]:
    """Split a residual into its top-level conjuncts (the active *roots* of
    §3.3, read conjunctively). ``TRUE`` splits to the empty tuple; a
    non-conjunction is a single root. Disjunctions are *not* split — a
    disjunctive residual is one atomic root."""
    if f.op is Op.AND:
        return split_conj(f.args[0]) + split_conj(f.args[1])
    if f.op is Op.TRUE:
        return ()
    return (f,)


def subformulae(f: Formula, acc: dict[str, Formula]) -> None:
    """Collect every distinct subformula of ``f`` (keyed by ``key``) into
    ``acc`` — the syntactic support the evaluation phase ranges over."""
    if f.key in acc:
        return
    acc[f.key] = f
    for a in f.args:
        subformulae(a, acc)
