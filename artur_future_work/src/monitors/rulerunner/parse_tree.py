"""Internal parse tree for RuleRunner.

Reuses ltlf2dfa's parser for the surface syntax (so the formula strings
accepted here match the rest of the codebase), then walks its AST into a
frozen-dataclass DAG with three properties the rules layer relies on:

1. Strictly binary children. ltlf2dfa keeps n-ary And/Or/Implies/Until/
   Release as flat tuples; we left-fold &/| and right-fold ->/U/R. The
   U/R right-association matches ltlf2dfa's own MONA translation
   (verified empirically: `a U b U c` and `a U (b U c)` produce the
   same MONA formula).

2. Syntactic deduplication of shared subformulae. Two occurrences of the
   same subformula string yield the *same* Node instance, so the rules
   layer generates one R[psi] / [psi]V neuron per distinct subformula
   even if it appears in many places in the surface formula.

3. Precomputed depth. The convergence loop in step() needs an exact
   iteration cap equal to the parse-tree depth.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from ltlf2dfa.ltlf import (
    LTLfAlways,
    LTLfAnd,
    LTLfAtomic,
    LTLfEventually,
    LTLfImplies,
    LTLfNext,
    LTLfNot,
    LTLfOr,
    LTLfRelease,
    LTLfUntil,
    LTLfWeakNext,
)
from ltlf2dfa.parser.ltlf import LTLfParser


class Op(Enum):
    ATOM = "ATOM"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    IMPLIES = "IMPLIES"
    NEXT = "NEXT"
    WEAK_NEXT = "WEAK_NEXT"
    EVENTUALLY = "EVENTUALLY"
    ALWAYS = "ALWAYS"
    UNTIL = "UNTIL"
    RELEASE = "RELEASE"


@dataclass(frozen=True)
class Node:
    """One node in the parse-tree DAG.

    `key` is the canonical syntactic identity used for deduplication and
    as the name of the corresponding R[.] / [.]V neurons downstream.
    `atom` is set iff `op is Op.ATOM`. `depth` is 0 for atoms and
    1 + max(child.depth) otherwise.
    """

    op: Op
    children: tuple["Node", ...]
    key: str
    depth: int
    atom: str | None = None

    def subformulae(self) -> Iterator["Node"]:
        """Yield every distinct subformula bottom-up; root yielded last."""
        seen: set[str] = set()

        def walk(n: "Node") -> Iterator["Node"]:
            for c in n.children:
                yield from walk(c)
            if n.key not in seen:
                seen.add(n.key)
                yield n

        yield from walk(self)


_UNARY: dict[type, Op] = {
    LTLfNot: Op.NOT,
    LTLfNext: Op.NEXT,
    LTLfWeakNext: Op.WEAK_NEXT,
    LTLfEventually: Op.EVENTUALLY,
    LTLfAlways: Op.ALWAYS,
}

_BINARY_OP: dict[type, Op] = {
    LTLfAnd: Op.AND,
    LTLfOr: Op.OR,
    LTLfImplies: Op.IMPLIES,
    LTLfUntil: Op.UNTIL,
    LTLfRelease: Op.RELEASE,
}

_LEFT_ASSOC: frozenset[type] = frozenset({LTLfAnd, LTLfOr})

_INFIX: dict[Op, str] = {
    Op.AND: "&",
    Op.OR: "|",
    Op.IMPLIES: "->",
    Op.UNTIL: "U",
    Op.RELEASE: "R",
}


def parse(formula: str) -> Node:
    """Parse an LTLf formula string into a deduplicated parse-tree DAG."""
    ast = LTLfParser()(formula)
    return _walk(ast, {})


def _make(
    op: Op,
    children: tuple[Node, ...],
    key: str,
    atom: str | None,
    cache: dict[str, Node],
) -> Node:
    cached = cache.get(key)
    if cached is not None:
        return cached
    depth = 1 + max((c.depth for c in children), default=-1)
    node = Node(op=op, children=children, key=key, depth=depth, atom=atom)
    cache[key] = node
    return node


def _walk(ast, cache: dict[str, Node]) -> Node:
    surface_key = str(ast)
    cached = cache.get(surface_key)
    if cached is not None:
        return cached

    if isinstance(ast, LTLfAtomic):
        return _make(Op.ATOM, (), surface_key, ast.s, cache)

    cls = type(ast)
    if cls in _UNARY:
        child = _walk(ast.f, cache)
        return _make(_UNARY[cls], (child,), surface_key, None, cache)

    if cls in _BINARY_OP:
        op = _BINARY_OP[cls]
        kids = [_walk(c, cache) for c in ast.formulas]
        infix = _INFIX[op]
        if cls in _LEFT_ASSOC:
            reduced = kids[0]
            for nxt in kids[1:]:
                pair_key = f"({reduced.key} {infix} {nxt.key})"
                reduced = _make(op, (reduced, nxt), pair_key, None, cache)
        else:
            reduced = kids[-1]
            for nxt in reversed(kids[:-1]):
                pair_key = f"({nxt.key} {infix} {reduced.key})"
                reduced = _make(op, (nxt, reduced), pair_key, None, cache)
        cache[surface_key] = reduced
        return reduced

    raise TypeError(f"Unsupported LTLf AST node type: {cls.__name__}")
