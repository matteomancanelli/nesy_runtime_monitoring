"""Tests for the RuleRunner parse-tree DAG."""

from __future__ import annotations

from src.monitors.rulerunner.parse_tree import Op, parse


def test_atom() -> None:
    n = parse("a")
    assert n.op is Op.ATOM
    assert n.atom == "a"
    assert n.children == ()
    assert n.depth == 0


def test_not() -> None:
    n = parse("!a")
    assert n.op is Op.NOT
    assert len(n.children) == 1
    assert n.children[0].op is Op.ATOM and n.children[0].atom == "a"
    assert n.depth == 1


def test_and_or() -> None:
    a_and_b = parse("a & b")
    assert a_and_b.op is Op.AND
    assert tuple(c.atom for c in a_and_b.children) == ("a", "b")
    assert a_and_b.depth == 1

    a_or_b = parse("a | b")
    assert a_or_b.op is Op.OR


def test_implies() -> None:
    n = parse("a -> b")
    assert n.op is Op.IMPLIES
    assert n.depth == 1


def test_until_release() -> None:
    assert parse("a U b").op is Op.UNTIL
    assert parse("a R b").op is Op.RELEASE


def test_next_weak_next() -> None:
    assert parse("X a").op is Op.NEXT
    assert parse("WX a").op is Op.WEAK_NEXT


def test_eventually_always() -> None:
    assert parse("F a").op is Op.EVENTUALLY
    assert parse("G a").op is Op.ALWAYS


def test_nary_and_left_associated() -> None:
    """`a & b & c` folds left to `((a & b) & c)`."""
    n = parse("a & b & c")
    assert n.op is Op.AND
    left, right = n.children
    assert left.op is Op.AND
    assert right.op is Op.ATOM and right.atom == "c"
    assert tuple(c.atom for c in left.children) == ("a", "b")
    assert n.depth == 2


def test_nary_or_left_associated() -> None:
    n = parse("a | b | c | d")
    assert n.op is Op.OR
    # Spine is left-deep: (((a | b) | c) | d)
    assert n.children[1].atom == "d"
    assert n.children[0].op is Op.OR
    assert n.children[0].children[1].atom == "c"
    assert n.depth == 3


def test_until_right_associated() -> None:
    """`a U b U c` folds right to `(a U (b U c))` (matches ltlf2dfa->MONA)."""
    n = parse("a U b U c")
    assert n.op is Op.UNTIL
    left, right = n.children
    assert left.op is Op.ATOM and left.atom == "a"
    assert right.op is Op.UNTIL
    assert tuple(c.atom for c in right.children) == ("b", "c")
    assert n.depth == 2


def test_implies_right_associated() -> None:
    n = parse("a -> b -> c")
    assert n.op is Op.IMPLIES
    left, right = n.children
    assert left.atom == "a"
    assert right.op is Op.IMPLIES
    assert tuple(c.atom for c in right.children) == ("b", "c")


def test_subformula_sharing() -> None:
    """`b` appearing twice in `(F b) & (G b)` yields one shared Node."""
    n = parse("(F b) & (G b)")
    f_b, g_b = n.children
    assert f_b.op is Op.EVENTUALLY and g_b.op is Op.ALWAYS
    # Same identity, not just equality.
    assert f_b.children[0] is g_b.children[0]


def test_subformula_sharing_after_binarization() -> None:
    """Shared subformulae cross binarization boundaries."""
    n = parse("(a & b) & (a & b)")
    # Inner (a & b) appears twice in the AST; after walking, both
    # occurrences should point at the same Node.
    left, right = n.children
    assert left is right


def test_subformulae_iteration_distinct_and_bottom_up() -> None:
    n = parse("a -> F b")
    forms = list(n.subformulae())
    # Root yielded last.
    assert forms[-1] is n
    # No duplicates.
    keys = [s.key for s in forms]
    assert len(keys) == len(set(keys))
    # First yielded element is always a leaf atom.
    assert forms[0].op is Op.ATOM
    # Every yielded node's children appear earlier in the sequence.
    seen: set[str] = set()
    for node in forms:
        for c in node.children:
            assert c.key in seen
        seen.add(node.key)


def test_depth_nested_temporal() -> None:
    """`X (F (G a))` -> depth 3 (atom 0, G 1, F 2, X 3)."""
    n = parse("X (F (G a))")
    assert n.depth == 3


def test_depth_response_pattern() -> None:
    """`G (a -> F b)`: atoms 0, F 1, -> 2, G 3."""
    n = parse("G (a -> F b)")
    assert n.op is Op.ALWAYS
    assert n.depth == 3


def test_parse_is_pure() -> None:
    """Repeated calls produce structurally equal DAGs with identical keys."""
    n1 = parse("(a & b) U (F c)")
    n2 = parse("(a & b) U (F c)")
    assert n1.key == n2.key
    assert n1.depth == n2.depth


def test_node_is_hashable() -> None:
    """Frozen-dataclass Nodes can sit in sets/dicts (rules layer needs this)."""
    n = parse("a U b")
    s = {n, n.children[0], n.children[1]}
    assert len(s) == 3
