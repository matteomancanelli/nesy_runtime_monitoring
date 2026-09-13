"""Exhaustive, implementation-independent finite-trace semantic checks.

Unlike the historical equivalence tests, the oracle below does not use MONA,
RuleRunner rules, progression, or a transition tensor.  It evaluates the LTLf
definition directly.  This prevents several implementations generated from the
same flawed intermediate representation from merely agreeing with each other.

All traces of lengths 0--3 over {a,b} are enumerated.  Observations are sparse:
only true atoms are present, so the tests also enforce the project-wide
"missing means false" API convention.

Scope of the independence claim: the *semantics* below are independent, but
the parse tree is shared with the RuleRunner and progression implementations.
A front-end defect -- operator precedence, the associativity of a folded n-ary
operator, or the binarization itself -- could in principle hide here, because
the oracle would misread the formula in the same way.  ``SymbolicDFAMonitor``
closes that gap: it is compiled by ltlf2dfa/MONA from the formula *string* and
never sees our parse tree, so the association-sensitive entries at the end of
``_COMPLETE_DOMAIN`` fail whenever the two front ends disagree.
"""

from __future__ import annotations

from functools import cache
from itertools import product

import pytest

from src.monitors.base import Verdict
from src.monitors.deep_dfa import DeepDFAMonitorDense, DeepDFAMonitorFactored
from src.monitors.progression import (
    ProgressionEngine,
    ProgressionRuleRunnerEagerMonitor,
    ProgressionRuleRunnerMonitor,
    ProgressionRuleRunnerStructuredMonitor,
)
from src.monitors.rulerunner.cilp import CILPRunner
from src.monitors.rulerunner.engine import RuleEngine
from src.monitors.rulerunner.parse_tree import Node, Op, parse
from src.monitors.rulerunner.structured import StructuredCILPRunner
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


def _all_sparse_traces(max_length: int = 3):
    valuations = [
        {atom: True for atom, bit in zip(("a", "b"), bits) if bit}
        for bits in product((False, True), repeat=2)
    ]
    for length in range(max_length + 1):
        for cells in product(valuations, repeat=length):
            yield list(cells)


def _direct_verdict(formula: str, trace: list[dict[str, bool]]) -> Verdict:
    """Evaluate the finite-trace semantics directly from the parse tree."""
    root = parse(formula)
    n = len(trace)

    @cache
    def empty(node: Node) -> bool:
        if node.op is Op.ATOM:
            return node.atom == "true"
        if node.op is Op.NOT:
            return not empty(node.children[0])
        if node.op is Op.AND:
            return empty(node.children[0]) and empty(node.children[1])
        if node.op is Op.OR:
            return empty(node.children[0]) or empty(node.children[1])
        if node.op is Op.IMPLIES:
            return not empty(node.children[0]) or empty(node.children[1])
        if node.op in (Op.NEXT, Op.EVENTUALLY, Op.UNTIL):
            return False
        if node.op in (Op.WEAK_NEXT, Op.ALWAYS, Op.RELEASE):
            return True
        raise AssertionError(node.op)

    @cache
    def holds(node: Node, i: int) -> bool:
        if i >= n:
            return empty(node)
        if node.op is Op.ATOM:
            if node.atom == "true":
                return True
            if node.atom == "false":
                return False
            return bool(trace[i].get(node.atom, False))
        if node.op is Op.NOT:
            return not holds(node.children[0], i)
        if node.op is Op.AND:
            return holds(node.children[0], i) and holds(node.children[1], i)
        if node.op is Op.OR:
            return holds(node.children[0], i) or holds(node.children[1], i)
        if node.op is Op.IMPLIES:
            return not holds(node.children[0], i) or holds(node.children[1], i)
        if node.op is Op.NEXT:
            return i + 1 < n and holds(node.children[0], i + 1)
        if node.op is Op.WEAK_NEXT:
            return i + 1 >= n or holds(node.children[0], i + 1)
        if node.op is Op.EVENTUALLY:
            return any(holds(node.children[0], j) for j in range(i, n))
        if node.op is Op.ALWAYS:
            return all(holds(node.children[0], j) for j in range(i, n))
        if node.op is Op.UNTIL:
            left, right = node.children
            return any(
                holds(right, j) and all(holds(left, k) for k in range(i, j))
                for j in range(i, n)
            )
        if node.op is Op.RELEASE:
            # phi R psi = psi & (phi | W(phi R psi)) on finite traces.
            left, right = node.children
            return holds(right, i) and (
                holds(left, i) or i + 1 >= n or holds(node, i + 1)
            )
        raise AssertionError(node.op)

    return Verdict.SATISFY if holds(root, 0) else Verdict.VIOLATE


_FAITHFUL_RR_DOMAIN = [
    "true",
    "false",
    "a",
    "!a",
    "a & b",
    "a | b",
    "a -> b",
    "F a",
    "G a",
    "X a",
    "WX a",
    "X X a",
    "WX X a",
    "X WX a",
    "a U b",
    "a R b",
    "F (a & b)",
    "G (a | b)",
    "X (F a)",
    "G true",
    "F false",
]


@pytest.mark.parametrize("formula", _FAITHFUL_RR_DOMAIN)
def test_original_rulerunner_realizations_against_direct_oracle(formula: str) -> None:
    """Covers the repaired paper baseline, including the old nested-X holes."""
    monitors = [
        RuleEngine.from_formula(formula),
        CILPRunner.from_formula(formula),
        StructuredCILPRunner.from_formula(formula),
    ]
    for trace in _all_sparse_traces():
        expected = _direct_verdict(formula, trace)
        dense_trace = [
            {"a": cell.get("a", False), "b": cell.get("b", False)} for cell in trace
        ]
        for monitor in monitors:
            assert monitor.run(trace) is expected, (formula, trace, type(monitor))
            assert monitor.run(dense_trace) is expected, (
                formula,
                dense_trace,
                type(monitor),
            )


_COMPLETE_DOMAIN = _FAITHFUL_RR_DOMAIN + [
    "F (a & X b)",
    "G (a -> F b)",
    "G (a -> X b)",
    "a U (b & X a)",
    "F G a",
    "G F a",
    # Association- and precedence-sensitive: these disagree with MONA's own
    # reading of the string if our binarization or precedence is wrong.
    "a U b U a",
    "a -> b -> a",
    "a & b | a",
    "a | b & a",
    "!a & b",
    "F a U b",
    "a R b R a",
    "X a & b",
]


@pytest.mark.parametrize("formula", _COMPLETE_DOMAIN)
def test_complete_architectures_against_direct_oracle(formula: str) -> None:
    monitors = [
        SymbolicDFAMonitor.compile(formula),
        DeepDFAMonitorDense.compile(formula),
        DeepDFAMonitorFactored.compile(formula),
        ProgressionEngine.compile(formula),
        ProgressionRuleRunnerEagerMonitor.compile(formula),
        ProgressionRuleRunnerMonitor.compile(formula),
        ProgressionRuleRunnerStructuredMonitor.compile(formula),
    ]
    for trace in _all_sparse_traces():
        expected = _direct_verdict(formula, trace)
        for monitor in monitors:
            assert monitor.run(trace) is expected, (formula, trace, type(monitor))
