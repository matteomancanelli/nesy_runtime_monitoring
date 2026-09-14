"""RuleRunner rule system: per-operator templates + builder.

A RuleSystem is pure data — a set of evaluation rules (cell-internal,
truth-value propagation) and reactivation rules (cell-to-cell carry-over
of rule names). The semantic executor lives in `engine.py`; the CILP
encoding lives in `cilp.py`. This file only knows how to *build* the
rule set from a parse tree, faithful to Perotti, Garcez, Boella
IJCNN 2014 §III.

Literal naming convention (strings only — modes are baked in):
  obs:a               atom a is observed in the current cell
  R[<key>]            rule name without mode (single-mode operator)
  R[<key>]^<mode>     rule name with mode (B/L/R for and/or/implies,
                      A/B/L/R for until, M for next/weak_next)
  [<key>]T            truth value: subformula <key> is true
  [<key>]F            truth value: subformula <key> is false
  [<key>]?            truth value: undecided (single-mode operator)
  [<key>]?^<mode>     truth value: undecided + state-machine mode

A Literal is `(name, negated)`. Negation appears only in atom-evaluation
bodies (~obs:a); rule names and truth values are always positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from src.monitors.rulerunner.parse_tree import Node, Op


@dataclass(frozen=True)
class Literal:
    name: str
    negated: bool = False

    def __str__(self) -> str:
        return ("~" if self.negated else "") + self.name


@dataclass(frozen=True)
class Rule:
    body: frozenset[Literal]
    head: Literal


@dataclass(frozen=True)
class RuleSystem:
    eval_rules: tuple[Rule, ...]
    react_rules: tuple[Rule, ...]
    initial_state: frozenset[Literal]
    atoms: tuple[str, ...]
    root_key: str


_MODES_BLR: tuple[str, ...] = ("B", "L", "R")
_MODES_UNTIL: tuple[str, ...] = ("A", "B", "L", "R")
_MODES_NEXT: tuple[str, ...] = ("", "M")


def _R(key: str, mode: str = "") -> Literal:
    return Literal(f"R[{key}]" + (f"^{mode}" if mode else ""))


def _T(key: str) -> Literal:
    return Literal(f"[{key}]T")


def _F(key: str) -> Literal:
    return Literal(f"[{key}]F")


def _U(key: str, mode: str = "") -> Literal:
    return Literal(f"[{key}]?" + (f"^{mode}" if mode else ""))


def _obs(atom: str, observed: bool) -> Literal:
    return Literal(f"obs:{atom}", negated=not observed)


def _initial_mode(node: Node) -> str:
    if node.op in (Op.AND, Op.OR, Op.IMPLIES):
        return "B"
    if node.op is Op.UNTIL:
        return "A"
    return ""


def _undecided_modes(node: Node) -> tuple[str, ...]:
    if node.op is Op.ATOM:
        return ()  # atoms always resolve to T or F at their cell
    if node.op in (Op.AND, Op.OR, Op.IMPLIES):
        return _MODES_BLR
    if node.op is Op.UNTIL:
        return _MODES_UNTIL
    if node.op in (Op.NEXT, Op.WEAK_NEXT):
        return _MODES_NEXT
    if node.op is Op.ALWAYS:
        return ("", "K")
    return ("",)


def _value_literals(node: Node, value: str) -> list[Literal]:
    """For value in {'T','F','?'} return all matching literals for `node`.

    A `?` enumerates over the node's undecided modes (so e.g. an AND
    child contributes three literals — ?^B, ?^L, ?^R — and a rule that
    consumes the child as `?` expands to three rules.).
    """
    if value == "T":
        return [_T(node.key)]
    if value == "F":
        return [_F(node.key)]
    if value == "?":
        return [_U(node.key, m) for m in _undecided_modes(node)]
    raise ValueError(value)


def _initial_activation(node: Node) -> set[Literal]:
    """Published RuleRunner initial state for the subsystem rooted at ``node``.

    This is deliberately *not* every syntactic descendant.  In particular,
    ``X phi`` and ``W phi`` initially activate only their own rule: ``phi`` is
    installed by reactivation in the following cell (Algorithm 2, lines 24--29).
    The same construction is reused whenever a temporal operator starts a fresh
    operand instance.
    """
    root = {_R(node.key, _initial_mode(node))}
    if node.op is Op.ATOM:
        return root
    if node.op is Op.NOT:
        child = node.children[0]
        # The published grammar is in NNF and evaluates !a directly from the
        # observation.  General negation is a conservative implementation
        # extension and therefore needs its child subsystem.
        if child.op is Op.ATOM and child.atom not in ("true", "false"):
            return root
        return root | _initial_activation(child)
    if node.op in (Op.NEXT, Op.WEAK_NEXT):
        return root
    out = set(root)
    for child in node.children:
        out |= _initial_activation(child)
    return out


def _subtree_reinstall(node: Node) -> list[Literal]:
    """Re-install ``node`` using the same activation boundary as INITIALISE."""
    return list(_initial_activation(node))


# ---------------- per-operator templates ----------------


def _atom(node: Node) -> tuple[list[Rule], list[Rule]]:
    assert node.atom is not None
    R = _R(node.key)
    if node.atom == "true":
        return ([Rule(frozenset({R}), _T(node.key))], [])
    if node.atom == "false":
        return ([Rule(frozenset({R}), _F(node.key))], [])
    return (
        [
            Rule(frozenset({R, _obs(node.atom, True)}), _T(node.key)),
            Rule(frozenset({R, _obs(node.atom, False)}), _F(node.key)),
        ],
        [],
    )


def _not(node: Node) -> tuple[list[Rule], list[Rule]]:
    (child,) = node.children
    R = _R(node.key)
    if child.op is Op.ATOM and child.atom not in ("true", "false"):
        assert child.atom is not None
        return (
            [
                Rule(frozenset({R, _obs(child.atom, True)}), _F(node.key)),
                Rule(frozenset({R, _obs(child.atom, False)}), _T(node.key)),
            ],
            [],
        )
    eval_rules: list[Rule] = [
        Rule(frozenset({R, _T(child.key)}), _F(node.key)),
        Rule(frozenset({R, _F(child.key)}), _T(node.key)),
    ]
    for m in _undecided_modes(child):
        eval_rules.append(Rule(frozenset({R, _U(child.key, m)}), _U(node.key)))
    react_rules = [Rule(frozenset({_U(node.key)}), R)]
    return eval_rules, react_rules


def _binary_propositional(
    node: Node,
    table_B: dict[tuple[str, str], str],
) -> tuple[list[Rule], list[Rule]]:
    """Shared expander for AND/OR/IMPLIES.

    `table_B` is the 3×3 truth table for mode B. Each value in
    {"T","F","?B","?L","?R"} names the resulting head. Modes L and R are
    derived automatically: mode L watches only [φ] (replicates the φ
    column of the mode-B table, dropping ψ); mode R watches only [ψ].
    """
    phi, psi = node.children
    K = node.key
    eval_rules: list[Rule] = []

    def head_for(tag: str) -> Literal:
        if tag == "T":
            return _T(K)
        if tag == "F":
            return _F(K)
        return _U(K, tag[1])  # "?B" -> ?^B, "?L" -> ?^L, "?R" -> ?^R

    R_B = _R(K, "B")
    for (pv, qv), tag in table_B.items():
        head = head_for(tag)
        for p_lit, q_lit in product(_value_literals(phi, pv), _value_literals(psi, qv)):
            eval_rules.append(Rule(frozenset({R_B, p_lit, q_lit}), head))

    # Mode L is entered from B when ψ settled at a specific definite
    # value (psi_pin) — so L's truth table reads B's column at that pin,
    # not at ψ=?. Same for R, with the roles reversed.
    psi_pin = next(qv for (pv, qv), tag in table_B.items() if pv == "?" and tag == "?L")
    phi_pin = next(pv for (pv, qv), tag in table_B.items() if qv == "?" and tag == "?R")

    R_L = _R(K, "L")
    for pv in ("T", "F", "?"):
        tag = table_B.get((pv, psi_pin))
        if tag is None:
            continue
        head = _U(K, "L") if tag.startswith("?") else head_for(tag)
        for p_lit in _value_literals(phi, pv):
            eval_rules.append(Rule(frozenset({R_L, p_lit}), head))

    R_R = _R(K, "R")
    for qv in ("T", "F", "?"):
        tag = table_B.get((phi_pin, qv))
        if tag is None:
            continue
        head = _U(K, "R") if tag.startswith("?") else head_for(tag)
        for q_lit in _value_literals(psi, qv):
            eval_rules.append(Rule(frozenset({R_R, q_lit}), head))

    react_rules = [Rule(frozenset({_U(K, m)}), _R(K, m)) for m in _MODES_BLR]
    return eval_rules, react_rules


def _and(node: Node) -> tuple[list[Rule], list[Rule]]:
    table = {
        ("T", "T"): "T",
        ("T", "F"): "F",
        ("T", "?"): "?R",
        ("F", "T"): "F",
        ("F", "F"): "F",
        ("F", "?"): "F",
        ("?", "T"): "?L",
        ("?", "F"): "F",
        ("?", "?"): "?B",
    }
    return _binary_propositional(node, table)


def _or(node: Node) -> tuple[list[Rule], list[Rule]]:
    table = {
        ("T", "T"): "T",
        ("T", "F"): "T",
        ("T", "?"): "T",
        ("F", "T"): "T",
        ("F", "F"): "F",
        ("F", "?"): "?R",
        ("?", "T"): "T",
        ("?", "F"): "?L",
        ("?", "?"): "?B",
    }
    return _binary_propositional(node, table)


def _implies(node: Node) -> tuple[list[Rule], list[Rule]]:
    table = {
        ("T", "T"): "T",
        ("T", "F"): "F",
        ("T", "?"): "?R",
        ("F", "T"): "T",
        ("F", "F"): "T",
        ("F", "?"): "T",
        ("?", "T"): "T",
        ("?", "F"): "?L",
        ("?", "?"): "?B",
    }
    return _binary_propositional(node, table)


def _eventually(node: Node) -> tuple[list[Rule], list[Rule]]:
    (child,) = node.children
    R = _R(node.key)
    eval_rules: list[Rule] = [
        Rule(frozenset({R, _T(child.key)}), _T(node.key)),
        Rule(frozenset({R, _F(child.key)}), _U(node.key)),
    ]
    for m in _undecided_modes(child):
        eval_rules.append(Rule(frozenset({R, _U(child.key, m)}), _U(node.key)))

    # Re-install operand subtree fresh next cell — [φ]F at this cell does
    # NOT fire φ's own reactivation, so we must restart φ-monitoring.
    react_body = frozenset({_U(node.key)})
    react_rules = [Rule(react_body, R)] + [
        Rule(react_body, lit) for lit in _subtree_reinstall(child)
    ]
    return eval_rules, react_rules


def _always(node: Node) -> tuple[list[Rule], list[Rule]]:
    (child,) = node.children
    R = _R(node.key)
    eval_rules: list[Rule] = [
        Rule(frozenset({R, _T(child.key)}), _U(node.key, "K")),
        Rule(frozenset({R, _F(child.key)}), _F(node.key)),
    ]
    for m in _undecided_modes(child):
        eval_rules.append(Rule(frozenset({R, _U(child.key, m)}), _U(node.key)))

    react_rules: list[Rule] = []
    for mode in ("", "K"):
        react_body = frozenset({_U(node.key, mode)})
        react_rules.append(Rule(react_body, R))
        react_rules.extend(Rule(react_body, lit) for lit in _subtree_reinstall(child))
    return eval_rules, react_rules


def _until_release(
    node: Node, table: dict[tuple[str, str], str]
) -> tuple[list[Rule], list[Rule]]:
    """Shared expander for U and R (single-mode binary temporals)."""
    phi, psi = node.children
    R = _R(node.key)
    eval_rules: list[Rule] = []

    def head_for(tag: str) -> Literal:
        if tag == "T":
            return _T(node.key)
        if tag == "F":
            return _F(node.key)
        return _U(node.key)

    for (pv, qv), tag in table.items():
        head = head_for(tag)
        for p_lit, q_lit in product(_value_literals(phi, pv), _value_literals(psi, qv)):
            eval_rules.append(Rule(frozenset({R, p_lit, q_lit}), head))

    react_body = frozenset({_U(node.key)})
    react_rules = [Rule(react_body, R)] + [
        Rule(react_body, lit)
        for lit in _subtree_reinstall(phi) + _subtree_reinstall(psi)
    ]
    return eval_rules, react_rules


def _until(node: Node) -> tuple[list[Rule], list[Rule]]:
    """The four published U evaluation tables (Figure 1: UA/UB/UL/UR)."""
    phi, psi = node.children
    K = node.key
    eval_rules: list[Rule] = []

    def head(tag: str) -> Literal:
        if tag == "T":
            return _T(K)
        if tag == "F":
            return _F(K)
        return _U(K, tag[1])

    tables = {
        "A": {
            ("T", "T"): "T",
            ("T", "?"): "?A",
            ("T", "F"): "?A",
            ("?", "T"): "T",
            ("?", "?"): "?A",
            ("?", "F"): "?B",
            ("F", "T"): "T",
            ("F", "?"): "?R",
            ("F", "F"): "F",
        },
        "B": {
            ("T", "T"): "T",
            ("T", "?"): "?A",
            ("T", "F"): "?A",
            ("?", "T"): "?L",
            ("?", "?"): "?B",
            ("?", "F"): "?B",
            ("F", "T"): "F",
            ("F", "?"): "F",
            ("F", "F"): "F",
        },
    }
    for mode, table in tables.items():
        for (pv, qv), tag in table.items():
            for p_lit, q_lit in product(
                _value_literals(phi, pv), _value_literals(psi, qv)
            ):
                eval_rules.append(
                    Rule(frozenset({_R(K, mode), p_lit, q_lit}), head(tag))
                )

    for mode, child in (("L", phi), ("R", psi)):
        for value, tag in (("T", "T"), ("?", f"?{mode}"), ("F", "F")):
            for lit in _value_literals(child, value):
                eval_rules.append(Rule(frozenset({_R(K, mode), lit}), head(tag)))

    react_rules: list[Rule] = []
    reinstall = _subtree_reinstall(phi) + _subtree_reinstall(psi)
    for mode in _MODES_UNTIL:
        body = frozenset({_U(K, mode)})
        react_rules.append(Rule(body, _R(K, mode)))
        react_rules.extend(Rule(body, lit) for lit in reinstall)
    return eval_rules, react_rules


def _release(node: Node) -> tuple[list[Rule], list[Rule]]:
    table = {
        ("T", "T"): "T",
        ("T", "F"): "F",
        ("T", "?"): "?",
        ("F", "T"): "?",
        ("F", "F"): "F",
        ("F", "?"): "?",
        ("?", "T"): "?",
        ("?", "F"): "F",
        ("?", "?"): "?",
    }
    return _until_release(node, table)


def _next_like(node: Node) -> tuple[list[Rule], list[Rule]]:
    """Shared expander for X and WX (semantics differ only at end-of-trace,
    which is handled by the wrapper, not here)."""
    (child,) = node.children
    K = node.key
    R_initial = _R(K)
    R_monitor = _R(K, "M")

    # Initial cell: defer unconditionally.  The unqualified `?` is what the
    # END table distinguishes from monitoring mode `?M`.
    eval_rules: list[Rule] = [Rule(frozenset({R_initial}), _U(K))]

    # Monitoring mode (cell 2+): mirror phi's truth value.
    eval_rules.append(Rule(frozenset({R_monitor, _T(child.key)}), _T(K)))
    eval_rules.append(Rule(frozenset({R_monitor, _F(child.key)}), _F(K)))
    for m in _undecided_modes(child):
        eval_rules.append(Rule(frozenset({R_monitor, _U(child.key, m)}), _U(K, "M")))

    # Reactivation:
    # ? (just deferred) installs monitoring mode and the child's *published
    # initial state*.  ?M keeps only the wrapper alive; an undecided child
    # reactivates its own active subsystem.
    react_rules: list[Rule] = []
    init_body = frozenset({_U(K)})
    react_rules.append(Rule(init_body, R_monitor))
    react_rules.extend(Rule(init_body, lit) for lit in _subtree_reinstall(child))
    react_rules.append(Rule(frozenset({_U(K, "M")}), R_monitor))
    return eval_rules, react_rules


_TEMPLATES = {
    Op.ATOM: _atom,
    Op.NOT: _not,
    Op.AND: _and,
    Op.OR: _or,
    Op.IMPLIES: _implies,
    Op.EVENTUALLY: _eventually,
    Op.ALWAYS: _always,
    Op.UNTIL: _until,
    Op.RELEASE: _release,
    Op.NEXT: _next_like,
    Op.WEAK_NEXT: _next_like,
}


def build_rules(root: Node) -> RuleSystem:
    """Walk the parse-tree DAG and assemble the full RuleSystem."""
    eval_rules: list[Rule] = []
    react_rules: list[Rule] = []
    atoms: set[str] = set()

    for node in root.subformulae():
        e, r = _TEMPLATES[node.op](node)
        eval_rules.extend(e)
        react_rules.extend(r)
        if node.op is Op.ATOM and node.atom not in ("true", "false"):
            assert node.atom is not None
            atoms.add(node.atom)

    initial_state = frozenset(_initial_activation(root))

    # Dedup by (body, head); CILP doesn't need duplicate clauses.
    eval_rules = list({(r.body, r.head): r for r in eval_rules}.values())
    react_rules = list({(r.body, r.head): r for r in react_rules}.values())

    return RuleSystem(
        eval_rules=tuple(eval_rules),
        react_rules=tuple(react_rules),
        initial_state=initial_state,
        atoms=tuple(sorted(atoms)),
        root_key=root.key,
    )
