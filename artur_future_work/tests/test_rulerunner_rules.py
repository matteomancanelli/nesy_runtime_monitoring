"""Structural tests for the RuleRunner rule system.

We check that build_rules() instantiates the right templates with the
right literal names, and reproduces the IJCNN 2014 worked example for
`a ∨ ◇b`. Semantic equivalence with SymbolicDFAMonitor is tested in a
later step once the rule executor exists.
"""

from __future__ import annotations

from src.monitors.rulerunner.parse_tree import parse
from src.monitors.rulerunner.rules import Literal, Rule, build_rules


def _has_rule(rules: tuple[Rule, ...], body_names: set[str], head_name: str) -> bool:
    body_lits = {Literal(n[1:], negated=True) if n.startswith("~") else Literal(n)
                 for n in body_names}
    head_lit = Literal(head_name)
    return any(r.body == frozenset(body_lits) and r.head == head_lit for r in rules)


# ----------------- atoms -----------------


def test_atom_yields_two_eval_rules_and_no_react() -> None:
    rs = build_rules(parse("a"))
    assert len(rs.eval_rules) == 2
    assert len(rs.react_rules) == 0
    assert _has_rule(rs.eval_rules, {"R[a]", "obs:a"}, "[a]T")
    assert _has_rule(rs.eval_rules, {"R[a]", "~obs:a"}, "[a]F")
    assert rs.initial_state == frozenset({Literal("R[a]")})
    assert rs.atoms == ("a",)


# ----------------- NOT -----------------


def test_not_atom() -> None:
    rs = build_rules(parse("!a"))
    # NOT eval: T->F, F->T; the [a]? case is pruned (atoms never undecided).
    assert _has_rule(rs.eval_rules, {"R[!(a)]", "[a]T"}, "[!(a)]F")
    assert _has_rule(rs.eval_rules, {"R[!(a)]", "[a]F"}, "[!(a)]T")
    assert _has_rule(rs.react_rules, {"[!(a)]?"}, "R[!(a)]")
    # Atom's `?` literal is pruned, so no NOT rule consumes [a]?.
    not_rules = [r for r in rs.eval_rules if r.head.name.startswith("[!(a)]")]
    assert len(not_rules) == 2


def test_not_over_temporal_has_undecided_consumer() -> None:
    """NOT over a temporal child should consume [child]? (child IS undecidable)."""
    rs = build_rules(parse("!(F a)"))
    assert _has_rule(rs.eval_rules, {"R[!(F(a))]", "[F(a)]?"}, "[!(F(a))]?")


# ----------------- AND / OR / IMPLIES -----------------


def test_and_truth_table_mode_B() -> None:
    rs = build_rules(parse("a & b"))
    K = "(a & b)"
    # Definite cells of the 3x3 — all four atom×atom combinations.
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]T", "[b]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]T", "[b]F"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]F", "[b]T"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]F", "[b]F"}, f"[{K}]F")
    # `?` cells: atoms are never `?`, so no rule consuming [a]? or [b]? exists.
    assert not any("[a]?" in {b.name for b in r.body} for r in rs.eval_rules)


def test_and_modes_L_and_R_derive_from_correct_pin() -> None:
    """AND mode L is entered when ψ pins to T (the value that satisfies right
    so the conjunction depends solely on left). Same value pins R."""
    rs = build_rules(parse("(F a) & (F b)"))
    K = "(F(a) & F(b))"
    # Mode L's truth table: row at ψ=T -> {T→T, F→F, ?→?L}.
    assert _has_rule(rs.eval_rules, {f"R[{K}]^L", "[F(a)]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^L", "[F(a)]F"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^L", "[F(a)]?"}, f"[{K}]?^L")
    # Mode R symmetric on ψ.
    assert _has_rule(rs.eval_rules, {f"R[{K}]^R", "[F(b)]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^R", "[F(b)]F"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^R", "[F(b)]?"}, f"[{K}]?^R")


def test_or_modes_pin_to_F() -> None:
    """OR enters mode L when ψ pinned to F (left becomes the only hope)."""
    rs = build_rules(parse("(F a) | (F b)"))
    K = "(F(a) | F(b))"
    # Mode L: row at ψ=F -> {T→T, F→F, ?→?L}.
    assert _has_rule(rs.eval_rules, {f"R[{K}]^L", "[F(a)]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^L", "[F(a)]F"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^L", "[F(a)]?"}, f"[{K}]?^L")


def test_implies_truth_table() -> None:
    """φ → ψ: vacuously true when φ is F; falsifiable only when φ T and ψ F."""
    rs = build_rules(parse("a -> b"))
    K = "(a -> b)"
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]T", "[b]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]T", "[b]F"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]F", "[b]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]F", "[b]F"}, f"[{K}]T")


def test_binary_react_rules_have_three_modes() -> None:
    rs = build_rules(parse("a & b"))
    K = "(a & b)"
    react_heads = {(r.head.name,) for r in rs.react_rules}
    assert (f"R[{K}]^B",) in react_heads
    assert (f"R[{K}]^L",) in react_heads
    assert (f"R[{K}]^R",) in react_heads


# ----------------- EVENTUALLY / ALWAYS -----------------


def test_eventually_eval_and_react() -> None:
    rs = build_rules(parse("F a"))
    K = "F(a)"
    assert _has_rule(rs.eval_rules, {f"R[{K}]", "[a]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]", "[a]F"}, f"[{K}]?")
    # Reactivation re-installs the operand subtree (here just `a`).
    assert _has_rule(rs.react_rules, {f"[{K}]?"}, f"R[{K}]")
    assert _has_rule(rs.react_rules, {f"[{K}]?"}, "R[a]")


def test_always_eval_and_react() -> None:
    rs = build_rules(parse("G a"))
    K = "G(a)"
    assert _has_rule(rs.eval_rules, {f"R[{K}]", "[a]T"}, f"[{K}]?")
    assert _has_rule(rs.eval_rules, {f"R[{K}]", "[a]F"}, f"[{K}]F")
    assert _has_rule(rs.react_rules, {f"[{K}]?"}, "R[a]")


def test_eventually_reactivation_reinstalls_full_subtree() -> None:
    """For `F(a & b)`, the reactivation must re-install R[a], R[b], R[a&b]^B."""
    rs = build_rules(parse("F (a & b)"))
    K = "F((a & b))"
    body = {f"[{K}]?"}
    assert _has_rule(rs.react_rules, body, "R[a]")
    assert _has_rule(rs.react_rules, body, "R[b]")
    assert _has_rule(rs.react_rules, body, "R[(a & b)]^B")
    assert _has_rule(rs.react_rules, body, f"R[{K}]")


# ----------------- UNTIL / RELEASE -----------------


def test_until_truth_table() -> None:
    rs = build_rules(parse("a U b"))
    K = "(a U b)"
    R = f"R[{K}]"
    # (T,T)->T, (F,T)->T, (T,F)->?, (F,F)->F
    assert _has_rule(rs.eval_rules, {R, "[a]T", "[b]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {R, "[a]F", "[b]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {R, "[a]T", "[b]F"}, f"[{K}]?")
    assert _has_rule(rs.eval_rules, {R, "[a]F", "[b]F"}, f"[{K}]F")


def test_release_truth_table() -> None:
    rs = build_rules(parse("a R b"))
    K = "(a R b)"
    R = f"R[{K}]"
    # (T,T)->T, (T,F)->F, (F,T)->?, (F,F)->F
    assert _has_rule(rs.eval_rules, {R, "[a]T", "[b]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {R, "[a]T", "[b]F"}, f"[{K}]F")
    assert _has_rule(rs.eval_rules, {R, "[a]F", "[b]T"}, f"[{K}]?")
    assert _has_rule(rs.eval_rules, {R, "[a]F", "[b]F"}, f"[{K}]F")


# ----------------- NEXT / WEAK_NEXT -----------------


def test_next_mode_B_defers_unconditionally() -> None:
    rs = build_rules(parse("X a"))
    K = "X(a)"
    # The B-mode rule has just R[X(a)]^B in the body (no observation).
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B"}, f"[{K}]?^I")


def test_next_mode_A_mirrors_phi() -> None:
    rs = build_rules(parse("X a"))
    K = "X(a)"
    R_A = f"R[{K}]^A"
    assert _has_rule(rs.eval_rules, {R_A, "[a]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {R_A, "[a]F"}, f"[{K}]F")


def test_next_mode_I_react_installs_A_mode_and_phi_subtree() -> None:
    rs = build_rules(parse("X (F a)"))
    K = "X(F(a))"
    body = {f"[{K}]?^I"}
    # Transition B -> A re-installs the φ subtree fresh.
    assert _has_rule(rs.react_rules, body, f"R[{K}]^A")
    assert _has_rule(rs.react_rules, body, "R[F(a)]")
    assert _has_rule(rs.react_rules, body, "R[a]")
    # Mode-A reactivation only re-installs itself (φ's own ? handles φ).
    assert _has_rule(rs.react_rules, {f"[{K}]?^A"}, f"R[{K}]^A")


def test_weak_next_same_rules_as_next() -> None:
    """X and WX differ only at end-of-trace, which is handled by the wrapper."""
    x_rs = build_rules(parse("X a"))
    w_rs = build_rules(parse("WX a"))
    # Both produce the same number of rules with the same body/head shapes.
    assert len(x_rs.eval_rules) == len(w_rs.eval_rules)
    assert len(x_rs.react_rules) == len(w_rs.react_rules)


# ----------------- IJCNN 2014 worked example reproduction -----------------


def test_ijcnn2014_initial_state() -> None:
    """Paper §III worked example for a ∨ ◇b:
    initial state = {R[a], R[b], R[♦b], R[a∨♦b]B}."""
    rs = build_rules(parse("a | F b"))
    expected = frozenset({
        Literal("R[a]"),
        Literal("R[b]"),
        Literal("R[F(b)]"),
        Literal("R[(a | F(b))]^B"),
    })
    assert rs.initial_state == expected


def test_ijcnn2014_eval_rules_present() -> None:
    """The key eval rules from the worked example trace."""
    rs = build_rules(parse("a | F b"))
    K = "(a | F(b))"
    # The four rules the paper lists for monitoring the trace [c, a, b]:
    assert _has_rule(rs.eval_rules, {"R[a]", "~obs:a"}, "[a]F")
    assert _has_rule(rs.eval_rules, {"R[b]", "obs:b"}, "[b]T")
    assert _has_rule(rs.eval_rules, {"R[F(b)]", "[b]T"}, "[F(b)]T")
    assert _has_rule(rs.eval_rules, {"R[F(b)]", "[b]F"}, "[F(b)]?")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^B", "[a]F", "[F(b)]?"}, f"[{K}]?^R")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^R", "[F(b)]T"}, f"[{K}]T")
    assert _has_rule(rs.eval_rules, {f"R[{K}]^R", "[F(b)]?"}, f"[{K}]?^R")


def test_ijcnn2014_react_rules_present() -> None:
    rs = build_rules(parse("a | F b"))
    K = "(a | F(b))"
    # [♦b]? → R[b], R[♦b]
    assert _has_rule(rs.react_rules, {"[F(b)]?"}, "R[b]")
    assert _has_rule(rs.react_rules, {"[F(b)]?"}, "R[F(b)]")
    # [a∨♦b]?R → R[a∨♦b]R
    assert _has_rule(rs.react_rules, {f"[{K}]?^R"}, f"R[{K}]^R")


# ----------------- DAG sharing -----------------


def test_shared_subformula_yields_one_rule_set() -> None:
    """`(F b) & (F b)` after dedup has one Node for `F(b)`, so only one
    set of EVENTUALLY rules is emitted."""
    rs = build_rules(parse("(F b) & (F b)"))
    fb_rules = [r for r in rs.eval_rules if r.head.name == "[F(b)]T"
                or r.head.name == "[F(b)]?"]
    # Two eval rules for F(b): one for [b]T, one for [b]F.
    assert len(fb_rules) == 2


def test_no_duplicate_rules() -> None:
    """After dedup, no two rules share both body and head."""
    rs = build_rules(parse("G ((a | b) -> F (c & d))"))
    seen: set[tuple[frozenset, Literal]] = set()
    for r in rs.eval_rules + rs.react_rules:
        key = (r.body, r.head)
        assert key not in seen, f"duplicate rule: {r}"
        seen.add(key)


# ----------------- coverage of every operator -----------------


def test_build_rules_handles_every_operator() -> None:
    formulas = [
        "a", "!a", "a & b", "a | b", "a -> b",
        "X a", "WX a", "F a", "G a", "a U b", "a R b",
        "G (a -> F b)",
        "F ((a & X b) | (c & WX d))",
    ]
    for f in formulas:
        rs = build_rules(parse(f))
        assert len(rs.eval_rules) > 0, f
        assert rs.root_key == parse(f).key
        # Initial state lists every distinct subformula's R[.] literal.
        assert len(rs.initial_state) == len(list(parse(f).subformulae()))
