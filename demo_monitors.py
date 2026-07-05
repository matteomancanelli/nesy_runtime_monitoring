#!/usr/bin/env python
"""Quick demo: visualise the Symbolic DFA and the RuleRunner derivation.

TEMPORARY presentation aid (not part of the test suite) — a way to show
colleagues, given an LTLf formula, (1) the minimal DFA the symbolic /
DeepDFA paradigms walk, rendered with graphviz, and (2) the step-by-step
RuleRunner run: which evaluation and reactivation rules fire in each cell,
in the same tabular style as the example in latex/3_monitoring.tex.

Defaults reproduce that LaTeX example: the two formulas F(a & b) and
F(a & X b) over the trace ({a}, {}, {b}) — the good/bad pair that exposes
RuleRunner's nested-temporal limitation. Run it to check the paper's hand
derivation against the actual engine.

Usage:
    python demo_monitors.py                       # the two example formulas
    python demo_monitors.py --formula "F(a & X(b))" --trace "a; ; b"
    python demo_monitors.py --formula "G(a -> F(b))" --trace "a b; ; b"

Trace syntax: cells separated by ';', atoms within a cell by whitespace,
an empty cell means no atom holds. So "a; ; b" is ({a}, {}, {b}).

DFA images (+ .dot sources) are written to ./demo_output/.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from ltlf2dfa.parser.ltlf import LTLfParser

from src.monitors.base import Verdict
from src.monitors.rulerunner.engine import RuleEngine
from src.monitors.rulerunner.parse_tree import Node, parse
from src.monitors.rulerunner.rules import Literal, Rule, RuleSystem, build_rules
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

OUT_DIR = Path(__file__).parent / "demo_output"


# --------------------------------------------------------------------------
# Trace parsing / pretty-printing
# --------------------------------------------------------------------------

def parse_trace(spec: str, atoms: tuple[str, ...]) -> list[dict[str, bool]]:
    """"a; ; b" -> [{a:True,...}, {...}, {b:True,...}] over the given atoms."""
    cells = []
    for chunk in spec.split(";"):
        present = set(chunk.split())
        cells.append({atom: (atom in present) for atom in atoms})
    return cells


def show_cell(cell: dict[str, bool]) -> str:
    on = sorted(a for a, v in cell.items() if v)
    return "{" + ", ".join(on) + "}" if on else "{}"


# --------------------------------------------------------------------------
# DFA rendering (ltlf2dfa gives DOT directly; render with the `dot` CLI)
# --------------------------------------------------------------------------

def render_dfa(formula: str, slug: str) -> None:
    dot = LTLfParser()(formula).to_dfa()
    OUT_DIR.mkdir(exist_ok=True)
    dot_path = OUT_DIR / f"dfa_{slug}.dot"
    dot_path.write_text(dot)

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        print(f"  [DFA] wrote {dot_path} (install graphviz's `dot` to get a PNG)")
        return
    png_path = OUT_DIR / f"dfa_{slug}.png"
    subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    print(f"  [DFA] {png_path}")


# --------------------------------------------------------------------------
# Instrumented RuleRunner run (mirrors engine.py, but records what fires)
# --------------------------------------------------------------------------

def fmt_literal(lit: Literal) -> str:
    name = lit.name
    if name.startswith("obs:"):
        return ("~" if lit.negated else "") + f"obs({name[4:]})"
    return ("~" if lit.negated else "") + name


def fmt_rule(rule: Rule) -> str:
    # R[...] names first, then observations, then truth-value literals.
    def sort_key(lit: Literal) -> tuple[int, str]:
        if lit.name.startswith("R["):
            return (0, lit.name)
        if lit.name.startswith("obs:"):
            return (1, lit.name)
        return (2, lit.name)

    body = ", ".join(fmt_literal(l) for l in sorted(rule.body, key=sort_key))
    return f"{body}  ->  {fmt_literal(rule.head)}"


def fire_new(rules: tuple[Rule, ...], state: set[Literal]) -> list[Rule]:
    """Rules whose body is satisfied and whose head is not yet present."""
    fired = []
    for rule in rules:
        ok = all(
            (Literal(l.name) not in state) if l.negated else (l in state)
            for l in rule.body
        )
        if ok and rule.head not in state:
            fired.append(rule)
    return fired


def derive(root: Node, formula: str, trace: list[dict[str, bool]]) -> None:
    rules: RuleSystem = build_rules(root)
    depth = root.depth
    state: set[Literal] = set(rules.initial_state)
    last_cell: frozenset[Literal] = frozenset(state)
    decided: Verdict | None = None

    for t, obs in enumerate(trace):
        print(f"  t={t}   sigma_t = {show_cell(obs)}")

        cell_state = set(state)
        for atom in rules.atoms:
            if obs.get(atom, False):
                cell_state.add(Literal(f"obs:{atom}"))

        # Evaluation phase: fire to a fixpoint, recording newly derived facts.
        print("    evaluation:")
        any_eval = False
        for _ in range(depth + 1):
            fired = fire_new(rules.eval_rules, cell_state)
            if not fired:
                break
            for rule in fired:
                print(f"      {fmt_rule(rule)}")
                cell_state.add(rule.head)
                any_eval = True
        if not any_eval:
            print("      (nothing new)")

        last_cell = frozenset(cell_state)

        # Root verdict?
        k = rules.root_key
        if Literal(f"[{k}]T") in cell_state:
            decided = Verdict.SATISFY
        elif Literal(f"[{k}]F") in cell_state:
            decided = Verdict.VIOLATE
        if decided is not None:
            print(f"    --> root decided: {decided.name}\n")
            break

        # Reactivation phase: group fired rules by body (paper style).
        print("    reactivation:")
        fired_react = [r for r in rules.react_rules if _body_ok(r, cell_state)]
        grouped: dict[str, list[Literal]] = {}
        for r in fired_react:
            key = ", ".join(fmt_literal(l) for l in sorted(r.body, key=lambda x: x.name))
            grouped.setdefault(key, []).append(r.head)
        if grouped:
            for body, heads in grouped.items():
                head_str = ", ".join(fmt_literal(h) for h in heads)
                print(f"      {body}  ->  {head_str}")
        else:
            print("      (none)")
        state = {r.head for r in fired_react if r.head.name.startswith("R[")}
        print()

    # End-of-trace resolution.
    if decided is None:
        resolved = RuleEngine(root)._resolve(root, last_cell)
        verdict = Verdict.SATISFY if resolved else Verdict.VIOLATE
        print(f"  END of trace, root undecided -> resolve subformulae -> {verdict.name}\n")


def _body_ok(rule: Rule, state: set[Literal]) -> bool:
    return all(
        (Literal(l.name) not in state) if l.negated else (l in state)
        for l in rule.body
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_one(formula: str, trace_spec: str) -> None:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", formula).strip("_")
    root = parse(formula)
    atoms = tuple(sorted({n.atom for n in root.subformulae() if n.atom is not None}))
    trace = parse_trace(trace_spec, atoms)

    print("=" * 72)
    print(f"Formula: {formula}")
    print(f"Trace:   {' '.join(show_cell(c) for c in trace)}")
    print("=" * 72)

    # Ground-truth / DFA verdict and the RuleRunner verdict, for comparison.
    dfa_verdict = SymbolicDFAMonitor.compile(formula).run(trace)
    rr_verdict = RuleEngine(root).run(trace)
    print(f"Symbolic DFA verdict : {dfa_verdict.name}")
    print(f"RuleRunner  verdict  : {rr_verdict.name}"
          + ("   <-- DIVERGES from DFA" if rr_verdict != dfa_verdict else ""))
    print()

    render_dfa(formula, slug)
    print()
    print("RuleRunner derivation")
    print("-" * 72)
    derive(root, formula, trace)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formula", help="LTLf formula, e.g. 'F(a & X(b))'")
    ap.add_argument("--trace", help="cells ';'-separated, atoms whitespace-separated,"
                                    " e.g. 'a; ; b'")
    args = ap.parse_args()

    if args.formula:
        run_one(args.formula, args.trace or "a; ; b")
    else:
        # The latex/3_monitoring.tex good/bad example pair.
        run_one("F(a & b)", "a; ; b")
        print()
        run_one("F(a & X(b))", "a; ; b")


if __name__ == "__main__":
    main()
