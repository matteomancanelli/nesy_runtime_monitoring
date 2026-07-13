"""Symbolic executor for a RuleRunner rule system.

This is Algorithm 2 from Perotti, Garcez, Boella, IJCNN 2014 in plain
Python: each cell goes through an evaluation phase (parallel rule
firing, repeated up to parse-tree depth — see step-2 plan rationale)
followed by a single reactivation pass that produces the next cell's
rule-name state.

Purpose: a reference semantics. Step 3 (CILP) will produce the same
verdicts via a neural network; this engine is the oracle we compare
against. It is also what the equivalence test against
`SymbolicDFAMonitor` uses to catch template bugs in `rules.py`.

End-of-trace handling lives here, not in `rules.py`, because it needs
the parse-tree DAG (per-operator end semantics + the pin values for
binary operators in modes L/R, where one child was settled mid-trace
and only the mode marker remains in state).

Known limitation — nested temporal under F/G/U/R.
=================================================
The IJCNN 2014 encoding uses a single literal per subformula. For a
formula like `F(a & X b)`, F's reactivation creates a fresh
`(a & X b)` instance at each cell while X-b instances from prior
cells are still resolving. Both instances share the literal
`[X b]`: mode A produces the prior-cell instance's resolution
(definite T/F at cell N), mode B produces the fresh defer `?^I`.
The binary operator's mode-R rules cannot tell which instance
each `[X b]` literal belongs to and fire on both, corrupting the
carry-over.

A correct fix would scope literals by cell offset (e.g. `[X b @ now]`
vs `[X b @ prev]`), which is a structural redesign that goes beyond
what IJCNN 2014 documents. We accept the limitation here: the
engine matches `SymbolicDFAMonitor` on flat-temporal and on
temporal-under-propositional formulas (the IJCNN scalability suite
this project benchmarks), and disagrees on nested-temporal-under-
F/G/U/R formulas like the BPM response pattern `G(a → F b)`. This is
itself a relevant finding for Paper A — the DFA-based monitor has no
such restriction because its single canonical state machine doesn't
conflate concurrent instances.
"""

from __future__ import annotations

from src.formula.compiler import Observation
from src.monitors.base import Verdict
from src.monitors.rulerunner.parse_tree import Node, Op, parse
from src.monitors.rulerunner.rules import Literal, Rule, RuleSystem, build_rules


class RuleEngine:
    """Pure-Python executor for a RuleSystem."""

    def __init__(self, root: Node) -> None:
        self._root = root
        self._rules: RuleSystem = build_rules(root)
        self._depth = root.depth
        self._state: set[Literal] = set(self._rules.initial_state)
        self._last_cell: frozenset[Literal] = frozenset(self._rules.initial_state)
        self._decided: Verdict | None = None

    @classmethod
    def from_formula(cls, formula: str) -> "RuleEngine":
        return cls(parse(formula))

    def reset(self) -> None:
        self._state = set(self._rules.initial_state)
        self._last_cell = frozenset(self._rules.initial_state)
        self._decided = None

    # ---------------- per-cell step ----------------

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided

        cell_state: set[Literal] = set(self._state)
        for atom in self._rules.atoms:
            if obs.get(atom, False):
                cell_state.add(Literal(f"obs:{atom}"))

        # Evaluation phase: parallel rule firing. Truth values propagate
        # one parse-tree level per pass; we need depth+1 passes (atom
        # eval fires from cell-input level into level 0, then one pass
        # per internal node up to and including the root).
        for _ in range(self._depth + 1):
            produced = self._fire(self._rules.eval_rules, cell_state)
            new_facts = produced - cell_state
            if not new_facts:
                break
            cell_state |= new_facts

        self._last_cell = frozenset(cell_state)
        verdict = self._read_root_verdict(cell_state)
        if verdict is not Verdict.UNDECIDED:
            self._decided = verdict
            return verdict

        # Reactivation: single parallel pass. We carry only R[.] literals,
        # and crucially we rebuild from scratch (no skip against the
        # current cell state — those R[.]s belong to *this* cell, not the
        # next one).
        next_facts = self._fire(self._rules.react_rules, cell_state)
        self._state = {lit for lit in next_facts if lit.name.startswith("R[")}
        return Verdict.UNDECIDED

    def run(self, trace) -> Verdict:
        self.reset()
        for obs in trace:
            v = self.step(obs)
            if v is not Verdict.UNDECIDED:
                return v
        return self.final_verdict()

    # ---------------- end-of-trace resolution ----------------

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        resolved = self._resolve(self._root, self._last_cell)
        return Verdict.SATISFY if resolved else Verdict.VIOLATE

    # ---------------- internals ----------------

    @staticmethod
    def _fire(rules: tuple[Rule, ...], state: set[Literal]) -> set[Literal]:
        """Return the heads of every rule whose body is satisfied by
        `state`. Does NOT skip heads already in `state` — that's a
        caller-side concern (eval wants to detect a fixed point;
        react wants every firing rule's head)."""
        produced: set[Literal] = set()
        for rule in rules:
            satisfied = True
            for lit in rule.body:
                if lit.negated:
                    if Literal(lit.name) in state:
                        satisfied = False
                        break
                else:
                    if lit not in state:
                        satisfied = False
                        break
            if satisfied:
                produced.add(rule.head)
        return produced

    def _read_root_verdict(self, state: set[Literal]) -> Verdict:
        k = self._rules.root_key
        if Literal(f"[{k}]T") in state:
            return Verdict.SATISFY
        if Literal(f"[{k}]F") in state:
            return Verdict.VIOLATE
        return Verdict.UNDECIDED

    def _resolve(self, node: Node, state: frozenset[Literal]) -> bool:
        """Recursive end-of-trace resolution. Definite values in `state`
        win; otherwise apply per-operator end semantics."""
        K = node.key
        if Literal(f"[{K}]T") in state:
            return True
        if Literal(f"[{K}]F") in state:
            return False

        if node.op is Op.ATOM:
            # R[a] was not active at the last cell; the atom's value is
            # irrelevant to the active mode of any parent.
            return False
        if node.op is Op.NOT:
            return not self._resolve(node.children[0], state)
        if node.op in (Op.AND, Op.OR, Op.IMPLIES):
            return self._resolve_binary_propositional(node, state)
        # Single-temporal operators whose end-of-trace value tracks a
        # specific child's end-of-trace value:
        #   F φ end = φ end          (vacuous over the trace iff φ ends T)
        #   G φ end = φ end          (dual)
        #   φ U ψ end = ψ end        (U requires ψ at the last cell)
        #   φ R ψ end = ψ end        (R requires ψ throughout, last cell pins)
        if node.op in (Op.EVENTUALLY, Op.ALWAYS):
            return self._resolve(node.children[0], state)
        if node.op in (Op.UNTIL, Op.RELEASE):
            return self._resolve(node.children[1], state)
        if node.op is Op.NEXT:
            return False  # strong next: cell i+1 doesn't exist -> F
        if node.op is Op.WEAK_NEXT:
            return True
        raise ValueError(node.op)

    # End-of-trace pin values: when a binary op is in mode L, the right
    # child was settled mid-trace at the pin value (the value that
    # triggered the L transition in the B-mode table). Same for mode R.
    _PIN_RIGHT_FOR_L: dict[Op, bool] = {Op.AND: True, Op.OR: False, Op.IMPLIES: False}
    _PIN_LEFT_FOR_R: dict[Op, bool] = {Op.AND: True, Op.OR: False, Op.IMPLIES: True}

    def _resolve_binary_propositional(
        self, node: Node, state: frozenset[Literal]
    ) -> bool:
        K = node.key
        phi, psi = node.children

        if Literal(f"[{K}]?^L") in state:
            left = self._resolve(phi, state)
            right = self._PIN_RIGHT_FOR_L[node.op]
        elif Literal(f"[{K}]?^R") in state:
            left = self._PIN_LEFT_FOR_R[node.op]
            right = self._resolve(psi, state)
        else:
            # ?^B or completely missing: recurse both children.
            left = self._resolve(phi, state)
            right = self._resolve(psi, state)

        if node.op is Op.AND:
            return left and right
        if node.op is Op.OR:
            return left or right
        return (not left) or right  # IMPLIES
