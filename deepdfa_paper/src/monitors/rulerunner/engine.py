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

Known limitation — shared-register temporal-instance conflation.
================================================================
The IJCNN 2014 encoding uses a single literal per subformula. For a
formula like `F(a & X b)`, F's reactivation creates a fresh
`(a & X b)` instance at each cell while X-b instances from prior
cells are still resolving. Both instances share the literal `[X b]`: the
fresh instance produces an unqualified `?`, while monitoring mode `M`
produces the prior-cell instance's resolution (definite T/F or `?^M`).
The binary operator's mode-R rules cannot tell which instance
each `[X b]` literal belongs to and fire on both, corrupting the
carry-over.

Any correct repair must distinguish the temporal meanings that this
state has merged, which is a structural redesign beyond IJCNN 2014's
one-slot address space.  This module intentionally remains the faithful
published baseline.  ``certify_rule_runner`` decides its correctness
formula-by-formula; nesting alone is neither necessary nor sufficient for
failure.  The bounded-event monitor adds finite offset-indexed event modules
for bounded islands, while the progression monitor carries residual roots and
is complete for the supported LTLf syntax.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.formula.compiler import Observation
from src.monitors.base import Verdict
from src.monitors.rulerunner.parse_tree import Node, Op, parse
from src.monitors.rulerunner.rules import Literal, Rule, RuleSystem, build_rules


@dataclass(frozen=True)
class RuleEngineState:
    """The complete mutable state of a :class:`RuleEngine`.

    Exhaustive analyses (``equivalence.py``, ``bounded_extrapolation.py``)
    save and restore engine states instead of reaching into private
    attributes.  Two invariants make that safe:

    * ``step`` reads only ``active`` and ``decided``.  Code that explores the
      reachable state graph may therefore use that pair as its key.
    * ``last_cell`` and ``seen_cell`` are read only by ``final_verdict``, and
      ``step`` always rewrites them.  A restored state whose ``last_cell`` is
      unknown is usable as long as ``final_verdict`` is called only after a
      subsequent ``step``.

    ``RuleEngine._STATE_FIELDS`` pins the attributes covered here; a test
    fails if the engine grows another one, so a new field cannot silently
    escape the round trip.
    """

    active: frozenset[Literal]
    last_cell: frozenset[Literal]
    seen_cell: bool
    decided: Verdict | None


class RuleEngine:
    """Pure-Python executor for a RuleSystem."""

    #: Mutable attributes carried by :class:`RuleEngineState`.  The remaining
    #: attributes (``_root``/``_rules``/``_depth``) are compile-time constants.
    _STATE_FIELDS = ("_state", "_last_cell", "_decided", "_seen_cell")
    _CONSTANT_FIELDS = ("_root", "_rules", "_depth")

    def __init__(self, root: Node) -> None:
        self._root = root
        self._rules: RuleSystem = build_rules(root)
        self._depth = root.depth
        self._state: set[Literal] = set(self._rules.initial_state)
        self._last_cell: frozenset[Literal] = frozenset(self._rules.initial_state)
        self._decided: Verdict | None = None
        self._seen_cell = False

    @classmethod
    def from_formula(cls, formula: str) -> "RuleEngine":
        return cls(parse(formula))

    def reset(self) -> None:
        self._state = set(self._rules.initial_state)
        self._last_cell = frozenset(self._rules.initial_state)
        self._decided = None
        self._seen_cell = False

    @property
    def rule_system(self) -> RuleSystem:
        """The compiled rule system (read-only view for analyses)."""
        return self._rules

    @property
    def atoms(self) -> frozenset[str]:
        """Observable atoms the rule system reads."""
        return frozenset(self._rules.atoms)

    # ---------------- explicit state round trip ----------------

    def state(self) -> RuleEngineState:
        """Snapshot the complete mutable state."""
        return RuleEngineState(
            active=frozenset(self._state),
            last_cell=self._last_cell,
            seen_cell=self._seen_cell,
            decided=self._decided,
        )

    def load_state(self, state: RuleEngineState) -> None:
        """Restore a state produced by :meth:`state`."""
        self._state = set(state.active)
        self._last_cell = state.last_cell
        self._seen_cell = state.seen_cell
        self._decided = state.decided

    # ---------------- per-cell step ----------------

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        self._seen_cell = True

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
        if not self._seen_cell:
            resolved = self._holds_empty(self._root)
            return Verdict.SATISFY if resolved else Verdict.VIOLATE
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

    def _holds_empty(self, node: Node) -> bool:
        if node.op is Op.ATOM:
            return node.atom == "true"
        if node.op is Op.NOT:
            return not self._holds_empty(node.children[0])
        if node.op is Op.AND:
            return self._holds_empty(node.children[0]) and self._holds_empty(
                node.children[1]
            )
        if node.op is Op.OR:
            return self._holds_empty(node.children[0]) or self._holds_empty(
                node.children[1]
            )
        if node.op is Op.IMPLIES:
            return not self._holds_empty(node.children[0]) or self._holds_empty(
                node.children[1]
            )
        if node.op in (Op.NEXT, Op.EVENTUALLY, Op.UNTIL):
            return False
        if node.op in (Op.WEAK_NEXT, Op.ALWAYS, Op.RELEASE):
            return True
        raise ValueError(node.op)

    def _resolve(self, node: Node, state: frozenset[Literal]) -> bool:
        """Recursive end-of-trace resolution. Definite values in `state`
        win; otherwise apply per-operator end semantics."""
        K = node.key
        if Literal(f"[{K}]T") in state:
            return True
        if Literal(f"[{K}]F") in state:
            return False

        if node.op is Op.ATOM:
            if node.atom == "true":
                return True
            if node.atom == "false":
                return False
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
            # Initial mode has no successor and is false.  Monitoring mode
            # already moved to the successor cell and therefore mirrors phi,
            # including phi's own END resolution.
            if Literal(f"[{K}]?^M") in state:
                return self._resolve(node.children[0], state)
            return False
        if node.op is Op.WEAK_NEXT:
            if Literal(f"[{K}]?^M") in state:
                return self._resolve(node.children[0], state)
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
