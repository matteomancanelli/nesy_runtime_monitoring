"""Structured per-node CILP network — IJCNN 2015, Fig. 5 ("flattened tree").

A self-contained, *alternative* realization of the RuleRunner neural monitor
that mirrors the parse tree literally: one CILP subnetwork per parse-tree node
(per subformula/operator), composed by an explicit bottom-up (post-order)
sweep, instead of the single flat network + ``depth+1`` fixpoint loop used by
[cilp.CILPRunner](cilp.py).

Motivation
----------
`cilp.CILPRunner` pools every rule into one network and reaches the per-cell
fixpoint by iterating the whole network ``depth+1`` times. IJCNN 2015 (Perotti,
d'Avila Garcez, Boella — "Neural-Symbolic Monitoring and Adaptation", Fig. 5)
instead keeps one subnetwork per parse-tree node and composes them
horizontally: child-node outputs feed parent-node inputs, while recurrent
connections carry the reactivation (``R[.]``) state across cells. That per-node
decomposition is precisely what their *local* (single-node) adaptation operates
on — so this module is kept for the deferred Paper-B adaptation work.

Status: NOT wired into the package API, `RuleRunnerMonitor`, or the
experiments. It is a drop-in alternative kept on the shelf. By construction it
returns the *same* per-cell verdicts as `CILPRunner` and the symbolic
`RuleEngine`: partitioning the rules by node is purely organizational, and a
single post-order sweep computes the same least model as the flat fixpoint,
because each node's truth value depends only on its children (which precede it
in the sweep) plus its own carried mode and the cell observations.

Reuse: the CILP numerics (`_layer_matrices`, sign activation) and the
end-of-trace recipe (`_resolve_end`) are imported from `cilp`, so the two
realizations are guaranteed numerically identical and there is no second copy
to keep in sync.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.rulerunner.cilp import (
    _layer_matrices,
    _resolve_end,
    _step_activation,
)
from src.monitors.rulerunner.parse_tree import Node, parse
from src.monitors.rulerunner.rules import Rule, RuleSystem, build_rules

# A per-layer CILP weight bundle: (W_ih, b_h, W_ho, b_o).
_Layer = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _key_of(literal_name: str) -> str:
    """Return the subformula key embedded in a literal name.

    Truth literals are ``[<key>]...`` and rule names are ``R[<key>]...``;
    keys never contain square brackets (they are ``str(ast)`` rendered with
    parentheses), so the key is whatever sits between the first ``[`` and the
    next ``]``.
    """
    i = literal_name.index("[")
    j = literal_name.index("]", i)
    return literal_name[i + 1 : j]


def _partition_rules(
    rs: RuleSystem, nodes: list[Node]
) -> tuple[dict[str, list[Rule]], dict[str, list[Rule]]]:
    """Group evaluation/reactivation rules by the parse-tree node that owns them.

    An evaluation rule is owned by the node whose truth value it writes (its
    head ``[<key>]...``). A reactivation rule is owned by the node whose
    undecided literal triggers it (its single body literal ``[<key>]?...``);
    its heads may install *other* nodes' rule names (the operand-subtree
    reinstall of F/G/U/R and X/WX), which is fine — every node's reactivation
    outputs are OR-ed into the one shared next-cell state, so it does not
    matter which subnetwork emits a given ``R[.]``.
    """
    eval_by: dict[str, list[Rule]] = {n.key: [] for n in nodes}
    react_by: dict[str, list[Rule]] = {n.key: [] for n in nodes}

    for r in rs.eval_rules:
        eval_by[_key_of(r.head.name)].append(r)
    for r in rs.react_rules:
        (body_lit,) = r.body  # reactivation bodies are always a single `?` literal
        react_by[_key_of(body_lit.name)].append(r)
    return eval_by, react_by


class StructuredCILPRunner:
    """Per-node (IJCNN 2015 Fig. 5) realization of the RuleRunner monitor.

    Same public interface as `cilp.CILPRunner` (``from_formula``, ``step``,
    ``run``, ``final_verdict``, ``reset``), so it is drop-in swappable. The
    per-cell evaluation is an explicit bottom-up sweep over one subnetwork per
    node rather than a flat fixpoint loop; verdicts are identical to
    `CILPRunner` by construction.

    The per-node subnetworks are kept addressable in ``self.eval_net`` /
    ``self.react_net`` (keyed by ``Node.key``) so a future per-node adaptation
    step can train a single operator's weights in isolation.
    """

    def __init__(self, root: Node) -> None:
        self._root = root
        self._rs: RuleSystem = build_rules(root)
        # `subformulae()` yields each distinct node bottom-up (children before
        # parents, root last) — exactly the post-order the sweep needs, valid
        # even for the shared-subformula DAG.
        self._nodes: list[Node] = list(root.subformulae())
        self._build()

        self._state: torch.Tensor = self._initial_x.clone()
        self._last_cell: torch.Tensor = self._initial_x.clone()
        self._decided: Verdict | None = None

    @classmethod
    def from_formula(cls, formula: str) -> "StructuredCILPRunner":
        return cls(parse(formula))

    # -- compilation --

    def _build(self) -> None:
        rs = self._rs
        lit_names: set[str] = set()
        for r in rs.eval_rules + rs.react_rules:
            lit_names.update(lit.name for lit in r.body)
            lit_names.add(r.head.name)
        lit_names.update(lit.name for lit in rs.initial_state)
        lit_names.add(f"[{rs.root_key}]T")
        lit_names.add(f"[{rs.root_key}]F")
        lit_names.update(f"obs:{a}" for a in rs.atoms)

        self._index = {n: i for i, n in enumerate(sorted(lit_names))}
        self._n = len(self._index)

        eval_by, react_by = _partition_rules(rs, self._nodes)
        # One CILP subnetwork per node, over the shared literal space. Each
        # subnetwork's outputs are silent (-1) everywhere except the literals
        # that node writes, so OR-accumulating a node's output into the global
        # vector touches only that node's slots.
        self.eval_net: dict[str, _Layer] = {}
        self.react_net: dict[str, _Layer] = {}
        for node in self._nodes:
            self.eval_net[node.key] = _layer_matrices(
                tuple(eval_by[node.key]), self._index, self._n
            )
            self.react_net[node.key] = _layer_matrices(
                tuple(react_by[node.key]), self._index, self._n
            )

        self._R_mask = torch.zeros(self._n, dtype=torch.bool)
        for name, idx in self._index.items():
            if name.startswith("R["):
                self._R_mask[idx] = True

        self._initial_x = torch.full((self._n,), -1.0)
        for lit in rs.initial_state:
            self._initial_x[self._index[lit.name]] = 1.0

        self._T_idx = self._index[f"[{rs.root_key}]T"]
        self._F_idx = self._index[f"[{rs.root_key}]F"]

    @staticmethod
    def _forward(layer: _Layer, x: torch.Tensor) -> torch.Tensor:
        """One CILP rule-application step for a single node's subnetwork."""
        W_ih, b_h, W_ho, b_o = layer
        h = _step_activation(W_ih @ x + b_h)
        return _step_activation(W_ho @ h + b_o)

    # -- public API (mirrors CILPRunner) --

    def reset(self) -> None:
        self._state = self._initial_x.clone()
        self._last_cell = self._initial_x.clone()
        self._decided = None

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided

        # Cell input: -1 everywhere, carry the R[.] recurrent state, clamp obs.
        x = torch.full((self._n,), -1.0)
        x[self._R_mask] = self._state[self._R_mask]
        for atom in self._rs.atoms:
            x[self._index[f"obs:{atom}"]] = 1.0 if obs.get(atom, False) else -1.0

        # Evaluation: a single bottom-up sweep. Each node is visited after its
        # children, so the one pass through its subnetwork sees their final
        # truth values — no global fixpoint loop is needed.
        for node in self._nodes:
            x = torch.maximum(x, self._forward(self.eval_net[node.key], x))

        self._last_cell = x.clone()
        if x[self._T_idx] > 0:
            self._decided = Verdict.SATISFY
            return Verdict.SATISFY
        if x[self._F_idx] > 0:
            self._decided = Verdict.VIOLATE
            return Verdict.VIOLATE

        # Reactivation: every node emits its R[.] heads; OR them (recurrent
        # connections) into the next cell's state.
        next_state = torch.full((self._n,), -1.0)
        for node in self._nodes:
            y = self._forward(self.react_net[node.key], x)
            next_state[self._R_mask] = torch.maximum(
                next_state[self._R_mask], y[self._R_mask]
            )
        self._state = next_state
        return Verdict.UNDECIDED

    def run(self, trace: Iterable[Observation]) -> Verdict:
        self.reset()
        for obs in trace:
            v = self.step(obs)
            if v is not Verdict.UNDECIDED:
                return v
        return self.final_verdict()

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        last = self._last_cell

        def in_state(name: str) -> bool:
            idx = self._index.get(name)
            return idx is not None and last[idx].item() > 0

        resolved = _resolve_end(self._root, in_state)
        return Verdict.SATISFY if resolved else Verdict.VIOLATE


class StructuredRuleRunnerMonitor(Monitor):
    """`Monitor`-ABC adapter over `StructuredCILPRunner`.

    Not exported from the package ``__init__`` and not used by the
    experiments — provided so the structured network can be slotted into any
    `Monitor`-based harness on demand, exactly like `RuleRunnerMonitor` wraps
    `CILPRunner`.
    """

    def __init__(self, runner: StructuredCILPRunner) -> None:
        self._runner = runner

    @classmethod
    def compile(cls, formula: str) -> "StructuredRuleRunnerMonitor":
        return cls(StructuredCILPRunner.from_formula(formula))

    def step(self, obs: Observation) -> Verdict:
        return self._runner.step(obs)

    def final_verdict(self) -> Verdict:
        return self._runner.final_verdict()

    def reset(self) -> None:
        self._runner.reset()
