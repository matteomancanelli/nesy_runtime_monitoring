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

Status: wired into the package API (`StructuredRuleRunnerMonitor`) and the
experiments as a second RuleRunner data point. It is device-aware and batches
across traces (`batch_run` runs a batched per-node sweep on CPU or CUDA). By
construction it returns the *same* per-cell verdicts as `CILPRunner` and the
symbolic `RuleEngine`: partitioning the rules by node is purely organizational,
and a single post-order sweep computes the same least model as the flat
fixpoint, because each node's truth value depends only on its children (which
precede it in the sweep) plus its own carried mode and the cell observations.

Reuse: the CILP numerics (`_layer_matrices`, sign activation) and the
end-of-trace recipe (`_resolve_end`) are imported from `cilp`, so the two
realizations are guaranteed numerically identical and there is no second copy
to keep in sync.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
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

    def __init__(self, root: Node, device: str | torch.device = "cpu") -> None:
        self._root = root
        self._rs: RuleSystem = build_rules(root)
        self._device = torch.device(device)
        # `subformulae()` yields each distinct node bottom-up (children before
        # parents, root last) — exactly the post-order the sweep needs, valid
        # even for the shared-subformula DAG.
        self._nodes: list[Node] = list(root.subformulae())
        self._build()

        self._state: torch.Tensor = self._initial_x.clone()
        self._last_cell: torch.Tensor = self._initial_x.clone()
        self._decided: Verdict | None = None

    @classmethod
    def from_formula(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "StructuredCILPRunner":
        return cls(parse(formula), device=device)

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
        # vector touches only that node's slots. Every weight/bias is placed on
        # the target device once, at build time, so step()/batch_run() never
        # copy per call.
        dev = self._device
        self.eval_net: dict[str, _Layer] = {}
        self.react_net: dict[str, _Layer] = {}
        for node in self._nodes:
            self.eval_net[node.key] = tuple(
                t.to(dev)
                for t in _layer_matrices(tuple(eval_by[node.key]), self._index, self._n)
            )
            self.react_net[node.key] = tuple(
                t.to(dev)
                for t in _layer_matrices(
                    tuple(react_by[node.key]), self._index, self._n
                )
            )

        self._R_mask = torch.zeros(self._n, dtype=torch.bool)
        for name, idx in self._index.items():
            if name.startswith("R["):
                self._R_mask[idx] = True
        self._R_mask = self._R_mask.to(dev)

        self._initial_x = torch.full((self._n,), -1.0)
        for lit in rs.initial_state:
            self._initial_x[self._index[lit.name]] = 1.0
        self._initial_x = self._initial_x.to(dev)

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
        x = torch.full((self._n,), -1.0, device=self._device)
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
        next_state = torch.full((self._n,), -1.0, device=self._device)
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
        return self._end_verdict(self._last_cell.cpu())

    def _end_verdict(self, cell_state: torch.Tensor) -> Verdict:
        """End-of-trace resolution for one (CPU) state vector."""

        def in_state(name: str) -> bool:
            idx = self._index.get(name)
            return idx is not None and cell_state[idx].item() > 0

        resolved = _resolve_end(self._root, in_state)
        return Verdict.SATISFY if resolved else Verdict.VIOLATE

    # -- batched, device-aware path (CPU or CUDA) --

    def batch_run(
        self,
        traces: Iterable[Iterable[Observation]],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Vectorised cross-trace monitoring on ``self._device`` (CPU/CUDA).

        Equivalent to ``[self.run(t) for t in traces]`` — identical per-trace
        early-termination and end-of-trace verdicts — but every cell is a single
        bottom-up sweep of batched matmuls over the whole batch, one small
        matmul pair per parse-tree node. This is the cross-trace parallelism the
        IJCNN 2015 structured network is meant to exploit: the per-node
        subnetworks stay on the device and the trace axis is the batch dimension.

        The per-cell node sweep is still *sequential across parse-tree levels*
        (a parent reads its children's fresh truth values, so it must run after
        them). Only the trace axis is parallelised; the within-cell dependency
        is intrinsic to the encoding, exactly as in ``CILPRunner.batch_run``.
        Fusing independent same-level sibling nodes into one kernel — the tree
        parallelism IJCNN 2015 intends — is a further optimisation this naive
        sweep does not perform.

        All traces advance uniformly each cell (decided traces are not frozen);
        verdicts are reconstructed per trace afterwards. ``early_termination`` is
        accepted for interface parity but does not change the compute — every
        cell of every trace is always processed.
        """
        dev = self._device
        n = self._n
        trace_list = [list(t) for t in traces]
        B = len(trace_list)
        if B == 0:
            return []
        lengths = [len(t) for t in trace_list]
        maxL = max(lengths)
        if maxL == 0:  # all-empty traces -> end-of-trace on the initial state
            return [self._end_verdict(self._initial_x.cpu())] * B

        atoms = self._rs.atoms
        n_atoms = len(atoms)
        if n_atoms:
            obs_cols = torch.tensor(
                [self._index[f"obs:{a}"] for a in atoms],
                dtype=torch.long,
                device=dev,
            )
            # (maxL, B, n_atoms) in {+1,-1}; built once (the batch-encoding cost).
            arr = np.full((maxL, B, n_atoms), -1.0, dtype=np.float32)
            for b, t in enumerate(trace_list):
                for i, obs in enumerate(t):
                    for j, a in enumerate(atoms):
                        if obs.get(a, False):
                            arr[i, b, j] = 1.0
            clamp = torch.from_numpy(arr).to(dev)

        Rmask = self._R_mask
        state = self._initial_x.unsqueeze(0).repeat(B, 1)
        last_state = self._initial_x.unsqueeze(0).repeat(B, 1).clone()
        verdict_code = torch.zeros(B, maxL, dtype=torch.long, device=dev)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=dev)

        for i in range(maxL):
            x = torch.full((B, n), -1.0, device=dev)
            x[:, Rmask] = state[:, Rmask]
            if n_atoms:
                x[:, obs_cols] = clamp[i]

            # Evaluation: a single bottom-up sweep, one batched matmul pair per
            # node; each node OR-accumulates into the shared activation.
            for node in self._nodes:
                W_ih, b_h, W_ho, b_o = self.eval_net[node.key]
                h = _step_activation(x @ W_ih.t() + b_h)
                y = _step_activation(h @ W_ho.t() + b_o)
                x = torch.maximum(x, y)

            sat = x[:, self._T_idx] > 0
            vio = (~sat) & (x[:, self._F_idx] > 0)
            verdict_code[:, i] = sat.long() + 2 * vio.long()

            islast = lengths_t == (i + 1)
            if bool(islast.any()):
                last_state = torch.where(islast.unsqueeze(1), x, last_state)

            # Reactivation: every node emits its R[.] heads; OR them into the
            # next cell's state.
            nxt = torch.full((B, n), -1.0, device=dev)
            for node in self._nodes:
                W_ih, b_h, W_ho, b_o = self.react_net[node.key]
                h = _step_activation(x @ W_ih.t() + b_h)
                y = _step_activation(h @ W_ho.t() + b_o)
                nxt[:, Rmask] = torch.maximum(nxt[:, Rmask], y[:, Rmask])
            state = nxt

        # First decided cell within each trace's length wins; else EOT.
        ar = torch.arange(maxL, device=dev).unsqueeze(0)
        valid = ar < lengths_t.unsqueeze(1)
        vc = torch.where(valid, verdict_code, torch.zeros_like(verdict_code))
        decided = vc != 0
        has_dec = decided.any(dim=1).cpu()
        first_idx = torch.argmax(decided.to(torch.int8), dim=1)
        first_v = vc.gather(1, first_idx.unsqueeze(1)).squeeze(1).cpu()
        last_state = last_state.cpu()

        results: list[Verdict] = []
        for b in range(B):
            if bool(has_dec[b]):
                results.append(
                    Verdict.SATISFY if int(first_v[b]) == 1 else Verdict.VIOLATE
                )
            else:
                results.append(self._end_verdict(last_state[b]))
        return results


class StructuredRuleRunnerMonitor(Monitor):
    """`Monitor`-ABC adapter over `StructuredCILPRunner`.

    The structured (IJCNN 2015 Fig. 5) variant of the RuleRunner network, slotted
    into the `Monitor` harness as a second RuleRunner data point alongside
    `RuleRunnerMonitor` (CILP).

    **Device-aware and cross-trace batched (CPU/CUDA).** The per-node
    subnetworks live on the device passed to `compile` and `batch_run`
    parallelises the trace axis with batched matmuls, exactly like
    `CILPRunner.batch_run` — this is the point of keeping the structured variant:
    the per-node network is meant to run like a network, in parallel, on a GPU.
    `effective_device` reports where the weights actually compute.

    Caveat on the *within-cell* cost: the per-cell evaluation is a *sequential
    sweep over parse-tree nodes* (each a tiny matmul), so it issues more, smaller
    kernels per cell than the flat encoding's `depth+1` whole-network passes —
    likely *less* GPU-friendly per cell, unless the independent same-level
    sibling nodes are also fused (the tree-parallelism IJCNN 2015 intends, which
    this naive sweep does not yet exploit). The cross-trace batch dimension is
    parallelised regardless.
    """

    def __init__(self, runner: StructuredCILPRunner) -> None:
        self._runner = runner

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "StructuredRuleRunnerMonitor":
        return cls(StructuredCILPRunner.from_formula(formula, device=device))

    @property
    def effective_device(self) -> str:
        """Device the per-node CILP weight matrices actually live / compute on."""
        return "cuda" if self._runner._device.type == "cuda" else "cpu"

    def step(self, obs: Observation) -> Verdict:
        return self._runner.step(obs)

    def final_verdict(self) -> Verdict:
        return self._runner.final_verdict()

    def reset(self) -> None:
        self._runner.reset()

    def batch_run(
        self,
        traces: Iterable[Observation],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Vectorised cross-trace path (CPU/CUDA). Delegates to the runner,
        which parallelises the trace axis with batched matmuls; identical
        verdicts to the sequential default. ``early_termination`` is accepted
        for interface parity but does not change the compute (see
        ``StructuredCILPRunner.batch_run``)."""
        return self._runner.batch_run(traces, early_termination=early_termination)
