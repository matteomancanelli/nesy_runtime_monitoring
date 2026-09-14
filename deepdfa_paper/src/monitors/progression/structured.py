"""Factorized structured CILP network for progression RuleRunner.

This is the progression counterpart of
``rulerunner.structured.StructuredCILPRunner``.  Both realizations use the same
two-phase organization:

* a bottom-up collection of per-node CILP evaluation subnetworks; and
* a collection of per-root reactivation subnetworks whose outputs are OR-ed
  into the recurrent state for the next trace cell.

The difference is exactly the RuleRunner repair.  The original monitor indexes
its recurrent registers by subformulae of the input parse tree; this monitor
indexes them by roots in the *progression closure*.  For each active residual
root ``chi`` and local observation symbol, its reactivation module emits
``split_conj(nf(prog(chi, obs)))``.  Since progression distributes over
conjunction, unioning those outputs represents the exact successor residual.
No hidden unit has to recognize the complete active root set.

Cross-root simplification is deliberately not part of recurrence.  A state
such as ``{a, !a}`` denotes the correct (unsatisfiable) conjunction even when it
is not rewritten to the literal ``FALSE``.  Whole reachable root sets are
enumerated only at compile time to classify accepting sinks and traps.  A
small, fixed state-label head provides exact early verdicts; it does not drive
the recurrent update.  This preserves local, syntactically-owned transition
modules while retaining exact online monitoring.

As in the old structured implementation, the modules are executed explicitly
in Python rather than literally concatenated into one sparse recurrent tensor.
They are a functionally equivalent factorization: every module is a CILP layer
over a shared literal space, and their outputs are combined monotonically.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.progression.eager import (
    MAX_GUARD_ATOMS,
    _sink_trap_labels,
)
from src.monitors.progression.formula import (
    Formula,
    Op,
    atoms_of,
    from_node,
    simplify,
    split_conj,
)
from src.monitors.progression.progression import holds_empty, prog
from src.monitors.rulerunner.cilp import _layer_matrices, _step_activation
from src.monitors.rulerunner.parse_tree import parse
from src.monitors.rulerunner.rules import Literal, Rule

# A per-node CILP weight bundle: (W_ih, b_h, W_ho, b_o).
_Layer = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class FactorizedProgressionGraph:
    """Compile-time graph supporting a root-local recurrent realization.

    ``root_trans`` is the actual recurrence: one source root and its local
    observation symbol map to a set of successor roots. ``states`` / ``trans``
    enumerate reachable *sets* of roots only to compute exact online labels.
    """

    formula: Formula
    atoms: tuple[str, ...]
    roots: tuple[Formula, ...]
    initial: frozenset[int]
    relevant: tuple[tuple[str, ...], ...]
    root_trans: tuple[dict[int, frozenset[int]], ...]
    closure: tuple[Formula, ...]
    states: tuple[frozenset[int], ...]
    trans: tuple[dict[int, int], ...]
    accepting: frozenset[int]
    trap_states: frozenset[int]
    accepting_sinks: frozenset[int]

    def root_symbol(self, root: int, obs: Observation) -> int:
        symbol = 0
        for j, atom in enumerate(self.relevant[root]):
            if obs.get(atom, False):
                symbol |= 1 << j
        return symbol


def _postorder(f: Formula, seen: set[str], out: list[Formula]) -> None:
    if f.key in seen:
        return
    seen.add(f.key)
    for child in f.args:
        _postorder(child, seen, out)
    out.append(f)


@lru_cache(maxsize=None)
def build_factorized_progression_graph(
    formula: str, max_guard_atoms: int = MAX_GUARD_ATOMS
) -> FactorizedProgressionGraph:
    """Build root-local transitions and the reachable aggregate label graph."""
    phi = simplify(from_node(parse(formula)))
    atoms = tuple(sorted(atoms_of(phi)))

    roots: list[Formula] = []
    root_index: dict[str, int] = {}

    def intern_root(root: Formula) -> int:
        index = root_index.get(root.key)
        if index is None:
            index = len(roots)
            root_index[root.key] = index
            roots.append(root)
        return index

    initial = frozenset(intern_root(root) for root in split_conj(phi))
    relevant: list[tuple[str, ...]] = []
    root_trans: list[dict[int, frozenset[int]]] = []

    head = 0
    while head < len(roots):
        root = roots[head]
        head += 1
        rel = tuple(sorted(atoms_of(root)))
        if len(rel) > max_guard_atoms:
            raise ValueError(
                f"Residual root {root.key!r} has {len(rel)} guard atoms; "
                f"enumerating 2^{len(rel)} observations exceeds "
                f"MAX_GUARD_ATOMS={max_guard_atoms}."
            )
        row: dict[int, frozenset[int]] = {}
        for symbol in range(1 << len(rel)):
            obs = {rel[j]: bool((symbol >> j) & 1) for j in range(len(rel))}
            successor = simplify(prog(root, obs))
            row[symbol] = frozenset(
                intern_root(next_root) for next_root in split_conj(successor)
            )
        relevant.append(rel)
        root_trans.append(row)

    closure: list[Formula] = []
    seen_closure: set[str] = set()
    for root in roots:
        _postorder(root, seen_closure, closure)

    # Enumerate aggregate root sets only for exact sink/trap classification.
    states: list[frozenset[int]] = [initial]
    state_index: dict[frozenset[int], int] = {initial: 0}
    trans: list[dict[int, int]] = []
    head = 0
    while head < len(states):
        state = states[head]
        head += 1
        rel = tuple(sorted({atom for root in state for atom in relevant[root]}))
        if len(rel) > max_guard_atoms:
            raise ValueError(
                f"Factorized residual state has {len(rel)} guard atoms; "
                f"enumerating 2^{len(rel)} observations exceeds "
                f"MAX_GUARD_ATOMS={max_guard_atoms}."
            )
        row: dict[int, int] = {}
        for symbol in range(1 << len(rel)):
            obs = {rel[j]: bool((symbol >> j) & 1) for j in range(len(rel))}
            successor_roots: set[int] = set()
            for root in state:
                local_symbol = 0
                for j, atom in enumerate(relevant[root]):
                    if obs.get(atom, False):
                        local_symbol |= 1 << j
                successor_roots.update(root_trans[root][local_symbol])
            successor = frozenset(successor_roots)
            next_index = state_index.get(successor)
            if next_index is None:
                next_index = len(states)
                state_index[successor] = next_index
                states.append(successor)
            row[symbol] = next_index
        trans.append(row)

    accepting = frozenset(
        i
        for i, state in enumerate(states)
        if all(holds_empty(roots[root]) for root in state)
    )
    trap_states, accepting_sinks = _sink_trap_labels(trans, accepting)
    return FactorizedProgressionGraph(
        formula=phi,
        atoms=atoms,
        roots=tuple(roots),
        initial=initial,
        relevant=tuple(relevant),
        root_trans=tuple(root_trans),
        closure=tuple(closure),
        states=tuple(states),
        trans=tuple(trans),
        accepting=accepting,
        trap_states=trap_states,
        accepting_sinks=accepting_sinks,
    )


def _last_rules(node: Formula) -> list[Rule]:
    """Horn clauses computing ``last(node, obs)`` from its children's truths.

    ``last`` (single-cell / end-of-trace truth) is compositional, so each node
    is a tiny Boolean gate over its children's truth literals (keyed by
    ``Formula.key``). Atoms are clamped from the observation (no rule);
    constants and the two next-operators are constant-truth (``X`` false, ``WX``
    true). The recipe matches ``progression.last`` exactly.
    """
    head = Literal(node.key)
    op = node.op
    if op is Op.ATOM or op is Op.NEXT or op is Op.FALSE:
        # last = obs(atom) [clamped, no rule]; last(X x) = False; last(FALSE) = False.
        return []
    if op is Op.WEAK_NEXT or op is Op.TRUE:
        # last(WX x) = True; last(TRUE) = True — an always-firing empty-body rule.
        return [Rule(frozenset(), head)]
    if op is Op.NOT:
        (x,) = node.args
        return [Rule(frozenset({Literal(x.key, negated=True)}), head)]
    if op is Op.AND:
        x, y = node.args
        return [Rule(frozenset({Literal(x.key), Literal(y.key)}), head)]
    if op is Op.OR:
        x, y = node.args
        return [
            Rule(frozenset({Literal(x.key)}), head),
            Rule(frozenset({Literal(y.key)}), head),
        ]
    if op is Op.EVENTUALLY or op is Op.ALWAYS:
        (x,) = node.args
        return [Rule(frozenset({Literal(x.key)}), head)]
    if op is Op.UNTIL or op is Op.RELEASE:
        _x, y = node.args
        return [Rule(frozenset({Literal(y.key)}), head)]
    raise TypeError(f"Cannot build last-rules for op {op}")


class _StructuredNet:
    """Per-node evaluation plus per-root progression/reactivation modules."""

    def __init__(self, graph: FactorizedProgressionGraph, device: torch.device) -> None:
        self.graph = graph
        self.device = device
        self.roots = graph.roots
        self.n_roots = len(self.roots)
        self.root_index = {root.key: i for i, root in enumerate(self.roots)}

        # Truth-literal space over the closure nodes (children before parents).
        self.closure = graph.closure
        self.truth_index: dict[str, int] = {
            n.key: i for i, n in enumerate(self.closure)
        }
        self.n_truth = len(self.truth_index)

        # Atom truth slots are clamped straight from the observation.
        self.atom_cols: dict[str, int] = {
            n.key: self.truth_index[n.key] for n in self.closure if n.op is Op.ATOM
        }

        # Same organization as the old structured runner: one addressable
        # evaluation module per closure node, all over one shared truth space.
        self.eval_net: dict[str, _Layer] = {}
        for node in self.closure:
            rules = _last_rules(node)
            layer = _layer_matrices(tuple(rules), self.truth_index, self.n_truth)
            self.eval_net[node.key] = tuple(t.to(device) for t in layer)

        # Root-local progression clauses. A module tests only whether its own
        # source root is active and the atoms relevant to that root. It never
        # inspects any other root register.
        self.n_atoms = len(graph.atoms)
        self.atom_index = {atom: i for i, atom in enumerate(graph.atoms)}
        self.react_index = {
            **{f"R[{root.key}]": i for i, root in enumerate(self.roots)},
            **{f"obs:{atom}": self.n_roots + i for i, atom in enumerate(graph.atoms)},
        }
        react_width = self.n_roots + self.n_atoms
        self.react_net: dict[str, _Layer] = {}
        for source, root in enumerate(self.roots):
            rules: list[Rule] = []
            rel = graph.relevant[source]
            for symbol, successors in graph.root_trans[source].items():
                body = {Literal(f"R[{root.key}]")}
                body.update(
                    Literal(f"obs:{atom}", negated=not bool((symbol >> j) & 1))
                    for j, atom in enumerate(rel)
                )
                for successor in successors:
                    rules.append(
                        Rule(
                            frozenset(body),
                            Literal(f"R[{self.roots[successor].key}]"),
                        )
                    )
            layer = _layer_matrices(tuple(rules), self.react_index, react_width)
            self.react_net[root.key] = tuple(t.to(device) for t in layer)

        # Map each residual-root register to its truth slot for the boundary bit.
        self.root_truth_idx = torch.tensor(
            [self.truth_index[root.key] for root in self.roots],
            dtype=torch.long,
            device=device,
        )

        initial = torch.full((self.n_roots,), -1.0, device=device)
        for root in graph.initial:
            initial[root] = 1.0
        self.initial_state = initial

        # Fixed global label head. It recognizes reachable aggregate root sets
        # after the local recurrence, solely to expose exact sink/trap verdicts.
        n_states = len(graph.states)
        W_label_ih = torch.empty((n_states, self.n_roots), device=device)
        for i, state in enumerate(graph.states):
            for root in range(self.n_roots):
                W_label_ih[i, root] = 1.0 if root in state else -1.0
        b_label_h = torch.full((n_states,), -(self.n_roots - 0.5), device=device)
        W_label_ho = torch.zeros((2, n_states), device=device)
        for state in graph.accepting_sinks:
            W_label_ho[0, state] = 1.0
        for state in graph.trap_states:
            W_label_ho[1, state] = 1.0
        b_label_o = torch.tensor(
            [
                max(0, len(graph.accepting_sinks)) - 1.0,
                max(0, len(graph.trap_states)) - 1.0,
            ],
            device=device,
        )
        if not graph.accepting_sinks:
            b_label_o[0] = -1.0
        if not graph.trap_states:
            b_label_o[1] = -1.0
        self.label_layer: _Layer = (
            W_label_ih,
            b_label_h,
            W_label_ho,
            b_label_o,
        )

    def _eval(self, truth0: torch.Tensor) -> torch.Tensor:
        """Bottom-up per-node sweep. ``truth0`` is the truth vector with atom
        slots already clamped (``(B, n_truth)`` in {+1,-1}, non-atoms -1);
        returns the completed truth vector (all closure nodes resolved)."""
        x = truth0
        for node in self.closure:
            W_ih, b_h, W_ho, b_o = self.eval_net[node.key]
            h = _step_activation(x @ W_ih.t() + b_h)
            y = _step_activation(h @ W_ho.t() + b_o)
            x = torch.maximum(x, y)
        return x

    @staticmethod
    def _forward(layer: _Layer, x: torch.Tensor) -> torch.Tensor:
        W_ih, b_h, W_ho, b_o = layer
        hidden = _step_activation(x @ W_ih.t() + b_h)
        return _step_activation(hidden @ W_ho.t() + b_o)

    def advance(
        self, state: torch.Tensor, truth0: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One batched cell. ``state`` (B, n_roots), ``truth0`` (B, n_truth)
        (closure-atom slots clamped, rest -1), and ``atoms`` (B, n_atoms) the
        recurrence guard encoding — all in {+1,-1}. Returns
        (next_state, sat, vio, last), the last three as (B,) bool tensors.

        Evaluation and recurrence are both factorized into syntactically-owned
        modules. Only the final sink/trap readout examines the whole root set."""
        # Evaluation: one bottom-up node sweep, as in the old structured runner.
        truth = self._eval(truth0)
        root_truth = truth.index_select(1, self.root_truth_idx)  # (B, n_roots)
        active = state > 0
        last = ((~active) | (root_truth > 0)).all(dim=1)

        # Reactivation: each active root independently emits its progressed
        # successor roots; OR the module outputs into the next multi-hot state.
        react_input = torch.cat([state, atoms], dim=1)
        nxt = torch.full_like(state, -1.0)
        for root in self.roots:
            output = self._forward(self.react_net[root.key], react_input)
            nxt = torch.maximum(nxt, output[:, : self.n_roots])

        labels = self._forward(self.label_layer, nxt)
        sat = labels[:, 0] > 0
        vio = labels[:, 1] > 0
        return nxt, sat, vio, last


class ProgressionRuleRunnerStructuredMonitor(Monitor):
    """Structured (per-closure-node) progression-based RuleRunner.

    The modular counterpart to ``ProgressionRuleRunnerMonitor`` (flat): both
    evaluation and residual reactivation are split into addressable CILP
    modules, following the same execution pattern as the old structured
    RuleRunner while replacing its input-subformula state with residual roots.
    Exact online labels come from a fixed aggregate-state readout.

    Kept as the **local-learning** data point (Paper B): each ``eval_net`` or
    ``react_net`` subnetwork is an addressable, syntactically-local set of
    weights. CPU/CUDA and single/batched are implementation choices, not
    fundamentals.
    """

    def __init__(self, net: _StructuredNet) -> None:
        self._net = net
        self._state = net.initial_state.clone()
        self._last_v: bool | None = None
        self._decided: Verdict | None = None

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "ProgressionRuleRunnerStructuredMonitor":
        graph = build_factorized_progression_graph(formula)
        return cls(_StructuredNet(graph, torch.device(device)))

    @property
    def effective_device(self) -> str:
        """Device the (torch) subnetworks actually compute on."""
        return "cuda" if self._net.device.type == "cuda" else "cpu"

    def reset(self) -> None:
        self._state = self._net.initial_state.clone()
        self._last_v = None
        self._decided = None

    def _truth_row(self, obs: Observation) -> torch.Tensor:
        """(1, n_truth) truth vector: -1 everywhere, atom slots from ``obs``."""
        row = torch.full((1, self._net.n_truth), -1.0, device=self._net.device)
        for key, j in self._net.atom_cols.items():
            if obs.get(key, False):
                row[0, j] = 1.0
        return row

    def _atom_row(self, obs: Observation) -> torch.Tensor:
        """(1, n_atoms) observation vector used by recurrence guards."""
        row = torch.full((1, self._net.n_atoms), -1.0, device=self._net.device)
        for a, j in self._net.atom_index.items():
            if obs.get(a, False):
                row[0, j] = 1.0
        return row

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        truth0 = self._truth_row(obs)
        atoms = self._atom_row(obs)
        nxt, sat, vio, last = self._net.advance(self._state.unsqueeze(0), truth0, atoms)
        self._state = nxt.squeeze(0)
        self._last_v = bool(last.item())
        if bool(sat.item()):
            self._decided = Verdict.SATISFY
            return Verdict.SATISFY
        if bool(vio.item()):
            self._decided = Verdict.VIOLATE
            return Verdict.VIOLATE
        return Verdict.UNDECIDED

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        if self._last_v is None:
            empty = holds_empty(self._net.graph.formula)
            return Verdict.SATISFY if empty else Verdict.VIOLATE
        return Verdict.SATISFY if self._last_v else Verdict.VIOLATE

    # -- batched, device-aware path (CPU or CUDA) --

    def batch_run(
        self,
        traces: Iterable[Iterable[Observation]],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Vectorised cross-trace monitoring on ``self._net.device``.

        Each cell is one bottom-up sweep of batched per-node matmuls plus the
        root-local recurrence modules, over the whole batch; identical verdicts to
        ``[run(t) ...]``. All traces advance uniformly (decided traces are not
        frozen); per-trace early-termination / end-of-trace are replayed
        afterwards — the first decided cell within a trace's length wins, else
        the ``last`` bit at its final cell. ``early_termination`` is accepted for
        interface parity but does not change the compute (the within-cell
        per-node sweep is sequential across parse-tree levels regardless — see
        the module docstring; only the trace axis is parallelised)."""
        net = self._net
        dev = net.device
        trace_list = [list(t) for t in traces]
        B = len(trace_list)
        if B == 0:
            return []
        lengths = [len(t) for t in trace_list]
        maxL = max(lengths)
        if maxL == 0:
            v = Verdict.SATISFY if holds_empty(net.graph.formula) else Verdict.VIOLATE
            return [v] * B

        # Encode observations once, into both the closure-truth space (atom
        # slots) and the recurrence's atom space: (maxL, B, .) in {+1,-1}.
        truth_arr = np.full((maxL, B, net.n_truth), -1.0, dtype=np.float32)
        atom_arr = np.full((maxL, B, net.n_atoms), -1.0, dtype=np.float32)
        for b, t in enumerate(trace_list):
            for i, obs in enumerate(t):
                for key, j in net.atom_cols.items():
                    if obs.get(key, False):
                        truth_arr[i, b, j] = 1.0
                for a, j in net.atom_index.items():
                    if obs.get(a, False):
                        atom_arr[i, b, j] = 1.0
        truth_clamp = torch.from_numpy(truth_arr).to(dev)
        atom_clamp = torch.from_numpy(atom_arr).to(dev)

        state = net.initial_state.unsqueeze(0).repeat(B, 1)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=dev)
        verdict_code = torch.zeros(B, maxL, dtype=torch.long, device=dev)
        last_bits = torch.zeros(B, maxL, dtype=torch.bool, device=dev)

        for i in range(maxL):
            nxt, sat, vio, last = net.advance(state, truth_clamp[i], atom_clamp[i])
            verdict_code[:, i] = sat.long() + 2 * (vio & ~sat).long()
            last_bits[:, i] = last
            state = nxt

        ar = torch.arange(maxL, device=dev).unsqueeze(0)
        valid = ar < lengths_t.unsqueeze(1)
        vc = torch.where(valid, verdict_code, torch.zeros_like(verdict_code))
        decided = vc != 0
        has_dec = decided.any(dim=1).cpu()
        first_idx = torch.argmax(decided.to(torch.int8), dim=1)
        first_v = vc.gather(1, first_idx.unsqueeze(1)).squeeze(1).cpu()
        final_idx = (lengths_t - 1).clamp(min=0)
        last_final = last_bits.gather(1, final_idx.unsqueeze(1)).squeeze(1).cpu()

        results: list[Verdict] = []
        for b in range(B):
            if lengths[b] == 0:
                results.append(
                    Verdict.SATISFY
                    if holds_empty(net.graph.formula)
                    else Verdict.VIOLATE
                )
                continue
            if bool(has_dec[b]):
                results.append(
                    Verdict.SATISFY if int(first_v[b]) == 1 else Verdict.VIOLATE
                )
            else:
                results.append(
                    Verdict.SATISFY if bool(last_final[b]) else Verdict.VIOLATE
                )
        return results
