"""Structured (per-closure-node) CILP network for the progression RuleRunner.

The *flat* progression monitor (``flat.py``) enumerates every residual
transition into one hidden layer that identifies the current residual state
and reads the verdict off precomputed heads. This module is the **structured**
counterpart, analogous to the original RuleRunner's
[structured.py](../rulerunner/structured.py): it decomposes the per-cell
computation into **one small CILP subnetwork per node of the residual closure
C_phi**, composed by an explicit bottom-up (post-order) sweep. That per-node
decomposition is the substrate for *local* (single-node) learning — the
Paper-B adaptation setting — where parameters attach to syntactically-local
subnetworks rather than to an opaque flat transition layer.

What is structured, and what is not (the honest boundary)
---------------------------------------------------------
For the *original* RuleRunner both phases — evaluation and reactivation —
decompose cleanly per parse-tree node. For the **progression** reformulation
that is only half true, because the state update is formula *progression*
followed by Boolean *canonicalization*, and canonicalization is inherently
**cross-root**: e.g. the residual ``(X a) & (X ~a)`` progresses to
``a & ~a`` which simplifies to ``FALSE`` (a VIOLATE that neither root produces
alone). A per-root progression that merely unions each root's successors would
miss such collapses, so the transition *cannot* be a sound per-node circuit.
We therefore split the cell as:

* **Evaluation — genuinely per node.** A bottom-up sweep of one CILP
  subnetwork per closure node computes each node's ``last`` truth value (its
  single-cell / end-of-trace semantics) from its children's truth values plus
  the cell observation. This is the modular, locally-learnable part, and it
  mirrors the original structured monitor's evaluation sweep exactly.

* **Recurrence — the shared global canonicalization.** The next residual state
  (a multi-hot vector over the roots ``R_phi``, the same state representation as
  ``flat.py``) is produced by the compiled state-identifying transition — this
  is the irreducibly-global step. We reuse ``flat._FlatNet`` for it.

The online verdict is then *derived* from those two, so the per-node evaluation
genuinely drives the verdict rather than being dead weight:

* **SATISFY** iff the next state is the accepting sink (``TRUE`` — an *empty*
  root set) **and** the end-of-trace ``last`` bit holds;
* **VIOLATE** iff the next state is the trap (``FALSE`` root active) **and** the
  ``last`` bit fails.

Both conditions are exactly the eager table's ``online`` codes (verified
bit-for-bit against the eager / lazy / flat / symbolic monitors), so the
structured monitor is verdict-for-verdict identical to the others while keeping
the per-node organization Paper B needs.

Framing (mirrors the original structured monitor). This is the
**modular / local-learning contrast, not the throughput path**: CPU/GPU and
single-vs-batched are *implementation* choices, not fundamentals. It is
device-aware and batches across traces exactly like ``flat.py``, but its per-cell
evaluation is a *sequential post-order sweep over closure nodes* (many tiny
matmuls), so it issues more, smaller kernels per cell than the flat network's
single state-identify pass — deliberately, to keep the per-node subnetworks
addressable for adaptation. ``effective_device`` truthfully reports where the
(torch) subnetworks compute.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.progression.eager import (
    ProgressionDFA,
    build_progression_dfa,
)
from src.monitors.progression.flat import _FlatNet
from src.monitors.progression.formula import Formula, Op
from src.monitors.rulerunner.cilp import _layer_matrices, _step_activation
from src.monitors.rulerunner.rules import Literal, Rule

# A per-node CILP weight bundle: (W_ih, b_h, W_ho, b_o).
_Layer = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
    """Per-closure-node CILP evaluation + reused flat recurrence.

    Evaluation: one subnetwork per closure node over a shared truth-literal
    space (indexed by ``Formula.key``), swept bottom-up so each node sees its
    children's fresh truth. Recurrence: delegated to ``flat._FlatNet`` (the
    global canonicalization). The current residual state is a multi-hot vector
    over the flat network's roots ``R_phi``, carried across cells.
    """

    def __init__(self, dfa: ProgressionDFA, device: torch.device) -> None:
        self.dfa = dfa
        self.device = device
        self.flat = _FlatNet(dfa, device)  # shared recurrence + root indexing

        # Truth-literal space over the closure nodes (children before parents).
        self.closure = dfa.closure
        self.truth_index: dict[str, int] = {
            n.key: i for i, n in enumerate(self.closure)
        }
        self.n_truth = len(self.truth_index)

        # Atom truth slots are clamped straight from the observation.
        self.atom_cols: dict[str, int] = {
            n.key: self.truth_index[n.key]
            for n in self.closure
            if n.op is Op.ATOM
        }

        # One CILP subnetwork per non-atom node, in post-order. Each writes only
        # its own truth slot; OR-accumulating into the shared vector touches only
        # that slot. Placed on the device once, at build time.
        self.eval_layers: list[_Layer] = []
        for node in self.closure:
            rules = _last_rules(node)
            if node.op is Op.ATOM:
                continue  # clamped, not computed
            layer = _layer_matrices(tuple(rules), self.truth_index, self.n_truth)
            self.eval_layers.append(tuple(t.to(device) for t in layer))

        # Map each flat root register -> its truth slot (for the last bit), and
        # locate the trap/sink markers in the flat root indexing.
        inv = {i: k for k, i in self.flat.root_keys.items()}
        self.root_truth_idx = torch.tensor(
            [self.truth_index[inv[r]] for r in range(self.flat.n_roots)],
            dtype=torch.long,
            device=device,
        )
        self.false_root = self.flat.root_keys.get("false")  # trap root, or None

        self.n_atoms = self.flat.n_atoms
        self.atom_index = self.flat.atom_index  # atom -> flat recurrence column
        self.initial_state = self.flat.initial_state

    def _eval(self, truth0: torch.Tensor) -> torch.Tensor:
        """Bottom-up per-node sweep. ``truth0`` is the truth vector with atom
        slots already clamped (``(B, n_truth)`` in {+1,-1}, non-atoms -1);
        returns the completed truth vector (all closure nodes resolved)."""
        x = truth0
        for W_ih, b_h, W_ho, b_o in self.eval_layers:
            h = _step_activation(x @ W_ih.t() + b_h)
            y = _step_activation(h @ W_ho.t() + b_o)
            x = torch.maximum(x, y)
        return x

    def advance(
        self, state: torch.Tensor, truth0: torch.Tensor, flat_atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One batched cell. ``state`` (B, n_roots), ``truth0`` (B, n_truth)
        (closure-atom slots clamped, rest -1), ``flat_atoms`` (B, n_atoms) the
        recurrence's atom encoding — all in {+1,-1}. Returns
        (next_state, sat, vio, last), the last three as (B,) bool tensors.

        The recurrence (next_state) comes from the shared flat transition; the
        ``last`` bit and online verdicts are derived from the per-node eval
        sweep and the next state's absorbing type."""
        # Recurrence: reuse the flat state-identify transition (global step).
        # We take only the next state and re-derive the verdicts structurally.
        nxt, _sat_f, _vio_f, _last_f = self.flat.advance(state, flat_atoms)

        # Evaluation: per-node last-truth of every closure node.
        truth = self._eval(truth0)
        # last bit = AND over active roots r of truth[r]; inactive roots impose
        # nothing (an active root must be true, matching last(AND of roots)).
        root_truth = truth.index_select(1, self.root_truth_idx)  # (B, n_roots)
        active = state > 0
        last = ((~active) | (root_truth > 0)).all(dim=1)

        # Online verdict from the next state's absorbing type:
        #   next state empty  <=> TRUE  sink  -> SATISFY (with last)
        #   FALSE root active  <=> FALSE trap  -> VIOLATE (with not last)
        next_empty = (nxt <= 0).all(dim=1)
        if self.false_root is None:
            next_false = torch.zeros_like(next_empty)
        else:
            next_false = nxt[:, self.false_root] > 0
        sat = next_empty & last
        vio = next_false & ~last
        return nxt, sat, vio, last


class ProgressionRuleRunnerStructuredMonitor(Monitor):
    """Structured (per-closure-node) progression-based RuleRunner.

    The modular counterpart to ``ProgressionRuleRunnerMonitor`` (flat): the
    per-cell residual truth is evaluated by one CILP subnetwork per closure
    node, swept bottom-up, while the residual-state recurrence reuses the flat
    transition (see the module docstring for why the recurrence stays global).
    Verdict-for-verdict identical to the flat / eager / lazy / symbolic monitors.

    Kept as the **local-learning** data point (Paper B): each ``eval_layers``
    subnetwork is an addressable, syntactically-local set of weights. CPU/CUDA
    and single/batched are implementation choices, not fundamentals.
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
        dfa = build_progression_dfa(formula)
        return cls(_StructuredNet(dfa, torch.device(device)))

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
        """(1, n_atoms) recurrence atom vector, in the flat network's ordering."""
        row = torch.full((1, self._net.n_atoms), -1.0, device=self._net.device)
        for a, j in self._net.atom_index.items():
            if obs.get(a, False):
                row[0, j] = 1.0
        return row

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        truth0 = self._truth_row(obs)
        flat_atoms = self._atom_row(obs)
        nxt, sat, vio, last = self._net.advance(
            self._state.unsqueeze(0), truth0, flat_atoms
        )
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
            from src.monitors.progression.progression import holds_empty

            empty = holds_empty(self._net.dfa.formula)
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
        shared flat recurrence, over the whole batch; identical verdicts to
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
            from src.monitors.progression.progression import holds_empty

            v = Verdict.SATISFY if holds_empty(net.dfa.formula) else Verdict.VIOLATE
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
            if bool(has_dec[b]):
                results.append(
                    Verdict.SATISFY if int(first_v[b]) == 1 else Verdict.VIOLATE
                )
            else:
                results.append(
                    Verdict.SATISFY if bool(last_final[b]) else Verdict.VIOLATE
                )
        return results
