"""CILP encoding of a RuleSystem as a torch network.

Garcez & Zaverucha 1999's translation algorithm, adapted for the
two-phase eval/react loop of IJCNN 2014's RuleRunner. Each rule
becomes one hidden unit; positive body literals contribute +W weight,
negated body literals contribute -W; hidden bias is set so the unit
activates iff every body literal is satisfied. Output literals are an
OR of their incoming hidden units: output bias = W*(k-1) ensures any
firing hidden flips the output to +1.

Phases share the same literal-index vector space but have separate
weight matrices. State across cells is just the +1 positions of R[.]
literals; [.]V truth values are transient per cell.

Correctness oracle: the engine in [engine.py](engine.py). The
equivalence sweep at the bottom of `test_rulerunner_cilp.py` checks
that the network produces the same per-cell verdict as the engine
(and therefore as `SymbolicDFAMonitor`, modulo the documented
nested-temporal limitation).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from src.formula.compiler import Observation
from src.monitors.base import Verdict
from src.monitors.rulerunner.parse_tree import Node, Op, parse
from src.monitors.rulerunner.rules import Rule, RuleSystem, build_rules

# Sign activation works for any W > 0; W=1 is simplest.
_W = 1.0


def _step_activation(x: torch.Tensor) -> torch.Tensor:
    """Sign activation with 0 mapping to -1 (the CILP convention: a
    silent neuron is "false")."""
    return torch.where(x > 0, 1.0, -1.0)


# ---------------- compilation ----------------


def _layer_matrices(
    rules: tuple[Rule, ...],
    literal_index: dict[str, int],
    n_lits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (W_ih, b_h, W_ho, b_o) for one rule set.

    One hidden unit per rule. Hidden bias = -W*(n - 0.5) where n is
    the body length. Output bias = W*(k - 1) where k counts the
    hidden units targeting that output literal (k=0 outputs get a
    negative bias so they stay -1 even when no rule fires)."""
    n_clauses = len(rules)
    W_ih = torch.zeros(n_clauses, n_lits)
    b_h = torch.zeros(n_clauses)
    W_ho = torch.zeros(n_lits, n_clauses)
    k_per_output = [0] * n_lits

    for c, rule in enumerate(rules):
        for lit in rule.body:
            W_ih[c, literal_index[lit.name]] = -_W if lit.negated else _W
        b_h[c] = -_W * (len(rule.body) - 0.5)
        head_idx = literal_index[rule.head.name]
        W_ho[head_idx, c] = _W
        k_per_output[head_idx] += 1

    b_o = torch.full((n_lits,), -1.0)
    for i, k in enumerate(k_per_output):
        if k > 0:
            b_o[i] = _W * (k - 1)
    return W_ih, b_h, W_ho, b_o


# ---------------- end-of-trace resolution ----------------

# Same recipe as engine._resolve, but parameterised on `in_state(name) -> bool`
# so it works against either a set[Literal] or a torch tensor + index.

_PIN_RIGHT_FOR_L: dict[Op, bool] = {Op.AND: True, Op.OR: False, Op.IMPLIES: False}
_PIN_LEFT_FOR_R: dict[Op, bool] = {Op.AND: True, Op.OR: False, Op.IMPLIES: True}


def _resolve_end(node: Node, in_state) -> bool:
    K = node.key
    if in_state(f"[{K}]T"):
        return True
    if in_state(f"[{K}]F"):
        return False

    if node.op is Op.ATOM:
        return False
    if node.op is Op.NOT:
        return not _resolve_end(node.children[0], in_state)
    if node.op in (Op.AND, Op.OR, Op.IMPLIES):
        return _resolve_binary(node, in_state)
    if node.op in (Op.EVENTUALLY, Op.ALWAYS):
        return _resolve_end(node.children[0], in_state)
    if node.op in (Op.UNTIL, Op.RELEASE):
        return _resolve_end(node.children[1], in_state)
    if node.op is Op.NEXT:
        return False
    if node.op is Op.WEAK_NEXT:
        return True
    raise ValueError(node.op)


def _resolve_binary(node: Node, in_state) -> bool:
    K = node.key
    phi, psi = node.children
    if in_state(f"[{K}]?^L"):
        left = _resolve_end(phi, in_state)
        right = _PIN_RIGHT_FOR_L[node.op]
    elif in_state(f"[{K}]?^R"):
        left = _PIN_LEFT_FOR_R[node.op]
        right = _resolve_end(psi, in_state)
    else:
        left = _resolve_end(phi, in_state)
        right = _resolve_end(psi, in_state)

    if node.op is Op.AND:
        return left and right
    if node.op is Op.OR:
        return left or right
    return (not left) or right


# ---------------- runner ----------------


class CILPRunner:
    """CILP-network-driven monitor; same interface as RuleEngine.

    Step-by-step verdicts are guaranteed identical to RuleEngine's by
    construction (each rule -> one hidden unit, sign activation
    matches set-membership semantics). The equivalence test in
    `test_rulerunner_cilp.py` is the formal verification.
    """

    def __init__(self, root: Node, device: str | torch.device = "cpu") -> None:
        self._root = root
        self._rs: RuleSystem = build_rules(root)
        self._depth = root.depth
        self._device = torch.device(device)
        self._build()

        self._state: torch.Tensor = self._initial_x.clone()
        self._last_cell: torch.Tensor = self._initial_x.clone()
        self._decided: Verdict | None = None

    @classmethod
    def from_formula(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "CILPRunner":
        return cls(parse(formula), device=device)

    # -- compilation --

    def _build(self) -> None:
        rs = self._rs
        lit_names: set[str] = set()
        for r in rs.eval_rules + rs.react_rules:
            lit_names.update(lit.name for lit in r.body)
            lit_names.add(r.head.name)
        lit_names.update(lit.name for lit in rs.initial_state)
        # Guarantee the verdict literals exist even if no rule references
        # them in this particular formula (e.g. atom-only formulas).
        lit_names.add(f"[{rs.root_key}]T")
        lit_names.add(f"[{rs.root_key}]F")
        # Guarantee an obs:a literal per atom even if pruning removed it.
        lit_names.update(f"obs:{a}" for a in rs.atoms)

        self._literal_index = {n: i for i, n in enumerate(sorted(lit_names))}
        self._n_lits = len(self._literal_index)

        self._W_ih_eval, self._b_h_eval, self._W_ho_eval, self._b_o_eval = (
            _layer_matrices(rs.eval_rules, self._literal_index, self._n_lits)
        )
        self._W_ih_react, self._b_h_react, self._W_ho_react, self._b_o_react = (
            _layer_matrices(rs.react_rules, self._literal_index, self._n_lits)
        )

        self._R_mask = torch.zeros(self._n_lits, dtype=torch.bool)
        for name, idx in self._literal_index.items():
            if name.startswith("R["):
                self._R_mask[idx] = True

        self._initial_x = torch.full((self._n_lits,), -1.0)
        for lit in rs.initial_state:
            self._initial_x[self._literal_index[lit.name]] = 1.0

        # Move every weight/bias/mask onto the target device once, at build
        # time, so step() and batch_run() never copy per call.
        dev = self._device
        self._W_ih_eval = self._W_ih_eval.to(dev)
        self._b_h_eval = self._b_h_eval.to(dev)
        self._W_ho_eval = self._W_ho_eval.to(dev)
        self._b_o_eval = self._b_o_eval.to(dev)
        self._W_ih_react = self._W_ih_react.to(dev)
        self._b_h_react = self._b_h_react.to(dev)
        self._W_ho_react = self._W_ho_react.to(dev)
        self._b_o_react = self._b_o_react.to(dev)
        self._R_mask = self._R_mask.to(dev)
        self._initial_x = self._initial_x.to(dev)

        self._T_idx = self._literal_index[f"[{rs.root_key}]T"]
        self._F_idx = self._literal_index[f"[{rs.root_key}]F"]

    # -- public API --

    def reset(self) -> None:
        self._state = self._initial_x.clone()
        self._last_cell = self._initial_x.clone()
        self._decided = None

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided

        # Build the cell's input vector: -1 everywhere, copy R-state across,
        # then clamp obs:a literals from `obs`.
        x = torch.full((self._n_lits,), -1.0, device=self._device)
        x[self._R_mask] = self._state[self._R_mask]
        for atom in self._rs.atoms:
            obs_idx = self._literal_index[f"obs:{atom}"]
            x[obs_idx] = 1.0 if obs.get(atom, False) else -1.0

        # Evaluation phase: depth+1 sign-activated forward passes, OR-accumulate.
        for _ in range(self._depth + 1):
            h = _step_activation(self._W_ih_eval @ x + self._b_h_eval)
            y = _step_activation(self._W_ho_eval @ h + self._b_o_eval)
            x_new = torch.maximum(x, y)
            if torch.equal(x_new, x):
                break
            x = x_new

        self._last_cell = x.clone()

        if x[self._T_idx] > 0:
            self._decided = Verdict.SATISFY
            return Verdict.SATISFY
        if x[self._F_idx] > 0:
            self._decided = Verdict.VIOLATE
            return Verdict.VIOLATE

        # Reactivation phase: single forward pass; keep only R[.] positions.
        h = _step_activation(self._W_ih_react @ x + self._b_h_react)
        y = _step_activation(self._W_ho_react @ h + self._b_o_react)
        next_state = torch.full((self._n_lits,), -1.0, device=self._device)
        next_state[self._R_mask] = y[self._R_mask]
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

    # -- batched, device-aware path (CPU or CUDA) --

    def batch_run(
        self,
        traces: Iterable[Iterable[Observation]],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Vectorised cross-trace monitoring on ``self._device`` (CPU/CUDA).

        Equivalent to ``[self.run(t) for t in traces]`` — identical
        per-trace early-termination and end-of-trace verdicts — but every
        cell is one batched pair of matmuls over the whole batch (the eval
        convergence is ``depth+1`` batched passes, faithful to IJCNN 2014's
        matrix-matrix formulation). The within-step sequential ``depth+1``
        dependency is intrinsic to the rule system and remains; only the
        cross-trace axis is parallelised.

        All traces advance uniformly each cell (decided traces are not
        frozen); verdicts are reconstructed per trace afterwards — the first
        decided cell within a trace's length wins, otherwise end-of-trace
        resolution runs on that trace's final-cell state. This is exact
        because the batched state evolution coincides with the sequential
        one up to and including each trace's first decision.

        ``early_termination`` is accepted for interface parity with the base
        ``Monitor`` but does not change the compute: the batched path already
        processes *all* cells of *every* trace, so RuleRunner always pays the
        full per-cell cost regardless. The per-trace early-termination replay
        below only recovers the correct verdict and runs either way.
        """
        dev = self._device
        n = self._n_lits
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
                [self._literal_index[f"obs:{a}"] for a in atoms],
                dtype=torch.long,
                device=dev,
            )
            # (maxL, B, n_atoms) in {+1,-1}; built once (the batch-encoding
            # cost, analogous to DeepDFA's per-step symbol tensor).
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

            # Evaluation: depth+1 batched passes to the fixpoint.
            for _ in range(self._depth + 1):
                h = _step_activation(x @ self._W_ih_eval.t() + self._b_h_eval)
                y = _step_activation(h @ self._W_ho_eval.t() + self._b_o_eval)
                x_new = torch.maximum(x, y)
                if torch.equal(x_new, x):
                    break
                x = x_new

            sat = x[:, self._T_idx] > 0
            vio = (~sat) & (x[:, self._F_idx] > 0)
            verdict_code[:, i] = sat.long() + 2 * vio.long()

            islast = lengths_t == (i + 1)
            if bool(islast.any()):
                last_state = torch.where(islast.unsqueeze(1), x, last_state)

            # Reactivation: one batched pass -> next cell's R-state.
            h = _step_activation(x @ self._W_ih_react.t() + self._b_h_react)
            y = _step_activation(h @ self._W_ho_react.t() + self._b_o_react)
            nxt = torch.full((B, n), -1.0, device=dev)
            nxt[:, Rmask] = y[:, Rmask]
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

    def _end_verdict(self, cell_state: torch.Tensor) -> Verdict:
        """End-of-trace resolution for one (CPU) state vector."""

        def in_state(name: str) -> bool:
            idx = self._literal_index.get(name)
            return idx is not None and cell_state[idx].item() > 0

        resolved = _resolve_end(self._root, in_state)
        return Verdict.SATISFY if resolved else Verdict.VIOLATE
