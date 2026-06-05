"""Paradigm 3: DeepDFA — differentiable transition-tensor monitor.

This reimplements the DeepDFA forward pass (Mezini et al.) against 
our own `DFA` dataclass and `Monitor` interface. We deliberately 
do NOT vendor their code (see CLAUDE.md § Paradigm 3 for the full rationale):

  * Their DeepDFA assumes the BPM *mutual-exclusivity* assumption — exactly
    one atom (activity) true per step, so the alphabet is the set of atoms.
    Our benchmark family `◇ V(a_0 ∧ a_i)` requires *conjunctions* of
    simultaneously-true atoms, so that encoding is unusable here.
  * DeepDFA must be the canonical, exactly-correct monitor in the three-way
    comparison: it has to agree with SymbolicDFAMonitor on *every* trace,
    including nested-temporal formulas where RuleRunner diverges.

Two representations of the same transition function are provided:

  * dense    — T (|Q|, 2^|AP|, |Q|) one-hot transition tensor. One matmul
               per step, trivial GPU batching. Build/storage cost is
               exponential in |AP| (this is DeepDFA's structural scaling
               weakness, dual to RuleRunner's nested-temporal limit and the
               symbolic DFA's state blowup). Best for small |AP| and for
               the batching showcase (Exp 3, ijcnn_n8 -> 256 symbols).

  * factored — no 2^|AP| tensor. Each edge guard is compiled to a closure
               that computes its satisfaction probability recursively over
               the boolean structure, assuming atom independence:
                   P(a)=p_a   P(¬φ)=1-P(φ)
                   P(φ∧ψ)=P(φ)P(ψ)   P(φ∨ψ)=1-(1-P(φ))(1-P(ψ))
               With **crisp** 0/1 inputs this is exact for *any* guard
               (product=AND, 1-∏(1-·)=OR exactly), so factored crisp
               monitoring scales to large |AP| (Exp 2). With *fractional*
               probabilities it is exact only for read-once guards (the
               IJCNN family is read-once after MONA's factoring); this is
               the differentiable path for the deferred adaptation PoC.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from sympy import symbols, sympify
from sympy.logic.boolalg import And, BooleanFalse, BooleanTrue, Not, Or

from src.formula.compiler import DFA, Observation, compile_ltlf
from src.monitors.base import Monitor, Verdict

# A guard probability function: maps p (..., |AP|) -> prob (...).
ProbFn = Callable[[torch.Tensor], torch.Tensor]


def _compile_guard_prob(label: str, atom_index: dict[str, int]) -> ProbFn:
    """Compile a MONA guard label into a torch-evaluable probability fn."""
    stripped = label.strip()
    if stripped == "true":
        return lambda p: torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    if stripped == "false":
        return lambda p: torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)

    locals_map = dict(zip(atom_index, symbols(list(atom_index))))
    expr = sympify(stripped, locals=locals_map)
    return _expr_to_prob(expr, atom_index)


def _expr_to_prob(expr, atom_index: dict[str, int]) -> ProbFn:
    if isinstance(expr, BooleanTrue):
        return lambda p: torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    if isinstance(expr, BooleanFalse):
        return lambda p: torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
    if expr.is_Symbol:
        i = atom_index[str(expr)]
        return lambda p: p[..., i]
    if isinstance(expr, Not):
        sub = _expr_to_prob(expr.args[0], atom_index)
        return lambda p: 1.0 - sub(p)
    if isinstance(expr, And):
        subs = [_expr_to_prob(a, atom_index) for a in expr.args]

        def _and(p: torch.Tensor) -> torch.Tensor:
            out = subs[0](p)
            for s in subs[1:]:
                out = out * s(p)
            return out

        return _and
    if isinstance(expr, Or):
        subs = [_expr_to_prob(a, atom_index) for a in expr.args]

        def _or(p: torch.Tensor) -> torch.Tensor:
            comp = 1.0 - subs[0](p)
            for s in subs[1:]:
                comp = comp * (1.0 - s(p))
            return 1.0 - comp

        return _or
    raise TypeError(f"Unsupported guard expression node: {expr!r}")


class DeepDFATensor:
    """Tensorization of a `DFA` shared by the dense and factored paths.

    States map to 0..|Q|-1 by sorted order; in the dense path symbols are
    integers in [0, 2^|AP|) with bit i = truth of `atoms[i]`.
    """

    def __init__(
        self,
        dfa: DFA,
        mode: str = "dense",
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in ("dense", "factored"):
            raise ValueError(f"mode must be 'dense' or 'factored', got {mode!r}")
        self.dfa = dfa
        self.mode = mode
        self.device = torch.device(device)
        self.atoms = dfa.atoms
        self.n_atoms = len(self.atoms)
        self.atom_index = {a: i for i, a in enumerate(self.atoms)}

        self.state_list = sorted(dfa.states)
        self.state_idx = {s: i for i, s in enumerate(self.state_list)}
        self.n_states = len(self.state_list)

        mu = torch.zeros(self.n_states, device=self.device)
        mu[self.state_idx[dfa.initial]] = 1.0
        self.mu = mu

        acc = torch.zeros(self.n_states, device=self.device)
        for s in dfa.accepting:
            acc[self.state_idx[s]] = 1.0
        self.accepting = acc

        self.trap_idx = frozenset(self.state_idx[s] for s in dfa.trap_states)
        self.sink_idx = frozenset(self.state_idx[s] for s in dfa.accepting_sinks)

        if mode == "dense":
            self.n_symbols = 1 << self.n_atoms
            self._build_dense()
        else:
            self._build_factored()

    # ----- dense -----

    def _build_dense(self) -> None:
        T = torch.zeros(
            self.n_states, self.n_symbols, self.n_states, device=self.device
        )
        for src in self.state_list:
            si = self.state_idx[src]
            for sigma in range(self.n_symbols):
                obs = {a: bool((sigma >> i) & 1) for i, a in enumerate(self.atoms)}
                T[si, sigma, self.state_idx[self.dfa.step(src, obs)]] = 1.0
        self.T = T

    def symbol_index(self, obs: Observation) -> int:
        idx = 0
        for i, a in enumerate(self.atoms):
            if obs.get(a, False):
                idx |= 1 << i
        return idx

    # ----- factored -----

    def _build_factored(self) -> None:
        # (src_idx, dst_idx, prob_fn) per DFA transition.
        self._edges: list[tuple[int, int, ProbFn]] = [
            (
                self.state_idx[t.src],
                self.state_idx[t.dst],
                _compile_guard_prob(t.label, self.atom_index),
            )
            for t in self.dfa.transitions
        ]

    def soft_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Build the (..., |Q|, |Q|) transition matrix for atom-prob p (..., |AP|).

        Differentiable in p. With crisp 0/1 p this is the exact transition
        matrix; with fractional p it is exact for read-once guards.
        """
        batch_shape = p.shape[:-1]
        M = torch.zeros(*batch_shape, self.n_states, self.n_states, device=p.device)
        for si, di, fn in self._edges:
            M[..., si, di] = M[..., si, di] + fn(p)
        return M

    def prob_vector(self, obs: Observation) -> torch.Tensor:
        """Crisp atom-probability vector (|AP|,) from an observation."""
        p = torch.zeros(self.n_atoms, device=self.device)
        for i, a in enumerate(self.atoms):
            if obs.get(a, False):
                p[i] = 1.0
        return p


class DeepDFAMonitor(Monitor):
    """Paradigm 3: monitor driven by the DeepDFA transition tensor.

    `step` performs one state-vector / transition-matrix product, then reads
    the three-valued verdict off the precomputed trap / accepting-sink
    labels. `batch_run` overrides the base class to process a whole batch of
    traces with batched matmuls — where DeepDFA's GPU advantage shows up.
    """

    def __init__(self, tensor: DeepDFATensor) -> None:
        self._dt = tensor
        self._q = tensor.mu.clone()
        self._decided: Verdict | None = None

    @classmethod
    def compile(
        cls,
        formula: str,
        mode: str = "dense",
        device: str | torch.device = "cpu",
    ) -> "DeepDFAMonitor":
        return cls(DeepDFATensor(compile_ltlf(formula), mode=mode, device=device))

    def reset(self) -> None:
        self._q = self._dt.mu.clone()
        self._decided = None

    def _advance(self, q: torch.Tensor, obs: Observation) -> torch.Tensor:
        dt = self._dt
        if dt.mode == "dense":
            return q @ dt.T[:, dt.symbol_index(obs), :]
        return q @ dt.soft_matrix(dt.prob_vector(obs))

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        self._q = self._advance(self._q, obs)
        state_i = int(torch.argmax(self._q))
        if state_i in self._dt.trap_idx:
            self._decided = Verdict.VIOLATE
            return Verdict.VIOLATE
        if state_i in self._dt.sink_idx:
            self._decided = Verdict.SATISFY
            return Verdict.SATISFY
        return Verdict.UNDECIDED

    def final_verdict(self) -> Verdict:
        if self._decided is not None:
            return self._decided
        state_i = int(torch.argmax(self._q))
        return (
            Verdict.SATISFY if self._dt.accepting[state_i] > 0 else Verdict.VIOLATE
        )

    # ----- batched GPU path -----

    def batch_run(
        self, traces: Iterable[Iterable[Observation]]
    ) -> list[Verdict]:
        """Process all traces with batched matmuls.

        Equivalent to ``[self.run(t) for t in traces]`` (same early-termination
        and end-of-trace semantics), but the per-step update is one batched
        ``bmm`` over the whole batch. Works in both modes; the dense mode is
        the GPU-batching showcase.
        """
        dt = self._dt
        trace_list = [list(t) for t in traces]
        if not trace_list:
            return []
        lengths = [len(t) for t in trace_list]
        B, L = len(trace_list), max(lengths)

        q = dt.mu.unsqueeze(0).expand(B, -1).clone()  # (B, |Q|)
        states = torch.empty(B, L, dtype=torch.long, device=dt.device)

        if dt.mode == "dense":
            sym = torch.zeros(B, L, dtype=torch.long, device=dt.device)
            for b, t in enumerate(trace_list):
                for i, obs in enumerate(t):
                    sym[b, i] = dt.symbol_index(obs)
            for i in range(L):
                sel = dt.T[:, sym[:, i], :].permute(1, 0, 2)  # (B, |Q|, |Q|)
                q = torch.bmm(q.unsqueeze(1), sel).squeeze(1)
                states[:, i] = q.argmax(dim=1)
        else:
            # Stack per-step atom-probability vectors: (L, B, |AP|).
            P = torch.zeros(L, B, dt.n_atoms, device=dt.device)
            for b, t in enumerate(trace_list):
                for i, obs in enumerate(t):
                    P[i, b] = dt.prob_vector(obs)
            for i in range(L):
                M = dt.soft_matrix(P[i])  # (B, |Q|, |Q|)
                q = torch.bmm(q.unsqueeze(1), M).squeeze(1)
                states[:, i] = q.argmax(dim=1)

        states_cpu = states.cpu()
        return [
            self._verdict_from_path(states_cpu[b], lengths[b]) for b in range(B)
        ]

    def _verdict_from_path(self, path: torch.Tensor, length: int) -> Verdict:
        dt = self._dt
        for i in range(length):
            s = int(path[i])
            if s in dt.trap_idx:
                return Verdict.VIOLATE
            if s in dt.sink_idx:
                return Verdict.SATISFY
        last = dt.state_idx[dt.dfa.initial] if length == 0 else int(path[length - 1])
        return Verdict.SATISFY if dt.accepting[last] > 0 else Verdict.VIOLATE
