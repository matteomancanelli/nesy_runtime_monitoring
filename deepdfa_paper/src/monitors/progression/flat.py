"""Flat CILP network for the progression-based RuleRunner (Part 2b).

The batched, device-aware realization. Faithful to §3.3's rule form: the state
carried across time is the **set of active roots** — a multi-hot vector over
``R_phi`` (formula registers), *not* an anonymous one-hot DFA state — and the
per-cell update is a single-hidden-layer CILP pass (Garcez & Zaverucha sign
activation), so it vectorizes over a whole batch of traces as matmuls on CPU or
CUDA.

Construction. Compiled from the eager ``ProgressionDFA`` (``eager.py``). Each
hidden unit is one ``(residual-state, guard-symbol)`` transition clause: it
pins **every** root register (``+1`` active / ``-1`` inactive, so it identifies
the current residual state exactly) and its state's guard atoms, and thus fires
iff the carried roots equal that state *and* the observation matches that
symbol. Because the carried state is always exactly one reachable residual
state and the guard symbol partitions the observations, **exactly one hidden
fires per cell**, and three output heads read off it:

* online verdict (SATISFY / VIOLATE — the accepting-sink / trap conditions),
* end-of-trace ``last`` accept bit,
* the next multi-hot root state (fed back).

So the network reproduces the eager transition **bit-for-bit** (verified in
tests against the eager, lazy, and symbolic monitors), while keeping the
RuleRunner-distinctive formula-register state and single-hidden-layer form —
and remaining differentiable under the usual ``sign -> tanh`` relaxation for
the adaptation setting.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from src.formula.compiler import Observation
from src.monitors.base import Monitor, Verdict
from src.monitors.progression.eager import (
    _SATISFY,
    _VIOLATE,
    ProgressionDFA,
    build_progression_dfa,
)
from src.monitors.progression.formula import split_conj

_W = 1.0


def _sign(x: torch.Tensor) -> torch.Tensor:
    """CILP sign activation: >0 -> +1, else -1 (a silent unit is 'false')."""
    return torch.where(x > 0, 1.0, -1.0)


class _FlatNet:
    """Weight matrices of the progression CILP network, built from an eager DFA."""

    def __init__(self, dfa: ProgressionDFA, device: torch.device) -> None:
        self.dfa = dfa
        self.device = device

        # Roots R_phi: distinct top-level conjuncts across all residual states.
        root_keys: dict[str, int] = {}
        state_roots: list[frozenset[int]] = []
        for st in dfa.states:
            idxs = set()
            for r in split_conj(st):
                if r.key not in root_keys:
                    root_keys[r.key] = len(root_keys)
                idxs.add(root_keys[r.key])
            state_roots.append(frozenset(idxs))
        # Exposed so the structured monitor (structured.py) can reuse this exact
        # recurrence with a consistent root indexing.
        self.root_keys = root_keys
        self.state_roots = state_roots
        self.n_roots = len(root_keys)
        self.atom_index = {a: i for i, a in enumerate(dfa.atoms)}
        self.n_atoms = len(dfa.atoms)
        D = self.n_roots + self.n_atoms

        # Enumerate hidden units: one per (state, guard symbol).
        # each entry: (state, guard symbol, next state, verdict code, last bit)
        hid: list[tuple[int, int, int, int, bool]] = []
        for i in range(dfa.n_states):
            for sym, nxt in dfa.trans[i].items():
                hid.append((i, sym, nxt, dfa.online[i][sym], dfa.last_accept[i][sym]))
        H = len(hid)

        W_ih = np.zeros((H, D), dtype=np.float32)
        b_h = np.zeros(H, dtype=np.float32)
        for h, (i, sym, _nxt, _code, _lv) in enumerate(hid):
            rs = state_roots[i]
            for r in range(self.n_roots):
                W_ih[h, r] = _W if r in rs else -_W
            rel = dfa.relevant[i]
            for j, a in enumerate(rel):
                W_ih[h, self.n_roots + self.atom_index[a]] = (
                    _W if (sym >> j) & 1 else -_W
                )
            n_body = self.n_roots + len(rel)
            b_h[h] = -_W * (n_body - 0.5)

        # Output heads (OR-of-hiddens: bias k-1, or -1 when no hidden targets it).
        # Exactly one hidden fires, so each head reads that hidden's precomputed
        # value.
        W_v = np.zeros((3, H), dtype=np.float32)   # rows: SAT, VIO, LAST
        for h, (_i, _sym, _nxt, code, lv) in enumerate(hid):
            if code == _SATISFY:
                W_v[0, h] = _W
            if code == _VIOLATE:
                W_v[1, h] = _W
            if lv:
                W_v[2, h] = _W
        b_v = np.array(
            [_W * (int((W_v[k] > 0).sum()) - 1) if (W_v[k] > 0).any() else -_W
             for k in range(3)],
            dtype=np.float32,
        )

        W_ns = np.zeros((self.n_roots, H), dtype=np.float32)  # next multi-hot roots
        for h, (_i, _sym, nxt, _code, _lv) in enumerate(hid):
            for r in state_roots[nxt]:
                W_ns[r, h] = _W
        b_ns = np.array(
            [_W * (int((W_ns[r] > 0).sum()) - 1) if (W_ns[r] > 0).any() else -_W
             for r in range(self.n_roots)],
            dtype=np.float32,
        )

        t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
        self.W_ih, self.b_h = t(W_ih), t(b_h)
        self.W_v, self.b_v = t(W_v), t(b_v)
        self.W_ns, self.b_ns = t(W_ns), t(b_ns)

        # Initial multi-hot roots (state 0), as a (n_roots,) +/-1 vector.
        init = -np.ones(self.n_roots, dtype=np.float32)
        for r in state_roots[dfa.initial]:
            init[r] = 1.0
        self.initial_state = t(init)

    def advance(
        self, state: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One batched cell. ``state`` (B, n_roots), ``atoms`` (B, n_atoms) in
        {+1,-1}. Returns (next_state, sat, vio, last), the last three as
        (B,) bool tensors."""
        x = torch.cat([state, atoms], dim=1)
        h = _sign(x @ self.W_ih.t() + self.b_h)
        v = h @ self.W_v.t() + self.b_v          # (B, 3)
        nxt = _sign(h @ self.W_ns.t() + self.b_ns)
        sat = v[:, 0] > 0
        vio = v[:, 1] > 0
        last = v[:, 2] > 0
        return nxt, sat, vio, last


class ProgressionRuleRunnerMonitor(Monitor):
    """Batched flat-CILP progression-based RuleRunner (multi-hot root state)."""

    def __init__(self, net: _FlatNet) -> None:
        self._net = net
        self._state = net.initial_state.clone()
        self._last_v: bool | None = None
        self._decided: Verdict | None = None

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "ProgressionRuleRunnerMonitor":
        dfa = build_progression_dfa(formula)
        return cls(_FlatNet(dfa, torch.device(device)))

    @property
    def effective_device(self) -> str:
        return "cuda" if self._net.device.type == "cuda" else "cpu"

    def reset(self) -> None:
        self._state = self._net.initial_state.clone()
        self._last_v = None
        self._decided = None

    def _atom_row(self, obs: Observation) -> torch.Tensor:
        row = torch.full((1, self._net.n_atoms), -1.0, device=self._net.device)
        for a, j in self._net.atom_index.items():
            if obs.get(a, False):
                row[0, j] = 1.0
        return row

    def step(self, obs: Observation) -> Verdict:
        if self._decided is not None:
            return self._decided
        atoms = self._atom_row(obs)
        nxt, sat, vio, last = self._net.advance(self._state.unsqueeze(0), atoms)
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

        One batched CILP pass (state-identify hidden + three output heads) per
        cell over the whole batch; identical verdicts to ``[run(t) ...]``. All
        traces advance uniformly (decided traces are not frozen); per-trace
        early-termination / end-of-trace are replayed afterwards — the first
        decided cell within a trace's length wins, else the ``last`` bit at its
        final cell. ``early_termination`` is accepted for interface parity but
        does not change the compute (every cell of every trace is processed)."""
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

        # Encode observations once: (maxL, B, n_atoms) in {+1,-1}.
        arr = np.full((maxL, B, net.n_atoms), -1.0, dtype=np.float32)
        for b, t in enumerate(trace_list):
            for i, obs in enumerate(t):
                for a, j in net.atom_index.items():
                    if obs.get(a, False):
                        arr[i, b, j] = 1.0
        clamp = torch.from_numpy(arr).to(dev)

        state = net.initial_state.unsqueeze(0).repeat(B, 1)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=dev)
        verdict_code = torch.zeros(B, maxL, dtype=torch.long, device=dev)
        last_bits = torch.zeros(B, maxL, dtype=torch.bool, device=dev)

        for i in range(maxL):
            nxt, sat, vio, last = net.advance(state, clamp[i])
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
        # last-cell accept bit per trace (its final valid cell).
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
