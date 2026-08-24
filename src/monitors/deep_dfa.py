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
    including formulas rejected by the original RuleRunner certifier because of
    shared-register temporal-instance conflation.

Two representations of the same transition function are provided:

  * dense    — T (|Q|, 2^|AP|, |Q|) one-hot transition tensor. One matmul
               per step, trivial GPU batching. Build/storage cost is
               exponential in |AP| (this is DeepDFA's structural scaling
               weakness, dual to original RuleRunner's shared-register limit
               and the symbolic DFA's state blowup). Best for small |AP| and for
               the batching showcase (Exp 3, ijcnn_n8 -> 256 symbols).

  * factored — no 2^|AP| tensor. Two complementary views of each edge guard:

      (1) Exact cube/WMC evaluation (the path Exp 1-3 time). Each guard is decomposed
          *once* at construction into a disjoint (orthogonal) cube cover by
          Shannon expansion, and stored as `require-true` / `require-false`
          integer masks over the atoms. The per-cell transition matrix is then
          a single vectorized tensor reduction over those masks — no per-cell
          Python recursion over sympy closures. This is what keeps the Exp 2
          factored curve **flat** in |AP| (Phase 0.2): the per-cell cost is a
          couple of batched tensor ops, not an O(formula-size) closure walk.
          It is exact for crisp 0/1 inputs and for fractional independent-
          Bernoulli atom probabilities: the latter is an exact weighted model
          count because the cubes are mutually exclusive.

      (2) Recursive approximation (`recursive_matrix`; historically exposed as
          `soft_matrix`). Each guard is compiled to a closure that applies
          independence identities locally over the boolean syntax:
                   P(a)=p_a   P(¬φ)=1-P(φ)
                   P(φ∧ψ)=P(φ)P(ψ)   P(φ∨ψ)=1-(1-P(φ))(1-P(ψ))
          With **crisp** 0/1 inputs this is exact for *any* guard; with
          *fractional* probabilities it is exact only for read-once guards
          (the IJCNN family is read-once after MONA's factoring). It remains as
          an explicit approximation/diagnostic, not the default soft monitor.

    Neither view materializes the 2^|AP| dense tensor, so factored crisp
    monitoring scales to large |AP| (Exp 2) where dense hits the alphabet wall.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from sympy import symbols, sympify
from sympy.logic.boolalg import And, BooleanFalse, BooleanTrue, Not, Or

from src.formula.compiler import DFA, Observation, compile_ltlf
from src.monitors.base import Monitor, Verdict

# A guard probability function: maps p (..., |AP|) -> prob (...).
ProbFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DeepDFAArtifactStats:
    """Stable, serializable diagnostics for a compiled DeepDFA artifact.

    Byte counts cover the transition-representation tensors themselves, not
    Python object overhead or temporary tensors allocated during a forward
    pass. ``None`` means that a quantity does not apply to the selected mode.
    """

    mode: str
    device: str
    dtype: str
    n_atoms: int
    alphabet_size: int
    n_states: int
    n_transitions: int
    n_accepting_states: int
    dense_tensor_elements: int | None
    dense_tensor_bytes: int | None
    cube_count: int | None
    cube_mask_elements: int | None
    cube_tensor_bytes: int | None


def _checked_probability_threshold(threshold: float) -> float:
    """Return a finite threshold in [0, 1], or raise a clear API error."""
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be finite and lie in [0, 1]")
    return value


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


def _guard_cubes(label: str, atom_index: dict[str, int]) -> list[dict[str, bool]]:
    """Disjoint (orthogonal) cube cover of a MONA guard label.

    Returns a list of cubes; each cube maps atom name -> required truth value
    (atoms absent from a cube are don't-cares). The cubes are mutually
    exclusive (Shannon expansion), so a guard's satisfaction equals the *sum*
    of the per-cube products — exact for crisp 0/1 inputs, and (because the
    cubes are disjoint) row-stochastic when summed over a state's out-edges.
    """
    stripped = label.strip()
    if stripped == "true":
        return [{}]  # one all-don't-care cube — always fires
    if stripped == "false":
        return []  # never fires
    locals_map = dict(zip(atom_index, symbols(list(atom_index))))
    expr = sympify(stripped, locals=locals_map)
    return [{str(v): val for v, val in cube.items()} for cube in _shannon_cubes(expr)]


def _shannon_cubes(expr) -> list[dict]:
    """Shannon-expand a boolean expr into disjoint cubes (paths to True)."""
    if isinstance(expr, BooleanTrue):
        return [{}]
    if isinstance(expr, BooleanFalse):
        return []
    v = sorted(expr.free_symbols, key=str)[0]
    cubes: list[dict] = []
    for cube in _shannon_cubes(expr.subs(v, BooleanTrue())):
        cubes.append({v: True, **cube})
    for cube in _shannon_cubes(expr.subs(v, BooleanFalse())):
        cubes.append({v: False, **cube})
    return cubes


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

    @property
    def artifact_stats(self) -> DeepDFAArtifactStats:
        """Describe the compiled representation without private-field access."""
        common = dict(
            mode=self.mode,
            device=str(self.device),
            dtype=str(self.mu.dtype),
            n_atoms=self.n_atoms,
            alphabet_size=1 << self.n_atoms,
            n_states=self.n_states,
            n_transitions=len(self.dfa.transitions),
            n_accepting_states=len(self.dfa.accepting),
        )
        if self.mode == "dense":
            return DeepDFAArtifactStats(
                **common,
                dense_tensor_elements=self.T.numel(),
                dense_tensor_bytes=self.T.numel() * self.T.element_size(),
                cube_count=None,
                cube_mask_elements=None,
                cube_tensor_bytes=None,
            )

        cube_count = self._cube_flat.numel()
        cube_tensors = (self._cube_rt, self._cube_rf, self._cube_flat)
        return DeepDFAArtifactStats(
            **common,
            dense_tensor_elements=None,
            dense_tensor_bytes=None,
            cube_count=cube_count,
            cube_mask_elements=self._cube_rt.numel() + self._cube_rf.numel(),
            cube_tensor_bytes=sum(t.numel() * t.element_size() for t in cube_tensors),
        )

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
        # (src_idx, dst_idx, prob_fn) per DFA transition — the recursive
        # approximation (read-once-exact; see recursive_matrix).
        self._edges: list[tuple[int, int, ProbFn]] = [
            (
                self.state_idx[t.src],
                self.state_idx[t.dst],
                _compile_guard_prob(t.label, self.atom_index),
            )
            for t in self.dfa.transitions
        ]

        # Vectorized crisp path: precompute require-true / require-false masks
        # for every cube of every edge's disjoint cover. Building these *once*
        # here replaces the per-cell sympy-closure walk, so exact_matrix stays
        # flat in |AP| (Phase 0.2).
        rt_rows: list[list[float]] = []
        rf_rows: list[list[float]] = []
        flat_idx: list[int] = []  # src * |Q| + dst, per cube
        for t in self.dfa.transitions:
            si, di = self.state_idx[t.src], self.state_idx[t.dst]
            for cube in _guard_cubes(t.label, self.atom_index):
                rt = [0.0] * self.n_atoms
                rf = [0.0] * self.n_atoms
                for atom, val in cube.items():
                    (rt if val else rf)[self.atom_index[atom]] = 1.0
                rt_rows.append(rt)
                rf_rows.append(rf)
                flat_idx.append(si * self.n_states + di)

        n_cubes = len(flat_idx)
        self._cube_rt = torch.zeros(n_cubes, self.n_atoms, device=self.device)
        self._cube_rf = torch.zeros(n_cubes, self.n_atoms, device=self.device)
        if n_cubes:
            self._cube_rt[:] = torch.tensor(rt_rows, device=self.device)
            self._cube_rf[:] = torch.tensor(rf_rows, device=self.device)
        self._cube_flat = torch.tensor(flat_idx, dtype=torch.long, device=self.device)

    def exact_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Exact (..., |Q|, |Q|) transition matrix from disjoint cubes.

        For crisp 0/1 atom values this is the deterministic DFA transition
        matrix. For fractional values it is the exact expected transition
        matrix under independent Bernoulli atoms: every cube is a conjunction,
        and the disjoint cubes can be summed without double-counting. Each
        cube's value is the differentiable product
            1 - require_true * (1 - p) - require_false * p
        (= 1 for a don't-care atom, p for a require-true atom, 1-p for a
        require-false atom), and the disjoint cubes are summed into the matrix.
        """
        batch_shape = p.shape[:-1]
        n_batch = 1
        for d in batch_shape:
            n_batch *= d
        pf = p.reshape(n_batch, self.n_atoms).unsqueeze(1)  # (B, 1, |AP|)
        rt = self._cube_rt.to(dtype=p.dtype)
        rf = self._cube_rf.to(dtype=p.dtype)
        factor = 1.0 - rt * (1.0 - pf) - rf * pf  # (B, C, |AP|)
        cube_val = factor.prod(dim=-1)  # (B, C)
        M = torch.zeros(
            n_batch, self.n_states * self.n_states, device=p.device, dtype=p.dtype
        )
        M.index_add_(1, self._cube_flat, cube_val)
        return M.view(*batch_shape, self.n_states, self.n_states)

    def crisp_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for :meth:`exact_matrix`.

        The old name described its use by the crisp benchmark, not its
        semantics: the disjoint-cube calculation is exact for fractional
        independent-Bernoulli inputs too.
        """
        return self.exact_matrix(p)

    def recursive_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Approximate transition matrix from recursive Boolean evaluation.

        Differentiable in p via the recursive read-once probability closures.
        With crisp 0/1 p this is the exact transition matrix; with fractional p
        it is exact for read-once guards only. Repeated variables can introduce
        correlations between subexpressions, so this method can double-count
        mass. Use :meth:`exact_matrix` for probabilistic monitoring.
        """
        batch_shape = p.shape[:-1]
        M = torch.zeros(
            *batch_shape,
            self.n_states,
            self.n_states,
            device=p.device,
            dtype=p.dtype,
        )
        for si, di, fn in self._edges:
            M[..., si, di] = M[..., si, di] + fn(p)
        return M

    def soft_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for the historical recursive approximation.

        New soft-monitoring code should use :meth:`exact_matrix`. This alias is
        retained so existing experiments that intentionally measure the
        recursive approximation do not silently change meaning.
        """
        return self.recursive_matrix(p)

    def prob_vector(self, obs: Observation) -> torch.Tensor:
        """Crisp atom-probability vector (|AP|,) from an observation."""
        p = torch.zeros(self.n_atoms, device=self.device)
        for i, a in enumerate(self.atoms):
            if obs.get(a, False):
                p[i] = 1.0
        return p

    def soft_prob_vector(self, obs: dict[str, float]) -> torch.Tensor:
        """Atom-probability vector (|AP|,) from a *soft* observation.

        Values are per-atom probabilities in [0, 1]; a missing atom is treated
        as probability 0 (false), matching :meth:`prob_vector`'s crisp default.
        """
        p = torch.zeros(self.n_atoms, device=self.device)
        for i, a in enumerate(self.atoms):
            p[i] = float(obs.get(a, 0.0))
        return p

    def encode_soft(
        self, trace_list: list[list[dict[str, float]]], L: int
    ) -> np.ndarray:
        """Atom-probability array (B, L, |AP|) for a padded batch of soft traces.

        Soft analogue of :meth:`encode_presence`: reads the float probability of
        each atom (missing -> 0.0) instead of a bool. Padding cells (beyond a
        trace's length) are left 0.0; the batched soft readout masks them out so
        they never affect a shorter trace's state distribution.
        """
        P = np.zeros((len(trace_list), L, self.n_atoms), dtype=np.float32)
        for b, trace in enumerate(trace_list):
            for i, obs in enumerate(trace):
                P[b, i] = [float(obs.get(a, 0.0)) for a in self.atoms]
        return P

    def encode_presence(
        self, trace_list: list[list[Observation]], L: int
    ) -> np.ndarray:
        """Crisp atom-presence array (B, L, |AP|) for a padded batch of traces.

        Vectorized batch encoder: builds the whole presence array in numpy in a
        single pass (one list-comprehension row per cell) instead of allocating
        a torch vector per cell. This keeps batch encoding out of the per-cell
        compute the timing measures, so the factored Exp 2 curve reflects the
        model cost, not Python tensor-allocation overhead (Phase 0.2).
        """
        pres = np.zeros((len(trace_list), L, self.n_atoms), dtype=np.float32)
        for b, trace in enumerate(trace_list):
            for i, obs in enumerate(trace):
                pres[b, i] = [obs.get(a, False) for a in self.atoms]
        return pres


# Above this many bytes for the (L·B·|Q|²) prefix stack, the parallel scan
# falls back to the sequential loop (which is O(1)-memory in trace length).
SCAN_MEM_LIMIT_BYTES = 4 * 1024**3  # 4 GB


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

    @property
    def effective_device(self) -> str:
        """Device the transition tensor actually lives / computes on."""
        return "cuda" if self._dt.device.type == "cuda" else "cpu"

    @property
    def artifact_stats(self) -> DeepDFAArtifactStats:
        """Public representation diagnostics for artifact reports and audits."""
        return self._dt.artifact_stats

    def _advance(self, q: torch.Tensor, obs: Observation) -> torch.Tensor:
        dt = self._dt
        if dt.mode == "dense":
            return q @ dt.T[:, dt.symbol_index(obs), :]
        return q @ dt.exact_matrix(dt.prob_vector(obs))

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
        self,
        traces: Iterable[Iterable[Observation]],
        early_termination: bool = True,
    ) -> list[Verdict]:
        """Process all traces with batched matmuls.

        Equivalent to ``[self.run(t) for t in traces]`` (same early-termination
        and end-of-trace semantics), but the per-step update is one batched
        ``bmm`` over the whole batch. Works in both modes; the dense mode is
        the GPU-batching showcase.

        ``early_termination`` is accepted for interface parity with the base
        ``Monitor`` but does not change the compute: the batched path already
        advances *every* trace through *all* its cells uniformly (decided
        traces are not frozen), so DeepDFA always pays the full per-cell cost.
        Early termination is only replayed afterwards, per trace, to recover
        the correct verdict (:meth:`_verdict_from_path`) — that reconstruction
        runs regardless of the flag, so verdicts stay correct either way.
        """
        dt = self._dt
        trace_list = [list(t) for t in traces]
        if not trace_list:
            return []
        lengths = [len(t) for t in trace_list]
        B, L = len(trace_list), max(lengths)

        q = dt.mu.unsqueeze(0).expand(B, -1).clone()  # (B, |Q|)
        states = torch.empty(B, L, dtype=torch.long, device=dt.device)

        pres = dt.encode_presence(trace_list, L)  # (B, L, |AP|), numpy float32

        if dt.mode == "dense":
            weights = (1 << np.arange(dt.n_atoms, dtype=np.int64))
            sym_np = (pres.astype(np.int64) * weights).sum(axis=2)  # (B, L)
            sym = torch.from_numpy(sym_np).to(dt.device)
            for i in range(L):
                sel = dt.T[:, sym[:, i], :].permute(1, 0, 2)  # (B, |Q|, |Q|)
                q = torch.bmm(q.unsqueeze(1), sel).squeeze(1)
                states[:, i] = q.argmax(dim=1)
        else:
            # Per-step atom-probability stack (L, B, |AP|) from the batch encoder.
            P = torch.from_numpy(pres).to(dt.device).permute(1, 0, 2).contiguous()
            for i in range(L):
                M = dt.exact_matrix(P[i])  # (B, |Q|, |Q|)
                q = torch.bmm(q.unsqueeze(1), M).squeeze(1)
                states[:, i] = q.argmax(dim=1)

        states_cpu = states.cpu()
        return [
            self._verdict_from_path(states_cpu[b], lengths[b]) for b in range(B)
        ]

    # ----- parallel prefix-scan batched run (Phase 0.6: kill the per-cell
    #        launch overhead by folding the whole trace into O(log L) matmuls) --
    #
    # The crisp update q_t = q_{t-1} @ M_t makes the state path a *prefix product*
    # of the per-cell transition matrices: q_t = q_0 @ (M_1 @ ... @ M_t). Matrix
    # product is associative, so those prefix products can be computed with an
    # associative (Hillis–Steele) scan in ceil(log2 L) batched matmuls over the
    # WHOLE time axis at once, instead of L sequential per-cell bmm()s. The
    # sequential path's dominant cost at small |Q| is the fixed per-launch
    # overhead paid L times (~1e2 µs/cell at batch 1 in Exp 3); the scan pays it
    # ~log2(L) times over much larger matmuls, which is the lever that can invert
    # the "symbolic always wins" trend when there is enough work (large batch
    # and/or large |Q|). Trade-off: it materializes all L·B transition matrices
    # (L·B·|Q|² floats), so it is memory-bound at large |Q| — we fall back to the
    # sequential loop past SCAN_MEM_LIMIT_BYTES.

    def _transition_stack(self, trace_list, L: int, B: int) -> torch.Tensor:
        """All per-cell transition matrices as (L, B, |Q|, |Q|)."""
        dt = self._dt
        pres = dt.encode_presence(trace_list, L)  # (B, L, |AP|)
        if dt.mode == "dense":
            weights = 1 << np.arange(dt.n_atoms, dtype=np.int64)
            sym_np = (pres.astype(np.int64) * weights).sum(axis=2)  # (B, L)
            sym = torch.from_numpy(sym_np).to(dt.device).transpose(0, 1)  # (L, B)
            # dt.T is (|Q|, S, |Q|); gather symbols -> (|Q|, L, B, |Q|) -> (L,B,|Q|,|Q|)
            return dt.T[:, sym, :].permute(1, 2, 0, 3).contiguous()
        # (L, B, |AP|) then flatten cells for one vectorized exact_matrix build.
        P = torch.from_numpy(pres).to(dt.device).permute(1, 0, 2).contiguous()
        M = dt.exact_matrix(P.reshape(L * B, dt.n_atoms))  # (L*B, |Q|, |Q|)
        return M.view(L, B, dt.n_states, dt.n_states)

    def _scan_states(self, trace_list, lengths, L: int, B: int) -> torch.Tensor:
        """Per-cell argmax state path (B, L) via an associative prefix scan."""
        dt = self._dt
        mm = self._transition_stack(trace_list, L, B)  # (L, B, |Q|, |Q|)
        # Hillis–Steele inclusive prefix product along the time axis (dim 0).
        d = 1
        while d < L:
            prev = mm
            mm = prev.clone()
            mm[d:] = torch.matmul(prev[: L - d], prev[d:])  # left operand = earlier
            d *= 2
        # q_t = q_0 @ P_t  (mu is one-hot on the initial state)
        q = torch.einsum("s,lbst->lbt", dt.mu, mm)  # (L, B, |Q|)
        return q.argmax(dim=-1).transpose(0, 1).contiguous().cpu()  # (B, L)

    def _batch_run_scan(self, traces, early_termination: bool = True) -> list[Verdict]:
        trace_list = [list(t) for t in traces]
        if not trace_list:
            return []
        lengths = [len(t) for t in trace_list]
        B, L = len(trace_list), max(lengths)
        dt = self._dt
        est_bytes = L * B * dt.n_states * dt.n_states * 4
        if est_bytes > SCAN_MEM_LIMIT_BYTES:
            # Too big to materialize the whole prefix stack; the sequential loop
            # is O(1)-memory in L. (A chunked scan would bound this; left for
            # later — the point here is to measure the scan where it fits.) Warn
            # loudly so a benchmark never *silently* reports the sequential path
            # under the "scan" label (e.g. Exp 6 at very large |Q|).
            warnings.warn(
                f"DeepDFAMonitorScan: prefix stack ~{est_bytes / 1e9:.1f} GB exceeds "
                f"SCAN_MEM_LIMIT_BYTES ({SCAN_MEM_LIMIT_BYTES / 1e9:.1f} GB) at "
                f"|Q|={dt.n_states}, L={L}, B={B}; falling back to the SEQUENTIAL "
                f"loop — this measurement is NOT the scan.",
                RuntimeWarning,
                stacklevel=2,
            )
            return super().batch_run(traces, early_termination=early_termination)
        states = self._scan_states(trace_list, lengths, L, B)
        return [self._verdict_from_path(states[b], lengths[b]) for b in range(B)]

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

    # ----- probabilistic readout -----
    #
    # Given independent per-atom probabilities, propagate the full state
    # distribution through exact disjoint-cube weighted model counting and read
    # the accepting mass at the end. There is deliberately no mid-trace argmax:
    # that would discard probability mass and cease to compute the marginal.

    def _require_soft(self) -> None:
        # The per-atom probabilistic readout is exposed by factored mode. Dense
        # stores matrices indexed by complete symbols, rather than a direct
        # per-atom probability interface.
        if self._dt.mode != "factored":
            raise ValueError(
                "soft readout requires factored mode; compile with "
                "mode='factored' (or use DeepDFAMonitorFactored)"
            )

    def acceptance_probability_tensor(
        self,
        probabilities: torch.Tensor,
        lengths: torch.Tensor | None = None,
        normalize: bool = False,
        method: str = "exact",
    ) -> torch.Tensor:
        """Return differentiable acceptance probabilities for soft traces.

        ``probabilities`` has shape ``(L, |AP|)`` for one trace or
        ``(B, L, |AP|)`` for a batch. Values are independent Bernoulli atom
        probabilities. ``lengths`` optionally gives the unpadded length of each
        batched trace; ended traces are frozen while later batch cells run.

        The default ``method="exact"`` uses disjoint-cube weighted model
        counting and is exact for arbitrary guards under that input model.
        ``method="recursive"`` retains the historical read-once-only
        approximation for diagnostics. The returned tensor remains connected
        to ``probabilities`` through autograd.
        """
        self._require_soft()
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("probabilities must be a torch.Tensor")
        if not probabilities.is_floating_point():
            raise TypeError("probabilities must have a floating-point dtype")
        if probabilities.ndim not in (2, 3):
            raise ValueError(
                "probabilities must have shape (L, |AP|) or (B, L, |AP|)"
            )

        dt = self._dt
        unbatched = probabilities.ndim == 2
        P = probabilities.unsqueeze(0) if unbatched else probabilities
        if P.shape[-1] != dt.n_atoms:
            raise ValueError(
                f"expected {dt.n_atoms} atom probabilities, got {P.shape[-1]}"
            )
        P = P.to(dt.device)
        if not bool(torch.isfinite(P).all()):
            raise ValueError("probabilities must contain only finite values")
        if bool(torch.any((P < 0.0) | (P > 1.0))):
            raise ValueError("probabilities must lie in [0, 1]")
        B, L, _ = P.shape

        if lengths is None:
            trace_lengths = torch.full(
                (B,), L, dtype=torch.long, device=dt.device
            )
        else:
            trace_lengths = torch.as_tensor(
                lengths, dtype=torch.long, device=dt.device
            )
            if trace_lengths.shape != (B,):
                raise ValueError(f"lengths must have shape ({B},)")
            if bool(torch.any((trace_lengths < 0) | (trace_lengths > L))):
                raise ValueError(f"lengths entries must lie in [0, {L}]")

        if method == "exact":
            matrix_fn = dt.exact_matrix
        elif method == "recursive":
            matrix_fn = dt.recursive_matrix
        else:
            raise ValueError("method must be 'exact' or 'recursive'")

        q = dt.mu.to(dtype=P.dtype).unsqueeze(0).expand(B, -1).clone()
        for i in range(L):
            M = matrix_fn(P[:, i, :])
            q_new = torch.bmm(q.unsqueeze(1), M).squeeze(1)
            q = torch.where((i < trace_lengths).unsqueeze(1), q_new, q)

        accepting = dt.accepting.to(dtype=P.dtype)
        score = (q * accepting).sum(dim=1)
        if normalize:
            mass = q.sum(dim=1)
            score = torch.where(mass > 0.0, score / mass, score)
        return score[0] if unbatched else score

    def acceptance_probability(
        self,
        soft_trace: Iterable[dict[str, float]],
        normalize: bool = False,
        method: str = "exact",
    ) -> float:
        """Python-float wrapper around :meth:`acceptance_probability_tensor`.

        Use the tensor-native method directly when gradients are required.
        """
        self._require_soft()
        dt = self._dt
        trace = list(soft_trace)
        P = torch.tensor(
            [[float(obs.get(a, 0.0)) for a in dt.atoms] for obs in trace],
            dtype=dt.mu.dtype,
            device=dt.device,
        ).reshape(len(trace), dt.n_atoms)
        score = self.acceptance_probability_tensor(
            P, normalize=normalize, method=method
        )
        return float(score.detach().cpu())

    def soft_verdict(
        self,
        soft_trace: Iterable[dict[str, float]],
        threshold: float = 0.5,
        normalize: bool = False,
        method: str = "exact",
    ) -> Verdict:
        """Binary verdict from the acceptance score at ``threshold``."""
        threshold = _checked_probability_threshold(threshold)
        p = self.acceptance_probability(
            soft_trace, normalize=normalize, method=method
        )
        return Verdict.SATISFY if p >= threshold else Verdict.VIOLATE

    def batch_acceptance_probability(
        self,
        soft_traces: Iterable[Iterable[dict[str, float]]],
        normalize: bool = False,
        method: str = "exact",
    ) -> list[float]:
        """Python-list wrapper for batched probabilistic monitoring.

        With the default exact method, normalization is a numerical no-op
        because every transition row has unit mass. It remains available for
        explicit ``method="recursive"`` compatibility experiments.
        """
        self._require_soft()
        dt = self._dt
        trace_list = [list(t) for t in soft_traces]
        if not trace_list:
            return []
        lengths = torch.tensor([len(t) for t in trace_list], device=dt.device)
        L = int(lengths.max())
        P = torch.from_numpy(dt.encode_soft(trace_list, L)).to(dt.device)
        scores = self.acceptance_probability_tensor(
            P, lengths=lengths, normalize=normalize, method=method
        )
        return scores.detach().cpu().tolist()

    def batch_soft_verdict(
        self,
        soft_traces: Iterable[Iterable[dict[str, float]]],
        threshold: float = 0.5,
        normalize: bool = False,
        method: str = "exact",
    ) -> list[Verdict]:
        """Binary verdicts for a batch of soft traces at ``threshold``."""
        threshold = _checked_probability_threshold(threshold)
        return [
            Verdict.SATISFY if p >= threshold else Verdict.VIOLATE
            for p in self.batch_acceptance_probability(
                soft_traces, normalize=normalize, method=method
            )
        ]


# ---------------------------------------------------------------------------
# Mode-fixed subclasses (reusable across experiments)
# ---------------------------------------------------------------------------
#
# The timing harness keys results by ``monitor_cls.__name__``, so these distinct
# names give dense and factored their own curves/CSV rows when both appear in an
# experiment's MONITORS list. ``DeepDFAMonitor`` itself still defaults to dense.


class DeepDFAMonitorDense(DeepDFAMonitor):
    """DeepDFA pinned to the dense ``2^|AP|`` transition tensor."""

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "DeepDFAMonitorDense":
        return super().compile(formula, mode="dense", device=device)


class DeepDFAMonitorFactored(DeepDFAMonitor):
    """DeepDFA pinned to the factored (vectorized cube-mask) crisp path."""

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "DeepDFAMonitorFactored":
        return super().compile(formula, mode="factored", device=device)


class DeepDFAMonitorScan(DeepDFAMonitor):
    """DeepDFA whose ``batch_run`` uses the parallel prefix-scan (dense tensor).

    Same verdicts as the sequential ``DeepDFAMonitor`` (verified against it), but
    the whole batched trace is folded into O(log L) big matmuls instead of L
    per-cell ``bmm``s, so it pays the fixed per-launch overhead ~log2(L) times
    rather than L times. This is the "single/parallel scan" direction: the lever
    that can beat the symbolic walk once there is enough work (large batch and/or
    large automaton). Memory-bound at large |Q| (falls back to the sequential
    loop past ``SCAN_MEM_LIMIT_BYTES``).
    """

    @classmethod
    def compile(
        cls, formula: str, device: str | torch.device = "cpu"
    ) -> "DeepDFAMonitorScan":
        return super().compile(formula, mode="dense", device=device)

    def batch_run(self, traces, early_termination: bool = True) -> list[Verdict]:
        return self._batch_run_scan(traces, early_termination=early_termination)
