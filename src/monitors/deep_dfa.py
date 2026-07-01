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

  * factored — no 2^|AP| tensor. Two complementary views of each edge guard:

      (1) Crisp monitoring (the path Exp 1-3 time). Each guard is decomposed
          *once* at construction into a disjoint (orthogonal) cube cover by
          Shannon expansion, and stored as `require-true` / `require-false`
          integer masks over the atoms. The per-cell transition matrix is then
          a single vectorized tensor reduction over those masks — no per-cell
          Python recursion over sympy closures. This is what keeps the Exp 2
          factored curve **flat** in |AP| (Phase 0.2): the per-cell cost is a
          couple of batched tensor ops, not an O(formula-size) closure walk.
          It is exact for crisp 0/1 inputs (each cube contributes 0/1 and the
          cubes are mutually exclusive, so they sum to a 0/1 transition).

      (2) Differentiable soft path (`soft_matrix`, for the deferred adaptation
          PoC). Each guard is compiled to a closure that computes its
          satisfaction probability recursively over the boolean structure,
          assuming atom independence:
                   P(a)=p_a   P(¬φ)=1-P(φ)
                   P(φ∧ψ)=P(φ)P(ψ)   P(φ∨ψ)=1-(1-P(φ))(1-P(ψ))
          With **crisp** 0/1 inputs this is exact for *any* guard; with
          *fractional* probabilities it is exact only for read-once guards
          (the IJCNN family is read-once after MONA's factoring). This soft
          path is kept separate so its read-once semantics are unchanged.

    Neither view materializes the 2^|AP| dense tensor, so factored crisp
    monitoring scales to large |AP| (Exp 2) where dense hits the alphabet wall.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
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
        # (src_idx, dst_idx, prob_fn) per DFA transition — the differentiable
        # soft path (read-once-exact fractional probabilities; see soft_matrix).
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
        # here replaces the per-cell sympy-closure walk, so crisp_matrix stays
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

    def crisp_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Vectorized (..., |Q|, |Q|) transition matrix from the cube masks.

        For crisp 0/1 atom values p (..., |AP|) this is the exact transition
        matrix; it is the flat, closure-free path used by the monitor. Each
        cube's value is the product over atoms of
            1 - require_true * (1 - p) - require_false * p
        (= 1 for a don't-care atom, p for a require-true atom, 1-p for a
        require-false atom), and the disjoint cubes are summed into the matrix.
        """
        batch_shape = p.shape[:-1]
        n_batch = 1
        for d in batch_shape:
            n_batch *= d
        pf = p.reshape(n_batch, self.n_atoms).unsqueeze(1)  # (B, 1, |AP|)
        factor = 1.0 - self._cube_rt * (1.0 - pf) - self._cube_rf * pf  # (B, C, |AP|)
        cube_val = factor.prod(dim=-1)  # (B, C)
        M = torch.zeros(
            n_batch, self.n_states * self.n_states, device=p.device, dtype=p.dtype
        )
        M.index_add_(1, self._cube_flat, cube_val)
        return M.view(*batch_shape, self.n_states, self.n_states)

    def soft_matrix(self, p: torch.Tensor) -> torch.Tensor:
        """Build the (..., |Q|, |Q|) transition matrix for atom-prob p (..., |AP|).

        Differentiable in p via the recursive read-once probability closures.
        With crisp 0/1 p this is the exact transition matrix; with fractional p
        it is exact for read-once guards. This is the path for the deferred
        adaptation PoC; the crisp monitor uses :meth:`crisp_matrix` instead.
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
        return q @ dt.crisp_matrix(dt.prob_vector(obs))

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
                M = dt.crisp_matrix(P[i])  # (B, |Q|, |Q|)
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

    # ----- soft readout (Phase 1.2: monitoring under perceptual uncertainty) -----
    #
    # Marginal acceptance probability (Option A, CLAUDE.md § Phase 1). Given a
    # *soft* trace (per-atom probabilities), propagate the full state
    # DISTRIBUTION through the differentiable `soft_matrix` (row-stochastic for
    # fractional inputs) and read the accepting mass at end-of-trace. Crucially
    # there is NO mid-trace argmax: collapsing to a single state each step would
    # discard the probability mass and reduce to the brittle "threshold-and-walk"
    # readout (Option B), which is exactly what the symbolic baseline does.
    #
    # This uses `soft_matrix` (recursive read-once guard probabilities), not
    # `crisp_matrix`: on non-read-once guards the two differ, and the read-once
    # `soft_matrix` is what makes calibration an *empirical* question there
    # rather than an exact-marginal identity (Phase 1.3 / 3.3).

    def _require_soft(self) -> None:
        # The soft readout is the factored mode's differentiable path: it needs
        # the per-edge guard-probability closures (`soft_matrix` / `_edges`),
        # which only the factored build materializes. Dense stores a 2^|AP|
        # one-hot tensor with no fractional-input semantics. Compile with
        # mode="factored" (or use DeepDFAMonitorFactored) for soft monitoring.
        if self._dt.mode != "factored":
            raise ValueError(
                "soft readout requires factored mode; compile with "
                "mode='factored' (or use DeepDFAMonitorFactored)"
            )

    def acceptance_probability(
        self, soft_trace: Iterable[dict[str, float]], normalize: bool = False
    ) -> float:
        """Acceptance score in [0, 1] (usually) for a single soft trace.

        Reference (unbatched) implementation; see
        :meth:`batch_acceptance_probability` for the fast path and for the
        meaning of ``normalize`` (the read-once vs non-read-once caveat).
        """
        self._require_soft()
        dt = self._dt
        q = dt.mu.clone()
        for obs in soft_trace:
            q = q @ dt.soft_matrix(dt.soft_prob_vector(obs))
        accept = float(q @ dt.accepting)
        if normalize:
            mass = float(q.sum())
            return accept / mass if mass > 0.0 else accept
        return accept

    def soft_verdict(
        self,
        soft_trace: Iterable[dict[str, float]],
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> Verdict:
        """Binary verdict from the acceptance score at ``threshold``."""
        p = self.acceptance_probability(soft_trace, normalize=normalize)
        return Verdict.SATISFY if p >= threshold else Verdict.VIOLATE

    def batch_acceptance_probability(
        self,
        soft_traces: Iterable[Iterable[dict[str, float]]],
        normalize: bool = False,
    ) -> list[float]:
        """Marginal acceptance score for a batch of soft traces.

        One batched ``soft_matrix`` + ``bmm`` per cell over the whole batch.
        Traces of unequal length are padded; padding cells are masked so a
        shorter trace's distribution is frozen once it ends. Equivalent to
        ``[self.acceptance_probability(t) for t in soft_traces]``.

        ``normalize`` (Capability Exp A, Phase 1.4): ``soft_matrix`` is only
        row-stochastic when every DFA guard is **read-once**. On a
        non-read-once guard (e.g. the 2-of-3 majority) the independence-
        assuming guard-probability product over-counts, so the row sums
        exceed 1 and the raw ``q_final @ accepting`` is *not* a valid
        probability (it can exceed 1). With ``normalize=True`` the score is
        divided by the total propagated mass ``q_final @ 1``, forcing a value
        in [0, 1]; this is exact/unchanged for read-once guards (mass == 1)
        and a heuristic renormalization otherwise. The raw score is the
        settled Option-A readout and is kept as the default so the
        non-stochasticity remains observable (it is a *finding*).
        """
        self._require_soft()
        dt = self._dt
        trace_list = [list(t) for t in soft_traces]
        if not trace_list:
            return []
        lengths = torch.tensor(
            [len(t) for t in trace_list], device=dt.device
        )
        B, L = len(trace_list), int(lengths.max())
        q = dt.mu.unsqueeze(0).expand(B, -1).clone()  # (B, |Q|)
        if L == 0:
            accept = (q * dt.accepting).sum(dim=1)
            return self._finish_score(q, accept, normalize)
        P = torch.from_numpy(dt.encode_soft(trace_list, L)).to(dt.device)
        for i in range(L):
            M = dt.soft_matrix(P[:, i, :])  # (B, |Q|, |Q|)
            q_new = torch.bmm(q.unsqueeze(1), M).squeeze(1)
            active = (i < lengths).unsqueeze(1)  # freeze ended traces
            q = torch.where(active, q_new, q)
        accept = (q * dt.accepting).sum(dim=1)
        return self._finish_score(q, accept, normalize)

    @staticmethod
    def _finish_score(
        q: torch.Tensor, accept: torch.Tensor, normalize: bool
    ) -> list[float]:
        if normalize:
            mass = q.sum(dim=1)
            accept = torch.where(mass > 0.0, accept / mass, accept)
        return accept.cpu().tolist()

    def batch_soft_verdict(
        self,
        soft_traces: Iterable[Iterable[dict[str, float]]],
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> list[Verdict]:
        """Binary verdicts for a batch of soft traces at ``threshold``."""
        return [
            Verdict.SATISFY if p >= threshold else Verdict.VIOLATE
            for p in self.batch_acceptance_probability(soft_traces, normalize=normalize)
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
