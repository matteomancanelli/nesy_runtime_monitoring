"""Observation-corruption models + oracle for Capability Exp A (Phase 1.1).

Monitoring under perceptual uncertainty: a neural perceptor emits
*probabilities*, not booleans. This module produces the inputs for that
experiment — ground-truth verdicts for crisp traces (the oracle) and
corrupted / soft versions of those traces at a controllable noise level ε.

Contract types (the interface the soft consumers in Phase 1.2 read):

    Observation      = dict[str, bool]    # crisp cell (from compiler.py)
    SoftObservation  = dict[str, float]   # per-atom probability in [0, 1]
    SoftTrace        = list[SoftObservation]

A crisp trace is the special case of a soft trace whose values are all
0.0 / 1.0, so every corruption model emits floats uniformly and the
downstream monitors take one input type.

Two corruption models, deliberately different in character:

  * ``BitFlipNoise`` — flips each atom's bit with probability ε and emits
    a crisp 0/1. This is the adversary the *thresholding symbolic
    baseline* actually sees: at ε it systematically corrupts bits, and at
    ε=1 every bit is flipped. Information is destroyed, not softened.

  * ``BetaNoise`` — replaces each bit with a Beta-distributed probability.
    The realistic "soft perceptor": at ε=0 it is a point mass on the true
    bit; as ε grows the mean drifts toward 0.5 (less informative) and the
    variance grows. This is the fractional-probability input only the soft
    paradigms (DeepDFA's ``soft_matrix``) can consume without thresholding.

See CLAUDE.md § Phase 1 for how these feed the accuracy-vs-ε and
calibration metrics (Phase 1.3), and why the calibration set must include
a non-read-once-guard formula.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.formula.compiler import Observation
from src.monitors.base import Verdict
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

SoftObservation = dict[str, float]
SoftTrace = list[SoftObservation]
CrispTrace = list[Observation]


# ---------------------------------------------------------------------------
# Oracle: ground-truth verdicts for crisp traces
# ---------------------------------------------------------------------------


def true_verdicts(formula: str, crisp_traces: list[CrispTrace]) -> list[Verdict]:
    """Ground-truth binary verdict for each crisp trace.

    The symbolic DFA is exact on every formula (no nested-temporal limit),
    so its end-of-trace verdict is the label the noisy monitors are scored
    against. Verdicts are binary (SATISFY / VIOLATE); ``run`` never returns
    UNDECIDED at end-of-trace.
    """
    monitor = SymbolicDFAMonitor.compile(formula)
    return [monitor.run(trace) for trace in crisp_traces]


# ---------------------------------------------------------------------------
# Symbolic soft-consumption baseline: threshold, then walk (the brittle path)
# ---------------------------------------------------------------------------


def threshold_trace(soft_trace: SoftTrace, threshold: float = 0.5) -> CrispTrace:
    """Threshold a soft trace to a crisp one at ``threshold``.

    This is the symbolic paradigm's only way to consume soft observations: it
    must collapse each probability to a bool and discard the confidence, then
    run the crisp DFA walk. It is the deliberately brittle baseline in
    Capability Exp A — it cannot emit a calibrated confidence, and on
    ``BitFlipNoise`` it simply monitors the corrupted bits.
    """
    return [
        {a: v >= threshold for a, v in obs.items()} for obs in soft_trace
    ]


# ---------------------------------------------------------------------------
# Corruption models
# ---------------------------------------------------------------------------


class NoiseModel(ABC):
    """Maps a crisp trace to a soft trace at a fixed noise level ε."""

    @abstractmethod
    def corrupt(self, trace: CrispTrace, rng: np.random.Generator) -> SoftTrace:
        """Return a soft version of ``trace``, drawing from ``rng`` (reproducible)."""

    def corrupt_all(
        self, traces: list[CrispTrace], rng: np.random.Generator
    ) -> list[SoftTrace]:
        """Corrupt a whole batch, threading the same ``rng`` through each trace."""
        return [self.corrupt(t, rng) for t in traces]


@dataclass(frozen=True)
class BitFlipNoise(NoiseModel):
    """Flip each atom's bit with probability ε; emit crisp 0/1 floats.

    ε=0 is the identity (up to bool→float); ε=1 flips every bit. The
    output is always in {0.0, 1.0} — this is the destroyed-information
    adversary the thresholding symbolic baseline sees, not a softened
    probability.
    """

    eps: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {self.eps}")

    def corrupt(self, trace: CrispTrace, rng: np.random.Generator) -> SoftTrace:
        out: SoftTrace = []
        for obs in trace:
            atoms = list(obs.keys())
            truth = np.array([obs[a] for a in atoms], dtype=bool)
            flip = rng.random(len(atoms)) < self.eps
            bits = np.where(flip, ~truth, truth)
            out.append({a: float(bits[i]) for i, a in enumerate(atoms)})
        return out


@dataclass(frozen=True)
class BetaNoise(NoiseModel):
    """Replace each bit with a Beta-distributed probability in [0, 1].

    The emitted probability has mean ``m = (1 - eps) * b + eps * 0.5`` and
    concentration ``ν = concentration`` (a Beta(m·ν, (1-m)·ν) sample).
    So at ε=0 it is a point mass on the true bit ``b`` (identity), and as ε
    grows the mean drifts toward 0.5 and the variance grows — a single-knob
    degradation from a confident-correct perceptor to an uninformative one.

    ``concentration`` sets the spread at fixed ε (higher = tighter around
    the mean); the default keeps low-ε samples visibly confident.
    """

    eps: float
    concentration: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {self.eps}")
        if self.concentration <= 0.0:
            raise ValueError(f"concentration must be > 0, got {self.concentration}")

    def corrupt(self, trace: CrispTrace, rng: np.random.Generator) -> SoftTrace:
        out: SoftTrace = []
        for obs in trace:
            atoms = list(obs.keys())
            b = np.array([1.0 if obs[a] else 0.0 for a in atoms])
            if self.eps == 0.0:
                p = b  # point mass on the true bit — exact identity
            else:
                mean = (1.0 - self.eps) * b + self.eps * 0.5
                alpha = mean * self.concentration
                beta = (1.0 - mean) * self.concentration
                p = rng.beta(alpha, beta)
            out.append({a: float(p[i]) for i, a in enumerate(atoms)})
        return out
