"""Benchmark-formula characterization utilities (Phase 3.3).

Two small, reusable functions that let the richer benchmark families
(`DECLARE_SUITE`, `NON_READ_ONCE_SUITE`, `STATE_BLOWUP_SUITE` in
:mod:`src.benchmarks.formulas`) declare properties that are *computed and
verified*, not hand-asserted:

  * :func:`guard_read_once` — is every DFA edge guard read-once (each atom
    appears at most once)? This is exactly the property that governs whether
    DeepDFA's ``soft_matrix`` is *exact* on fractional inputs: the recursive
    guard-probability product assumes atom independence, which is only correct
    when each atom is read once. So occurrence-counting on MONA's emitted guard
    string is the faithful check (a *finding* — see the non-read-once caveat in
    :mod:`src.monitors.deep_dfa`).

  * :func:`exact_marginal` — the brute-force probabilistic oracle: the true
    marginal acceptance probability of a *single* soft cell, obtained by
    enumerating all ``2^|atoms|`` crisp assignments and weighting by the
    independent-bit likelihood. This is the ground truth that ``soft_matrix``
    over-counts on non-read-once guards, and the reference curve for the Exp 7
    divergence panel.
"""

from __future__ import annotations

import re
from itertools import product

from src.formula.compiler import DFA
from src.monitors.base import Verdict
from src.monitors.symbolic_dfa import SymbolicDFAMonitor


def guard_read_once(dfa: DFA) -> tuple[bool, dict[str, int]]:
    """Whether every transition guard of ``dfa`` is read-once.

    A guard is read-once iff each atom appears at most once in its (MONA-emitted,
    Python ``not``/``and``/``or`` syntax) label. Occurrences are counted with a
    word-boundary regex per atom in ``dfa.atoms`` so multi-character atom names
    (``aa``, ``ab``) and substrings are handled correctly.

    Returns ``(is_read_once, worst)`` where ``worst`` maps each atom that ever
    exceeds one occurrence in *some* guard to its maximum multiplicity across all
    guards (empty when the DFA is read-once).
    """
    worst: dict[str, int] = {}
    patterns = {a: re.compile(rf"\b{re.escape(a)}\b") for a in dfa.atoms}
    for t in dfa.transitions:
        for atom, pat in patterns.items():
            count = len(pat.findall(t.label))
            if count > 1:
                worst[atom] = max(worst.get(atom, 0), count)
    return (not worst, worst)


def exact_marginal(formula: str, soft_cell: dict[str, float]) -> float:
    """True marginal acceptance probability of a single soft cell.

    Enumerates all ``2^|atoms|`` crisp assignments of one cell, weights each by
    the independent-bit likelihood ``∏ p_a^{x_a} (1-p_a)^{1-x_a}``, and sums the
    weight of the assignments the symbolic DFA accepts (as a length-1 trace).

    This is the ground-truth P(accept) that DeepDFA's ``soft_matrix``
    independence-product approximates — *exactly* for read-once guards, but with
    an over-count for non-read-once ones (the Phase 1.4 / Exp 7 finding). Kept
    single-cell because the divergence is a per-cell guard property and the
    threshold family is ``F(threshold)`` (|Q|=2, acceptance = P(threshold true)),
    which keeps the enumeration tractable.
    """
    return exact_marginal_trace(formula, [soft_cell])


def exact_marginal_trace(
    formula: str, soft_trace: list[dict[str, float]]
) -> float:
    """True marginal acceptance probability of a full soft trace.

    Brute-force over all ``2^(|atoms|·L)`` crisp traces, weighting each by the
    independent-bit likelihood. Exponential in trace length, so intended for
    short traces / few atoms only — the multi-state non-read-once anchor
    ``alt_response`` (2 atoms, L≈4) sits on the same over-count axis as the
    single-cell threshold family. :func:`exact_marginal` is the ``L=1`` special
    case (the one used by the calibration checks).
    """
    monitor = SymbolicDFAMonitor.compile(formula)
    if not soft_trace:
        monitor.reset()
        return 1.0 if monitor.run([]) is Verdict.SATISFY else 0.0
    atoms = tuple(soft_trace[0].keys())
    per_cell = list(product((False, True), repeat=len(atoms)))
    total = 0.0
    for combo in product(per_cell, repeat=len(soft_trace)):
        weight = 1.0
        crisp = []
        for bits, soft_cell in zip(combo, soft_trace):
            cell = dict(zip(atoms, bits))
            for atom, bit in cell.items():
                p = soft_cell[atom]
                weight *= p if bit else (1.0 - p)
            crisp.append(cell)
        if weight == 0.0:
            continue
        monitor.reset()
        if monitor.run(crisp) is Verdict.SATISFY:
            total += weight
    return total
