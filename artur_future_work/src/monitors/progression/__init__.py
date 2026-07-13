"""Progression-based RuleRunner (Paradigm 2, corrected).

The original RuleRunner conflates concurrent instances of a subformula
reinstalled from different temporal contexts (the nested-temporal
limitation). This package implements the progression-based reformulation
of latex/3_rulerunner.tex §3.3, which carries residual formulae obtained
by formula progression and is sound and complete for every LTLf formula.

Public neural monitor (mirrors the old RuleRunner's flat/structured pair):

* ``ProgressionRuleRunnerMonitor`` — the **flat CILP network** (``flat.py``),
  multi-hot-root state, batched on CPU/CUDA. This is the monitor the
  experiments run.
* ``ProgressionRuleRunnerStructuredMonitor`` — the **structured** encoding
  (``structured.py``), one CILP subnetwork per closure node ``C_phi``; the
  modular / local-learning contrast (Paper B), not the throughput path.

Reference / machinery (pure Python, CPU-only — no tensors, so honestly
``effective_device`` "cpu"; not headline experimental paradigms):

* ``ProgressionEngine`` — the lazy on-the-fly semantics oracle (``engine.py``);
* ``build_progression_dfa`` / ``ProgressionDFA`` — the eager construction the
  flat network is compiled from, carrying the cost-of-correctness metrics;
* ``ProgressionRuleRunnerEagerMonitor`` — table-driven correctness oracle.
"""

from src.monitors.progression.eager import (
    ProgressionDFA,
    ProgressionRuleRunnerEagerMonitor,
    build_progression_dfa,
)
from src.monitors.progression.engine import ProgressionEngine
from src.monitors.progression.flat import ProgressionRuleRunnerMonitor
from src.monitors.progression.structured import (
    ProgressionRuleRunnerStructuredMonitor,
)

__all__ = [
    "ProgressionRuleRunnerMonitor",
    "ProgressionRuleRunnerStructuredMonitor",
    "ProgressionEngine",
    "ProgressionRuleRunnerEagerMonitor",
    "ProgressionDFA",
    "build_progression_dfa",
]
