"""Progression-based RuleRunner (Paradigm 2, corrected).

The original RuleRunner conflates concurrent instances of a subformula
reinstalled from different temporal contexts (the shared-register
temporal-instance limitation). This package implements the progression-based
reformulation of latex/3_rulerunner.tex §3.3, which carries residual formulae
obtained by formula progression and is sound and complete for every LTLf formula.

Public neural monitor (mirrors the old RuleRunner's flat/structured pair):

* ``ProgressionRuleRunnerMonitor`` — the **flat CILP network** (``flat.py``),
  multi-hot-root state, batched on CPU/CUDA. This is the monitor the
  experiments run.
* ``ProgressionRuleRunnerStructuredMonitor`` — the **structured factorized**
  encoding (``structured.py``): per-closure-node evaluation and per-root
  progression/reactivation CILP modules, following the same explicit two-phase
  organization as the old structured RuleRunner. Whole root sets are used only
  by a fixed exact sink/trap readout.

Reference / machinery (pure Python, CPU-only — no tensors, so honestly
``effective_device`` "cpu"; not headline experimental paradigms):

* ``ProgressionEngine`` — the lazy on-the-fly semantics oracle (``engine.py``);
* ``build_progression_dfa`` / ``ProgressionDFA`` — the eager construction the
  flat network is compiled from, carrying the cost-of-correctness metrics;
* ``build_factorized_progression_graph`` / ``FactorizedProgressionGraph`` —
  root-local transitions plus the aggregate graph used by the structured
  monitor's exact verdict readout;
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
    FactorizedProgressionGraph,
    ProgressionRuleRunnerStructuredMonitor,
    build_factorized_progression_graph,
)

__all__ = [
    "ProgressionRuleRunnerMonitor",
    "ProgressionRuleRunnerStructuredMonitor",
    "ProgressionEngine",
    "ProgressionRuleRunnerEagerMonitor",
    "ProgressionDFA",
    "build_progression_dfa",
    "FactorizedProgressionGraph",
    "build_factorized_progression_graph",
]
