"""RuleRunner-style LTLf monitors (Paradigm 2).

Pipeline: LTLf parse tree -> RuleRunner rule system (eval + reactivation
rules) -> CILP-encoded neural network -> monitor wrapper. Faithful to
Perotti, Garcez, Boella, IJCNN 2014.  This package also exposes the exact
product certifier and the bounded-event middle repair.  The complete
progression repair lives in :mod:`src.monitors.progression`.  See
``docs/rulerunner_status.md`` for the cross-version status.
"""

from src.monitors.rulerunner.bounded import (
    BoundedEventPipeline,
    BoundedIsland,
    EventizedFormula,
    bounded_horizon,
    derive_event_cell,
    derive_event_trace,
    evaluate_bounded,
    eventize_bounded_islands,
)
from src.monitors.rulerunner.bounded_cilp import (
    BoundedEventRuleRunnerMonitor,
    BoundedEventStructuredRuleRunnerMonitor,
    UnsafeRuleRunnerSkeleton,
)
from src.monitors.rulerunner.bounded_extrapolation import (
    BoundedExtrapolationCILP,
    ExtrapolationLimitExceeded,
)
from src.monitors.rulerunner.equivalence import (
    EquivalenceWitness,
    RuleRunnerEquivalenceResult,
    certify_rule_runner,
)
from src.monitors.rulerunner.monitor import RuleRunnerMonitor
from src.monitors.rulerunner.structured import StructuredRuleRunnerMonitor

__all__ = [
    "BoundedEventPipeline",
    "BoundedEventRuleRunnerMonitor",
    "BoundedEventStructuredRuleRunnerMonitor",
    "BoundedExtrapolationCILP",
    "BoundedIsland",
    "EquivalenceWitness",
    "EventizedFormula",
    "ExtrapolationLimitExceeded",
    "RuleRunnerEquivalenceResult",
    "RuleRunnerMonitor",
    "StructuredRuleRunnerMonitor",
    "UnsafeRuleRunnerSkeleton",
    "bounded_horizon",
    "certify_rule_runner",
    "derive_event_cell",
    "derive_event_trace",
    "evaluate_bounded",
    "eventize_bounded_islands",
]
