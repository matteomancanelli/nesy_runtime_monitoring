"""RuleRunner-style LTLf monitors (Paradigm 2), exposed lazily.

The lightweight initializer lets parser/rule-only code remain independent of
Torch allocation. Public imports keep their historical names through
``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BoundedEventPipeline": ("src.monitors.rulerunner.bounded", "BoundedEventPipeline"),
    "BoundedIsland": ("src.monitors.rulerunner.bounded", "BoundedIsland"),
    "EventizedFormula": ("src.monitors.rulerunner.bounded", "EventizedFormula"),
    "bounded_horizon": ("src.monitors.rulerunner.bounded", "bounded_horizon"),
    "derive_event_cell": ("src.monitors.rulerunner.bounded", "derive_event_cell"),
    "derive_event_trace": ("src.monitors.rulerunner.bounded", "derive_event_trace"),
    "evaluate_bounded": ("src.monitors.rulerunner.bounded", "evaluate_bounded"),
    "eventize_bounded_islands": (
        "src.monitors.rulerunner.bounded",
        "eventize_bounded_islands",
    ),
    "BoundedEventRuleRunnerMonitor": (
        "src.monitors.rulerunner.bounded_cilp",
        "BoundedEventRuleRunnerMonitor",
    ),
    "BoundedEventStructuredRuleRunnerMonitor": (
        "src.monitors.rulerunner.bounded_cilp",
        "BoundedEventStructuredRuleRunnerMonitor",
    ),
    "UnsafeRuleRunnerSkeleton": (
        "src.monitors.rulerunner.bounded_cilp",
        "UnsafeRuleRunnerSkeleton",
    ),
    "BoundedExtrapolationCILP": (
        "src.monitors.rulerunner.bounded_extrapolation",
        "BoundedExtrapolationCILP",
    ),
    "ExtrapolationLimitExceeded": (
        "src.monitors.rulerunner.bounded_extrapolation",
        "ExtrapolationLimitExceeded",
    ),
    "EquivalenceWitness": (
        "src.monitors.rulerunner.equivalence",
        "EquivalenceWitness",
    ),
    "RuleRunnerEquivalenceResult": (
        "src.monitors.rulerunner.equivalence",
        "RuleRunnerEquivalenceResult",
    ),
    "certify_rule_runner": (
        "src.monitors.rulerunner.equivalence",
        "certify_rule_runner",
    ),
    "RuleRunnerMonitor": ("src.monitors.rulerunner.monitor", "RuleRunnerMonitor"),
    "StructuredRuleRunnerMonitor": (
        "src.monitors.rulerunner.structured",
        "StructuredRuleRunnerMonitor",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
