"""Public monitor API with lazy imports.

Keeping this package initializer lightweight is measurement-critical: importing
one symbolic/parser submodule in a fresh E2 worker must not allocate every Torch
backend before the requested monitor begins compilation.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Monitor": ("src.monitors.base", "Monitor"),
    "Verdict": ("src.monitors.base", "Verdict"),
    "SymbolicDFAMonitor": ("src.monitors.symbolic_dfa", "SymbolicDFAMonitor"),
    "RuleRunnerMonitor": ("src.monitors.rulerunner", "RuleRunnerMonitor"),
    "StructuredRuleRunnerMonitor": (
        "src.monitors.rulerunner",
        "StructuredRuleRunnerMonitor",
    ),
    "BoundedEventRuleRunnerMonitor": (
        "src.monitors.rulerunner",
        "BoundedEventRuleRunnerMonitor",
    ),
    "BoundedEventStructuredRuleRunnerMonitor": (
        "src.monitors.rulerunner",
        "BoundedEventStructuredRuleRunnerMonitor",
    ),
    "ProgressionRuleRunnerMonitor": (
        "src.monitors.progression",
        "ProgressionRuleRunnerMonitor",
    ),
    "ProgressionRuleRunnerStructuredMonitor": (
        "src.monitors.progression",
        "ProgressionRuleRunnerStructuredMonitor",
    ),
    "DeepDFAArtifactStats": ("src.monitors.deep_dfa", "DeepDFAArtifactStats"),
    "DeepDFAMonitor": ("src.monitors.deep_dfa", "DeepDFAMonitor"),
    "DeepDFAMonitorDense": ("src.monitors.deep_dfa", "DeepDFAMonitorDense"),
    "DeepDFAMonitorFactored": (
        "src.monitors.deep_dfa",
        "DeepDFAMonitorFactored",
    ),
    "DeepDFAMonitorScan": ("src.monitors.deep_dfa", "DeepDFAMonitorScan"),
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
