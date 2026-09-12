"""Versioned schema and resource policy for the final evaluation.

The legacy timing harness writes one aggregate row per benchmark cell. Phase E0
defines the replacement contract here: formula structure is characterized once,
raw repetitions remain separate, failures are explicit outcomes, and provenance
is part of every record. Later phases can extend the schema only by incrementing
``RESULT_SCHEMA_VERSION`` and providing a migration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from src.benchmarks.formulas import BenchmarkFormula, ParameterValue
from src.monitors.rulerunner.parse_tree import Node, Op, parse

RESULT_SCHEMA_VERSION = "e0.v1"


class RunStatus(str, Enum):
    """Every attempted benchmark cell terminates in exactly one status."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    OOM = "oom"
    UNSUPPORTED = "unsupported"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass(frozen=True)
class ResourceBudgets:
    """Portable default resource policy for one attempted benchmark cell.

    Memory limits are fractions because the final CPU and GPU machines may have
    different capacities. A run manifest must resolve them to byte counts from
    the actual host before launching subprocesses.
    """

    cold_compile_timeout_s: float = 300.0
    execution_timeout_s: float = 300.0
    max_host_memory_fraction: float = 0.80
    max_gpu_memory_fraction: float = 0.80
    max_atoms_for_exact_certification: int = 12

    def __post_init__(self) -> None:
        if self.cold_compile_timeout_s <= 0 or self.execution_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        for name in ("max_host_memory_fraction", "max_gpu_memory_fraction"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.max_atoms_for_exact_certification < 1:
            raise ValueError("max_atoms_for_exact_certification must be positive")


DEFAULT_RESOURCE_BUDGETS = ResourceBudgets()


_TEMPORAL_OPS = frozenset(
    {
        Op.NEXT,
        Op.WEAK_NEXT,
        Op.EVENTUALLY,
        Op.ALWAYS,
        Op.UNTIL,
        Op.RELEASE,
    }
)


def _occurrence_count(node: Node) -> int:
    """Count syntax-tree occurrences, including repeated shared DAG nodes."""
    return 1 + sum(_occurrence_count(child) for child in node.children)


def _temporal_depth(node: Node) -> int:
    """Maximum number of temporal operators on a root-to-leaf path."""
    child_depth = max((_temporal_depth(child) for child in node.children), default=0)
    return child_depth + int(node.op in _TEMPORAL_OPS)


@dataclass(frozen=True)
class FormulaStructure:
    """Declared and compiler-derived structural identity of one formula."""

    schema_version: str
    formula_id: str
    formula: str
    family: str
    source: str
    parameters: tuple[tuple[str, ParameterValue], ...]
    tree_shape: str
    roles: tuple[str, ...]
    n_atoms: int
    ast_nodes: int
    distinct_subformulae: int
    ast_depth: int
    temporal_depth: int
    dfa_states: int | None = None
    dfa_transitions: int | None = None

    def flat_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly representation with JSON compound fields."""
        row = asdict(self)
        row["parameters"] = json.dumps(dict(self.parameters), sort_keys=True)
        row["roles"] = json.dumps(self.roles)
        return row


def characterize_formula(
    benchmark: BenchmarkFormula, *, include_dfa: bool = False
) -> FormulaStructure:
    """Compute stable structural statistics for ``benchmark``.

    The binary parse-tree statistics use the exact front end consumed by
    RuleRunner, including explicit association and syntactic sharing. DFA
    characterization is opt-in because it invokes MONA and belongs outside
    cheap registry import.
    """
    root = parse(benchmark.formula)
    dfa_states = benchmark.dfa_states
    dfa_transitions: int | None = None
    if include_dfa:
        from src.formula.compiler import compile_ltlf

        dfa = compile_ltlf(benchmark.formula)
        dfa_states = len(dfa.states)
        dfa_transitions = len(dfa.transitions)

    return FormulaStructure(
        schema_version=RESULT_SCHEMA_VERSION,
        formula_id=benchmark.formula_id,
        formula=benchmark.formula,
        family=benchmark.family,
        source=benchmark.source,
        parameters=benchmark.parameters,
        tree_shape=benchmark.tree_shape,
        roles=benchmark.roles,
        n_atoms=benchmark.n_atoms,
        ast_nodes=_occurrence_count(root),
        distinct_subformulae=len(tuple(root.subformulae())),
        ast_depth=root.depth,
        temporal_depth=_temporal_depth(root),
        dfa_states=dfa_states,
        dfa_transitions=dfa_transitions,
    )


@dataclass(frozen=True)
class RunProvenance:
    """Immutable environment identity shared by all records in one run."""

    schema_version: str
    run_id: str
    experiment_id: str
    started_at_utc: str
    git_commit: str
    worktree_fingerprint: str
    hostname: str
    operating_system: str
    python_version: str
    torch_version: str
    cuda_version: str
    mona_version: str
    ltlf2dfa_version: str
    cpu_model: str
    cpu_count: int
    host_memory_bytes: int
    gpu_name: str
    gpu_memory_bytes: int | None
    torch_threads: int


@dataclass(frozen=True)
class RawResultRecord:
    """One raw repetition, including failed and unsupported attempts."""

    schema_version: str
    run_id: str
    experiment_id: str
    formula_id: str
    monitor_name: str
    monitor_backend: str
    requested_device: str
    effective_device: str
    dtype: str
    trace_seed: int
    repetition: int
    trace_length: int
    batch_size: int
    early_termination: bool
    online_label_mode: str
    status: RunStatus
    duration_s: float | None = None
    compile_total_s: float | None = None
    compile_stages_json: str = "{}"
    offered_traces: int = 0
    offered_cells: int = 0
    processed_traces: int | None = None
    processed_cells: int | None = None
    persistent_bytes: int | None = None
    peak_host_rss_bytes: int | None = None
    peak_gpu_allocated_bytes: int | None = None
    semantic_decision_index: int | None = None
    reported_decision_index: int | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"record schema {self.schema_version!r} does not match "
                f"{RESULT_SCHEMA_VERSION!r}"
            )
        if self.trace_length < 0 or self.batch_size < 0 or self.repetition < 0:
            raise ValueError("trace_length, batch_size, and repetition must be >= 0")
        if self.offered_traces < 0 or self.offered_cells < 0:
            raise ValueError("offered_traces and offered_cells must be >= 0")
        for name in ("processed_traces", "processed_cells"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when present")
        if self.status is RunStatus.SUCCESS and self.duration_s is None:
            raise ValueError("successful runtime records require duration_s")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if self.compile_total_s is not None and self.compile_total_s < 0:
            raise ValueError("compile_total_s must be non-negative")
        try:
            stages = json.loads(self.compile_stages_json)
        except json.JSONDecodeError as exc:
            raise ValueError("compile_stages_json must be valid JSON") from exc
        if not isinstance(stages, dict):
            raise ValueError("compile_stages_json must encode an object")

    @property
    def decision_lag(self) -> int | None:
        if self.semantic_decision_index is None or self.reported_decision_index is None:
            return None
        return self.reported_decision_index - self.semantic_decision_index

    def flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        row["decision_lag"] = self.decision_lag
        return row
