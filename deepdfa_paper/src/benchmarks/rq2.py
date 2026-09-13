"""Phase E3 protocol for the RuleRunner cost-of-correctness study."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

from src.benchmarks.e2 import (
    MONITOR_SPEC_BY_ID,
    ROOT,
    E2CellRecord,
    WorkerRequest,
    _failure_record,
    _host_memory_bytes,
    supervise_command,
)
from src.benchmarks.schema import DEFAULT_RESOURCE_BUDGETS, ResourceBudgets, RunStatus

RQ2_SCHEMA_VERSION = "rq2.v1"


@dataclass(frozen=True)
class RQ2Case:
    formula_id: str
    stratum: str
    rationale: str


RQ2_CASES: tuple[RQ2Case, ...] = (
    RQ2Case(
        "nested_next_control",
        "all_applicable",
        "Safe finite-horizon control with a nonzero bounded pipeline.",
    ),
    RQ2Case(
        "ijcnn_balanced_n4",
        "all_applicable",
        "Paper-faithful breadth control with no extracted event horizon.",
    ),
    RQ2Case(
        "eventual_next_alias",
        "bounded_repair",
        "Overlapping-instance repair whose bounded default labels are exact.",
    ),
    RQ2Case(
        "next_offset_alias",
        "bounded_repair_late_default",
        "Two-offset collision with a sound but late bounded-default label.",
    ),
    RQ2Case(
        "globally_next_alias",
        "bounded_repair_late_default",
        "Repeated bounded obligation with a late bounded-default label.",
    ),
    RQ2Case(
        "until_next_alias",
        "bounded_repair_late_default",
        "Until repair with a late bounded-default label.",
    ),
    RQ2Case(
        "response",
        "progression_only",
        "Unbounded overlapping obligations rejected by the bounded repair.",
    ),
)

RQ2_MONITOR_IDS = (
    "rulerunner_original_flat",
    "rulerunner_original_structured",
    "rulerunner_bounded_flat_default",
    "rulerunner_bounded_flat_exact",
    "rulerunner_bounded_structured_default",
    "rulerunner_bounded_structured_exact",
    "rulerunner_progression_flat",
    "rulerunner_progression_structured",
)


@dataclass(frozen=True)
class RQ2WorkerRequest:
    compile_request: WorkerRequest
    stratum: str
    trace_seed: int
    trace_length: int
    batch_sizes: tuple[int, ...]
    warmup_iterations: int
    minimum_timed_s: float
    maximum_inner_iterations: int


@dataclass(frozen=True)
class RQ2RuntimeRecord:
    schema_version: str
    run_id: str
    cell_id: str
    formula_id: str
    stratum: str
    monitor_id: str
    monitor_name: str
    construction: str
    encoding: str
    online_label_mode: str
    requested_device: str
    effective_device: str
    trace_seed: int
    repetition: int
    trace_length: int
    batch_size: int
    early_termination: bool
    warmup_iterations: int
    inner_iterations: int
    status: RunStatus
    duration_s_per_batch: float | None = None
    duration_s_per_trace: float | None = None
    duration_s_per_cell: float | None = None
    throughput_traces_s: float | None = None
    failure_reason: str = ""
    warnings_json: str = "[]"

    def __post_init__(self) -> None:
        if self.schema_version != RQ2_SCHEMA_VERSION:
            raise ValueError(f"expected schema {RQ2_SCHEMA_VERSION!r}")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.status in (RunStatus.SUCCESS, RunStatus.FALLBACK):
            if self.duration_s_per_batch is None:
                raise ValueError("successful runtime row requires a duration")
        if self.batch_size <= 0 or self.trace_length <= 0:
            raise ValueError("runtime batch size and trace length must be positive")
        json.loads(self.warnings_json)

    def flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RQ2RuntimeRecord":
        values = dict(payload)
        values["status"] = RunStatus(values["status"])
        return cls(**values)


@dataclass(frozen=True)
class RQ2DecisionRecord:
    schema_version: str
    formula_id: str
    stratum: str
    online_label_mode: str
    trace_length: int
    trace_id: int
    trace_json: str
    semantic_decision_index: int
    reported_decision_index: int
    decision_lag_cells: int
    additional_monitor_compute_s: float
    final_verdict: str

    def flat_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decision_index(monitor: object, trace: list[dict[str, bool]]) -> tuple[int, str]:
    from src.monitors.base import Verdict

    monitor.reset()  # type: ignore[attr-defined]
    for index, observation in enumerate(trace):
        verdict = monitor.step(observation)  # type: ignore[attr-defined]
        if verdict is not Verdict.UNDECIDED:
            return index, verdict.name
    return len(trace), monitor.final_verdict().name  # type: ignore[attr-defined]


def _timed_reported_index(
    monitor: object,
    trace: list[dict[str, bool]],
    semantic_index: int,
) -> tuple[int, str, float]:
    from src.monitors.base import Verdict

    monitor.reset()  # type: ignore[attr-defined]
    elapsed = 0.0
    elapsed_at_semantic = 0.0
    for index, observation in enumerate(trace):
        start = time.perf_counter()
        verdict = monitor.step(observation)  # type: ignore[attr-defined]
        elapsed += time.perf_counter() - start
        if index == semantic_index:
            elapsed_at_semantic = elapsed
        if verdict is not Verdict.UNDECIDED:
            return index, verdict.name, max(0.0, elapsed - elapsed_at_semantic)
    start = time.perf_counter()
    verdict = monitor.final_verdict()  # type: ignore[attr-defined]
    elapsed += time.perf_counter() - start
    if semantic_index == len(trace):
        elapsed_at_semantic = elapsed
    return len(trace), verdict.name, max(0.0, elapsed - elapsed_at_semantic)


def characterize_decision_lag(
    formula_id: str, *, trace_length: int
) -> list[RQ2DecisionRecord]:
    """Exhaustively compare bounded labels with canonical permanent labels."""
    from src.benchmarks.rq1 import RQ1_CORPUS
    from src.monitors.rulerunner.bounded_cilp import BoundedEventRuleRunnerMonitor
    from src.monitors.symbolic_dfa import SymbolicDFAMonitor

    case = case_by_id(formula_id)
    benchmark = next(
        item.benchmark for item in RQ1_CORPUS if item.benchmark.formula_id == formula_id
    )
    canonical = SymbolicDFAMonitor.compile(benchmark.formula)
    bounded = {
        "sound_default": BoundedEventRuleRunnerMonitor.compile(
            benchmark.formula, certificate="cached", exact_online=False
        ),
        "exact_online": BoundedEventRuleRunnerMonitor.compile(
            benchmark.formula, certificate="cached", exact_online=True
        ),
    }
    valuations = tuple(product((False, True), repeat=len(benchmark.atoms)))
    records: list[RQ2DecisionRecord] = []
    for trace_id, symbols in enumerate(product(valuations, repeat=trace_length)):
        trace = [
            dict(zip(benchmark.atoms, valuation, strict=True)) for valuation in symbols
        ]
        semantic_index, final_verdict = _decision_index(canonical, trace)
        for label_mode, monitor in bounded.items():
            reported_index, reported_final, additional_compute = _timed_reported_index(
                monitor, trace, semantic_index
            )
            if reported_final != final_verdict:
                raise RuntimeError(
                    f"RQ2 final-verdict mismatch for {formula_id}:{label_mode}"
                )
            lag = reported_index - semantic_index
            if lag < 0:
                raise RuntimeError(
                    f"RQ2 prefix-soundness violation for {formula_id}:{label_mode}"
                )
            if label_mode == "exact_online" and lag:
                raise RuntimeError(f"exact-online head is late for {formula_id}")
            records.append(
                RQ2DecisionRecord(
                    schema_version=RQ2_SCHEMA_VERSION,
                    formula_id=formula_id,
                    stratum=case.stratum,
                    online_label_mode=label_mode,
                    trace_length=trace_length,
                    trace_id=trace_id,
                    trace_json=json.dumps(
                        [
                            [atom for atom, value in cell.items() if value]
                            for cell in trace
                        ]
                    ),
                    semantic_decision_index=semantic_index,
                    reported_decision_index=reported_index,
                    decision_lag_cells=lag,
                    additional_monitor_compute_s=additional_compute,
                    final_verdict=final_verdict,
                )
            )
    return records


def case_by_id(formula_id: str) -> RQ2Case:
    return next(case for case in RQ2_CASES if case.formula_id == formula_id)


def _rq1_gate() -> dict[tuple[str, str], bool]:
    artifact = ROOT / "results" / "rq1" / "rq1_characterization.json"
    rows = json.loads(artifact.read_text())["records"]
    return {
        (row["formula_id"], row["construction"]): bool(row["applicable"])
        for row in rows
    }


def construction_gate_name(monitor_id: str) -> str:
    construction = MONITOR_SPEC_BY_ID[monitor_id].construction
    if construction == "original":
        return "original"
    if construction == "bounded_default":
        return "bounded_default"
    if construction == "bounded_exact_online":
        return "bounded_exact_online"
    if construction == "progression":
        return "progression"
    raise ValueError(f"monitor {monitor_id!r} is not an RQ2 RuleRunner configuration")


def encoding_name(monitor_id: str) -> str:
    """Return the orthogonal flat/structured encoding axis."""
    return "structured" if "structured" in monitor_id else "flat"


def applicable_monitor_ids(formula_id: str) -> tuple[str, ...]:
    gate = _rq1_gate()
    return tuple(
        monitor_id
        for monitor_id in RQ2_MONITOR_IDS
        if gate[(formula_id, construction_gate_name(monitor_id))]
    )


def _runtime_failure_rows(
    request: RQ2WorkerRequest, status: RunStatus, reason: str
) -> list[RQ2RuntimeRecord]:
    spec = MONITOR_SPEC_BY_ID[request.compile_request.monitor_id]
    encoding = encoding_name(spec.monitor_id)
    label_mode = (
        "exact_online"
        if spec.construction == "bounded_exact_online"
        else "sound_default"
        if spec.construction == "bounded_default"
        else "exact"
    )
    return [
        RQ2RuntimeRecord(
            schema_version=RQ2_SCHEMA_VERSION,
            run_id=request.compile_request.run_id,
            cell_id=f"{request.compile_request.cell_id}::b{batch_size}",
            formula_id=request.compile_request.formula_id,
            stratum=request.stratum,
            monitor_id=spec.monitor_id,
            monitor_name=spec.monitor_name,
            construction=spec.construction,
            encoding=encoding,
            online_label_mode=label_mode,
            requested_device=request.compile_request.requested_device,
            effective_device="",
            trace_seed=request.trace_seed,
            repetition=request.compile_request.repetition,
            trace_length=request.trace_length,
            batch_size=batch_size,
            early_termination=False,
            warmup_iterations=request.warmup_iterations,
            inner_iterations=0,
            status=status,
            failure_reason=reason,
        )
        for batch_size in request.batch_sizes
    ]


def run_rq2_cell(
    formula_id: str,
    monitor_id: str,
    *,
    trace_seed: int,
    repetition: int,
    trace_length: int = 32,
    batch_sizes: tuple[int, ...] = (1, 64),
    requested_device: str = "cpu",
    warmup_iterations: int = 2,
    minimum_timed_s: float = 0.05,
    maximum_inner_iterations: int = 4096,
    budgets: ResourceBudgets = DEFAULT_RESOURCE_BUDGETS,
    run_id: str | None = None,
) -> tuple[E2CellRecord, list[RQ2RuntimeRecord]]:
    """Compile once in a fresh process and time paired runtime workloads."""
    from src.benchmarks.rq1 import RQ1_CORPUS

    if monitor_id not in RQ2_MONITOR_IDS:
        raise ValueError(f"monitor {monitor_id!r} is outside the RQ2 registry")
    case = case_by_id(formula_id)
    formula = next(
        item.benchmark for item in RQ1_CORPUS if item.benchmark.formula_id == formula_id
    )
    run_id = run_id or str(uuid.uuid4())
    cell_id = f"{formula_id}::{monitor_id}::{requested_device}::r{repetition}"
    compile_request = WorkerRequest(
        run_id=run_id,
        cell_id=cell_id,
        formula_id=formula_id,
        formula=formula.formula,
        atoms=formula.atoms,
        monitor_id=monitor_id,
        requested_device=requested_device,
        repetition=repetition,
    )
    request = RQ2WorkerRequest(
        compile_request=compile_request,
        stratum=case.stratum,
        trace_seed=trace_seed,
        trace_length=trace_length,
        batch_sizes=batch_sizes,
        warmup_iterations=warmup_iterations,
        minimum_timed_s=minimum_timed_s,
        maximum_inner_iterations=maximum_inner_iterations,
    )
    spec = MONITOR_SPEC_BY_ID[monitor_id]
    if monitor_id not in applicable_monitor_ids(formula_id):
        reason = "rejected by frozen rq1.v1 semantic applicability gate"
        compile_row = _failure_record(
            compile_request, spec, RunStatus.UNSUPPORTED, reason
        )
        return compile_row, _runtime_failure_rows(
            request, RunStatus.UNSUPPORTED, reason
        )

    host_memory = _host_memory_bytes()
    memory_limit = (
        int(host_memory * budgets.max_host_memory_fraction) if host_memory else None
    )
    with tempfile.NamedTemporaryFile("w", suffix=".phase", delete=False) as phase:
        phase_path = Path(phase.name)
    compile_request = replace(compile_request, phase_path=str(phase_path))
    request = replace(request, compile_request=compile_request)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        request_path = Path(handle.name)
        json.dump(asdict(request), handle)
    try:
        outcome = supervise_command(
            [
                sys.executable,
                str(ROOT / "src/benchmarks/rq2_worker.py"),
                str(request_path),
            ],
            timeout_s=budgets.cold_compile_timeout_s + budgets.execution_timeout_s,
            memory_limit_bytes=memory_limit,
            phase_path=phase_path,
        )
    finally:
        request_path.unlink(missing_ok=True)
        phase_path.unlink(missing_ok=True)

    if outcome.terminal_status is not None or outcome.returncode != 0:
        status = outcome.terminal_status or RunStatus.ERROR
        reason = outcome.failure_reason or (
            f"worker exited {outcome.returncode}: {outcome.stderr.strip()}"
        )
        compile_row = _failure_record(
            compile_request,
            spec,
            status,
            reason,
            peak_rss_bytes=outcome.peak_rss_bytes,
            memory_limit_bytes=memory_limit,
        )
        return compile_row, _runtime_failure_rows(request, status, reason)
    try:
        payload = json.loads(outcome.stdout.strip().splitlines()[-1])
        compile_row = E2CellRecord.from_dict(payload["compilation"])
        runtime_rows = [RQ2RuntimeRecord.from_dict(row) for row in payload["runtime"]]
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        reason = f"invalid RQ2 worker output: {exc}; stderr={outcome.stderr.strip()!r}"
        compile_row = _failure_record(
            compile_request,
            spec,
            RunStatus.ERROR,
            reason,
            memory_limit_bytes=memory_limit,
        )
        return compile_row, _runtime_failure_rows(request, RunStatus.ERROR, reason)
    compile_row = replace(
        compile_row,
        peak_compile_host_rss_bytes=max(
            outcome.peak_compile_rss_bytes,
            compile_row.peak_compile_host_rss_bytes or 0,
        ),
        peak_observed_host_rss_bytes=max(
            outcome.peak_rss_bytes,
            compile_row.peak_compile_host_rss_bytes or 0,
        ),
        host_memory_limit_bytes=memory_limit,
    )
    return compile_row, runtime_rows


def run_certificate_generation(
    formula_id: str,
    *,
    repetition: int,
    run_id: str,
    budgets: ResourceBudgets = DEFAULT_RESOURCE_BUDGETS,
) -> dict[str, Any]:
    """Measure offline exact certification separately from cached compilation."""
    from src.benchmarks.rq1 import RQ1_CORPUS

    benchmark = next(
        item.benchmark for item in RQ1_CORPUS if item.benchmark.formula_id == formula_id
    )
    host_memory = _host_memory_bytes()
    memory_limit = (
        int(host_memory * budgets.max_host_memory_fraction) if host_memory else None
    )
    with tempfile.NamedTemporaryFile("w", suffix=".phase", delete=False) as phase:
        phase_path = Path(phase.name)
    request = {
        "run_id": run_id,
        "formula_id": formula_id,
        "formula": benchmark.formula,
        "repetition": repetition,
        "phase_path": str(phase_path),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        request_path = Path(handle.name)
        json.dump(request, handle)
    try:
        outcome = supervise_command(
            [
                sys.executable,
                str(ROOT / "src/benchmarks/rq2_certificate_worker.py"),
                str(request_path),
            ],
            timeout_s=budgets.cold_compile_timeout_s,
            memory_limit_bytes=memory_limit,
            phase_path=phase_path,
        )
    finally:
        request_path.unlink(missing_ok=True)
        phase_path.unlink(missing_ok=True)
    if outcome.terminal_status is not None or outcome.returncode != 0:
        return {
            "schema_version": RQ2_SCHEMA_VERSION,
            "run_id": run_id,
            "formula_id": formula_id,
            "repetition": repetition,
            "status": (outcome.terminal_status or RunStatus.ERROR).value,
            "failure_reason": outcome.failure_reason
            or f"worker exited {outcome.returncode}: {outcome.stderr.strip()}",
            "peak_compile_host_rss_bytes": outcome.peak_compile_rss_bytes,
            "host_memory_limit_bytes": memory_limit,
        }
    row = json.loads(outcome.stdout.strip().splitlines()[-1])
    row["peak_compile_host_rss_bytes"] = max(
        outcome.peak_compile_rss_bytes,
        row["worker_peak_host_rss_bytes"],
    )
    row["host_memory_limit_bytes"] = memory_limit
    row["failure_reason"] = ""
    return row
