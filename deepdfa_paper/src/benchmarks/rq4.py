"""Versioned Phase E5 cross-architecture efficiency protocol.

RQ4 keeps semantic workload, early-termination policy, timing boundary, and
requested device as explicit independent dimensions.  Each cell is compiled
and timed in a fresh subprocess behind the frozen RQ1 applicability gate.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from src.benchmarks.e2 import (
    MONITOR_SPEC_BY_ID,
    ROOT,
    E2CellRecord,
    WorkerRequest,
    _failure_record,
    _host_memory_bytes,
    _rq1_rows,
    supervise_command,
)
from src.benchmarks.rq1 import RQ1_CORPUS
from src.benchmarks.schema import DEFAULT_RESOURCE_BUDGETS, ResourceBudgets, RunStatus

RQ4_SCHEMA_VERSION = "rq4.v1"

RQ4_MONITOR_IDS: tuple[str, ...] = (
    "symbolic_guarded",
    "rulerunner_original_flat",
    "rulerunner_bounded_flat_default",
    "rulerunner_bounded_flat_exact",
    "rulerunner_progression_flat",
    "deepdfa_dense",
    "deepdfa_factored",
)


@dataclass(frozen=True)
class RQ4Case:
    formula_id: str
    deployment_profile: str
    rationale: str


RQ4_CASES: tuple[RQ4Case, ...] = (
    RQ4Case(
        "atomic_control",
        "early_late_boundary",
        "Minimal early-decision control and fixed-overhead floor.",
    ),
    RQ4Case(
        "nested_next_control",
        "fixed_horizon_balanced",
        "Finite-horizon control with decisions after a fixed three-cell prefix.",
    ),
    RQ4Case(
        "ijcnn_balanced_n4",
        "early_late_boundary",
        "Paper-faithful RuleRunner formula shared by every headline architecture.",
    ),
)


@dataclass(frozen=True)
class RQ4Mode:
    mode_id: str
    workload_mode: str
    timing_boundary: str
    early_termination: bool
    workloads: tuple[tuple[int, int], ...]
    boundary_detail: str


RQ4_MODES: tuple[RQ4Mode, ...] = (
    RQ4Mode(
        "capacity_end_to_end",
        "capacity",
        "end_to_end",
        False,
        ((64, 1), (64, 32), (64, 128)),
        "numpy boolean cube -> Observation mappings -> public monitor verdicts",
    ),
    RQ4Mode(
        "capacity_core_native",
        "capacity",
        "core_native",
        False,
        ((64, 1), (64, 32), (64, 128)),
        (
            "predecoded native Observation mappings -> public monitor API; "
            "backend-owned tensorization and device transfer remain included"
        ),
    ),
    RQ4Mode(
        "deployment_end_to_end",
        "deployment",
        "end_to_end",
        True,
        ((64, 1), (64, 64)),
        "stratified boolean cube -> Observation mappings -> public monitor verdicts",
    ),
)


@dataclass(frozen=True)
class RQ4WorkerRequest:
    compile_request: WorkerRequest
    workload_mode: str
    timing_boundary: str
    boundary_detail: str
    early_termination: bool
    deployment_profile: str
    formula_family: str
    trace_seed: int
    workloads: tuple[tuple[int, int], ...]
    warmup_iterations: int
    measurement_blocks: int
    minimum_timed_s: float
    maximum_inner_iterations: int
    torch_threads: int
    torch_interop_threads: int


@dataclass(frozen=True)
class RQ4RuntimeRecord:
    schema_version: str
    run_id: str
    cell_id: str
    formula_id: str
    formula_family: str
    monitor_id: str
    monitor_name: str
    architecture: str
    construction: str
    backend: str
    requested_device: str
    effective_device: str
    workload_mode: str
    timing_boundary: str
    boundary_detail: str
    trace_profile: str
    trace_seed: int
    repetition: int
    trace_length: int
    batch_size: int
    early_termination: bool
    warmup_iterations: int
    measurement_blocks: int
    inner_iterations: int
    status: RunStatus
    median_duration_s_per_batch: float | None = None
    p95_duration_s_per_batch: float | None = None
    median_duration_s_per_trace: float | None = None
    median_duration_s_per_cell: float | None = None
    throughput_traces_s: float | None = None
    throughput_cells_s: float | None = None
    offered_traces: int = 0
    offered_cells: int = 0
    processed_traces: int | None = None
    processed_cells: int | None = None
    processed_cell_fraction: float | None = None
    semantic_decision_indices_json: str = "[]"
    strata_counts_json: str = "{}"
    timing_samples_json: str = "[]"
    persistent_tensor_bytes: int | None = None
    persistent_host_rss_delta_bytes: int | None = None
    peak_runtime_host_rss_bytes: int | None = None
    peak_runtime_gpu_allocated_bytes: int | None = None
    peak_runtime_gpu_reserved_bytes: int | None = None
    warnings_json: str = "[]"
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != RQ4_SCHEMA_VERSION:
            raise ValueError(f"expected schema {RQ4_SCHEMA_VERSION!r}")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.status in (RunStatus.SUCCESS, RunStatus.FALLBACK):
            if self.median_duration_s_per_batch is None:
                raise ValueError("successful runtime row requires a duration")
        for value in (
            self.semantic_decision_indices_json,
            self.strata_counts_json,
            self.timing_samples_json,
            self.warnings_json,
        ):
            json.loads(value)

    def flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RQ4RuntimeRecord":
        values = dict(payload)
        values["status"] = RunStatus(values["status"])
        return cls(**values)


def case_by_id(formula_id: str) -> RQ4Case:
    return next(case for case in RQ4_CASES if case.formula_id == formula_id)


def mode_by_id(mode_id: str) -> RQ4Mode:
    return next(mode for mode in RQ4_MODES if mode.mode_id == mode_id)


def _benchmark(formula_id: str):
    return next(
        case.benchmark for case in RQ1_CORPUS if case.benchmark.formula_id == formula_id
    )


def _rq1_gate(formula_id: str, monitor_id: str) -> str | None:
    spec = MONITOR_SPEC_BY_ID[monitor_id]
    if spec.rq1_construction is None:
        return None
    row = _rq1_rows().get((formula_id, spec.rq1_construction))
    if row is None or not row.get("applicable", False):
        return (
            f"excluded by frozen rq1.v1 gate: {formula_id}/"
            f"{spec.rq1_construction} is not applicable"
        )
    return None


def _failed_runtime_rows(
    request: RQ4WorkerRequest,
    status: RunStatus,
    reason: str,
) -> list[RQ4RuntimeRecord]:
    spec = MONITOR_SPEC_BY_ID[request.compile_request.monitor_id]
    return [
        RQ4RuntimeRecord(
            schema_version=RQ4_SCHEMA_VERSION,
            run_id=request.compile_request.run_id,
            cell_id=(
                f"{request.compile_request.cell_id}::l{trace_length}::b{batch_size}"
            ),
            formula_id=request.compile_request.formula_id,
            formula_family=request.formula_family,
            monitor_id=spec.monitor_id,
            monitor_name=spec.monitor_name,
            architecture=spec.architecture,
            construction=spec.construction,
            backend=spec.backend,
            requested_device=request.compile_request.requested_device,
            effective_device="",
            workload_mode=request.workload_mode,
            timing_boundary=request.timing_boundary,
            boundary_detail=request.boundary_detail,
            trace_profile=(
                request.deployment_profile
                if request.workload_mode == "deployment"
                else "uniform_bernoulli_p0.5"
            ),
            trace_seed=request.trace_seed,
            repetition=request.compile_request.repetition,
            trace_length=trace_length,
            batch_size=batch_size,
            early_termination=request.early_termination,
            warmup_iterations=request.warmup_iterations,
            measurement_blocks=request.measurement_blocks,
            inner_iterations=0,
            status=status,
            offered_traces=batch_size,
            offered_cells=batch_size * trace_length,
            failure_reason=reason,
        )
        for trace_length, batch_size in request.workloads
    ]


def run_rq4_cell(
    formula_id: str,
    monitor_id: str,
    mode_id: str,
    *,
    trace_seed: int,
    repetition: int,
    requested_device: str = "cpu",
    warmup_iterations: int = 2,
    measurement_blocks: int = 7,
    minimum_timed_s: float = 0.03,
    maximum_inner_iterations: int = 4096,
    torch_threads: int = 10,
    torch_interop_threads: int = 10,
    budgets: ResourceBudgets = DEFAULT_RESOURCE_BUDGETS,
    run_id: str | None = None,
) -> tuple[E2CellRecord, list[RQ4RuntimeRecord]]:
    """Compile and execute one isolated RQ4 formula/monitor/mode cell."""
    if monitor_id not in RQ4_MONITOR_IDS:
        raise ValueError(f"{monitor_id!r} is not in the RQ4 headline matrix")
    case = case_by_id(formula_id)
    mode = mode_by_id(mode_id)
    benchmark = _benchmark(formula_id)
    run_id = run_id or str(uuid.uuid4())
    cell_id = f"{mode_id}::{formula_id}::{monitor_id}::r{repetition}"
    compile_request = WorkerRequest(
        run_id=run_id,
        cell_id=cell_id,
        formula_id=formula_id,
        formula=benchmark.formula,
        atoms=benchmark.atoms,
        monitor_id=monitor_id,
        requested_device=requested_device,
        repetition=repetition,
    )
    request = RQ4WorkerRequest(
        compile_request=compile_request,
        workload_mode=mode.workload_mode,
        timing_boundary=mode.timing_boundary,
        boundary_detail=mode.boundary_detail,
        early_termination=mode.early_termination,
        deployment_profile=case.deployment_profile,
        formula_family=benchmark.family,
        trace_seed=trace_seed,
        workloads=mode.workloads,
        warmup_iterations=warmup_iterations,
        measurement_blocks=measurement_blocks,
        minimum_timed_s=minimum_timed_s,
        maximum_inner_iterations=maximum_inner_iterations,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
    )
    spec = MONITOR_SPEC_BY_ID[monitor_id]
    gate_reason = _rq1_gate(formula_id, monitor_id)
    if gate_reason:
        compilation = _failure_record(
            compile_request, spec, RunStatus.UNSUPPORTED, gate_reason
        )
        return compilation, _failed_runtime_rows(
            request, RunStatus.UNSUPPORTED, gate_reason
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
                str(ROOT / "src/benchmarks/rq4_worker.py"),
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
        compilation = _failure_record(
            compile_request,
            spec,
            status,
            reason,
            peak_rss_bytes=outcome.peak_rss_bytes,
            memory_limit_bytes=memory_limit,
        )
        return compilation, _failed_runtime_rows(request, status, reason)

    payload = json.loads(outcome.stdout.strip().splitlines()[-1])
    compilation = E2CellRecord.from_dict(payload["compilation"])
    compilation = replace(
        compilation,
        peak_compile_host_rss_bytes=max(
            outcome.peak_compile_rss_bytes,
            compilation.peak_compile_host_rss_bytes or 0,
        ),
        peak_observed_host_rss_bytes=max(
            outcome.peak_rss_bytes,
            compilation.peak_compile_host_rss_bytes or 0,
        ),
        host_memory_limit_bytes=memory_limit,
    )
    runtime = [
        replace(
            RQ4RuntimeRecord.from_dict(row),
            peak_runtime_host_rss_bytes=(outcome.peak_runtime_rss_bytes or None),
        )
        for row in payload["runtime"]
    ]
    if not runtime and compilation.status not in (
        RunStatus.SUCCESS,
        RunStatus.FALLBACK,
    ):
        runtime = _failed_runtime_rows(
            request, compilation.status, compilation.failure_reason
        )
    return compilation, runtime
