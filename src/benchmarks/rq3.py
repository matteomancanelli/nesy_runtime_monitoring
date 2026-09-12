"""Versioned Phase E4 structural-scaling corpus and isolated executor."""

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
    supervise_command,
)
from src.benchmarks.formulas import (
    GUARD_COMPLEXITY_SUITE,
    BenchmarkFormula,
    bounded_response,
    ijcnn_formula,
    kth_from_last,
)
from src.benchmarks.schema import DEFAULT_RESOURCE_BUDGETS, ResourceBudgets, RunStatus

RQ3_SCHEMA_VERSION = "rq3.v1"


@dataclass(frozen=True)
class RQ3Case:
    panel: str
    benchmark: BenchmarkFormula
    monitor_ids: tuple[str, ...]
    workloads: tuple[tuple[int, int], ...]  # (trace_length, batch_size)


def rq3_cases() -> tuple[RQ3Case, ...]:
    cases: list[RQ3Case] = []
    for n in (4, 8, 16, 32):
        for shape in ("balanced", "left_deep"):
            cases.append(
                RQ3Case(
                    "tree_shape",
                    ijcnn_formula(n, tree_shape=shape),
                    ("rulerunner_original_flat", "rulerunner_original_structured"),
                    ((32, 1), (32, 64)),
                )
            )
    for benchmark in GUARD_COMPLEXITY_SUITE:
        cases.append(
            RQ3Case(
                "guard_complexity",
                benchmark,
                ("deepdfa_dense", "deepdfa_factored"),
                ((32, 64),),
            )
        )
    scan_workloads = tuple((length, batch) for length in (32, 128) for batch in (1, 32))
    for deadline in (2, 6, 10):
        cases.append(
            RQ3Case(
                "scan_phase",
                bounded_response(deadline),
                ("deepdfa_dense", "deepdfa_scan"),
                scan_workloads,
            )
        )
    for deadline in (2, 6, 10, 14):
        cases.append(
            RQ3Case(
                "linear_state",
                bounded_response(deadline),
                ("symbolic_guarded", "deepdfa_dense", "deepdfa_factored"),
                ((32, 32),),
            )
        )
    for depth in (2, 4, 6, 8, 10):
        cases.append(
            RQ3Case(
                "exponential_state",
                kth_from_last(depth),
                ("symbolic_guarded", "deepdfa_dense", "deepdfa_factored"),
                ((16, 8),),
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class RQ3WorkerRequest:
    compile_request: WorkerRequest
    panel: str
    formula_family: str
    parameters_json: str
    tree_shape: str
    trace_seed: int
    workloads: tuple[tuple[int, int], ...]
    warmup_iterations: int
    minimum_timed_s: float
    maximum_inner_iterations: int


@dataclass(frozen=True)
class RQ3RuntimeRecord:
    schema_version: str
    run_id: str
    cell_id: str
    panel: str
    formula_id: str
    formula_family: str
    parameters_json: str
    tree_shape: str
    monitor_id: str
    monitor_name: str
    backend: str
    requested_device: str
    effective_device: str
    trace_seed: int
    repetition: int
    trace_length: int
    batch_size: int
    warmup_iterations: int
    inner_iterations: int
    status: RunStatus
    duration_s_per_batch: float | None = None
    duration_s_per_trace: float | None = None
    duration_s_per_cell: float | None = None
    throughput_traces_s: float | None = None
    estimated_scan_stack_bytes: int | None = None
    warnings_json: str = "[]"
    failure_reason: str = ""

    def flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RQ3RuntimeRecord":
        values = dict(payload)
        values["status"] = RunStatus(values["status"])
        return cls(**values)


def run_rq3_cell(
    case: RQ3Case,
    monitor_id: str,
    *,
    trace_seed: int,
    repetition: int,
    requested_device: str = "cpu",
    warmup_iterations: int = 2,
    minimum_timed_s: float = 0.05,
    maximum_inner_iterations: int = 4096,
    budgets: ResourceBudgets = DEFAULT_RESOURCE_BUDGETS,
    run_id: str | None = None,
) -> tuple[E2CellRecord, list[RQ3RuntimeRecord]]:
    if monitor_id not in case.monitor_ids:
        raise ValueError(f"{monitor_id!r} is not configured for {case.panel}")
    benchmark = case.benchmark
    run_id = run_id or str(uuid.uuid4())
    cell_id = f"{case.panel}::{benchmark.formula_id}::{monitor_id}::r{repetition}"
    compile_request = WorkerRequest(
        run_id=run_id,
        cell_id=cell_id,
        formula_id=benchmark.formula_id,
        formula=benchmark.formula,
        atoms=benchmark.atoms,
        monitor_id=monitor_id,
        requested_device=requested_device,
        repetition=repetition,
    )
    request = RQ3WorkerRequest(
        compile_request=compile_request,
        panel=case.panel,
        formula_family=benchmark.family,
        parameters_json=json.dumps(dict(benchmark.parameters), sort_keys=True),
        tree_shape=benchmark.tree_shape,
        trace_seed=trace_seed,
        workloads=case.workloads,
        warmup_iterations=warmup_iterations,
        minimum_timed_s=minimum_timed_s,
        maximum_inner_iterations=maximum_inner_iterations,
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
    spec = MONITOR_SPEC_BY_ID[monitor_id]
    try:
        outcome = supervise_command(
            [
                sys.executable,
                str(ROOT / "src/benchmarks/rq3_worker.py"),
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
        return (
            _failure_record(
                compile_request,
                spec,
                status,
                reason,
                peak_rss_bytes=outcome.peak_rss_bytes,
                memory_limit_bytes=memory_limit,
            ),
            [],
        )
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
    runtime = [RQ3RuntimeRecord.from_dict(row) for row in payload["runtime"]]
    return compilation, runtime
