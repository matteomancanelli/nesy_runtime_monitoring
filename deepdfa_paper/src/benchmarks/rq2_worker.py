"""Fresh-process worker for one Phase E3 compilation and runtime block."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import warnings
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _current_rss_bytes() -> int:
    try:
        rows = Path("/proc/self/status").read_text().splitlines()
        value = next(row for row in rows if row.startswith("VmRSS:"))
        return int(value.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return 0


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    startup_rss = _current_rss_bytes()
    payload = json.loads(Path(sys.argv[1]).read_text())

    import numpy as np
    import torch

    from src.benchmarks.e2 import (
        MONITOR_SPEC_BY_ID,
        E2CellRecord,
        StageRecorder,
        WorkerRequest,
        _artifact_stats,
        _compile_monitor,
        _record_base,
        _set_worker_phase,
        _worker_provenance,
    )
    from src.benchmarks.rq2 import (
        RQ2_SCHEMA_VERSION,
        RQ2RuntimeRecord,
        RQ2WorkerRequest,
        encoding_name,
    )
    from src.benchmarks.schema import RunStatus

    compile_payload = payload["compile_request"]
    allowed = {field.name for field in fields(WorkerRequest)}
    compile_request = WorkerRequest(
        **{key: value for key, value in compile_payload.items() if key in allowed}
    )
    request = RQ2WorkerRequest(
        compile_request=compile_request,
        stratum=payload["stratum"],
        trace_seed=int(payload["trace_seed"]),
        trace_length=int(payload["trace_length"]),
        batch_sizes=tuple(int(value) for value in payload["batch_sizes"]),
        warmup_iterations=int(payload["warmup_iterations"]),
        minimum_timed_s=float(payload["minimum_timed_s"]),
        maximum_inner_iterations=int(payload["maximum_inner_iterations"]),
    )
    spec = MONITOR_SPEC_BY_ID[compile_request.monitor_id]
    base = _record_base(compile_request, spec)
    base["worker_pid"] = os.getpid()
    baseline_rss = _current_rss_bytes()
    stages = StageRecorder(compile_request.requested_device, torch)
    baseline_gpu = None
    if compile_request.requested_device == "cuda":
        if not torch.cuda.is_available():
            compilation = E2CellRecord(
                **base,
                status=RunStatus.UNSUPPORTED,
                failure_stage="device_validation",
                failure_reason="CUDA was requested but is unavailable",
                startup_host_rss_bytes=startup_rss,
                baseline_host_rss_bytes=baseline_rss,
                peak_compile_host_rss_bytes=_peak_rss_bytes(),
                provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
            )
            print(json.dumps({"compilation": compilation.flat_dict(), "runtime": []}))
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_gpu = torch.cuda.memory_allocated()

    compile_start = time.perf_counter()
    try:
        _set_worker_phase(compile_request, "compile")
        monitor = _compile_monitor(spec, compile_request, stages)
        if monitor.effective_device == "cuda":
            torch.cuda.synchronize()
        compile_total = time.perf_counter() - compile_start
        peak_compile_rss = _peak_rss_bytes()
        peak_gpu_allocated = (
            torch.cuda.max_memory_allocated()
            if monitor.effective_device == "cuda"
            else None
        )
        peak_gpu_reserved = (
            torch.cuda.max_memory_reserved()
            if monitor.effective_device == "cuda"
            else None
        )
        _set_worker_phase(compile_request, "post_compile")
        artifact = _artifact_stats(monitor, spec)
        compilation = E2CellRecord(
            **{**base, "effective_device": monitor.effective_device},
            status=RunStatus.SUCCESS,
            compile_total_s=compile_total,
            compile_stages_json=json.dumps(stages.durations, sort_keys=True),
            artifact_stats_json=json.dumps(artifact, sort_keys=True),
            startup_host_rss_bytes=startup_rss,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=peak_compile_rss,
            baseline_gpu_allocated_bytes=baseline_gpu,
            peak_compile_gpu_allocated_bytes=peak_gpu_allocated,
            peak_compile_gpu_reserved_bytes=peak_gpu_reserved,
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )
    except Exception as exc:
        _set_worker_phase(compile_request, "post_compile")
        oom_types = (MemoryError,)
        if hasattr(torch.cuda, "OutOfMemoryError"):
            oom_types = oom_types + (torch.cuda.OutOfMemoryError,)
        status = RunStatus.OOM if isinstance(exc, oom_types) else RunStatus.ERROR
        if type(exc).__name__ in {
            "MissingCertificate",
            "UnsafeRuleRunnerSkeleton",
            "ExtrapolationLimitExceeded",
            "MonaFailure",
        }:
            status = RunStatus.UNSUPPORTED
        compilation = E2CellRecord(
            **base,
            status=status,
            failure_stage=stages.failed_stage or stages.active_stage or "compilation",
            failure_reason=f"{type(exc).__name__}: {exc}",
            compile_total_s=time.perf_counter() - compile_start,
            compile_stages_json=json.dumps(stages.durations, sort_keys=True),
            startup_host_rss_bytes=startup_rss,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=_peak_rss_bytes(),
            baseline_gpu_allocated_bytes=baseline_gpu,
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )
        print(json.dumps({"compilation": compilation.flat_dict(), "runtime": []}))
        return

    rng = np.random.default_rng(request.trace_seed)
    maximum_batch = max(request.batch_sizes)
    bits = rng.integers(
        0,
        2,
        size=(maximum_batch, request.trace_length, len(compile_request.atoms)),
        dtype=np.int8,
    )
    all_traces = [
        [
            {
                atom: bool(bits[trace_index, cell_index, atom_index])
                for atom_index, atom in enumerate(compile_request.atoms)
            }
            for cell_index in range(request.trace_length)
        ]
        for trace_index in range(maximum_batch)
    ]
    encoding = encoding_name(spec.monitor_id)
    label_mode = (
        "exact_online"
        if spec.construction == "bounded_exact_online"
        else "sound_default"
        if spec.construction == "bounded_default"
        else "exact"
    )
    runtime_rows: list[RQ2RuntimeRecord] = []
    _set_worker_phase(compile_request, "runtime")
    for batch_size in request.batch_sizes:
        traces = all_traces[:batch_size]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(request.warmup_iterations):
                    monitor.batch_run(traces, early_termination=False)
                if monitor.effective_device == "cuda":
                    torch.cuda.synchronize()

                inner = 1
                while True:
                    start = time.perf_counter()
                    for _ in range(inner):
                        monitor.batch_run(traces, early_termination=False)
                    if monitor.effective_device == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if (
                        elapsed >= request.minimum_timed_s
                        or inner >= request.maximum_inner_iterations
                    ):
                        break
                    multiplier = max(
                        2, int(request.minimum_timed_s / max(elapsed, 1e-9))
                    )
                    inner = min(request.maximum_inner_iterations, inner * multiplier)
            messages = [str(item.message) for item in caught]
            status = (
                RunStatus.FALLBACK
                if any("falling back" in message.lower() for message in messages)
                else RunStatus.SUCCESS
            )
            duration_batch = elapsed / inner
            runtime_rows.append(
                RQ2RuntimeRecord(
                    schema_version=RQ2_SCHEMA_VERSION,
                    run_id=compile_request.run_id,
                    cell_id=f"{compile_request.cell_id}::b{batch_size}",
                    formula_id=compile_request.formula_id,
                    stratum=request.stratum,
                    monitor_id=spec.monitor_id,
                    monitor_name=spec.monitor_name,
                    construction=spec.construction,
                    encoding=encoding,
                    online_label_mode=label_mode,
                    requested_device=compile_request.requested_device,
                    effective_device=monitor.effective_device,
                    trace_seed=request.trace_seed,
                    repetition=compile_request.repetition,
                    trace_length=request.trace_length,
                    batch_size=batch_size,
                    early_termination=False,
                    warmup_iterations=request.warmup_iterations,
                    inner_iterations=inner,
                    status=status,
                    duration_s_per_batch=duration_batch,
                    duration_s_per_trace=duration_batch / batch_size,
                    duration_s_per_cell=duration_batch
                    / (batch_size * request.trace_length),
                    throughput_traces_s=batch_size / duration_batch,
                    warnings_json=json.dumps(messages),
                )
            )
        except Exception as exc:
            status = RunStatus.OOM if isinstance(exc, MemoryError) else RunStatus.ERROR
            runtime_rows.append(
                RQ2RuntimeRecord(
                    schema_version=RQ2_SCHEMA_VERSION,
                    run_id=compile_request.run_id,
                    cell_id=f"{compile_request.cell_id}::b{batch_size}",
                    formula_id=compile_request.formula_id,
                    stratum=request.stratum,
                    monitor_id=spec.monitor_id,
                    monitor_name=spec.monitor_name,
                    construction=spec.construction,
                    encoding=encoding,
                    online_label_mode=label_mode,
                    requested_device=compile_request.requested_device,
                    effective_device=monitor.effective_device,
                    trace_seed=request.trace_seed,
                    repetition=compile_request.repetition,
                    trace_length=request.trace_length,
                    batch_size=batch_size,
                    early_termination=False,
                    warmup_iterations=request.warmup_iterations,
                    inner_iterations=0,
                    status=status,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    _set_worker_phase(compile_request, "done")
    print(
        json.dumps(
            {
                "compilation": compilation.flat_dict(),
                "runtime": [row.flat_dict() for row in runtime_rows],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
