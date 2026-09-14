"""Fresh-process worker for one Phase E4 structural-scaling cell."""

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


def _rss() -> int:
    try:
        row = next(
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
        return int(row.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return 0


def _peak() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    startup_rss = _rss()
    payload = json.loads(Path(sys.argv[1]).read_text())
    compile_payload = payload["compile_request"]

    import numpy as np

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
    from src.benchmarks.rq3 import RQ3_SCHEMA_VERSION, RQ3RuntimeRecord
    from src.benchmarks.schema import RunStatus

    allowed = {field.name for field in fields(WorkerRequest)}
    request = WorkerRequest(
        **{key: value for key, value in compile_payload.items() if key in allowed}
    )
    spec = MONITOR_SPEC_BY_ID[request.monitor_id]
    torch = None
    if spec.uses_requested_device:
        import torch as torch_module

        torch = torch_module
    base = _record_base(request, spec)
    base["worker_pid"] = os.getpid()
    baseline_rss = _rss()
    stages = StageRecorder(request.requested_device, torch)
    if request.requested_device == "cuda" and (
        torch is None or not torch.cuda.is_available()
    ):
        compilation = E2CellRecord(
            **base,
            status=RunStatus.UNSUPPORTED,
            failure_stage="device_validation",
            failure_reason="CUDA was requested but is unavailable",
            startup_host_rss_bytes=startup_rss,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=_peak(),
        )
        print(json.dumps({"compilation": compilation.flat_dict(), "runtime": []}))
        return
    baseline_gpu = None
    if torch is not None and request.requested_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_gpu = torch.cuda.memory_allocated()
    start = time.perf_counter()
    try:
        _set_worker_phase(request, "compile")
        monitor = _compile_monitor(spec, request, stages)
        if monitor.effective_device == "cuda":
            torch.cuda.synchronize()
        compile_total = time.perf_counter() - start
        peak_compile = _peak()
        gpu_allocated = (
            torch.cuda.max_memory_allocated()
            if torch is not None and monitor.effective_device == "cuda"
            else None
        )
        gpu_reserved = (
            torch.cuda.max_memory_reserved()
            if torch is not None and monitor.effective_device == "cuda"
            else None
        )
        _set_worker_phase(request, "post_compile")
        artifact = _artifact_stats(monitor, spec)
        compilation = E2CellRecord(
            **{**base, "effective_device": monitor.effective_device},
            status=RunStatus.SUCCESS,
            compile_total_s=compile_total,
            compile_stages_json=json.dumps(stages.durations, sort_keys=True),
            artifact_stats_json=json.dumps(artifact, sort_keys=True),
            startup_host_rss_bytes=startup_rss,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=peak_compile,
            baseline_gpu_allocated_bytes=baseline_gpu,
            peak_compile_gpu_allocated_bytes=gpu_allocated,
            peak_compile_gpu_reserved_bytes=gpu_reserved,
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )
    except Exception as exc:
        _set_worker_phase(request, "done")
        status = RunStatus.OOM if isinstance(exc, MemoryError) else RunStatus.ERROR
        compilation = E2CellRecord(
            **base,
            status=status,
            failure_stage=stages.failed_stage or stages.active_stage or "compilation",
            failure_reason=f"{type(exc).__name__}: {exc}",
            compile_total_s=time.perf_counter() - start,
            compile_stages_json=json.dumps(stages.durations, sort_keys=True),
            startup_host_rss_bytes=startup_rss,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=_peak(),
        )
        print(json.dumps({"compilation": compilation.flat_dict(), "runtime": []}))
        return

    rng = np.random.default_rng(int(payload["trace_seed"]))
    runtime = []
    _set_worker_phase(request, "runtime")
    for trace_length, batch_size in payload["workloads"]:
        bits = rng.integers(
            0,
            2,
            size=(batch_size, trace_length, len(request.atoms)),
            dtype=np.int8,
        )
        traces = [
            [
                {
                    atom: bool(bits[b, cell, atom_index])
                    for atom_index, atom in enumerate(request.atoms)
                }
                for cell in range(trace_length)
            ]
            for b in range(batch_size)
        ]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(int(payload["warmup_iterations"])):
                    monitor.batch_run(traces, early_termination=False)
                if monitor.effective_device == "cuda":
                    torch.cuda.synchronize()
                inner = 1
                while True:
                    timed = time.perf_counter()
                    for _ in range(inner):
                        monitor.batch_run(traces, early_termination=False)
                    if monitor.effective_device == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - timed
                    if elapsed >= float(payload["minimum_timed_s"]) or inner >= int(
                        payload["maximum_inner_iterations"]
                    ):
                        break
                    factor = max(
                        2,
                        int(float(payload["minimum_timed_s"]) / max(elapsed, 1e-9)),
                    )
                    inner = min(
                        int(payload["maximum_inner_iterations"]), inner * factor
                    )
            messages = [str(item.message) for item in caught]
            status = (
                RunStatus.FALLBACK
                if any("falling back" in message.lower() for message in messages)
                else RunStatus.SUCCESS
            )
            duration = elapsed / inner
            n_states = int(artifact.get("n_states", artifact.get("dfa_states", 0)))
            runtime.append(
                RQ3RuntimeRecord(
                    schema_version=RQ3_SCHEMA_VERSION,
                    run_id=request.run_id,
                    cell_id=f"{request.cell_id}::l{trace_length}::b{batch_size}",
                    panel=payload["panel"],
                    formula_id=request.formula_id,
                    formula_family=payload["formula_family"],
                    parameters_json=payload["parameters_json"],
                    tree_shape=payload["tree_shape"],
                    monitor_id=spec.monitor_id,
                    monitor_name=spec.monitor_name,
                    backend=spec.backend,
                    requested_device=request.requested_device,
                    effective_device=monitor.effective_device,
                    trace_seed=int(payload["trace_seed"]),
                    repetition=request.repetition,
                    trace_length=trace_length,
                    batch_size=batch_size,
                    warmup_iterations=int(payload["warmup_iterations"]),
                    inner_iterations=inner,
                    status=status,
                    duration_s_per_batch=duration,
                    duration_s_per_trace=duration / batch_size,
                    duration_s_per_cell=duration / (batch_size * trace_length),
                    throughput_traces_s=batch_size / duration,
                    estimated_scan_stack_bytes=(
                        trace_length * batch_size * n_states * n_states * 4
                        if spec.adapter == "deepdfa_scan"
                        else None
                    ),
                    warnings_json=json.dumps(messages),
                )
            )
        except Exception as exc:
            runtime.append(
                RQ3RuntimeRecord(
                    schema_version=RQ3_SCHEMA_VERSION,
                    run_id=request.run_id,
                    cell_id=f"{request.cell_id}::l{trace_length}::b{batch_size}",
                    panel=payload["panel"],
                    formula_id=request.formula_id,
                    formula_family=payload["formula_family"],
                    parameters_json=payload["parameters_json"],
                    tree_shape=payload["tree_shape"],
                    monitor_id=spec.monitor_id,
                    monitor_name=spec.monitor_name,
                    backend=spec.backend,
                    requested_device=request.requested_device,
                    effective_device=monitor.effective_device,
                    trace_seed=int(payload["trace_seed"]),
                    repetition=request.repetition,
                    trace_length=trace_length,
                    batch_size=batch_size,
                    warmup_iterations=int(payload["warmup_iterations"]),
                    inner_iterations=0,
                    status=RunStatus.ERROR,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    _set_worker_phase(request, "done")
    print(
        json.dumps(
            {
                "compilation": compilation.flat_dict(),
                "runtime": [row.flat_dict() for row in runtime],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
