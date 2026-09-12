"""Fresh-process worker for one Phase E5 cross-architecture cell."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import warnings
from collections import Counter
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


def _bits_to_traces(bits, atoms: tuple[str, ...]):
    batch, length, _ = bits.shape
    return [
        [
            {
                atom: bool(bits[b, cell, atom_index])
                for atom_index, atom in enumerate(atoms)
            }
            for cell in range(length)
        ]
        for b in range(batch)
    ]


def _deployment_bits(
    formula_id: str,
    atoms: tuple[str, ...],
    length: int,
    batch: int,
    seed: int,
):
    import numpy as np

    bits = np.zeros((batch, length, len(atoms)), dtype=np.int8)
    strata: list[str] = []
    offset = seed % 3
    for b in range(batch):
        stratum = (b + offset) % 3
        if formula_id == "nested_next_control":
            label = (
                "fixed_horizon_satisfy",
                "fixed_horizon_violate",
                "fixed_horizon_satisfy",
            )[stratum]
            bits[b, 2, 0] = int(label.endswith("satisfy"))
        else:
            label = ("early_satisfy", "late_satisfy", "boundary_violate")[stratum]
            if label != "boundary_violate":
                position = 0 if label == "early_satisfy" else length - 1
                if formula_id == "atomic_control":
                    bits[b, position, 0] = 1
                else:
                    bits[b, position, 0] = 1
                    bits[b, position, 1] = 1
        strata.append(label)
    return bits, strata


def _decision_indices(formula: str, traces):
    from src.monitors.base import Verdict
    from src.monitors.symbolic_dfa import SymbolicDFAMonitor

    canonical = SymbolicDFAMonitor.compile(formula)
    indices: list[int] = []
    verdicts: list[Verdict] = []
    for trace in traces:
        canonical.reset()
        decided = Verdict.UNDECIDED
        decision = len(trace)
        for index, obs in enumerate(trace):
            decided = canonical.step(obs)
            if decided is not Verdict.UNDECIDED:
                decision = index
                break
        if decided is Verdict.UNDECIDED:
            decided = canonical.final_verdict()
        indices.append(decision)
        verdicts.append(decided)
    return indices, verdicts


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
    from src.benchmarks.rq4 import RQ4_SCHEMA_VERSION, RQ4RuntimeRecord
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
        torch.set_num_threads(int(payload["torch_threads"]))
        torch.set_num_interop_threads(int(payload["torch_interop_threads"]))
    base = _record_base(request, spec)
    base["worker_pid"] = os.getpid()
    baseline_rss = _rss()
    stages = StageRecorder(request.requested_device, torch)
    if (
        request.requested_device == "cuda"
        and spec.uses_requested_device
        and (torch is None or not torch.cuda.is_available())
    ):
        compilation = E2CellRecord(
            **base,
            status=RunStatus.UNSUPPORTED,
            failure_stage="device_validation",
            failure_reason="CUDA was requested but is unavailable",
            startup_host_rss_bytes=startup_rss,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=_peak(),
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
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
            peak_compile_gpu_allocated_bytes=(
                torch.cuda.max_memory_allocated()
                if monitor.effective_device == "cuda"
                else None
            ),
            peak_compile_gpu_reserved_bytes=(
                torch.cuda.max_memory_reserved()
                if monitor.effective_device == "cuda"
                else None
            ),
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )
    except Exception as exc:
        _set_worker_phase(request, "done")
        oom_types = (MemoryError,)
        if torch is not None and hasattr(torch.cuda, "OutOfMemoryError"):
            oom_types += (torch.cuda.OutOfMemoryError,)
        status = RunStatus.OOM if isinstance(exc, oom_types) else RunStatus.ERROR
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
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )
        print(json.dumps({"compilation": compilation.flat_dict(), "runtime": []}))
        return

    persistent_tensor_bytes = artifact.get("persistent_tensor_allocated_bytes")
    if spec.adapter == "symbolic":
        persistent_tensor_bytes = None
    persistent_rss_delta = max(0, _rss() - baseline_rss)
    runtime = []
    rng = np.random.default_rng(int(payload["trace_seed"]))
    _set_worker_phase(request, "runtime")
    for trace_length, batch_size in payload["workloads"]:
        if payload["workload_mode"] == "deployment":
            bits, strata = _deployment_bits(
                request.formula_id,
                request.atoms,
                trace_length,
                batch_size,
                int(payload["trace_seed"]),
            )
            trace_profile = payload["deployment_profile"]
        else:
            bits = rng.integers(
                0,
                2,
                size=(batch_size, trace_length, len(request.atoms)),
                dtype=np.int8,
            )
            strata = ["uniform_bernoulli_p0.5"] * batch_size
            trace_profile = "uniform_bernoulli_p0.5"
        traces = _bits_to_traces(bits, request.atoms)
        indices, canonical_verdicts = _decision_indices(request.formula, traces)

        def invoke():
            selected = (
                _bits_to_traces(bits, request.atoms)
                if payload["timing_boundary"] == "end_to_end"
                else traces
            )
            return monitor.batch_run(
                selected, early_termination=bool(payload["early_termination"])
            )

        try:
            actual = invoke()
            if actual != canonical_verdicts:
                raise RuntimeError("timed monitor disagrees with canonical DFA")
            for _ in range(int(payload["warmup_iterations"])):
                invoke()
            if monitor.effective_device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            inner = 1
            while True:
                timed = time.perf_counter()
                for _ in range(inner):
                    invoke()
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
                inner = min(int(payload["maximum_inner_iterations"]), inner * factor)

            samples: list[float] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(int(payload["measurement_blocks"])):
                    timed = time.perf_counter()
                    for _ in range(inner):
                        invoke()
                    if monitor.effective_device == "cuda":
                        torch.cuda.synchronize()
                    samples.append((time.perf_counter() - timed) / inner)
            messages = [str(item.message) for item in caught]
            status = (
                RunStatus.FALLBACK
                if any("falling back" in item.lower() for item in messages)
                else RunStatus.SUCCESS
            )
            median = float(np.median(samples))
            p95 = float(np.percentile(samples, 95))
            offered_cells = batch_size * trace_length
            if bool(payload["early_termination"]) and spec.adapter == "symbolic":
                processed_cells = sum(min(trace_length, index + 1) for index in indices)
            else:
                processed_cells = offered_cells
            runtime.append(
                RQ4RuntimeRecord(
                    schema_version=RQ4_SCHEMA_VERSION,
                    run_id=request.run_id,
                    cell_id=f"{request.cell_id}::l{trace_length}::b{batch_size}",
                    formula_id=request.formula_id,
                    formula_family=payload["formula_family"],
                    monitor_id=spec.monitor_id,
                    monitor_name=spec.monitor_name,
                    architecture=spec.architecture,
                    construction=spec.construction,
                    backend=spec.backend,
                    requested_device=request.requested_device,
                    effective_device=monitor.effective_device,
                    workload_mode=payload["workload_mode"],
                    timing_boundary=payload["timing_boundary"],
                    boundary_detail=payload["boundary_detail"],
                    trace_profile=trace_profile,
                    trace_seed=int(payload["trace_seed"]),
                    repetition=request.repetition,
                    trace_length=trace_length,
                    batch_size=batch_size,
                    early_termination=bool(payload["early_termination"]),
                    warmup_iterations=int(payload["warmup_iterations"]),
                    measurement_blocks=int(payload["measurement_blocks"]),
                    inner_iterations=inner,
                    status=status,
                    median_duration_s_per_batch=median,
                    p95_duration_s_per_batch=p95,
                    median_duration_s_per_trace=median / batch_size,
                    median_duration_s_per_cell=median / offered_cells,
                    throughput_traces_s=batch_size / median,
                    throughput_cells_s=processed_cells / median,
                    offered_traces=batch_size,
                    offered_cells=offered_cells,
                    processed_traces=batch_size,
                    processed_cells=processed_cells,
                    processed_cell_fraction=processed_cells / offered_cells,
                    semantic_decision_indices_json=json.dumps(indices),
                    strata_counts_json=json.dumps(
                        dict(Counter(strata)), sort_keys=True
                    ),
                    timing_samples_json=json.dumps(samples),
                    persistent_tensor_bytes=persistent_tensor_bytes,
                    persistent_host_rss_delta_bytes=persistent_rss_delta,
                    peak_runtime_gpu_allocated_bytes=(
                        torch.cuda.max_memory_allocated()
                        if monitor.effective_device == "cuda"
                        else None
                    ),
                    peak_runtime_gpu_reserved_bytes=(
                        torch.cuda.max_memory_reserved()
                        if monitor.effective_device == "cuda"
                        else None
                    ),
                    warnings_json=json.dumps(messages),
                )
            )
        except Exception as exc:
            runtime.append(
                RQ4RuntimeRecord(
                    schema_version=RQ4_SCHEMA_VERSION,
                    run_id=request.run_id,
                    cell_id=f"{request.cell_id}::l{trace_length}::b{batch_size}",
                    formula_id=request.formula_id,
                    formula_family=payload["formula_family"],
                    monitor_id=spec.monitor_id,
                    monitor_name=spec.monitor_name,
                    architecture=spec.architecture,
                    construction=spec.construction,
                    backend=spec.backend,
                    requested_device=request.requested_device,
                    effective_device=monitor.effective_device,
                    workload_mode=payload["workload_mode"],
                    timing_boundary=payload["timing_boundary"],
                    boundary_detail=payload["boundary_detail"],
                    trace_profile=trace_profile,
                    trace_seed=int(payload["trace_seed"]),
                    repetition=request.repetition,
                    trace_length=trace_length,
                    batch_size=batch_size,
                    early_termination=bool(payload["early_termination"]),
                    warmup_iterations=int(payload["warmup_iterations"]),
                    measurement_blocks=int(payload["measurement_blocks"]),
                    inner_iterations=0,
                    status=RunStatus.ERROR,
                    offered_traces=batch_size,
                    offered_cells=batch_size * trace_length,
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
