"""Run Phase E5/RQ4 cross-architecture capacity and deployment blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks.e2 import _host_memory_bytes  # noqa: E402
from src.benchmarks.rq1 import RQ1_CORPUS  # noqa: E402
from src.benchmarks.rq4 import (  # noqa: E402
    RQ4_CASES,
    RQ4_MODES,
    RQ4_MONITOR_IDS,
    RQ4_SCHEMA_VERSION,
    run_rq4_cell,
)
from src.benchmarks.schema import (  # noqa: E402
    DEFAULT_RESOURCE_BUDGETS,
    characterize_formula,
)

OUTPUT_DIR = ROOT / "results" / "rq4"
SEEDS = (1301, 2411, 3527, 4637, 5741)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(1409)
    boot = np.median(rng.choice(array, size=(5000, len(array)), replace=True), axis=1)
    return {
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _absolute_summary(
    compilation: list[dict[str, Any]], runtime: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comp_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in compilation:
        if row["status"] in {"success", "fallback"}:
            comp_groups.setdefault(
                (
                    row["formula_id"],
                    row["monitor_id"],
                    row["mode_id"],
                    row["requested_device"],
                ),
                [],
            ).append(row)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in runtime:
        if row["status"] in {"success", "fallback"}:
            key = (
                row["formula_id"],
                row["monitor_id"],
                row["workload_mode"],
                row["timing_boundary"],
                row["requested_device"],
                row["effective_device"],
                int(row["trace_length"]),
                int(row["batch_size"]),
            )
            groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        (
            formula_id,
            monitor_id,
            workload_mode,
            timing_boundary,
            requested_device,
            effective_device,
            length,
            batch,
        ) = key
        mode_id = next(
            row["mode_id"]
            for row in compilation
            if (
                row["formula_id"] == formula_id
                and row["monitor_id"] == monitor_id
                and row["requested_device"] == requested_device
                and row["workload_mode"] == workload_mode
                and row["timing_boundary"] == timing_boundary
            )
        )
        comp_rows = comp_groups[(formula_id, monitor_id, mode_id, requested_device)]
        latency = _ci([float(row["median_duration_s_per_trace"]) for row in rows])
        throughput = _ci([float(row["throughput_traces_s"]) for row in rows])
        compile_ci = _ci([float(row["compile_total_s"]) for row in comp_rows])
        samples = [
            float(sample)
            for row in rows
            for sample in json.loads(row["timing_samples_json"])
        ]
        processed = [float(row["processed_cell_fraction"]) for row in rows]
        persistent_rss = [
            int(row["persistent_host_rss_delta_bytes"])
            for row in rows
            if row["persistent_host_rss_delta_bytes"] not in (None, "")
        ]
        peak_rss = [
            int(row["peak_runtime_host_rss_bytes"])
            for row in rows
            if row["peak_runtime_host_rss_bytes"] not in (None, "")
        ]
        output.append(
            {
                "schema_version": RQ4_SCHEMA_VERSION,
                "formula_id": formula_id,
                "monitor_id": monitor_id,
                "architecture": rows[0]["architecture"],
                "construction": rows[0]["construction"],
                "backend": rows[0]["backend"],
                "workload_mode": workload_mode,
                "timing_boundary": timing_boundary,
                "requested_device": requested_device,
                "effective_device": effective_device,
                "trace_profile": rows[0]["trace_profile"],
                "trace_length": length,
                "batch_size": batch,
                "repetitions": len(rows),
                "timing_sample_count": len(samples),
                **{
                    f"latency_s_per_trace_{name}": value
                    for name, value in latency.items()
                },
                "latency_s_per_batch_p95": float(np.percentile(samples, 95)),
                **{
                    f"throughput_traces_s_{name}": value
                    for name, value in throughput.items()
                },
                **{f"compile_s_{name}": value for name, value in compile_ci.items()},
                "processed_cell_fraction_median": float(np.median(processed)),
                "persistent_host_rss_delta_bytes_median": (
                    float(np.median(persistent_rss)) if persistent_rss else None
                ),
                "peak_runtime_host_rss_bytes_median": (
                    float(np.median(peak_rss)) if peak_rss else None
                ),
                "fallbacks": sum(row["status"] == "fallback" for row in rows),
            }
        )
    return output


def _symbolic_comparisons(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        (
            row["formula_id"],
            row["workload_mode"],
            row["timing_boundary"],
            row["requested_device"],
            row["effective_device"],
            row["trace_length"],
            row["batch_size"],
        ): row
        for row in summary
        if row["monitor_id"] == "symbolic_guarded"
    }
    output: list[dict[str, Any]] = []
    for row in summary:
        key = (
            row["formula_id"],
            row["workload_mode"],
            row["timing_boundary"],
            row["requested_device"],
            row["effective_device"],
            row["trace_length"],
            row["batch_size"],
        )
        reference = baseline.get(key)
        if reference is None:
            continue
        base_runtime = float(reference["latency_s_per_trace_median"])
        runtime = float(row["latency_s_per_trace_median"])
        compile_delta = float(row["compile_s_median"]) - float(
            reference["compile_s_median"]
        )
        runtime_delta = runtime - base_runtime
        if compile_delta <= 0 and runtime_delta <= 0:
            break_even = None
            break_even_status = "monitor_always_or_equal"
        elif compile_delta >= 0 and runtime_delta >= 0:
            break_even = None
            break_even_status = "symbolic_always_or_equal"
        else:
            break_even = -compile_delta / runtime_delta
            if break_even <= 0:
                raise RuntimeError("invalid cold-start crossover geometry")
            break_even_status = (
                "monitor_wins_after" if runtime_delta < 0 else "monitor_wins_before"
            )
        output.append(
            {
                "schema_version": RQ4_SCHEMA_VERSION,
                "formula_id": row["formula_id"],
                "monitor_id": row["monitor_id"],
                "baseline_monitor_id": "symbolic_guarded",
                "workload_mode": row["workload_mode"],
                "timing_boundary": row["timing_boundary"],
                "requested_device": row["requested_device"],
                "effective_device": row["effective_device"],
                "trace_length": row["trace_length"],
                "batch_size": row["batch_size"],
                "symbolic_over_monitor_runtime_ratio": base_runtime / runtime,
                "cold_start_break_even_traces": break_even,
                "break_even_status": break_even_status,
            }
        )
    return output


def _pareto(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summary:
        key = (
            row["formula_id"],
            row["workload_mode"],
            row["timing_boundary"],
            row["requested_device"],
            row["effective_device"],
            row["trace_length"],
            row["batch_size"],
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        for row in rows:
            metrics = (
                float(row["latency_s_per_trace_median"]),
                float(row["compile_s_median"]),
                float(row["persistent_host_rss_delta_bytes_median"] or math.inf),
            )
            dominated_by = []
            for other in rows:
                if other is row:
                    continue
                candidate = (
                    float(other["latency_s_per_trace_median"]),
                    float(other["compile_s_median"]),
                    float(other["persistent_host_rss_delta_bytes_median"] or math.inf),
                )
                weakly_better = all(
                    left <= right for left, right in zip(candidate, metrics)
                )
                strictly_better = any(
                    left < right for left, right in zip(candidate, metrics)
                )
                if weakly_better and strictly_better:
                    dominated_by.append(other["monitor_id"])
            output.append(
                {
                    "schema_version": RQ4_SCHEMA_VERSION,
                    "formula_id": key[0],
                    "workload_mode": key[1],
                    "timing_boundary": key[2],
                    "requested_device": key[3],
                    "effective_device": key[4],
                    "trace_length": key[5],
                    "batch_size": key[6],
                    "monitor_id": row["monitor_id"],
                    "latency_s_per_trace_median": metrics[0],
                    "compile_s_median": metrics[1],
                    "persistent_host_rss_delta_bytes_median": metrics[2],
                    "pareto_optimal": not dominated_by,
                    "dominated_by_json": json.dumps(dominated_by),
                }
            )
    return output


def _plot(summary: list[dict[str, Any]], pareto: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    cpu = [
        row
        for row in summary
        if row["requested_device"] == "cpu" and row["effective_device"] == "cpu"
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for formula_id, marker in (
        ("atomic_control", "o"),
        ("nested_next_control", "s"),
        ("ijcnn_balanced_n4", "^"),
    ):
        selected = [
            row
            for row in cpu
            if row["formula_id"] == formula_id
            and row["workload_mode"] == "capacity"
            and row["timing_boundary"] == "end_to_end"
            and row["batch_size"] == 1
        ]
        axes[0, 0].scatter(
            [row["monitor_id"] for row in selected],
            [row["latency_s_per_trace_median"] for row in selected],
            marker=marker,
            label=formula_id,
        )
    axes[0, 0].set(
        title="Capacity batch-1 latency",
        ylabel="seconds / trace",
        yscale="log",
    )
    axes[0, 0].tick_params(axis="x", rotation=55, labelsize=7)
    axes[0, 0].legend(fontsize=7)

    selected = [
        row
        for row in cpu
        if row["formula_id"] == "ijcnn_balanced_n4"
        and row["workload_mode"] == "capacity"
        and row["timing_boundary"] == "end_to_end"
        and row["batch_size"] == 128
    ]
    axes[0, 1].bar(
        [row["monitor_id"] for row in selected],
        [row["throughput_traces_s_median"] for row in selected],
        color="#0072B2",
    )
    axes[0, 1].set(
        title="IJCNN capacity throughput (B=128)",
        ylabel="traces / s",
        yscale="log",
    )
    axes[0, 1].tick_params(axis="x", rotation=55, labelsize=7)

    selected = [
        row
        for row in cpu
        if row["formula_id"] == "ijcnn_balanced_n4"
        and row["workload_mode"] == "deployment"
        and row["batch_size"] == 64
    ]
    axes[1, 0].bar(
        [row["monitor_id"] for row in selected],
        [row["processed_cell_fraction_median"] for row in selected],
        color="#009E73",
    )
    axes[1, 0].set(
        title="Deployment compute actually processed",
        ylabel="processed / offered cells",
        ylim=(0, 1.05),
    )
    axes[1, 0].tick_params(axis="x", rotation=55, labelsize=7)

    selected_ids = {
        row["monitor_id"]
        for row in pareto
        if row["formula_id"] == "ijcnn_balanced_n4"
        and row["workload_mode"] == "capacity"
        and row["timing_boundary"] == "end_to_end"
        and row["requested_device"] == "cpu"
        and row["batch_size"] == 128
        and row["pareto_optimal"]
    }
    selected = [
        row
        for row in cpu
        if row["formula_id"] == "ijcnn_balanced_n4"
        and row["workload_mode"] == "capacity"
        and row["timing_boundary"] == "end_to_end"
        and row["batch_size"] == 128
    ]
    for row in selected:
        axes[1, 1].scatter(
            row["persistent_host_rss_delta_bytes_median"] / 1024**2,
            row["latency_s_per_trace_median"],
            marker="*" if row["monitor_id"] in selected_ids else "o",
            s=90 if row["monitor_id"] in selected_ids else 35,
            label=row["monitor_id"],
        )
    axes[1, 1].set(
        title="Latency–memory landscape (stars: 3-D Pareto)",
        xlabel="persistent RSS delta (MiB)",
        ylabel="seconds / trace",
        yscale="log",
    )
    axes[1, 1].legend(fontsize=6)
    figure.suptitle("RQ4 cross-architecture landscape — controlled CPU candidate")
    figure.savefig(OUTPUT_DIR / "rq4_cross_architecture.png", dpi=180)
    plt.close(figure)


def _command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else "unavailable"


def _cpu_model() -> str:
    try:
        rows = Path("/proc/cpuinfo").read_text().splitlines()
        return next(
            row.split(":", 1)[1].strip() for row in rows if row.startswith("model name")
        )
    except (OSError, StopIteration, IndexError):
        return "unavailable"


def generate(
    repetitions: int,
    minimum_timed_s: float,
    measurement_blocks: int,
    devices: tuple[str, ...],
    torch_threads: int,
    torch_interop_threads: int,
) -> None:
    run_id = str(uuid.uuid4())
    compilation: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    benchmarks = {
        case.benchmark.formula_id: case.benchmark
        for case in RQ1_CORPUS
        if case.benchmark.formula_id in {item.formula_id for item in RQ4_CASES}
    }
    for requested_device in devices:
        for case in RQ4_CASES:
            benchmark = benchmarks[case.formula_id]
            structure = characterize_formula(benchmark)
            for mode in RQ4_MODES:
                for monitor_id in RQ4_MONITOR_IDS:
                    for repetition in range(repetitions):
                        print(
                            f"RQ4 {requested_device}: {mode.mode_id} / "
                            f"{case.formula_id} / {monitor_id} / r{repetition}",
                            flush=True,
                        )
                        comp, runs = run_rq4_cell(
                            case.formula_id,
                            monitor_id,
                            mode.mode_id,
                            trace_seed=SEEDS[repetition],
                            repetition=repetition,
                            requested_device=requested_device,
                            measurement_blocks=measurement_blocks,
                            minimum_timed_s=minimum_timed_s,
                            torch_threads=torch_threads,
                            torch_interop_threads=torch_interop_threads,
                            run_id=run_id,
                        )
                        artifact = json.loads(comp.artifact_stats_json)
                        row = comp.flat_dict()
                        row.update(
                            mode_id=mode.mode_id,
                            workload_mode=mode.workload_mode,
                            timing_boundary=mode.timing_boundary,
                            early_termination=mode.early_termination,
                            deployment_profile=case.deployment_profile,
                            formula_family=benchmark.family,
                            n_atoms=structure.n_atoms,
                            ast_nodes=structure.ast_nodes,
                            ast_depth=structure.ast_depth,
                            dfa_states=artifact.get(
                                "dfa_states", artifact.get("n_states")
                            ),
                            persistent_tensor_bytes=artifact.get(
                                "persistent_tensor_allocated_bytes"
                            ),
                        )
                        compilation.append(row)
                        runtime.extend(item.flat_dict() for item in runs)

    summary = _absolute_summary(compilation, runtime)
    comparisons = _symbolic_comparisons(summary)
    pareto = _pareto(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(OUTPUT_DIR / "rq4_compilation.csv", compilation)
    _write(OUTPUT_DIR / "rq4_runtime.csv", runtime)
    _write(OUTPUT_DIR / "rq4_absolute_summary.csv", summary)
    _write(OUTPUT_DIR / "rq4_symbolic_comparisons.csv", comparisons)
    _write(OUTPUT_DIR / "rq4_pareto.csv", pareto)
    _plot(summary, pareto)

    cuda_rows = [row for row in compilation if row["requested_device"] == "cuda"]
    cuda_success = [
        row
        for row in cuda_rows
        if row["status"] == "success" and row["effective_device"] == "cuda"
    ]
    cuda_status = "complete" if cuda_success else "unsupported"
    canonical = json.dumps(
        [compilation, runtime], sort_keys=True, separators=(",", ":")
    )
    manifest = {
        "schema_version": RQ4_SCHEMA_VERSION,
        "artifact_kind": "controlled_local_run_candidate",
        "artifact_id": hashlib.sha256(canonical.encode()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol": {
            "repetitions": repetitions,
            "trace_seeds": list(SEEDS[:repetitions]),
            "minimum_timed_s": minimum_timed_s,
            "measurement_blocks": measurement_blocks,
            "devices": list(devices),
            "torch_threads": torch_threads,
            "torch_interop_threads": torch_interop_threads,
            "modes": [asdict(mode) for mode in RQ4_MODES],
            "resource_budgets": asdict(DEFAULT_RESOURCE_BUDGETS),
        },
        "cpu_block": {
            "controlled": "cpu" in devices,
            "compilation_rows": sum(
                row["requested_device"] == "cpu" for row in compilation
            ),
            "runtime_rows": sum(row["requested_device"] == "cpu" for row in runtime),
        },
        "cuda_block": {
            "status": cuda_status,
            "reason": (
                ""
                if cuda_success
                else (
                    "CUDA unavailable on this host; requested CUDA attempts "
                    "are explicit unsupported rows."
                )
            ),
            "requested_rows": len(cuda_rows),
            "effective_cuda_success_rows": len(cuda_success),
        },
        "record_counts": {
            "compilation": len(compilation),
            "runtime": len(runtime),
            "absolute_summary": len(summary),
            "symbolic_comparisons": len(comparisons),
            "pareto": len(pareto),
        },
        "status_counts": {
            "compilation": dict(Counter(row["status"] for row in compilation)),
            "runtime": dict(Counter(row["status"] for row in runtime)),
        },
        "provenance": {
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "worktree_status_sha256": hashlib.sha256(
                _command(["git", "status", "--porcelain=v1"]).encode()
            ).hexdigest(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "mona": _command(["mona", "-v"]),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "host_memory_bytes": _host_memory_bytes(),
            "torch": _command(
                [sys.executable, "-c", "import torch; print(torch.__version__)"]
            ),
            "cuda_runtime": _command(
                [
                    sys.executable,
                    "-c",
                    "import torch; print(torch.version.cuda or 'unavailable')",
                ]
            ),
        },
        "summary": summary,
    }
    (OUTPUT_DIR / "rq4_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--minimum-timed-s", type=float, default=0.03)
    parser.add_argument("--measurement-blocks", type=int, default=7)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda")
    )
    parser.add_argument("--torch-threads", type=int, default=10)
    parser.add_argument("--torch-interop-threads", type=int, default=10)
    args = parser.parse_args()
    generate(
        args.repetitions,
        args.minimum_timed_s,
        args.measurement_blocks,
        tuple(args.devices),
        args.torch_threads,
        args.torch_interop_threads,
    )


if __name__ == "__main__":
    main()
