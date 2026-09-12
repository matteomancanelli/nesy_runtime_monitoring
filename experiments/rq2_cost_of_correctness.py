"""Run Phase E3: the RuleRunner cost-of-correctness experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
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

from src.benchmarks.e2 import MONITOR_SPEC_BY_ID  # noqa: E402
from src.benchmarks.rq2 import (  # noqa: E402
    RQ2_CASES,
    RQ2_SCHEMA_VERSION,
    applicable_monitor_ids,
    characterize_decision_lag,
    encoding_name,
    run_certificate_generation,
    run_rq2_cell,
)
from src.benchmarks.schema import DEFAULT_RESOURCE_BUDGETS  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "rq2"
TRACE_SEEDS = (1103, 2207, 3301, 4409, 5501)
DECISION_LENGTHS = {
    "eventual_next_alias": 4,
    "next_offset_alias": 4,
    "globally_next_alias": 5,
    "until_next_alias": 4,
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _interval(values: list[float], seed: int = 823) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(array, size=(5000, len(array)), replace=True), axis=1
    )
    return {
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
    }


def _summaries(
    compilation: list[dict[str, Any]], runtime: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    compile_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in compilation:
        if row["status"] == "success":
            compile_groups.setdefault(
                (row["formula_id"], row["monitor_id"]), []
            ).append(row)
    runtime_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in runtime:
        if row["status"] == "success":
            runtime_groups.setdefault(
                (row["formula_id"], row["monitor_id"], int(row["batch_size"])), []
            ).append(row)
    for (formula_id, monitor_id), rows in compile_groups.items():
        compile_ci = _interval([float(row["compile_total_s"]) for row in rows])
        rss_ci = _interval([float(row["peak_compile_host_rss_bytes"]) for row in rows])
        persistent_ci = _interval([float(row["persistent_bytes"]) for row in rows])
        for batch_size in (1, 64):
            runtime_rows = runtime_groups.get((formula_id, monitor_id, batch_size), [])
            if not runtime_rows:
                continue
            latency_ci = _interval(
                [float(row["duration_s_per_trace"]) for row in runtime_rows]
            )
            throughput_ci = _interval(
                [float(row["throughput_traces_s"]) for row in runtime_rows]
            )
            summaries.append(
                {
                    "schema_version": RQ2_SCHEMA_VERSION,
                    "formula_id": formula_id,
                    "stratum": rows[0]["stratum"],
                    "monitor_id": monitor_id,
                    "construction": rows[0]["construction"],
                    "encoding": rows[0]["encoding"],
                    "online_label_mode": rows[0]["online_label_mode"],
                    "batch_size": batch_size,
                    "repetitions": len(runtime_rows),
                    **{f"compile_s_{key}": value for key, value in compile_ci.items()},
                    **{
                        f"compile_peak_rss_{key}": value
                        for key, value in rss_ci.items()
                    },
                    **{
                        f"persistent_bytes_{key}": value
                        for key, value in persistent_ci.items()
                    },
                    **{
                        f"latency_s_per_trace_{key}": value
                        for key, value in latency_ci.items()
                    },
                    **{
                        f"throughput_traces_s_{key}": value
                        for key, value in throughput_ci.items()
                    },
                }
            )
    return summaries


def _decision_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["formula_id"], row["online_label_mode"]), []).append(row)
    output = []
    for (formula_id, mode), selected in groups.items():
        lags = np.asarray([row["decision_lag_cells"] for row in selected], dtype=float)
        compute = np.asarray(
            [row["additional_monitor_compute_s"] for row in selected], dtype=float
        )
        output.append(
            {
                "schema_version": RQ2_SCHEMA_VERSION,
                "formula_id": formula_id,
                "online_label_mode": mode,
                "traces": len(selected),
                "late_traces": int(np.count_nonzero(lags)),
                "late_fraction": float(np.mean(lags > 0)),
                "lag_cells_median": float(np.median(lags)),
                "lag_cells_p95": float(np.quantile(lags, 0.95)),
                "lag_cells_max": int(np.max(lags)),
                "additional_monitor_compute_s_median": float(np.median(compute)),
                "late_additional_monitor_compute_s_median": (
                    float(np.median(compute[lags > 0])) if np.any(lags > 0) else 0.0
                ),
            }
        )
    return output


def _plot_cost_figure(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    flat = [row for row in rows if row["encoding"] == "flat"]
    formula_ids = [case.formula_id for case in RQ2_CASES]
    monitor_ids = [
        "rulerunner_original_flat",
        "rulerunner_bounded_flat_default",
        "rulerunner_bounded_flat_exact",
        "rulerunner_progression_flat",
    ]
    labels = ["Original", "Bounded default", "Bounded exact", "Progression"]
    colors = ["#E69F00", "#56B4E9", "#0072B2", "#009E73"]
    metrics = [
        ("compile_s_median", "Cold compilation (s)"),
        ("persistent_bytes_median", "Persistent tensor storage (bytes)"),
        ("latency_s_per_trace_median", "Batch-1 latency (s/trace)"),
        ("throughput_traces_s_median", "Batch-64 throughput (traces/s)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(formula_ids), dtype=float)
    width = 0.19
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        batch_size = 64 if "throughput" in metric else 1
        for offset, (monitor_id, label, color) in enumerate(
            zip(monitor_ids, labels, colors, strict=True)
        ):
            values = []
            for formula_id in formula_ids:
                match = next(
                    (
                        row
                        for row in flat
                        if row["formula_id"] == formula_id
                        and row["monitor_id"] == monitor_id
                        and row["batch_size"] == batch_size
                    ),
                    None,
                )
                values.append(float(match[metric]) if match else np.nan)
            axis.bar(
                x + (offset - 1.5) * width,
                values,
                width,
                label=label,
                color=color,
            )
        axis.set_title(title)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, formula_ids, rotation=28, ha="right", fontsize=8)
    axes[0, 0].legend(ncols=2, fontsize=8)
    figure.suptitle("RQ2 cost of correctness — valid flat constructions only")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else "unavailable"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def generate(repetitions: int, minimum_timed_s: float) -> None:
    run_id = str(uuid.uuid4())
    compilation_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for case in RQ2_CASES:
        for monitor_id in applicable_monitor_ids(case.formula_id):
            for repetition in range(repetitions):
                seed = TRACE_SEEDS[repetition]
                print(
                    f"RQ2 runtime: {case.formula_id} / {monitor_id} / r{repetition}",
                    flush=True,
                )
                compilation, runtimes = run_rq2_cell(
                    case.formula_id,
                    monitor_id,
                    trace_seed=seed,
                    repetition=repetition,
                    run_id=run_id,
                    minimum_timed_s=minimum_timed_s,
                )
                spec = MONITOR_SPEC_BY_ID[monitor_id]
                artifact = json.loads(compilation.artifact_stats_json)
                compile_row = compilation.flat_dict()
                compile_row.update(
                    stratum=case.stratum,
                    encoding=encoding_name(monitor_id),
                    online_label_mode=(
                        "exact_online"
                        if spec.construction == "bounded_exact_online"
                        else "sound_default"
                        if spec.construction == "bounded_default"
                        else "exact"
                    ),
                    persistent_bytes=artifact.get(
                        "persistent_tensor_allocated_bytes", 0
                    ),
                )
                compilation_rows.append(compile_row)
                runtime_rows.extend(row.flat_dict() for row in runtimes)

    certificate_rows = []
    for case in RQ2_CASES:
        if not any(
            "bounded" in item for item in applicable_monitor_ids(case.formula_id)
        ):
            continue
        for repetition in range(repetitions):
            print(f"RQ2 certificate: {case.formula_id} / r{repetition}", flush=True)
            certificate_rows.append(
                run_certificate_generation(
                    case.formula_id, repetition=repetition, run_id=run_id
                )
            )

    decision_rows = []
    for formula_id, trace_length in DECISION_LENGTHS.items():
        print(f"RQ2 decision lag: {formula_id} / L={trace_length}", flush=True)
        decision_rows.extend(
            row.flat_dict()
            for row in characterize_decision_lag(formula_id, trace_length=trace_length)
        )

    summaries = _summaries(compilation_rows, runtime_rows)
    decision_summaries = _decision_summary(decision_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "rq2_compilation.csv", compilation_rows)
    _write_csv(OUTPUT_DIR / "rq2_runtime.csv", runtime_rows)
    _write_csv(OUTPUT_DIR / "rq2_certificate_generation.csv", certificate_rows)
    _write_csv(OUTPUT_DIR / "rq2_decision_lag.csv", decision_rows)
    _write_csv(OUTPUT_DIR / "rq2_summary.csv", summaries)
    _write_csv(OUTPUT_DIR / "rq2_decision_summary.csv", decision_summaries)
    _plot_cost_figure(summaries, OUTPUT_DIR / "rq2_cost_of_correctness.png")
    canonical = json.dumps(
        [compilation_rows, runtime_rows, certificate_rows, decision_rows],
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest = {
        "schema_version": RQ2_SCHEMA_VERSION,
        "artifact_kind": "controlled_cpu_run_candidate",
        "artifact_id": hashlib.sha256(canonical.encode()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol": {
            "repetitions": repetitions,
            "trace_seeds": list(TRACE_SEEDS[:repetitions]),
            "trace_length": 32,
            "batch_sizes": [1, 64],
            "early_termination": False,
            "minimum_timed_s": minimum_timed_s,
            "resource_budgets": asdict(DEFAULT_RESOURCE_BUDGETS),
        },
        "corpus": [asdict(case) for case in RQ2_CASES],
        "status_counts": {
            "compilation": dict(Counter(row["status"] for row in compilation_rows)),
            "runtime": dict(Counter(row["status"] for row in runtime_rows)),
            "certificate": dict(Counter(row["status"] for row in certificate_rows)),
        },
        "record_counts": {
            "compilation": len(compilation_rows),
            "runtime": len(runtime_rows),
            "certificate": len(certificate_rows),
            "decision_lag": len(decision_rows),
        },
        "provenance": {
            "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _version("torch"),
            "ltlf2dfa": _version("ltlf2dfa"),
            "mona": _command_output(["mona", "-v"]),
        },
        "summary": summaries,
        "decision_summary": decision_summaries,
    }
    (OUTPUT_DIR / "rq2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def resummarize_existing() -> None:
    """Rebuild derived fields without repeating unchanged raw measurements."""
    compilation = _read_csv(OUTPUT_DIR / "rq2_compilation.csv")
    runtime = _read_csv(OUTPUT_DIR / "rq2_runtime.csv")
    certificates = _read_csv(OUTPUT_DIR / "rq2_certificate_generation.csv")
    decisions = _read_csv(OUTPUT_DIR / "rq2_decision_lag.csv")
    for row in compilation + runtime:
        row["encoding"] = encoding_name(row["monitor_id"])
    summaries = _summaries(compilation, runtime)
    decision_summaries = _decision_summary(decisions)
    _write_csv(OUTPUT_DIR / "rq2_compilation.csv", compilation)
    _write_csv(OUTPUT_DIR / "rq2_runtime.csv", runtime)
    _write_csv(OUTPUT_DIR / "rq2_summary.csv", summaries)
    _write_csv(OUTPUT_DIR / "rq2_decision_summary.csv", decision_summaries)
    _plot_cost_figure(summaries, OUTPUT_DIR / "rq2_cost_of_correctness.png")
    manifest_path = OUTPUT_DIR / "rq2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical = json.dumps(
        [compilation, runtime, certificates, decisions],
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest.update(
        artifact_id=hashlib.sha256(canonical.encode()).hexdigest(),
        summary=summaries,
        decision_summary=decision_summaries,
        derived_fields_regenerated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--minimum-timed-s", type=float, default=0.05)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.summarize_existing:
        resummarize_existing()
    else:
        generate(args.repetitions, args.minimum_timed_s)


if __name__ == "__main__":
    main()
