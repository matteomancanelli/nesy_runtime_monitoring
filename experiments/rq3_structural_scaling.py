"""Run Phase E4/RQ3 structural-scaling panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from src.benchmarks.rq3 import (  # noqa: E402
    RQ3_SCHEMA_VERSION,
    rq3_cases,
    run_rq3_cell,
)
from src.benchmarks.schema import (  # noqa: E402
    DEFAULT_RESOURCE_BUDGETS,
    characterize_formula,
)

OUTPUT_DIR = ROOT / "results" / "rq3"
SEEDS = (1201, 2309, 3413, 4517, 5623)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(903)
    boot = np.median(rng.choice(array, size=(5000, len(array)), replace=True), axis=1)
    return {
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _summary(
    compilation: list[dict[str, Any]], runtime: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comp_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in compilation:
        if row["status"] == "success":
            comp_groups.setdefault(
                (row["panel"], row["formula_id"], row["monitor_id"]), []
            ).append(row)
    run_groups: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = {}
    for row in runtime:
        if row["status"] in {"success", "fallback"}:
            run_groups.setdefault(
                (
                    row["panel"],
                    row["formula_id"],
                    row["monitor_id"],
                    int(row["trace_length"]),
                    int(row["batch_size"]),
                ),
                [],
            ).append(row)
    output = []
    for key, run_rows in run_groups.items():
        panel, formula_id, monitor_id, length, batch = key
        comp_rows = comp_groups[(panel, formula_id, monitor_id)]
        compile_ci = _ci([float(row["compile_total_s"]) for row in comp_rows])
        runtime_ci = _ci([float(row["duration_s_per_trace"]) for row in run_rows])
        throughput_ci = _ci([float(row["throughput_traces_s"]) for row in run_rows])
        exemplar = comp_rows[0]
        output.append(
            {
                "schema_version": RQ3_SCHEMA_VERSION,
                "panel": panel,
                "formula_id": formula_id,
                "parameters_json": exemplar["parameters_json"],
                "tree_shape": exemplar["tree_shape"],
                "monitor_id": monitor_id,
                "trace_length": length,
                "batch_size": batch,
                "repetitions": len(run_rows),
                "n_atoms": exemplar["n_atoms"],
                "ast_depth": exemplar["ast_depth"],
                "dfa_states": exemplar["dfa_states"],
                "cube_count": exemplar["cube_count"],
                "dense_tensor_bytes": exemplar["dense_tensor_bytes"],
                "cube_tensor_bytes": exemplar["cube_tensor_bytes"],
                "persistent_bytes": exemplar["persistent_bytes"],
                "total_rules": exemplar["total_rules"],
                "fallbacks": sum(row["status"] == "fallback" for row in run_rows),
                **{f"compile_s_{name}": value for name, value in compile_ci.items()},
                **{
                    f"latency_s_per_trace_{name}": value
                    for name, value in runtime_ci.items()
                },
                **{
                    f"throughput_traces_s_{name}": value
                    for name, value in throughput_ci.items()
                },
            }
        )
    return output


def _validate(compilation: list[dict[str, Any]], runtime: list[dict[str, Any]]) -> None:
    bad = [row for row in compilation if row["status"] != "success"]
    bad += [row for row in runtime if row["status"] != "success"]
    if bad:
        raise RuntimeError(
            "RQ3 has unsuccessful/fallback cells: "
            + ", ".join(f"{row['cell_id']}={row['status']}" for row in bad[:10])
        )
    for row in compilation:
        if row["panel"] == "guard_complexity" and row["monitor_id"] == "deepdfa_dense":
            expected = int(row["dfa_states"]) ** 2 * (2 ** int(row["n_atoms"])) * 4
            if int(row["dense_tensor_bytes"]) != expected:
                raise RuntimeError("dense DeepDFA byte count violates its cost model")
        parameters = json.loads(row["parameters_json"])
        if row["panel"] == "linear_state" and row["dfa_states"]:
            if int(row["dfa_states"]) != int(parameters["deadline"]) + 2:
                raise RuntimeError("bounded-response DFA state count is not k+2")
        if row["panel"] == "exponential_state" and row["dfa_states"]:
            if int(row["dfa_states"]) != 2 ** int(parameters["temporal_depth"]) + 1:
                raise RuntimeError("kth-from-last DFA state count is not 2^k+1")


def _plot(summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    tree = [
        row
        for row in summary
        if row["panel"] == "tree_shape"
        and row["monitor_id"] == "rulerunner_original_flat"
        and row["batch_size"] == 64
    ]
    for shape, color in (("balanced", "#0072B2"), ("left_deep", "#D55E00")):
        selected = sorted(
            (row for row in tree if row["tree_shape"] == shape),
            key=lambda row: int(row["n_atoms"]),
        )
        axes[0, 0].plot(
            [int(row["ast_depth"]) for row in selected],
            [float(row["latency_s_per_trace_median"]) for row in selected],
            "o-",
            label=shape,
            color=color,
        )
    axes[0, 0].set(title="Tree shape", xlabel="AST depth", ylabel="s/trace")
    axes[0, 0].legend()

    guard = [
        row
        for row in summary
        if row["panel"] == "guard_complexity"
        and row["monitor_id"] == "deepdfa_factored"
    ]
    for kind, color in (
        ("read_once", "#009E73"),
        ("threshold", "#E69F00"),
        ("parity", "#CC79A7"),
    ):
        selected = sorted(
            (
                row
                for row in guard
                if json.loads(row["parameters_json"])["guard_kind"] == kind
            ),
            key=lambda row: int(row["cube_count"]),
        )
        axes[0, 1].plot(
            [int(row["cube_count"]) for row in selected],
            [float(row["latency_s_per_trace_median"]) for row in selected],
            "o-",
            label=kind,
            color=color,
        )
    axes[0, 1].set(
        title="Guard/cube complexity", xlabel="compiled cubes", ylabel="s/trace"
    )
    axes[0, 1].legend()

    scan = [row for row in summary if row["panel"] == "scan_phase"]
    dense = {
        (row["formula_id"], int(row["trace_length"]), int(row["batch_size"])): row
        for row in scan
        if row["monitor_id"] == "deepdfa_dense"
    }
    scans = [row for row in scan if row["monitor_id"] == "deepdfa_scan"]
    for (length, batch), marker, color in (
        ((32, 1), "o", "#0072B2"),
        ((128, 1), "s", "#CC79A7"),
        ((32, 32), "^", "#E69F00"),
        ((128, 32), "D", "#009E73"),
    ):
        selected = [
            row
            for row in scans
            if int(row["trace_length"]) == length and int(row["batch_size"]) == batch
        ]
        axes[0, 2].scatter(
            [int(row["dfa_states"]) for row in selected],
            [
                float(
                    dense[(row["formula_id"], length, batch)][
                        "latency_s_per_trace_median"
                    ]
                )
                / float(row["latency_s_per_trace_median"])
                for row in selected
            ],
            marker=marker,
            color=color,
            label=f"L={length}, B={batch}",
        )
    axes[0, 2].axhline(1, color="black", linewidth=1)
    axes[0, 2].set(
        title="Prefix scan phase", xlabel="DFA states", ylabel="sequential / scan"
    )
    axes[0, 2].legend(fontsize=7)

    for axis, panel, title in (
        (axes[1, 0], "linear_state", "Linear state growth"),
        (axes[1, 1], "exponential_state", "Exponential state growth"),
    ):
        selected_panel = [row for row in summary if row["panel"] == panel]
        for monitor_id, color in (
            ("symbolic_guarded", "#0072B2"),
            ("deepdfa_dense", "#D55E00"),
            ("deepdfa_factored", "#009E73"),
        ):
            selected = sorted(
                (row for row in selected_panel if row["monitor_id"] == monitor_id),
                key=lambda row: int(row["dfa_states"]),
            )
            axis.plot(
                [int(row["dfa_states"]) for row in selected],
                [float(row["latency_s_per_trace_median"]) for row in selected],
                "o-",
                label=monitor_id,
                color=color,
            )
        axis.set(
            title=title,
            xlabel="DFA states",
            ylabel="s/trace",
            xscale="log",
            yscale="log",
        )
    axes[1, 0].legend(fontsize=7)
    axes[1, 2].axis("off")
    figure.suptitle("RQ3 structural bottlenecks — controlled CPU candidate")
    figure.savefig(OUTPUT_DIR / "rq3_structural_scaling.png", dpi=180)
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


def generate(repetitions: int, minimum_timed_s: float) -> None:
    run_id = str(uuid.uuid4())
    compilation = []
    runtime = []
    cases = rq3_cases()
    for case in cases:
        structure = characterize_formula(case.benchmark)
        for monitor_id in case.monitor_ids:
            for repetition in range(repetitions):
                print(
                    f"RQ3 {case.panel}: {case.benchmark.formula_id} / "
                    f"{monitor_id} / r{repetition}",
                    flush=True,
                )
                comp, runs = run_rq3_cell(
                    case,
                    monitor_id,
                    trace_seed=SEEDS[repetition],
                    repetition=repetition,
                    run_id=run_id,
                    minimum_timed_s=minimum_timed_s,
                )
                artifact = json.loads(comp.artifact_stats_json)
                row = comp.flat_dict()
                row.update(
                    panel=case.panel,
                    formula_family=case.benchmark.family,
                    parameters_json=json.dumps(
                        dict(case.benchmark.parameters), sort_keys=True
                    ),
                    tree_shape=case.benchmark.tree_shape,
                    n_atoms=structure.n_atoms,
                    ast_depth=structure.ast_depth,
                    dfa_states=artifact.get("dfa_states", artifact.get("n_states")),
                    cube_count=artifact.get("cube_count"),
                    dense_tensor_bytes=artifact.get("dense_tensor_bytes"),
                    cube_tensor_bytes=artifact.get("cube_tensor_bytes"),
                    persistent_bytes=artifact.get(
                        "persistent_tensor_allocated_bytes", 0
                    ),
                    total_rules=artifact.get("total_rules"),
                )
                compilation.append(row)
                runtime.extend(item.flat_dict() for item in runs)
    _validate(compilation, runtime)
    summary = _summary(compilation, runtime)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(OUTPUT_DIR / "rq3_compilation.csv", compilation)
    _write(OUTPUT_DIR / "rq3_runtime.csv", runtime)
    _write(OUTPUT_DIR / "rq3_summary.csv", summary)
    _plot(summary)
    canonical = json.dumps(
        [compilation, runtime], sort_keys=True, separators=(",", ":")
    )
    manifest = {
        "schema_version": RQ3_SCHEMA_VERSION,
        "artifact_kind": "controlled_cpu_run_candidate",
        "artifact_id": hashlib.sha256(canonical.encode()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol": {
            "repetitions": repetitions,
            "trace_seeds": list(SEEDS[:repetitions]),
            "minimum_timed_s": minimum_timed_s,
            "early_termination": False,
            "resource_budgets": asdict(DEFAULT_RESOURCE_BUDGETS),
        },
        "panels": dict(Counter(case.panel for case in cases)),
        "record_counts": {"compilation": len(compilation), "runtime": len(runtime)},
        "status_counts": {
            "compilation": dict(Counter(row["status"] for row in compilation)),
            "runtime": dict(Counter(row["status"] for row in runtime)),
        },
        "provenance": {
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mona": _command(["mona", "-v"]),
        },
        "summary": summary,
    }
    (OUTPUT_DIR / "rq3_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--minimum-timed-s", type=float, default=0.05)
    args = parser.parse_args()
    generate(args.repetitions, args.minimum_timed_s)


if __name__ == "__main__":
    main()
