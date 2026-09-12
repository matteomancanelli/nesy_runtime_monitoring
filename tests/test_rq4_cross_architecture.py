"""Phase E5 cross-architecture protocol and checked-artifact invariants."""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmarks.rq4 import (
    RQ4_CASES,
    RQ4_MODES,
    RQ4_MONITOR_IDS,
    RQ4_SCHEMA_VERSION,
    run_rq4_cell,
)
from src.benchmarks.schema import RunStatus

ROOT = Path(__file__).parent.parent


def test_rq4_matrix_separates_modes_boundaries_and_label_modes() -> None:
    assert len(RQ4_CASES) == 3
    assert {mode.workload_mode for mode in RQ4_MODES} == {
        "capacity",
        "deployment",
    }
    assert {mode.timing_boundary for mode in RQ4_MODES} == {
        "end_to_end",
        "core_native",
    }
    assert "rulerunner_bounded_flat_default" in RQ4_MONITOR_IDS
    assert "rulerunner_bounded_flat_exact" in RQ4_MONITOR_IDS
    assert "deepdfa_dense" in RQ4_MONITOR_IDS
    assert "deepdfa_factored" in RQ4_MONITOR_IDS
    assert len(RQ4_MONITOR_IDS) == len(set(RQ4_MONITOR_IDS)) == 7


def test_real_rq4_worker_preserves_processed_cell_accounting() -> None:
    compilation, runtime = run_rq4_cell(
        "atomic_control",
        "symbolic_guarded",
        "deployment_end_to_end",
        trace_seed=0,
        repetition=0,
        measurement_blocks=2,
        minimum_timed_s=0.002,
        maximum_inner_iterations=32,
    )
    assert compilation.status is RunStatus.SUCCESS
    assert len(runtime) == 2
    assert all(row.status is RunStatus.SUCCESS for row in runtime)
    assert all(row.processed_cells < row.offered_cells for row in runtime)
    assert all(row.schema_version == RQ4_SCHEMA_VERSION for row in runtime)
    assert all(len(json.loads(row.timing_samples_json)) == 2 for row in runtime)


def test_cuda_unavailability_produces_runtime_rows_for_every_attempt() -> None:
    import torch

    if torch.cuda.is_available():
        return
    compilation, runtime = run_rq4_cell(
        "atomic_control",
        "deepdfa_dense",
        "capacity_core_native",
        trace_seed=0,
        repetition=0,
        requested_device="cuda",
        minimum_timed_s=0.001,
    )
    assert compilation.status is RunStatus.UNSUPPORTED
    assert len(runtime) == 3
    assert {row.status for row in runtime} == {RunStatus.UNSUPPORTED}
    assert all("CUDA" in row.failure_reason for row in runtime)


def test_checked_rq4_artifact_is_explicit_about_accelerator_status() -> None:
    path = ROOT / "results/rq4/rq4_manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == RQ4_SCHEMA_VERSION
    assert manifest["record_counts"] == {
        "absolute_summary": 192,
        "compilation": 630,
        "pareto": 192,
        "runtime": 1680,
        "symbolic_comparisons": 192,
    }
    assert manifest["status_counts"] == {
        "compilation": {"success": 360, "unsupported": 270},
        "runtime": {"success": 960, "unsupported": 720},
    }
    assert manifest["cpu_block"]["controlled"] is True
    assert manifest["cpu_block"]["compilation_rows"] == 315
    assert manifest["cpu_block"]["runtime_rows"] == 840
    assert manifest["cuda_block"]["status"] == "unsupported"
    assert manifest["cuda_block"]["effective_cuda_success_rows"] == 0
