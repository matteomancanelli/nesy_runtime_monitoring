"""Phase E2 fresh-process compilation and resource-accounting invariants."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from src.benchmarks.e2 import (
    E2_SCHEMA_VERSION,
    MONITOR_SPECS,
    E2CellRecord,
    run_rq1_cell,
    supervise_command,
)
from src.benchmarks.schema import RESULT_SCHEMA_VERSION, ResourceBudgets, RunStatus

ROOT = Path(__file__).parent.parent
SMOKE_JSON = ROOT / "results" / "e2" / "e2_instrumentation_smoke.json"


def _record(**overrides) -> E2CellRecord:
    values = dict(
        schema_version=E2_SCHEMA_VERSION,
        benchmark_schema_version=RESULT_SCHEMA_VERSION,
        run_id="run",
        cell_id="cell",
        formula_id="atomic_control",
        formula="a",
        monitor_id="symbolic_guarded",
        monitor_name="SymbolicDFAMonitor",
        architecture="symbolic_dfa",
        construction="canonical_dfa",
        backend="guarded_edges",
        requested_device="cpu",
        effective_device="cpu",
        repetition=0,
        worker_pid=123,
        status=RunStatus.SUCCESS,
        compile_total_s=0.1,
    )
    values.update(overrides)
    return E2CellRecord(**values)


def test_monitor_registry_is_stable_and_covers_every_family() -> None:
    ids = [spec.monitor_id for spec in MONITOR_SPECS]
    assert len(ids) == len(set(ids)) == 12
    assert {spec.architecture for spec in MONITOR_SPECS} == {
        "symbolic_dfa",
        "rulerunner",
        "deepdfa",
    }
    assert {spec.construction for spec in MONITOR_SPECS} >= {
        "original",
        "bounded_default",
        "bounded_exact_online",
        "progression",
        "fixed_dfa",
    }


def test_e2_schema_preserves_all_terminal_statuses() -> None:
    assert _record().flat_dict()["status"] == "success"
    assert _record(status=RunStatus.FALLBACK).flat_dict()["status"] == "fallback"
    for status in (
        RunStatus.TIMEOUT,
        RunStatus.OOM,
        RunStatus.UNSUPPORTED,
        RunStatus.ERROR,
    ):
        row = _record(status=status, compile_total_s=None)
        assert row.flat_dict()["status"] == status.value
    with pytest.raises(ValueError, match="requires compile_total_s"):
        _record(compile_total_s=None)
    with pytest.raises(ValueError, match="valid JSON"):
        _record(artifact_stats_json="not-json")


def test_parent_supervisor_records_timeout_and_host_oom() -> None:
    timeout = supervise_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=0.05,
        memory_limit_bytes=None,
    )
    assert timeout.terminal_status is RunStatus.TIMEOUT
    assert "wall-time budget" in timeout.failure_reason

    oom = supervise_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=2.0,
        memory_limit_bytes=1,
    )
    assert oom.terminal_status is RunStatus.OOM
    assert "host RSS budget" in oom.failure_reason


def test_parent_supervisor_delimits_runtime_rss(tmp_path: Path) -> None:
    phase = tmp_path / "runtime.phase"
    phase.write_text("runtime")
    outcome = supervise_command(
        [
            sys.executable,
            "-c",
            "import time; x = bytearray(1000000); time.sleep(0.05); print(len(x))",
        ],
        timeout_s=2.0,
        memory_limit_bytes=None,
        phase_path=phase,
    )
    assert outcome.returncode == 0
    assert outcome.peak_runtime_rss_bytes > 0
    assert outcome.peak_compile_rss_bytes == 0


def test_parent_supervisor_snapshots_python_environment() -> None:
    """C-only environment mutations must not leak into fresh E2 workers."""
    import ctypes

    variable = b"E2_C_ONLY_ENV_TEST"
    libc = ctypes.CDLL(None)
    assert libc.setenv(variable, b"must-not-leak", 1) == 0
    try:
        assert "E2_C_ONLY_ENV_TEST" not in os.environ
        outcome = supervise_command(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('E2_C_ONLY_ENV_TEST', 'clean'))",
            ],
            timeout_s=2.0,
            memory_limit_bytes=None,
        )
        assert outcome.returncode == 0
        assert outcome.stdout.strip() == "clean"
    finally:
        libc.unsetenv(variable)


def test_frozen_rq1_gate_rejects_unsafe_cell_without_compiling() -> None:
    record = run_rq1_cell("eventual_next_alias", "rulerunner_original_flat")
    assert record.status is RunStatus.UNSUPPORTED
    assert record.worker_pid is None
    assert record.compile_total_s is None
    assert "rq1.v1" in record.failure_reason


def test_cell_supervision_emits_timeout_and_oom_rows() -> None:
    timeout = run_rq1_cell(
        "atomic_control",
        "symbolic_guarded",
        budgets=ResourceBudgets(cold_compile_timeout_s=0.001),
    )
    assert timeout.status is RunStatus.TIMEOUT
    assert timeout.failure_stage == "worker_supervision"

    oom = run_rq1_cell(
        "atomic_control",
        "symbolic_guarded",
        budgets=ResourceBudgets(max_host_memory_fraction=1e-9),
    )
    assert oom.status is RunStatus.OOM
    assert oom.failure_stage == "worker_supervision"


def test_symbolic_cell_is_a_real_fresh_process() -> None:
    record = run_rq1_cell("atomic_control", "symbolic_guarded")
    stages = json.loads(record.compile_stages_json)
    artifact = json.loads(record.artifact_stats_json)

    assert record.status is RunStatus.SUCCESS
    assert record.worker_pid is not None
    assert record.worker_pid != 0
    assert record.compile_total_s is not None
    assert set(stages) == {"ltlf_to_minimal_dfa", "monitor_initialization"}
    assert artifact["dfa_states"] == 3
    assert artifact["persistent_tensor_allocated_bytes"] == 0
    assert record.startup_host_rss_bytes is not None
    assert record.baseline_host_rss_bytes is not None
    assert record.peak_compile_host_rss_bytes is not None
    assert record.startup_host_rss_bytes < record.baseline_host_rss_bytes
    assert record.peak_compile_host_rss_bytes >= record.baseline_host_rss_bytes
    assert record.peak_observed_host_rss_bytes >= record.peak_compile_host_rss_bytes


def test_cuda_measurement_or_unavailability_is_explicit() -> None:
    import torch

    record = run_rq1_cell(
        "atomic_control",
        "deepdfa_dense",
        requested_device="cuda",
    )
    if torch.cuda.is_available():
        assert record.status is RunStatus.SUCCESS
        assert record.effective_device == "cuda"
        assert record.baseline_gpu_allocated_bytes is not None
        assert record.peak_compile_gpu_allocated_bytes is not None
        assert record.peak_compile_gpu_reserved_bytes is not None
    else:
        assert record.status is RunStatus.UNSUPPORTED
        assert record.failure_stage == "device_validation"
        assert "CUDA" in record.failure_reason


def test_checked_in_smoke_covers_all_native_stage_and_artifact_adapters() -> None:
    artifact = json.loads(SMOKE_JSON.read_text())
    rows = artifact["records"]
    assert artifact["schema_version"] == E2_SCHEMA_VERSION
    assert artifact["artifact_kind"] == "instrumentation_smoke_not_paper_results"
    assert len(rows) == len(MONITOR_SPECS)
    assert {row["monitor_id"] for row in rows} == {
        spec.monitor_id for spec in MONITOR_SPECS
    }
    assert {row["status"] for row in rows} == {"success"}
    assert len({row["worker_pid"] for row in rows}) == len(rows)

    by_id = {row["monitor_id"]: row for row in rows}
    required = {
        "symbolic_guarded": ("ltlf_to_minimal_dfa", "dfa_states"),
        "rulerunner_original_flat": ("rules_and_cilp_lowering", "total_rules"),
        "rulerunner_original_structured": (
            "rules_and_structured_cilp_lowering",
            "modules",
        ),
        "rulerunner_bounded_flat_default": (
            "bounded_pipeline_lowering",
            "bounded_horizon",
        ),
        "rulerunner_bounded_flat_exact": (
            "bounded_exact_online_head",
            "exact_head_states",
        ),
        "rulerunner_bounded_structured_default": (
            "bounded_skeleton_lowering",
            "event_modules",
        ),
        "rulerunner_bounded_structured_exact": (
            "bounded_exact_online_head",
            "exact_head_states",
        ),
        "rulerunner_progression_flat": (
            "progression_residual_graph",
            "progression_states",
        ),
        "rulerunner_progression_structured": (
            "progression_factorized_graph",
            "progression_states",
        ),
        "deepdfa_dense": ("dfa_tensor_lowering", "dense_tensor_bytes"),
        "deepdfa_factored": ("dfa_tensor_lowering", "cube_tensor_bytes"),
        "deepdfa_scan": ("dfa_tensor_lowering", "dense_tensor_bytes"),
    }
    for monitor_id, (stage_name, statistic_name) in required.items():
        row = by_id[monitor_id]
        stages = json.loads(row["compile_stages_json"])
        stats = json.loads(row["artifact_stats_json"])
        assert stage_name in stages
        assert statistic_name in stats
        assert row["peak_observed_host_rss_bytes"] >= row["peak_compile_host_rss_bytes"]
