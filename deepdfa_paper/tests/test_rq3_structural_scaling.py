"""Phase E4 structural-corpus and checked-artifact invariants."""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmarks.formulas import (
    GUARD_COMPLEXITY_SUITE,
    guard_complexity_formula,
)
from src.benchmarks.rq3 import RQ3_SCHEMA_VERSION, rq3_cases, run_rq3_cell
from src.benchmarks.schema import RunStatus, characterize_formula

ROOT = Path(__file__).parent.parent


def test_guard_complexity_suite_has_matched_explicit_strata() -> None:
    assert len(GUARD_COMPLEXITY_SUITE) == 9
    identities = {
        (formula.parameter("guard_kind"), formula.parameter("n_atoms"))
        for formula in GUARD_COMPLEXITY_SUITE
    }
    assert identities == {
        (kind, n) for n in (3, 4, 5) for kind in ("read_once", "threshold", "parity")
    }
    assert guard_complexity_formula("read_once", 4).read_once
    assert not guard_complexity_formula("parity", 4).read_once


def test_rq3_cases_keep_five_structural_panels_separate() -> None:
    cases = rq3_cases()
    assert len(cases) == 29
    assert {case.panel for case in cases} == {
        "tree_shape",
        "guard_complexity",
        "scan_phase",
        "linear_state",
        "exponential_state",
    }
    assert sum(len(case.monitor_ids) for case in cases) == 67
    assert sum(len(case.monitor_ids) * len(case.workloads) for case in cases) == 101


def test_tree_shape_pair_changes_depth_but_not_occurrence_count() -> None:
    cases = [
        case
        for case in rq3_cases()
        if case.panel == "tree_shape" and case.benchmark.parameter("n_atoms") == 16
    ]
    structures = {
        case.benchmark.tree_shape: characterize_formula(case.benchmark)
        for case in cases
    }
    assert structures["balanced"].ast_nodes == structures["left_deep"].ast_nodes
    assert structures["balanced"].ast_depth < structures["left_deep"].ast_depth


def test_real_rq3_worker_emits_compilation_artifact_and_runtime() -> None:
    case = next(case for case in rq3_cases() if case.panel == "guard_complexity")
    compilation, runtime = run_rq3_cell(
        case,
        "deepdfa_factored",
        trace_seed=1201,
        repetition=0,
        minimum_timed_s=0.005,
    )
    artifact = json.loads(compilation.artifact_stats_json)
    assert compilation.status is RunStatus.SUCCESS
    assert artifact["cube_count"] > 0
    assert len(runtime) == 1
    assert runtime[0].schema_version == RQ3_SCHEMA_VERSION
    assert runtime[0].status is RunStatus.SUCCESS


def test_checked_rq3_artifact_is_complete_and_model_validated() -> None:
    manifest = json.loads((ROOT / "results/rq3/rq3_manifest.json").read_text())
    assert manifest["schema_version"] == RQ3_SCHEMA_VERSION
    assert manifest["record_counts"] == {"compilation": 335, "runtime": 505}
    assert manifest["status_counts"] == {
        "compilation": {"success": 335},
        "runtime": {"success": 505},
    }
