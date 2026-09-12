"""Phase E3 corpus, isolation, and decision-lag invariants."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.benchmarks.rq2 import (
    RQ2_CASES,
    RQ2_MONITOR_IDS,
    RQ2_SCHEMA_VERSION,
    applicable_monitor_ids,
    characterize_decision_lag,
    run_certificate_generation,
    run_rq2_cell,
)
from src.benchmarks.schema import RunStatus

ROOT = Path(__file__).parent.parent


def test_legacy_experiments_and_results_are_isolated() -> None:
    assert (ROOT / "old/experiments/exp1_single_trace.py").exists()
    assert (ROOT / "old/results/cpu/exp1_single_trace.csv").exists()
    assert not (ROOT / "experiments/exp1_single_trace.py").exists()
    assert not (ROOT / "results/cpu").exists()


def test_rq2_corpus_has_explicit_semantic_strata() -> None:
    assert len(RQ2_CASES) == 7
    assert {case.stratum for case in RQ2_CASES} == {
        "all_applicable",
        "bounded_repair",
        "bounded_repair_late_default",
        "progression_only",
    }
    assert len(RQ2_MONITOR_IDS) == 8
    assert len(applicable_monitor_ids("nested_next_control")) == 8
    assert len(applicable_monitor_ids("next_offset_alias")) == 6
    assert len(applicable_monitor_ids("response")) == 2


def test_rq2_worker_measures_paired_batches_in_one_fresh_process() -> None:
    compilation, runtime = run_rq2_cell(
        "nested_next_control",
        "rulerunner_original_flat",
        trace_seed=1103,
        repetition=0,
        trace_length=8,
        batch_sizes=(1, 4),
        minimum_timed_s=0.005,
    )
    assert compilation.status is RunStatus.SUCCESS
    assert compilation.worker_pid is not None
    assert json.loads(compilation.compile_stages_json)
    assert json.loads(compilation.artifact_stats_json)["total_rules"] > 0
    assert [row.batch_size for row in runtime] == [1, 4]
    assert all(row.status is RunStatus.SUCCESS for row in runtime)
    assert all(row.duration_s_per_trace > 0 for row in runtime)


def test_rq2_gate_excludes_semantically_invalid_original() -> None:
    compilation, runtime = run_rq2_cell(
        "next_offset_alias",
        "rulerunner_original_flat",
        trace_seed=1103,
        repetition=0,
        trace_length=4,
        batch_sizes=(1,),
    )
    assert compilation.status is RunStatus.UNSUPPORTED
    assert compilation.worker_pid is None
    assert runtime[0].status is RunStatus.UNSUPPORTED


def test_exact_head_removes_exhaustive_decision_lag() -> None:
    rows = characterize_decision_lag("next_offset_alias", trace_length=3)
    default = [row for row in rows if row.online_label_mode == "sound_default"]
    exact = [row for row in rows if row.online_label_mode == "exact_online"]
    assert len(default) == len(exact) == 8
    assert any(row.decision_lag_cells > 0 for row in default)
    assert all(row.decision_lag_cells == 0 for row in exact)


def test_certificate_generation_is_a_separate_fresh_process_measurement() -> None:
    row = run_certificate_generation(
        "next_offset_alias", repetition=0, run_id=str(uuid.uuid4())
    )
    assert row["schema_version"] == RQ2_SCHEMA_VERSION
    assert row["status"] == "success"
    assert row["certificate_generation_s"] > 0
    assert row["certificate_product_states"] > 0
    assert row["peak_compile_host_rss_bytes"] > 0
