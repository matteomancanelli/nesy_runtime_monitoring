"""Phase E1: frozen semantic-boundary corpus and artifact checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.benchmarks.rq1 import (
    RQ1_CONSTRUCTIONS,
    RQ1_CORPUS,
    RQ1_SCHEMA_VERSION,
    CharacterizationStatus,
    RQ1Record,
    characterize_corpus,
    corpus_category_counts,
)

ROOT = Path(__file__).parent.parent
FROZEN_JSON = ROOT / "results" / "rq1" / "rq1_characterization.json"


@pytest.fixture(scope="module")
def rq1_records() -> tuple[RQ1Record, ...]:
    return characterize_corpus()


def test_corpus_has_stable_unique_identity_and_required_strata() -> None:
    ids = [case.benchmark.formula_id for case in RQ1_CORPUS]
    formulas = [case.benchmark.formula for case in RQ1_CORPUS]
    counts = corpus_category_counts()

    assert len(RQ1_CORPUS) == 17
    assert len(ids) == len(set(ids))
    assert len(formulas) == len(set(formulas))
    assert counts["original_safe_control"] == 9
    assert counts["shared_register_counterexample"] == 8
    assert counts["bounded_repair_target"] == 6
    assert counts["bounded_rejection"] == 2
    assert counts["declare"] == 7


def test_each_case_has_one_row_per_construction(
    rq1_records: tuple[RQ1Record, ...],
) -> None:
    assert len(rq1_records) == len(RQ1_CORPUS) * len(RQ1_CONSTRUCTIONS)
    for offset in range(0, len(rq1_records), len(RQ1_CONSTRUCTIONS)):
        block = rq1_records[offset : offset + len(RQ1_CONSTRUCTIONS)]
        assert tuple(row.construction for row in block) == RQ1_CONSTRUCTIONS
        assert len({row.formula_id for row in block}) == 1
        assert all(row.schema_version == RQ1_SCHEMA_VERSION for row in block)


def test_original_boundary_and_shortest_witnesses_are_frozen(
    rq1_records: tuple[RQ1Record, ...],
) -> None:
    original = {
        row.formula_id: row for row in rq1_records if row.construction == "original"
    }
    safe = {formula_id for formula_id, row in original.items() if row.applicable}
    unsafe = set(original) - safe

    assert safe == {
        "atomic_control",
        "nested_next_control",
        "nested_eventually_control",
        "immediate_response_control",
        "flat_until_control",
        "ijcnn_balanced_n4",
        "precedence",
        "resp_existence",
        "not_coexistence",
    }
    assert unsafe == {
        "next_offset_alias",
        "globally_next_alias",
        "eventual_next_alias",
        "until_next_alias",
        "response",
        "chain_response",
        "alt_response",
        "chain_precedence",
    }
    assert all(original[formula_id].language_witness_json for formula_id in unsafe)
    assert all(original[formula_id].certificate_complete for formula_id in original)


def test_bounded_coverage_and_progression_completeness(
    rq1_records: tuple[RQ1Record, ...],
) -> None:
    bounded = {
        row.formula_id: row
        for row in rq1_records
        if row.construction == "bounded_default"
    }
    exact = {
        row.formula_id: row
        for row in rq1_records
        if row.construction == "bounded_exact_online"
    }
    progression = [row for row in rq1_records if row.construction == "progression"]

    rejected = {formula_id for formula_id, row in bounded.items() if not row.applicable}
    assert rejected == {"response", "alt_response"}
    assert all(
        bounded[formula_id].status is CharacterizationStatus.REJECTED
        for formula_id in rejected
    )
    assert all(
        row.final_language_equivalent and row.prefix_sound
        for row in bounded.values()
        if row.applicable
    )
    assert all(row.exact_online_equivalent for row in exact.values() if row.applicable)
    assert all(
        row.applicable
        and row.final_language_equivalent
        and row.prefix_sound
        and row.exact_online_equivalent
        for row in progression
    )

    default_late = {
        formula_id
        for formula_id, row in bounded.items()
        if row.applicable and not row.exact_online_equivalent
    }
    assert default_late == {
        "next_offset_alias",
        "globally_next_alias",
        "until_next_alias",
    }
    assert all(bounded[formula_id].online_witness_json for formula_id in default_late)


def test_no_characterization_silently_hits_a_resource_wall(
    rq1_records: tuple[RQ1Record, ...],
) -> None:
    counts = Counter(row.status for row in rq1_records)
    assert counts == {
        CharacterizationStatus.CHARACTERIZED: 64,
        CharacterizationStatus.REJECTED: 4,
    }


def test_checked_in_json_matches_exact_regeneration(
    rq1_records: tuple[RQ1Record, ...],
) -> None:
    artifact = json.loads(FROZEN_JSON.read_text())
    assert artifact["schema_version"] == RQ1_SCHEMA_VERSION
    assert artifact["formula_count"] == len(RQ1_CORPUS)
    assert artifact["record_count"] == len(rq1_records)
    assert artifact["records"] == [row.flat_dict() for row in rq1_records]
