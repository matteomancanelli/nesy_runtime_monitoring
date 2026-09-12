"""Generate the frozen RQ1 semantic-boundary artifact.

Run from the repository root with:

    python experiments/rq1_semantic_characterization.py

The output is semantic evidence, not a timing result.  Every formula is checked
exactly against the canonical DFA for original RuleRunner, bounded default,
bounded exact-online, and progression RuleRunner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.benchmarks.rq1 import (
    RQ1_CONSTRUCTIONS,
    RQ1_CORPUS,
    RQ1_SCHEMA_VERSION,
    CharacterizationStatus,
    characterize_corpus,
    corpus_category_counts,
)
from src.benchmarks.schema import DEFAULT_RESOURCE_BUDGETS, RESULT_SCHEMA_VERSION

DEFAULT_OUTPUT_DIR = ROOT / "results" / "rq1"
CSV_NAME = "rq1_characterization.csv"
JSON_NAME = "rq1_characterization.json"


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else "unavailable"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _semantic_gate(rows: list[dict[str, Any]]) -> None:
    """Fail artifact generation if a complete construction lost correctness."""
    bad_statuses = {
        CharacterizationStatus.ERROR.value,
        CharacterizationStatus.RESOURCE_LIMITED.value,
    }
    failures = [
        row
        for row in rows
        if row["status"] in bad_statuses
        or (
            row["construction"] == "progression"
            and (
                not row["final_language_equivalent"]
                or not row["prefix_sound"]
                or not row["exact_online_equivalent"]
            )
        )
        or (
            row["construction"] == "bounded_exact_online"
            and row["applicable"]
            and (
                not row["final_language_equivalent"]
                or not row["prefix_sound"]
                or not row["exact_online_equivalent"]
            )
        )
        or (
            row["construction"] == "bounded_default"
            and row["applicable"]
            and (not row["final_language_equivalent"] or not row["prefix_sound"])
        )
    ]
    if failures:
        identities = ", ".join(
            f"{row['formula_id']}:{row['construction']}" for row in failures
        )
        raise RuntimeError(f"RQ1 semantic gate failed for {identities}")


def _construction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for construction in RQ1_CONSTRUCTIONS:
        selected = [row for row in rows if row["construction"] == construction]
        summary[construction] = {
            "rows": len(selected),
            "applicable": sum(bool(row["applicable"]) for row in selected),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in selected).items())
            ),
            "final_language_equivalent": sum(
                row["final_language_equivalent"] is True for row in selected
            ),
            "prefix_sound": sum(row["prefix_sound"] is True for row in selected),
            "exact_online_equivalent": sum(
                row["exact_online_equivalent"] is True for row in selected
            ),
        }
    return summary


def _category_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = {
        row["formula_id"]: row for row in rows if row["construction"] == "original"
    }
    bounded = {
        row["formula_id"]: row
        for row in rows
        if row["construction"] == "bounded_default"
    }
    summary: dict[str, Any] = {}
    for category in corpus_category_counts():
        cases = [case for case in RQ1_CORPUS if category in case.categories]
        ids = [case.benchmark.formula_id for case in cases]
        summary[category] = {
            "formulas": len(ids),
            "original_language_equivalent": sum(
                original[formula_id]["final_language_equivalent"] is True
                for formula_id in ids
            ),
            "bounded_admitted": sum(
                bool(bounded[formula_id]["applicable"]) for formula_id in ids
            ),
        }
    return summary


def _manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_records = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    artifact_id = hashlib.sha256(canonical_records.encode()).hexdigest()
    return {
        "schema_version": RQ1_SCHEMA_VERSION,
        "benchmark_schema_version": RESULT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "experiments/rq1_semantic_characterization.py",
        "formula_count": len(RQ1_CORPUS),
        "record_count": len(rows),
        "constructions": list(RQ1_CONSTRUCTIONS),
        "resource_budgets": asdict(DEFAULT_RESOURCE_BUDGETS),
        "category_counts": corpus_category_counts(),
        "construction_summary": _construction_summary(rows),
        "category_summary": _category_summary(rows),
        "provenance": {
            "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ltlf2dfa": _package_version("ltlf2dfa"),
            "mona": _command_output(["mona", "-v"]),
        },
        "records": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(output_dir: Path) -> tuple[Path, Path]:
    records = characterize_corpus()
    rows = [record.flat_dict() for record in records]
    _semantic_gate(rows)
    manifest = _manifest(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_NAME
    json_path = output_dir / JSON_NAME
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


def check(output_dir: Path) -> None:
    json_path = output_dir / JSON_NAME
    if not json_path.exists():
        raise FileNotFoundError(f"missing frozen artifact: {json_path}")
    expected = json.loads(json_path.read_text())
    rows = [record.flat_dict() for record in characterize_corpus()]
    _semantic_gate(rows)
    if expected.get("schema_version") != RQ1_SCHEMA_VERSION:
        raise RuntimeError("frozen RQ1 schema version does not match the generator")
    if expected.get("records") != rows:
        raise RuntimeError("frozen RQ1 records are stale; regenerate the artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the frozen JSON records without rewriting the artifact",
    )
    args = parser.parse_args()

    if args.check:
        check(args.output_dir)
        print(f"RQ1 artifact is current: {args.output_dir / JSON_NAME}")
        return

    csv_path, json_path = generate(args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
