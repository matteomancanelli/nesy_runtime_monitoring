"""Validate E2 cold-compilation instrumentation across every monitor family.

This is a local infrastructure smoke test, not a paper timing sweep. Each row
is produced by a distinct fresh worker and is semantically admitted through the
frozen RQ1 artifact before compilation.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks.e2 import (  # noqa: E402
    E2_SCHEMA_VERSION,
    E2CellRecord,
    run_rq1_cell,
)
from src.benchmarks.schema import (  # noqa: E402
    DEFAULT_RESOURCE_BUDGETS,
    RESULT_SCHEMA_VERSION,
)

OUTPUT_DIR = ROOT / "results" / "e2"
CSV_PATH = OUTPUT_DIR / "e2_instrumentation_smoke.csv"
JSON_PATH = OUTPUT_DIR / "e2_instrumentation_smoke.json"

SMOKE_CELLS: tuple[tuple[str, str], ...] = (
    ("ijcnn_balanced_n4", "symbolic_guarded"),
    ("ijcnn_balanced_n4", "rulerunner_original_flat"),
    ("ijcnn_balanced_n4", "rulerunner_original_structured"),
    ("eventual_next_alias", "rulerunner_bounded_flat_default"),
    ("eventual_next_alias", "rulerunner_bounded_flat_exact"),
    ("eventual_next_alias", "rulerunner_bounded_structured_default"),
    ("eventual_next_alias", "rulerunner_bounded_structured_exact"),
    ("eventual_next_alias", "rulerunner_progression_flat"),
    ("eventual_next_alias", "rulerunner_progression_structured"),
    ("ijcnn_balanced_n4", "deepdfa_dense"),
    ("ijcnn_balanced_n4", "deepdfa_factored"),
    ("ijcnn_balanced_n4", "deepdfa_scan"),
)


def _validate(records: list[E2CellRecord]) -> None:
    failures = [record for record in records if record.status.value != "success"]
    if failures:
        details = ", ".join(
            f"{record.cell_id}={record.status.value}:{record.failure_reason}"
            for record in failures
        )
        raise RuntimeError(f"E2 smoke failed: {details}")
    worker_pids = [record.worker_pid for record in records]
    if None in worker_pids or len(worker_pids) != len(set(worker_pids)):
        raise RuntimeError("E2 cells did not use distinct fresh workers")
    for record in records:
        stages = json.loads(record.compile_stages_json)
        artifact = json.loads(record.artifact_stats_json)
        if not stages or not artifact:
            raise RuntimeError(f"missing E2 instrumentation for {record.cell_id}")
        if record.peak_observed_host_rss_bytes is None:
            raise RuntimeError(f"missing peak RSS for {record.cell_id}")


def main() -> None:
    run_id = str(uuid.uuid4())
    records: list[E2CellRecord] = []
    for formula_id, monitor_id in SMOKE_CELLS:
        print(f"E2 smoke: {formula_id} / {monitor_id}", flush=True)
        records.append(
            run_rq1_cell(
                formula_id,
                monitor_id,
                run_id=run_id,
                validation_trace_length=4,
                validation_batch_size=2,
            )
        )
    _validate(records)

    rows = [record.flat_dict() for record in records]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": E2_SCHEMA_VERSION,
        "benchmark_schema_version": RESULT_SCHEMA_VERSION,
        "artifact_kind": "instrumentation_smoke_not_paper_results",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "platform": platform.platform(),
        "resource_budgets": asdict(DEFAULT_RESOURCE_BUDGETS),
        "status_counts": dict(
            sorted(Counter(record.status.value for record in records).items())
        ),
        "records": rows,
    }
    JSON_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
