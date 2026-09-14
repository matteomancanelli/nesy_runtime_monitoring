"""Minimal fresh-process entry point for :mod:`src.benchmarks.e2`."""

from __future__ import annotations

import json
import resource
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _current_rss_bytes() -> int:
    try:
        rows = Path("/proc/self/status").read_text().splitlines()
        value = next(row for row in rows if row.startswith("VmRSS:"))
        return int(value.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return 0


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    startup_rss = _current_rss_bytes()
    startup_peak = _peak_rss_bytes()
    request_payload = json.loads(Path(sys.argv[1]).read_text())

    from src.benchmarks.e2 import WorkerRequest, execute_worker_request

    allowed = {field.name for field in fields(WorkerRequest)}
    request = WorkerRequest(
        **{key: value for key, value in request_payload.items() if key in allowed}
    )
    record = execute_worker_request(
        request,
        startup_host_rss_bytes=startup_rss,
        startup_peak_host_rss_bytes=startup_peak,
    )
    print(json.dumps(record.flat_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
