"""Fresh-process exact certificate-generation timer for Phase E3."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    phase_path = Path(request["phase_path"])

    from src.monitors.rulerunner.bounded import eventize_bounded_islands
    from src.monitors.rulerunner.equivalence import certify_rule_runner

    phase_path.write_text("compile")
    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    eventized = eventize_bounded_islands(request["formula"])
    eventization_s = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    certificate = certify_rule_runner(eventized.skeleton)
    certificate_s = time.perf_counter() - stage_start
    total_s = time.perf_counter() - total_start
    phase_path.write_text("done")
    print(
        json.dumps(
            {
                "schema_version": "rq2.v1",
                "run_id": request["run_id"],
                "formula_id": request["formula_id"],
                "repetition": request["repetition"],
                "status": "success",
                "eventization_s": eventization_s,
                "certificate_generation_s": certificate_s,
                "total_s": total_s,
                "bounded_horizon": eventized.horizon,
                "certificate_product_states": certificate.explored_product_states,
                "language_equivalent": certificate.language_equivalent,
                "prefix_sound": certificate.prefix_sound,
                "worker_peak_host_rss_bytes": _peak_rss_bytes(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
