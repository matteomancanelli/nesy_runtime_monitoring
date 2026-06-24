"""Phase 0.1 — the early_termination flag must not change verdicts.

`run(..., early_termination=False)` / `batch_run(..., early_termination=False)`
force every monitor to process all cells of every trace, so the per-cell-cost
figures (Exp 2/3) compare the same workload across paradigms instead of timing
how fast a crisp monitor gives up against a full batched pass (the
early-termination confound; CLAUDE.md Phase 0.1).

Forcing the full pass must be *verdict-preserving*: absorbing SATISFY/VIOLATE
states are sticky, so the first decided verdict is the same whether or not we
keep stepping afterwards. These tests pin that invariant for all three
paradigms on the IJCNN family, which early-terminates almost immediately on
random traces (the exact case the confound concerns).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.formulas import IJCNN_SUITE
from src.benchmarks.runner import random_traces, reset_if_stale, time_monitor
from src.monitors.deep_dfa import DeepDFAMonitor
from src.monitors.rulerunner import RuleRunnerMonitor
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

MONITORS = [SymbolicDFAMonitor, RuleRunnerMonitor, DeepDFAMonitor]


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_batch_run_modes_agree(monitor_cls):
    """early_termination=False yields the same verdicts as the default."""
    formula = IJCNN_SUITE[2]  # ijcnn_n8 — early-terminates on random traces
    traces = random_traces(formula.atoms, 50, 40, np.random.default_rng(0))
    monitor = monitor_cls.compile(formula.formula)

    eager = monitor.batch_run(traces, early_termination=True)
    full = monitor.batch_run(traces, early_termination=False)

    assert eager == full
    # sanity: the family really does decide these traces (else the test is vacuous)
    assert any(v.name != "UNDECIDED" for v in eager)


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_run_modes_agree(monitor_cls):
    formula = IJCNN_SUITE[1]
    traces = random_traces(formula.atoms, 30, 20, np.random.default_rng(1))
    monitor = monitor_cls.compile(formula.formula)
    for t in traces:
        assert monitor.run(t, early_termination=True) == monitor.run(
            t, early_termination=False
        )


def test_forced_full_pass_costs_more_for_symbolic():
    """The whole point of the flag: with early termination off the crisp walk
    actually processes all cells, so its measured per-cell cost is the real
    dict-lookup cost rather than the physically-impossible give-up artifact."""
    formula = IJCNN_SUITE[2]
    eager = time_monitor(
        SymbolicDFAMonitor, formula, trace_length=500, n_traces=20,
        n_repeats=3, n_warmup=1, early_termination=True,
    )
    full = time_monitor(
        SymbolicDFAMonitor, formula, trace_length=500, n_traces=20,
        n_repeats=3, n_warmup=1, early_termination=False,
    )
    assert full.early_termination is False
    assert full.mean_s_per_cell > eager.mean_s_per_cell


def test_reset_if_stale_drops_mismatched_mode(tmp_path):
    """A CSV measured in one mode is discarded when the other mode is requested
    (the two are different workloads and must not be resumed/mixed)."""
    import pandas as pd

    csv = tmp_path / "exp.csv"
    pd.DataFrame([{"monitor_name": "X", "early_termination": True}]).to_csv(
        csv, index=False
    )

    reset_if_stale(csv, early_termination=True)   # same mode -> kept
    assert csv.exists()

    reset_if_stale(csv, early_termination=False)  # other mode -> dropped
    assert not csv.exists()


def test_reset_if_stale_treats_legacy_csv_as_early_term(tmp_path):
    """Pre-Phase-0.1 CSVs lack the column; they were the confounded
    early-terminating measurement, so a full-pass run must discard them."""
    import pandas as pd

    csv = tmp_path / "legacy.csv"
    pd.DataFrame([{"monitor_name": "X", "mean_s_per_cell": 1e-10}]).to_csv(
        csv, index=False
    )
    reset_if_stale(csv, early_termination=False)
    assert not csv.exists()
