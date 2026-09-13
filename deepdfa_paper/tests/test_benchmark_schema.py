"""Phase E0: benchmark identity, structure, and result-schema invariants."""

from __future__ import annotations

import math
from collections import deque
from itertools import product

import pandas as pd
import pytest

from src.benchmarks.formulas import (
    IJCNN_ATOM_COUNTS,
    IJCNN_LEFTDEEP_SUITE,
    IJCNN_SUITE,
    bounded_response,
    ijcnn_formula,
    kth_from_last,
)
from src.benchmarks.legacy import load_legacy_timing
from src.benchmarks.runner import time_monitor
from src.benchmarks.schema import (
    DEFAULT_RESOURCE_BUDGETS,
    RESULT_SCHEMA_VERSION,
    RawResultRecord,
    ResourceBudgets,
    RunStatus,
    characterize_formula,
)
from src.formula.compiler import DFA, compile_ltlf


def _assert_language_equivalent(left: DFA, right: DFA) -> None:
    """Exact reachable-product check for the small IJCNN test instances."""
    atoms = tuple(sorted(set(left.atoms) | set(right.atoms)))
    symbols = [
        {atom: value for atom, value in zip(atoms, values)}
        for values in product((False, True), repeat=len(atoms))
    ]
    initial = (left.initial, right.initial)
    queue = deque([initial])
    seen = {initial}
    while queue:
        l_state, r_state = queue.popleft()
        assert (l_state in left.accepting) == (r_state in right.accepting)
        for symbol in symbols:
            successor = (left.step(l_state, symbol), right.step(r_state, symbol))
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)


def test_ijcnn_registry_uses_explicit_paper_faithful_shape() -> None:
    assert tuple(f.n_atoms for f in IJCNN_SUITE) == IJCNN_ATOM_COUNTS
    assert all(f.tree_shape == "balanced" for f in IJCNN_SUITE)
    assert all("ijcnn_balanced" in f.formula_id for f in IJCNN_SUITE)
    assert all(f.tree_shape == "left_deep" for f in IJCNN_LEFTDEEP_SUITE)


@pytest.mark.parametrize("n_atoms", IJCNN_ATOM_COUNTS)
def test_ijcnn_shape_controls_depth_without_changing_size(n_atoms: int) -> None:
    balanced = characterize_formula(ijcnn_formula(n_atoms, tree_shape="balanced"))
    left_deep = characterize_formula(ijcnn_formula(n_atoms, tree_shape="left_deep"))

    assert balanced.ast_nodes == left_deep.ast_nodes == 4 * n_atoms - 4
    assert balanced.distinct_subformulae == left_deep.distinct_subformulae
    assert balanced.ast_depth == 2 + math.ceil(math.log2(n_atoms - 1))
    assert left_deep.ast_depth == n_atoms
    assert balanced.temporal_depth == left_deep.temporal_depth == 1


@pytest.mark.parametrize("n_atoms", (2, 4, 8))
def test_ijcnn_shapes_are_exactly_language_equivalent(n_atoms: int) -> None:
    balanced = compile_ltlf(ijcnn_formula(n_atoms, tree_shape="balanced").formula)
    left_deep = compile_ltlf(ijcnn_formula(n_atoms, tree_shape="left_deep").formula)
    _assert_language_equivalent(balanced, left_deep)


def test_ijcnn_generator_rejects_invalid_shape_and_size() -> None:
    with pytest.raises(ValueError, match="n must be >= 2"):
        ijcnn_formula(1)
    with pytest.raises(ValueError, match="tree_shape"):
        ijcnn_formula(4, tree_shape="right_deep")


def test_declared_parameters_are_not_compiler_statistics() -> None:
    deadline = bounded_response(6)
    state_blowup = kth_from_last(4)

    assert deadline.parameter("deadline") == 6
    assert state_blowup.parameter("temporal_depth") == 4
    assert deadline.dfa_states is None
    assert state_blowup.dfa_states is None
    assert deadline.n_atoms == state_blowup.n_atoms == 2
    assert not hasattr(deadline, "n_leaves")


def test_formula_structure_flat_dict_is_csv_friendly() -> None:
    row = characterize_formula(bounded_response(2)).flat_dict()
    assert row["schema_version"] == RESULT_SCHEMA_VERSION
    assert row["parameters"] == '{"deadline": 2}'
    assert row["roles"].startswith("[")
    assert row["n_atoms"] == 2


def test_default_resource_policy_is_explicit_and_portable() -> None:
    assert DEFAULT_RESOURCE_BUDGETS.cold_compile_timeout_s == 300.0
    assert DEFAULT_RESOURCE_BUDGETS.execution_timeout_s == 300.0
    assert DEFAULT_RESOURCE_BUDGETS.max_host_memory_fraction == 0.80
    assert DEFAULT_RESOURCE_BUDGETS.max_gpu_memory_fraction == 0.80

    with pytest.raises(ValueError, match="timeouts"):
        ResourceBudgets(cold_compile_timeout_s=0)
    with pytest.raises(ValueError, match="max_host_memory_fraction"):
        ResourceBudgets(max_host_memory_fraction=1.1)


def _raw_record(**overrides) -> RawResultRecord:
    values = dict(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id="run-1",
        experiment_id="schema-test",
        formula_id="eventually",
        monitor_name="SymbolicDFAMonitor",
        monitor_backend="guarded",
        requested_device="cpu",
        effective_device="cpu",
        dtype="bool",
        trace_seed=7,
        repetition=0,
        trace_length=10,
        batch_size=2,
        early_termination=False,
        online_label_mode="exact",
        status=RunStatus.SUCCESS,
        duration_s=0.01,
        offered_traces=2,
        offered_cells=20,
    )
    values.update(overrides)
    return RawResultRecord(**values)


def test_raw_result_schema_keeps_failures_and_decision_lag_explicit() -> None:
    success = _raw_record(semantic_decision_index=3, reported_decision_index=5)
    assert success.decision_lag == 2
    assert success.flat_dict()["status"] == "success"
    assert success.flat_dict()["decision_lag"] == 2

    timeout = _raw_record(
        status=RunStatus.TIMEOUT,
        duration_s=None,
        failure_reason="execution budget exceeded",
    )
    assert timeout.flat_dict()["status"] == "timeout"


def test_successful_raw_record_requires_duration_and_valid_stage_json() -> None:
    with pytest.raises(ValueError, match="require duration_s"):
        _raw_record(duration_s=None)
    with pytest.raises(ValueError, match="valid JSON"):
        _raw_record(compile_stages_json="not-json")
    with pytest.raises(TypeError, match="RunStatus"):
        _raw_record(status="success")


class _NoopBenchmarkMonitor:
    effective_device = "cpu"

    @classmethod
    def compile(cls, formula: str, device: str = "cpu") -> _NoopBenchmarkMonitor:
        return cls()

    def batch_run(self, traces, *, early_termination: bool = True):
        return [None] * len(traces)


def test_legacy_timing_runner_emits_e0_structural_fields() -> None:
    formula = ijcnn_formula(4)
    result = time_monitor(
        _NoopBenchmarkMonitor,
        formula,
        trace_length=2,
        n_traces=2,
        n_repeats=1,
        n_warmup=0,
    )

    assert result.schema_version == RESULT_SCHEMA_VERSION
    assert result.formula_name == formula.formula_id
    assert result.formula_family == "ijcnn"
    assert result.tree_shape == "balanced"
    assert result.parameters_json == '{"n_atoms": 4}'
    assert result.n_atoms == 4
    assert result.ast_depth == 4


def test_plot_loader_migrates_legacy_n_leaves_without_conflating_axes(
    tmp_path,
) -> None:
    common = {
        "monitor_name": "SymbolicDFAMonitor",
        "trace_length": 10,
        "n_traces": 2,
        "mean_s_per_cell": 1e-6,
        "std_s_per_cell": 1e-7,
        "device": "cpu",
        "gpu_name": "",
    }
    rows = [
        {**common, "formula_name": "ijcnn_n4", "n_leaves": 4},
        {**common, "formula_name": "boundedresp_k5", "n_leaves": 7},
        {
            **common,
            "formula_name": "ijcnn_balanced_n8",
            "n_leaves": None,
            "n_atoms": 8,
            "dfa_states": 2,
        },
    ]
    csv_path = tmp_path / "mixed_schema.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    loaded = load_legacy_timing(csv_path).set_index("formula_name")
    assert loaded.loc["ijcnn_n4", "n_atoms"] == 4
    assert math.isnan(loaded.loc["ijcnn_n4", "dfa_states"])
    assert loaded.loc["boundedresp_k5", "n_atoms"] == 2
    assert loaded.loc["boundedresp_k5", "dfa_states"] == 7
    assert loaded.loc["ijcnn_balanced_n8", "n_atoms"] == 8
    assert loaded.loc["ijcnn_balanced_n8", "dfa_states"] == 2
