"""Phase E2 isolated compilation, representation, and memory instrumentation.

Each benchmark cell is compiled in a fresh Python process.  The parent enforces
wall-time and host-RSS budgets; the worker records native construction stages,
compiled representation statistics, peak process/CUDA memory, validation
fallbacks, and an explicit terminal status for every attempted cell.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from src.benchmarks.schema import (
    DEFAULT_RESOURCE_BUDGETS,
    RESULT_SCHEMA_VERSION,
    ResourceBudgets,
    RunStatus,
)

E2_SCHEMA_VERSION = "e2.v1"
ROOT = Path(__file__).resolve().parents[2]
RQ1_ARTIFACT = ROOT / "results" / "rq1" / "rq1_characterization.json"


@dataclass(frozen=True)
class MonitorSpec:
    monitor_id: str
    monitor_name: str
    architecture: str
    construction: str
    backend: str
    adapter: str
    rq1_construction: str | None
    uses_requested_device: bool
    compile_options: tuple[tuple[str, bool | int | float | str], ...] = ()

    @property
    def options(self) -> dict[str, bool | int | float | str]:
        return dict(self.compile_options)


MONITOR_SPECS: tuple[MonitorSpec, ...] = (
    MonitorSpec(
        "symbolic_guarded",
        "SymbolicDFAMonitor",
        "symbolic_dfa",
        "canonical_dfa",
        "guarded_edges",
        "symbolic",
        None,
        False,
    ),
    MonitorSpec(
        "rulerunner_original_flat",
        "RuleRunnerMonitor",
        "rulerunner",
        "original",
        "flat_cilp",
        "original_flat",
        "original",
        True,
    ),
    MonitorSpec(
        "rulerunner_original_structured",
        "StructuredRuleRunnerMonitor",
        "rulerunner",
        "original",
        "module_scheduled_cilp",
        "original_structured",
        "original",
        True,
    ),
    MonitorSpec(
        "rulerunner_bounded_flat_default",
        "BoundedEventRuleRunnerMonitor",
        "rulerunner",
        "bounded_default",
        "flat_cilp",
        "bounded_flat",
        "bounded_default",
        True,
        (("exact_online", False), ("certificate", "cached")),
    ),
    MonitorSpec(
        "rulerunner_bounded_flat_exact",
        "BoundedEventRuleRunnerMonitor",
        "rulerunner",
        "bounded_exact_online",
        "flat_cilp",
        "bounded_flat",
        "bounded_exact_online",
        True,
        (("exact_online", True), ("certificate", "cached")),
    ),
    MonitorSpec(
        "rulerunner_bounded_structured_default",
        "BoundedEventStructuredRuleRunnerMonitor",
        "rulerunner",
        "bounded_default",
        "structured_cilp",
        "bounded_structured",
        "bounded_default",
        True,
        (("exact_online", False), ("certificate", "cached")),
    ),
    MonitorSpec(
        "rulerunner_bounded_structured_exact",
        "BoundedEventStructuredRuleRunnerMonitor",
        "rulerunner",
        "bounded_exact_online",
        "structured_cilp",
        "bounded_structured",
        "bounded_exact_online",
        True,
        (("exact_online", True), ("certificate", "cached")),
    ),
    MonitorSpec(
        "rulerunner_progression_flat",
        "ProgressionRuleRunnerMonitor",
        "rulerunner",
        "progression",
        "flat_cilp",
        "progression_flat",
        "progression",
        True,
    ),
    MonitorSpec(
        "rulerunner_progression_structured",
        "ProgressionRuleRunnerStructuredMonitor",
        "rulerunner",
        "progression",
        "structured_cilp",
        "progression_structured",
        "progression",
        True,
    ),
    MonitorSpec(
        "deepdfa_dense",
        "DeepDFAMonitorDense",
        "deepdfa",
        "fixed_dfa",
        "dense",
        "deepdfa_dense",
        None,
        True,
    ),
    MonitorSpec(
        "deepdfa_factored",
        "DeepDFAMonitorFactored",
        "deepdfa",
        "fixed_dfa",
        "factored_cubes",
        "deepdfa_factored",
        None,
        True,
    ),
    MonitorSpec(
        "deepdfa_scan",
        "DeepDFAMonitorScan",
        "deepdfa",
        "fixed_dfa",
        "dense_prefix_scan",
        "deepdfa_scan",
        None,
        True,
    ),
)

MONITOR_SPEC_BY_ID = {spec.monitor_id: spec for spec in MONITOR_SPECS}


@dataclass(frozen=True)
class WorkerRequest:
    run_id: str
    cell_id: str
    formula_id: str
    formula: str
    atoms: tuple[str, ...]
    monitor_id: str
    requested_device: str
    repetition: int = 0
    validation_trace_length: int = 0
    validation_batch_size: int = 0
    phase_path: str = ""


@dataclass(frozen=True)
class E2CellRecord:
    """One isolated compilation attempt, including unsuccessful attempts."""

    schema_version: str
    benchmark_schema_version: str
    run_id: str
    cell_id: str
    formula_id: str
    formula: str
    monitor_id: str
    monitor_name: str
    architecture: str
    construction: str
    backend: str
    requested_device: str
    effective_device: str
    repetition: int
    worker_pid: int | None
    status: RunStatus
    failure_stage: str = ""
    failure_reason: str = ""
    compile_total_s: float | None = None
    compile_stages_json: str = "{}"
    artifact_stats_json: str = "{}"
    validation_duration_s: float | None = None
    warnings_json: str = "[]"
    startup_host_rss_bytes: int | None = None
    baseline_host_rss_bytes: int | None = None
    peak_compile_host_rss_bytes: int | None = None
    peak_observed_host_rss_bytes: int | None = None
    host_memory_limit_bytes: int | None = None
    baseline_gpu_allocated_bytes: int | None = None
    peak_compile_gpu_allocated_bytes: int | None = None
    peak_compile_gpu_reserved_bytes: int | None = None
    provenance_json: str = "{}"

    def __post_init__(self) -> None:
        if self.schema_version != E2_SCHEMA_VERSION:
            raise ValueError(f"expected schema {E2_SCHEMA_VERSION!r}")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        for field in (
            "compile_stages_json",
            "artifact_stats_json",
            "warnings_json",
            "provenance_json",
        ):
            try:
                json.loads(getattr(self, field))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be valid JSON") from exc
        if self.status in (RunStatus.SUCCESS, RunStatus.FALLBACK):
            if self.compile_total_s is None:
                raise ValueError(
                    "successful/fallback compilation requires compile_total_s"
                )
        for field in ("compile_total_s", "validation_duration_s"):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise ValueError(f"{field} must be non-negative")

    def flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "E2CellRecord":
        values = dict(payload)
        values["status"] = RunStatus(values["status"])
        return cls(**values)


class StageRecorder:
    """Synchronous native-stage timer used only inside a fresh worker."""

    def __init__(self, requested_device: str, torch_module: object | None) -> None:
        self._torch = torch_module
        self._sync_cuda = (
            requested_device == "cuda"
            and torch_module is not None
            and torch_module.cuda.is_available()  # type: ignore[union-attr]
        )
        self.durations: dict[str, float] = {}
        self.active_stage = "worker_import"
        self.failed_stage = ""

    def _sync(self) -> None:
        if self._sync_cuda:
            self._torch.cuda.synchronize()  # type: ignore[union-attr]

    def __call__(self, name: str, build):
        self.active_stage = name
        self._sync()
        start = time.perf_counter()
        try:
            return build()
        except Exception:
            self.failed_stage = name
            raise
        finally:
            self._sync()
            self.durations[name] = self.durations.get(name, 0.0) + (
                time.perf_counter() - start
            )
            self.active_stage = ""


def _compile_monitor(spec: MonitorSpec, request: WorkerRequest, stages: StageRecorder):
    """Compile through the same native components as each public ``compile``."""
    import torch

    formula = request.formula
    device = request.requested_device

    if spec.adapter == "symbolic":
        from src.formula.compiler import compile_ltlf
        from src.monitors.symbolic_dfa import SymbolicDFAMonitor

        dfa = stages("ltlf_to_minimal_dfa", lambda: compile_ltlf(formula))
        return stages("monitor_initialization", lambda: SymbolicDFAMonitor(dfa))

    if spec.adapter.startswith("deepdfa_"):
        from src.formula.compiler import compile_ltlf
        from src.monitors.deep_dfa import (
            DeepDFAMonitorDense,
            DeepDFAMonitorFactored,
            DeepDFAMonitorScan,
            DeepDFATensor,
        )

        dfa = stages("ltlf_to_minimal_dfa", lambda: compile_ltlf(formula))
        mode = "factored" if spec.adapter == "deepdfa_factored" else "dense"
        tensor = stages(
            "dfa_tensor_lowering",
            lambda: DeepDFATensor(dfa, mode=mode, device=device),
        )
        monitor_cls = {
            "deepdfa_dense": DeepDFAMonitorDense,
            "deepdfa_factored": DeepDFAMonitorFactored,
            "deepdfa_scan": DeepDFAMonitorScan,
        }[spec.adapter]
        return stages("monitor_initialization", lambda: monitor_cls(tensor))

    if spec.adapter.startswith("original_"):
        from src.monitors.rulerunner.cilp import CILPRunner
        from src.monitors.rulerunner.monitor import RuleRunnerMonitor
        from src.monitors.rulerunner.parse_tree import parse
        from src.monitors.rulerunner.structured import (
            StructuredCILPRunner,
            StructuredRuleRunnerMonitor,
        )

        root = stages("formula_parse", lambda: parse(formula))
        if spec.adapter == "original_flat":
            runner = stages(
                "rules_and_cilp_lowering", lambda: CILPRunner(root, device=device)
            )
            return stages("monitor_initialization", lambda: RuleRunnerMonitor(runner))
        runner = stages(
            "rules_and_structured_cilp_lowering",
            lambda: StructuredCILPRunner(root, device=device),
        )
        return stages(
            "monitor_initialization", lambda: StructuredRuleRunnerMonitor(runner)
        )

    if spec.adapter.startswith("bounded_"):
        from src.monitors.rulerunner.bounded_cilp import (
            BoundedEventRuleRunnerMonitor,
            BoundedEventStructuredRuleRunnerMonitor,
        )

        monitor_cls = (
            BoundedEventStructuredRuleRunnerMonitor
            if spec.adapter == "bounded_structured"
            else BoundedEventRuleRunnerMonitor
        )
        return monitor_cls.compile(
            formula,
            device=device,
            stage_callback=stages,
            **spec.options,
        )

    if spec.adapter == "progression_flat":
        from src.monitors.progression.eager import build_progression_dfa
        from src.monitors.progression.flat import (
            ProgressionRuleRunnerMonitor,
            _FlatNet,
        )

        graph = stages(
            "progression_residual_graph", lambda: build_progression_dfa(formula)
        )
        net = stages(
            "progression_flat_cilp_lowering",
            lambda: _FlatNet(graph, torch.device(device)),
        )
        return stages(
            "monitor_initialization", lambda: ProgressionRuleRunnerMonitor(net)
        )

    if spec.adapter == "progression_structured":
        from src.monitors.progression.structured import (
            ProgressionRuleRunnerStructuredMonitor,
            _StructuredNet,
            build_factorized_progression_graph,
        )

        graph = stages(
            "progression_factorized_graph",
            lambda: build_factorized_progression_graph(formula),
        )
        net = stages(
            "progression_structured_cilp_lowering",
            lambda: _StructuredNet(graph, torch.device(device)),
        )
        return stages(
            "monitor_initialization",
            lambda: ProgressionRuleRunnerStructuredMonitor(net),
        )

    raise KeyError(f"unknown compilation adapter {spec.adapter!r}")


def _tensor_inventory(root: object) -> dict[str, Any]:
    """Count unique persistent tensor storages reachable from a monitor."""
    import dataclasses
    import types

    import numpy as np
    import torch

    seen_objects: set[int] = set()
    storages: dict[tuple[str, int], int] = {}
    tensor_count = 0
    logical_elements = 0
    logical_bytes = 0
    nonzero_elements = 0
    numpy_bytes = 0
    dtypes: Counter[str] = Counter()
    devices: Counter[str] = Counter()

    def visit(value: object) -> None:
        nonlocal tensor_count, logical_elements, logical_bytes
        nonlocal nonzero_elements, numpy_bytes
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            elements = value.numel()
            logical_elements += elements
            logical_bytes += elements * value.element_size()
            nonzero_elements += int(torch.count_nonzero(value).item())
            dtypes[str(value.dtype)] += 1
            devices[str(value.device)] += 1
            storage = value.untyped_storage()
            pointer = storage.data_ptr() if storage.nbytes() else id(storage)
            storages[(str(value.device), pointer)] = storage.nbytes()
            return
        if isinstance(value, np.ndarray):
            numpy_bytes += value.nbytes
            return
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
            return
        if isinstance(value, (types.FunctionType, types.MethodType, type)):
            return
        module = type(value).__module__
        if dataclasses.is_dataclass(value) or module.startswith("src.monitors"):
            for item in vars(value).values():
                visit(item)

    visit(root)
    return {
        "persistent_tensor_count": tensor_count,
        "persistent_tensor_logical_elements": logical_elements,
        "persistent_tensor_logical_bytes": logical_bytes,
        "persistent_tensor_allocated_bytes": sum(storages.values()),
        "persistent_tensor_nonzero_elements": nonzero_elements,
        "persistent_numpy_bytes": numpy_bytes,
        "tensor_dtypes": dict(sorted(dtypes.items())),
        "tensor_devices": dict(sorted(devices.items())),
    }


def _rule_stats(runner: object) -> dict[str, Any]:
    rules = runner._rs  # type: ignore[attr-defined]
    index = getattr(runner, "_literal_index", getattr(runner, "_index", {}))
    modules = 2 if hasattr(runner, "_literal_index") else 2 * len(runner._nodes)
    return {
        "eval_rules": len(rules.eval_rules),
        "react_rules": len(rules.react_rules),
        "total_rules": len(rules.eval_rules) + len(rules.react_rules),
        "literals": len(index),
        "modules": modules,
        "ast_nodes": sum(1 for _ in runner._root.subformulae()),
        "ast_depth": runner._root.depth,
    }


def _artifact_stats(monitor: object, spec: MonitorSpec) -> dict[str, Any]:
    stats: dict[str, Any] = {"monitor_id": spec.monitor_id}
    if spec.adapter == "symbolic":
        stats.update(
            persistent_tensor_count=0,
            persistent_tensor_logical_elements=0,
            persistent_tensor_logical_bytes=0,
            persistent_tensor_allocated_bytes=0,
            persistent_tensor_nonzero_elements=0,
            persistent_numpy_bytes=0,
            tensor_dtypes={},
            tensor_devices={},
        )
        dfa = monitor._dfa  # type: ignore[attr-defined]
        stats.update(
            dfa_states=len(dfa.states),
            dfa_guarded_transitions=len(dfa.transitions),
            accepting_states=len(dfa.accepting),
            trap_states=len(dfa.trap_states),
            accepting_sinks=len(dfa.accepting_sinks),
        )
    else:
        stats.update(_tensor_inventory(monitor))

    if spec.adapter.startswith("deepdfa_"):
        stats.update(asdict(monitor.artifact_stats))  # type: ignore[attr-defined]
    elif spec.adapter.startswith("original_"):
        stats.update(_rule_stats(monitor._runner))  # type: ignore[attr-defined]
    elif spec.adapter.startswith("bounded_"):
        eventized = monitor.eventized  # type: ignore[attr-defined]
        stats.update(
            bounded_horizon=eventized.horizon,
            bounded_islands=len(eventized.islands),
            certificate_product_states=monitor.certificate.explored_product_states,
            certificate_source=monitor.certificate_source,
            exact_online=monitor.exact_online,
            exact_head_states=(
                monitor.extrapolation.n_states
                if monitor.extrapolation is not None
                else None
            ),
            event_modules=sum(
                len(window.owners) for window in monitor.event_net.windows.values()
            ),
            **{
                f"skeleton_{key}": value
                for key, value in _rule_stats(monitor.skeleton).items()
            },
        )
    elif spec.adapter == "progression_flat":
        net = monitor._net  # type: ignore[attr-defined]
        graph = net.dfa
        stats.update(
            progression_states=graph.n_states,
            progression_roots=graph.n_roots,
            progression_closure=graph.n_closure,
            progression_input_subformulae=graph.n_input_sub,
            transition_clauses=net.W_ih.shape[0],
            modules=1,
        )
    elif spec.adapter == "progression_structured":
        net = monitor._net  # type: ignore[attr-defined]
        graph = net.graph
        stats.update(
            progression_states=len(graph.states),
            progression_roots=len(graph.roots),
            progression_closure=len(graph.closure),
            eval_modules=len(net.eval_net),
            react_modules=len(net.react_net),
            label_modules=1,
        )
    return stats


def _peak_rss_bytes() -> int:
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/status").read_text().splitlines()
        row = next(line for line in fields if line.startswith("VmRSS:"))
        return int(row.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return _peak_rss_bytes()


def _worker_provenance() -> dict[str, Any]:
    import importlib.metadata
    import platform

    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "ltlf2dfa": version("ltlf2dfa"),
    }


def _set_worker_phase(request: WorkerRequest, phase: str) -> None:
    if request.phase_path:
        Path(request.phase_path).write_text(phase)


def _record_base(request: WorkerRequest, spec: MonitorSpec) -> dict[str, Any]:
    return {
        "schema_version": E2_SCHEMA_VERSION,
        "benchmark_schema_version": RESULT_SCHEMA_VERSION,
        "run_id": request.run_id,
        "cell_id": request.cell_id,
        "formula_id": request.formula_id,
        "formula": request.formula,
        "monitor_id": spec.monitor_id,
        "monitor_name": spec.monitor_name,
        "architecture": spec.architecture,
        "construction": spec.construction,
        "backend": spec.backend,
        "requested_device": request.requested_device,
        "effective_device": "",
        "repetition": request.repetition,
        "worker_pid": None,
    }


def execute_worker_request(
    request: WorkerRequest,
    *,
    startup_host_rss_bytes: int,
    startup_peak_host_rss_bytes: int,
) -> E2CellRecord:
    """Compile and characterize one cell; called only by the fresh worker."""
    import warnings

    spec = MONITOR_SPEC_BY_ID.get(request.monitor_id)
    if spec is None:
        placeholder = MonitorSpec(
            request.monitor_id,
            request.monitor_id,
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            None,
            False,
        )
        unknown_base = _record_base(request, placeholder)
        unknown_base["worker_pid"] = os.getpid()
        return E2CellRecord(
            **unknown_base,
            status=RunStatus.ERROR,
            failure_stage="request_validation",
            failure_reason=f"unknown monitor_id {request.monitor_id!r}",
            startup_host_rss_bytes=startup_host_rss_bytes,
            peak_compile_host_rss_bytes=startup_peak_host_rss_bytes,
        )

    base = _record_base(request, spec)
    base["worker_pid"] = os.getpid()
    torch = None
    if spec.uses_requested_device:
        import torch as torch_module

        torch = torch_module
    baseline_rss = _current_rss_bytes()
    stages = StageRecorder(request.requested_device, torch)
    if (
        request.requested_device == "cuda"
        and spec.uses_requested_device
        and (torch is None or not torch.cuda.is_available())
    ):
        return E2CellRecord(
            **base,
            status=RunStatus.UNSUPPORTED,
            failure_stage="device_validation",
            failure_reason="CUDA was requested but is unavailable",
            startup_host_rss_bytes=startup_host_rss_bytes,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=_peak_rss_bytes(),
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )

    baseline_gpu: int | None = None
    if (
        torch is not None
        and request.requested_device == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_gpu = torch.cuda.memory_allocated()

    start = time.perf_counter()
    try:
        _set_worker_phase(request, "compile")
        monitor = _compile_monitor(spec, request, stages)
        if (
            torch is not None
            and request.requested_device == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()
        compile_total = time.perf_counter() - start
        peak_compile_rss = _peak_rss_bytes()
        peak_gpu_allocated = (
            torch.cuda.max_memory_allocated()
            if torch is not None
            and request.requested_device == "cuda"
            and torch.cuda.is_available()
            else None
        )
        peak_gpu_reserved = (
            torch.cuda.max_memory_reserved()
            if torch is not None
            and request.requested_device == "cuda"
            and torch.cuda.is_available()
            else None
        )
        _set_worker_phase(request, "post_compile")
        artifact = _artifact_stats(monitor, spec)

        warning_messages: list[str] = []
        validation_duration: float | None = None
        status = RunStatus.SUCCESS
        if request.validation_batch_size and request.validation_trace_length:
            traces = [
                [dict.fromkeys(request.atoms, False)] * request.validation_trace_length
                for _ in range(request.validation_batch_size)
            ]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                validation_start = time.perf_counter()
                monitor.batch_run(traces, early_termination=False)
                if monitor.effective_device == "cuda" and torch is not None:
                    torch.cuda.synchronize()
                validation_duration = time.perf_counter() - validation_start
            warning_messages = [str(item.message) for item in caught]
            if any("falling back" in message.lower() for message in warning_messages):
                status = RunStatus.FALLBACK

        success_base = {**base, "effective_device": monitor.effective_device}
        return E2CellRecord(
            **success_base,
            status=status,
            compile_total_s=compile_total,
            compile_stages_json=json.dumps(stages.durations, sort_keys=True),
            artifact_stats_json=json.dumps(artifact, sort_keys=True),
            validation_duration_s=validation_duration,
            warnings_json=json.dumps(warning_messages),
            startup_host_rss_bytes=startup_host_rss_bytes,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=peak_compile_rss,
            baseline_gpu_allocated_bytes=baseline_gpu,
            peak_compile_gpu_allocated_bytes=peak_gpu_allocated,
            peak_compile_gpu_reserved_bytes=peak_gpu_reserved,
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )
    except Exception as exc:  # worker boundary: every attempt must yield a row
        _set_worker_phase(request, "post_compile")
        oom_types = (MemoryError,)
        if torch is not None and hasattr(torch.cuda, "OutOfMemoryError"):
            oom_types = oom_types + (torch.cuda.OutOfMemoryError,)
        status = RunStatus.OOM if isinstance(exc, oom_types) else RunStatus.ERROR
        unsupported_names = {
            "MissingCertificate",
            "UnsafeRuleRunnerSkeleton",
            "ExtrapolationLimitExceeded",
            "MonaFailure",
        }
        if type(exc).__name__ in unsupported_names:
            status = RunStatus.UNSUPPORTED
        return E2CellRecord(
            **base,
            status=status,
            failure_stage=stages.failed_stage or stages.active_stage or "compilation",
            failure_reason=f"{type(exc).__name__}: {exc}",
            compile_total_s=time.perf_counter() - start,
            compile_stages_json=json.dumps(stages.durations, sort_keys=True),
            startup_host_rss_bytes=startup_host_rss_bytes,
            baseline_host_rss_bytes=baseline_rss,
            peak_compile_host_rss_bytes=_peak_rss_bytes(),
            baseline_gpu_allocated_bytes=baseline_gpu,
            peak_compile_gpu_allocated_bytes=(
                torch.cuda.max_memory_allocated()
                if torch is not None
                and request.requested_device == "cuda"
                and torch.cuda.is_available()
                else None
            ),
            peak_compile_gpu_reserved_bytes=(
                torch.cuda.max_memory_reserved()
                if torch is not None
                and request.requested_device == "cuda"
                and torch.cuda.is_available()
                else None
            ),
            provenance_json=json.dumps(_worker_provenance(), sort_keys=True),
        )


def _host_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _process_tree_pids(root_pid: int) -> set[int]:
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text()
        except OSError:
            continue
        pending.extend(int(value) for value in children.split())
    return seen


def _process_tree_rss_bytes(root_pid: int) -> int:
    total = 0
    for pid in _process_tree_pids(root_pid):
        try:
            rows = Path(f"/proc/{pid}/status").read_text().splitlines()
            value = next(row for row in rows if row.startswith("VmRSS:"))
            total += int(value.split()[1]) * 1024
        except (OSError, StopIteration, ValueError):
            continue
    return total


@dataclass(frozen=True)
class SupervisedOutcome:
    returncode: int
    stdout: str
    stderr: str
    peak_rss_bytes: int
    peak_compile_rss_bytes: int
    peak_runtime_rss_bytes: int
    terminal_status: RunStatus | None
    failure_reason: str = ""


def supervise_command(
    command: list[str],
    *,
    timeout_s: float,
    memory_limit_bytes: int | None,
    phase_path: Path | None = None,
) -> SupervisedOutcome:
    """Run a process group while enforcing wall-time and RSS budgets.

    Passing an explicit environment snapshot is intentional. Native runtimes
    can call ``setenv`` without updating Python's ``os.environ`` mapping (MKL
    does this after some NumPy/PyTorch import orders). Inheriting the C-level
    environment implicitly would then make a supposedly fresh cell depend on
    which backends the parent process imported earlier.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=os.environ.copy(),
    )
    start = time.monotonic()
    peak_rss = 0
    peak_compile_rss = 0
    peak_runtime_rss = 0
    terminal_status: RunStatus | None = None
    reason = ""
    while process.poll() is None:
        current_rss = _process_tree_rss_bytes(process.pid)
        peak_rss = max(peak_rss, current_rss)
        if phase_path is not None:
            try:
                phase = phase_path.read_text()
            except OSError:
                phase = ""
            if phase == "compile":
                peak_compile_rss = max(peak_compile_rss, current_rss)
            elif phase == "runtime":
                peak_runtime_rss = max(peak_runtime_rss, current_rss)
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            terminal_status = RunStatus.TIMEOUT
            reason = f"wall-time budget exceeded ({timeout_s:.3f}s)"
        elif memory_limit_bytes and peak_rss > memory_limit_bytes:
            terminal_status = RunStatus.OOM
            reason = (
                f"host RSS budget exceeded ({peak_rss} > {memory_limit_bytes} bytes)"
            )
        if terminal_status is not None:
            os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
    return SupervisedOutcome(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        peak_rss_bytes=peak_rss,
        peak_compile_rss_bytes=peak_compile_rss,
        peak_runtime_rss_bytes=peak_runtime_rss,
        terminal_status=terminal_status,
        failure_reason=reason,
    )


def _failure_record(
    request: WorkerRequest,
    spec: MonitorSpec,
    status: RunStatus,
    reason: str,
    *,
    peak_rss_bytes: int = 0,
    memory_limit_bytes: int | None = None,
) -> E2CellRecord:
    return E2CellRecord(
        **_record_base(request, spec),
        status=status,
        failure_stage="worker_supervision",
        failure_reason=reason,
        peak_observed_host_rss_bytes=peak_rss_bytes or None,
        host_memory_limit_bytes=memory_limit_bytes,
    )


def _rq1_rows() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(RQ1_ARTIFACT.read_text())
    return {(row["formula_id"], row["construction"]): row for row in payload["records"]}


def run_rq1_cell(
    formula_id: str,
    monitor_id: str,
    *,
    requested_device: str = "cpu",
    repetition: int = 0,
    validation_trace_length: int = 0,
    validation_batch_size: int = 0,
    budgets: ResourceBudgets = DEFAULT_RESOURCE_BUDGETS,
    run_id: str | None = None,
) -> E2CellRecord:
    """Run one fresh-process compilation after consulting the RQ1 gate."""
    from src.benchmarks.rq1 import RQ1_CORPUS

    spec = MONITOR_SPEC_BY_ID[monitor_id]
    cases = {case.benchmark.formula_id: case.benchmark for case in RQ1_CORPUS}
    formula = cases[formula_id]
    run_id = run_id or str(uuid.uuid4())
    cell_id = f"{formula_id}::{monitor_id}::{requested_device}::r{repetition}"
    request = WorkerRequest(
        run_id=run_id,
        cell_id=cell_id,
        formula_id=formula.formula_id,
        formula=formula.formula,
        atoms=formula.atoms,
        monitor_id=monitor_id,
        requested_device=requested_device,
        repetition=repetition,
        validation_trace_length=validation_trace_length,
        validation_batch_size=validation_batch_size,
    )

    if spec.rq1_construction is not None:
        gate = _rq1_rows()[(formula_id, spec.rq1_construction)]
        if not gate["applicable"]:
            return _failure_record(
                request,
                spec,
                RunStatus.UNSUPPORTED,
                "rejected by frozen rq1.v1 semantic applicability gate",
            )

    host_memory = _host_memory_bytes()
    memory_limit = (
        int(host_memory * budgets.max_host_memory_fraction) if host_memory else None
    )
    with tempfile.NamedTemporaryFile("w", suffix=".phase", delete=False) as phase:
        phase_path = Path(phase.name)
    request = replace(request, phase_path=str(phase_path))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        request_path = Path(handle.name)
        json.dump(asdict(request), handle)
    try:
        outcome = supervise_command(
            [
                sys.executable,
                str(ROOT / "src" / "benchmarks" / "e2_worker.py"),
                str(request_path),
            ],
            timeout_s=budgets.cold_compile_timeout_s,
            memory_limit_bytes=memory_limit,
            phase_path=phase_path,
        )
    finally:
        request_path.unlink(missing_ok=True)
        phase_path.unlink(missing_ok=True)

    if outcome.terminal_status is not None:
        return _failure_record(
            request,
            spec,
            outcome.terminal_status,
            outcome.failure_reason,
            peak_rss_bytes=outcome.peak_rss_bytes,
            memory_limit_bytes=memory_limit,
        )
    if outcome.returncode != 0:
        return _failure_record(
            request,
            spec,
            RunStatus.ERROR,
            f"worker exited {outcome.returncode}: {outcome.stderr.strip()}",
            peak_rss_bytes=outcome.peak_rss_bytes,
            memory_limit_bytes=memory_limit,
        )
    try:
        payload = json.loads(outcome.stdout.strip().splitlines()[-1])
        record = E2CellRecord.from_dict(payload)
    except (IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _failure_record(
            request,
            spec,
            RunStatus.ERROR,
            f"invalid worker output: {exc}; stderr={outcome.stderr.strip()!r}",
            peak_rss_bytes=outcome.peak_rss_bytes,
            memory_limit_bytes=memory_limit,
        )
    return replace(
        record,
        peak_compile_host_rss_bytes=max(
            outcome.peak_compile_rss_bytes,
            record.peak_compile_host_rss_bytes or 0,
        ),
        peak_observed_host_rss_bytes=max(
            outcome.peak_rss_bytes,
            record.peak_compile_host_rss_bytes or 0,
        ),
        host_memory_limit_bytes=memory_limit,
    )
