"""CILP realizations of the certified bounded-event construction."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from src.monitors.base import Verdict
from src.monitors.rulerunner.bounded import evaluate_bounded
from src.monitors.rulerunner.bounded_cilp import (
    BoundedEventRuleRunnerMonitor,
    BoundedEventStructuredRuleRunnerMonitor,
    UnsafeRuleRunnerSkeleton,
    _ObservationShiftCILP,
)
from src.monitors.rulerunner.bounded_extrapolation import (
    ExtrapolationLimitExceeded,
)
from src.monitors.rulerunner.certificates import MissingCertificate
from src.monitors.rulerunner.cilp import CILPRunner
from src.monitors.rulerunner.structured import StructuredCILPRunner
from src.monitors.symbolic_dfa import SymbolicDFAMonitor

MONITORS = (
    BoundedEventRuleRunnerMonitor,
    BoundedEventStructuredRuleRunnerMonitor,
)


def _valuations(atoms: tuple[str, ...]) -> list[dict[str, bool]]:
    return [
        dict(zip(atoms, values))
        for values in product((False, True), repeat=len(atoms))
    ]


@pytest.mark.parametrize("monitor_cls", MONITORS)
@pytest.mark.parametrize(
    ("formula", "atoms", "max_length"),
    [
        ("WX a", ("a",), 4),
        ("F (a & b)", ("a", "b"), 3),
        ("X X a", ("a",), 4),
        ("G X a", ("a",), 4),
        ("F (a & X b)", ("a", "b"), 3),
        ("G (a -> X b)", ("a", "b"), 3),
        ("a U (b & X c)", ("a", "b", "c"), 3),
        ("F (a & X X b)", ("a", "b"), 3),
        ("(X a) & X (X a)", ("a",), 4),
    ],
)
def test_compiled_pipeline_matches_original_dfa_exhaustively(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
    formula: str,
    atoms: tuple[str, ...],
    max_length: int,
) -> None:
    monitor = monitor_cls.compile(formula)
    oracle = SymbolicDFAMonitor.compile(formula)
    alphabet = _valuations(atoms)

    for length in range(max_length + 1):
        for trace in product(alphabet, repeat=length):
            assert monitor.run(trace) is oracle.run(trace), (formula, trace)


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_neural_event_evaluator_matches_bounded_semantics(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
) -> None:
    monitor = monitor_cls.compile("G (a -> X (b | WX c))")
    trace = [
        {"a": True},
        {"b": True},
        {"c": True},
    ]
    tensors = [monitor._observation_tensor(cell) for cell in trace]

    for length in range(1, monitor.eventized.horizon + 2):
        events = monitor.event_net.evaluate(torch.stack(tensors[:length]))
        for index, island in enumerate(monitor.eventized.islands):
            expected = evaluate_bounded(island.formula, trace[:length])
            assert bool(events[index] > 0) is expected


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_batched_event_evaluator_equals_individual_windows(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
) -> None:
    monitor = monitor_cls.compile("F (a & X X b)")
    windows = torch.stack(
        [
            torch.stack(
                [monitor._observation_tensor(cell) for cell in trace]
            )
            for trace in (
                [{"a": True}, {}, {"b": True}],
                [{}, {"a": True}, {"b": True}],
                [{"a": True}, {"b": True}, {}],
            )
        ]
    )
    batched = monitor.event_net.evaluate_batch(windows)
    individual = torch.stack(
        [monitor.event_net.evaluate(window) for window in windows]
    )
    assert torch.equal(batched, individual)


def test_shift_register_is_a_recurrent_cilp_layer() -> None:
    shift = _ObservationShiftCILP(("a", "b"), horizon=3, device=torch.device("cpu"))
    history = shift.initial()
    observations = (
        torch.tensor([1.0, -1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([-1.0, -1.0]),
    )

    for current in observations:
        history = shift.advance(history, current)

    assert torch.equal(
        history,
        torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ]
        ),
    )


def test_flat_and_structured_compilers_have_distinct_rule_grouping() -> None:
    formula = "F (a & X X b)"
    flat = BoundedEventRuleRunnerMonitor.compile(formula)
    structured = BoundedEventStructuredRuleRunnerMonitor.compile(formula)

    assert isinstance(flat.skeleton, CILPRunner)
    assert isinstance(structured.skeleton, StructuredCILPRunner)
    for window in flat.event_net.windows.values():
        assert window.flat_layer is not None
        assert not window.structured_layers
    for window in structured.event_net.windows.values():
        assert window.flat_layer is None
        assert set(window.structured_layers) == {
            (node.key, offset) for node, offset in window.owners
        }


def test_unsafe_skeleton_is_rejected_with_exact_counterexample() -> None:
    with pytest.raises(UnsafeRuleRunnerSkeleton) as caught:
        BoundedEventRuleRunnerMonitor.compile("G (a -> F b)")

    certificate = caught.value.result
    assert certificate.complete
    assert not certificate.language_equivalent
    assert certificate.language_witness is not None


def test_extrapolation_compilation_guards_are_explicit() -> None:
    with pytest.raises(ExtrapolationLimitExceeded, match="max_atoms=1"):
        BoundedEventRuleRunnerMonitor.compile(
            "F (a & X b)",
            exact_online=True,
            max_extrapolation_atoms=1,
        )
    with pytest.raises(ExtrapolationLimitExceeded, match="max_states=1"):
        BoundedEventRuleRunnerMonitor.compile(
            "X a",
            exact_online=True,
            max_extrapolation_states=1,
        )


def test_exact_extrapolation_is_opt_in() -> None:
    """The head is a priced add-on, not part of the default construction."""
    default = BoundedEventRuleRunnerMonitor.compile("F (a & X b)")
    assert default.extrapolation is None
    assert not default.exact_online

    exact = BoundedEventRuleRunnerMonitor.compile("F (a & X b)", exact_online=True)
    assert exact.extrapolation is not None


@pytest.mark.parametrize("monitor_cls", MONITORS)
@pytest.mark.parametrize(
    ("formula", "atoms", "max_length"),
    [
        ("WX a", ("a",), 4),
        ("G X a", ("a",), 4),
        ("F (a & X b)", ("a", "b"), 3),
        ("G (a -> X b)", ("a", "b"), 3),
        ("a U (b & X c)", ("a", "b", "c"), 3),
    ],
)
def test_default_online_labels_are_sound_but_may_be_late(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
    formula: str,
    atoms: tuple[str, ...],
    max_length: int,
) -> None:
    """Without the head, every definite label still agrees with the DFA."""
    monitor = monitor_cls.compile(formula)
    oracle = SymbolicDFAMonitor.compile(formula)
    alphabet = _valuations(atoms)

    for length in range(1, max_length + 1):
        for trace in product(alphabet, repeat=length):
            monitor.reset()
            oracle.reset()
            for cell in trace:
                verdict = monitor.step(cell)
                expected = oracle.step(cell)
                if verdict is not Verdict.UNDECIDED:
                    assert verdict is expected, (formula, trace)
            assert monitor.final_verdict() is oracle.final_verdict()


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_skeleton_latency_is_not_bounded_by_the_horizon(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
) -> None:
    """Documents why the exact head exists at all.

    ``G(X a)`` is unsatisfiable on every non-empty trace, because its last
    position has no successor, so the DFA traps immediately.  The eventized
    skeleton ``G e`` cannot see this: a derived word whose events are all true
    is consistent with ``G e``, even though no original trace realizes it.  The
    skeleton's online label therefore lags by the whole trace, not by ``H`` --
    only the extrapolation head, which knows which continuations are
    realizable, recovers the DFA's timing.
    """
    trace = [{"a": True}] * 5
    delayed = monitor_cls.compile("G X a")
    exact = monitor_cls.compile("G X a", exact_online=True)

    assert all(delayed.step(cell) is Verdict.UNDECIDED for cell in trace)
    assert delayed.final_verdict() is Verdict.VIOLATE
    assert all(exact.step(cell) is Verdict.VIOLATE for cell in trace)


def test_cached_certificate_avoids_building_a_dfa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile-time cost must not hide the DFA construction it avoids."""
    import src.monitors.rulerunner.equivalence as equivalence

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("compile() certified the skeleton at runtime")

    monkeypatch.setattr(equivalence, "compile_ltlf", forbidden)
    monkeypatch.setattr(equivalence, "certify_rule_runner", forbidden)

    monitor = BoundedEventRuleRunnerMonitor.compile(
        "F (a & X b)", certificate="cached"
    )
    assert monitor.certificate_source == "cache"
    assert monitor.certificate.language_equivalent


def test_missing_certificate_is_explicit_in_cached_mode() -> None:
    with pytest.raises(MissingCertificate, match="--refresh"):
        BoundedEventRuleRunnerMonitor.compile(
            "(F a) U (G b)", certificate="cached"
        )


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_streaming_latency_and_finite_suffix_flush(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
) -> None:
    monitor = monitor_cls.compile("X X a")
    assert monitor.step({}) is Verdict.UNDECIDED
    assert monitor.step({}) is Verdict.UNDECIDED
    assert monitor.step({"a": True}) is Verdict.SATISFY

    monitor.reset()
    assert monitor.step({}) is Verdict.UNDECIDED
    assert monitor.final_verdict() is Verdict.VIOLATE


@pytest.mark.parametrize("monitor_cls", MONITORS)
@pytest.mark.parametrize(
    ("formula", "atoms", "max_length"),
    [
        ("WX a", ("a",), 4),
        ("F (a & b)", ("a", "b"), 3),
        ("G X a", ("a",), 4),
        ("F (a & X b)", ("a", "b"), 3),
        ("G (a -> X b)", ("a", "b"), 3),
        ("a U (b & X c)", ("a", "b", "c"), 3),
        ("(X a) & X (X a)", ("a",), 4),
    ],
)
def test_extrapolated_online_labels_match_original_dfa_exhaustively(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
    formula: str,
    atoms: tuple[str, ...],
    max_length: int,
) -> None:
    monitor = monitor_cls.compile(formula, exact_online=True)
    oracle = SymbolicDFAMonitor.compile(formula)
    alphabet = _valuations(atoms)

    for length in range(1, max_length + 1):
        for trace in product(alphabet, repeat=length):
            monitor.reset()
            oracle.reset()
            for cell in trace:
                assert monitor.step(cell) is oracle.step(cell), (formula, trace)


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_batch_api_matches_sequential_runs(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
) -> None:
    traces = [
        [],
        [{}],
        [{"a": True}],
        [{"a": True}, {"b": True}],
        [{}, {"b": True}, {"a": True}],
    ]
    batched = monitor_cls.compile("F (a & X b)").batch_run(traces)
    sequential = [monitor_cls.compile("F (a & X b)").run(t) for t in traces]
    assert batched == sequential


@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_batch_path_is_fused_and_never_calls_sequential_run(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = monitor_cls.compile("F (a & X b)")
    traces = [
        [{"a": True}, {"b": True}],
        [{}, {"b": True}, {"a": True}],
    ]
    expected = [SymbolicDFAMonitor.compile(monitor.formula).run(t) for t in traces]
    calls = 0
    skeleton_batch = monitor.skeleton.batch_run

    def batch_spy(*args: object, **kwargs: object) -> list[Verdict]:
        nonlocal calls
        calls += 1
        return skeleton_batch(*args, **kwargs)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the fused batch path called a sequential API")

    monkeypatch.setattr(monitor, "run", forbidden)
    monkeypatch.setattr(monitor.event_net, "evaluate", forbidden)
    monkeypatch.setattr(monitor.skeleton, "batch_run", batch_spy)

    assert monitor.batch_run(traces) == expected
    assert calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("monitor_cls", MONITORS)
def test_cuda_pipeline_matches_cpu(
    monitor_cls: type[BoundedEventRuleRunnerMonitor],
) -> None:
    formula = "G (a -> X b)"
    trace = [{"a": True}, {"b": True}, {}]
    cpu = monitor_cls.compile(formula, device="cpu")
    cuda = monitor_cls.compile(formula, device="cuda")
    assert cuda.effective_device == "cuda"
    assert cuda.run(trace) is cpu.run(trace)
    traces = [trace, [], [{"a": True}, {"b": False}]]
    assert cuda.batch_run(traces) == cpu.batch_run(traces)
