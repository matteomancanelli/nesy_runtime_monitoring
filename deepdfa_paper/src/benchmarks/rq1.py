"""Versioned RQ1 corpus and exact semantic characterization.

RQ1 is the semantic gate for every later performance experiment.  The corpus
contains safe controls, targeted shared-register counterexamples, bounded-event
repair/rejection cases, and the complete Declare suite.  Each construction is
checked against the canonical finite-trace DFA rather than inferred from a few
sample traces.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import product
from typing import Any

import torch

from src.benchmarks.formulas import DECLARE_SUITE, BenchmarkFormula, ijcnn_formula
from src.benchmarks.schema import (
    DEFAULT_RESOURCE_BUDGETS,
    RESULT_SCHEMA_VERSION,
    characterize_formula,
)
from src.formula.compiler import DFA, Observation, compile_ltlf
from src.monitors.base import Verdict
from src.monitors.progression.eager import ProgressionDFA, build_progression_dfa
from src.monitors.rulerunner.bounded import EventizedFormula, eventize_bounded_islands
from src.monitors.rulerunner.bounded_extrapolation import (
    BoundedExtrapolationCILP,
    ExtrapolationLimitExceeded,
)
from src.monitors.rulerunner.equivalence import (
    EquivalenceWitness,
    certify_rule_runner,
)

RQ1_SCHEMA_VERSION = "rq1.v1"
RQ1_CONSTRUCTIONS = (
    "original",
    "bounded_default",
    "bounded_exact_online",
    "progression",
)


class CharacterizationStatus(str, Enum):
    """Outcome of one formula/construction characterization."""

    CHARACTERIZED = "characterized"
    REJECTED = "rejected"
    RESOURCE_LIMITED = "resource_limited"
    ERROR = "error"


@dataclass(frozen=True)
class RQ1Case:
    benchmark: BenchmarkFormula
    categories: tuple[str, ...]
    rationale: str


def _case_formula(
    name: str,
    formula: str,
    atoms: tuple[str, ...],
    *,
    family: str,
    roles: tuple[str, ...],
) -> BenchmarkFormula:
    return BenchmarkFormula(
        name=name,
        formula=formula,
        atoms=atoms,
        family=family,
        source="RQ1 repair-ladder corpus",
        roles=roles,
    )


_DECLARE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "response": ("declare", "shared_register_counterexample", "bounded_rejection"),
    "chain_response": (
        "declare",
        "shared_register_counterexample",
        "bounded_repair_target",
    ),
    "precedence": ("declare", "original_safe_control"),
    "alt_response": (
        "declare",
        "shared_register_counterexample",
        "bounded_rejection",
    ),
    "resp_existence": ("declare", "original_safe_control"),
    "not_coexistence": ("declare", "original_safe_control"),
    "chain_precedence": (
        "declare",
        "shared_register_counterexample",
        "bounded_repair_target",
    ),
}


RQ1_CORPUS: tuple[RQ1Case, ...] = (
    RQ1Case(
        _case_formula(
            "atomic_control",
            "a",
            ("a",),
            family="repair_control",
            roles=("semantic_boundary",),
        ),
        ("original_safe_control",),
        "Propositional base case with no temporal recurrence.",
    ),
    RQ1Case(
        _case_formula(
            "nested_next_control",
            "X X a",
            ("a",),
            family="repair_control",
            roles=("semantic_boundary", "bounded_horizon"),
        ),
        ("original_safe_control", "bounded_repair_control"),
        "Nested Next is safe when its offsets have distinct subformula keys.",
    ),
    RQ1Case(
        _case_formula(
            "nested_eventually_control",
            "F F a",
            ("a",),
            family="repair_control",
            roles=("semantic_boundary",),
        ),
        ("original_safe_control",),
        "Nested temporal syntax that collapses semantically without aliasing.",
    ),
    RQ1Case(
        _case_formula(
            "immediate_response_control",
            "G (a -> b)",
            ("a", "b"),
            family="repair_control",
            roles=("semantic_boundary",),
        ),
        ("original_safe_control",),
        "A recurring obligation whose consequent is resolved in the same cell.",
    ),
    RQ1Case(
        _case_formula(
            "flat_until_control",
            "a U b",
            ("a", "b"),
            family="repair_control",
            roles=("semantic_boundary",),
        ),
        ("original_safe_control",),
        "Flat temporal control from the proved sufficient fragment.",
    ),
    RQ1Case(
        ijcnn_formula(4),
        ("original_safe_control", "ijcnn_control"),
        "Paper-faithful balanced IJCNN control used by later timing experiments.",
    ),
    RQ1Case(
        _case_formula(
            "next_offset_alias",
            "(X a) & X (X a)",
            ("a",),
            family="repair_counterexample",
            roles=("semantic_boundary", "bounded_horizon"),
        ),
        ("shared_register_counterexample", "bounded_repair_target"),
        "The smallest genuine two-offset shared-register collision.",
    ),
    RQ1Case(
        _case_formula(
            "globally_next_alias",
            "G X a",
            ("a",),
            family="repair_counterexample",
            roles=("semantic_boundary", "bounded_horizon"),
        ),
        ("shared_register_counterexample", "bounded_repair_target"),
        "Finite-boundary failure repeatedly reinstalling a bounded obligation.",
    ),
    RQ1Case(
        _case_formula(
            "eventual_next_alias",
            "F (a & X b)",
            ("a", "b"),
            family="repair_counterexample",
            roles=("semantic_boundary", "bounded_horizon"),
        ),
        ("shared_register_counterexample", "bounded_repair_target"),
        "Canonical overlapping-instance counterexample from the paper draft.",
    ),
    RQ1Case(
        _case_formula(
            "until_next_alias",
            "a U (b & X c)",
            ("a", "b", "c"),
            family="repair_counterexample",
            roles=("semantic_boundary", "bounded_horizon"),
        ),
        ("shared_register_counterexample", "bounded_repair_target"),
        "An Until obligation whose bounded consequent aliases across instances.",
    ),
    *tuple(
        RQ1Case(
            formula,
            _DECLARE_CATEGORIES[formula.name],
            "Declare template included to measure external-validity coverage.",
        )
        for formula in DECLARE_SUITE
    ),
)


@dataclass(frozen=True)
class SemanticWitness:
    """Shortest exact-product witness for one semantic comparison."""

    trace: tuple[frozenset[str], ...]
    construction: Verdict
    canonical: Verdict

    @property
    def length(self) -> int:
        return len(self.trace)

    def flat_json(self) -> str:
        return json.dumps(
            {
                "trace": [sorted(cell) for cell in self.trace],
                "length": self.length,
                "construction": self.construction.name,
                "canonical": self.canonical.name,
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class _ProductAnalysis:
    explored_product_states: int
    language_witness: SemanticWitness | None
    prefix_witness: SemanticWitness | None
    online_witness: SemanticWitness | None
    exact_online_witness: SemanticWitness | None = None

    @property
    def language_equivalent(self) -> bool:
        return self.language_witness is None

    @property
    def prefix_sound(self) -> bool:
        return self.prefix_witness is None

    @property
    def online_equivalent(self) -> bool:
        return self.online_witness is None

    @property
    def exact_head_equivalent(self) -> bool:
        return self.exact_online_witness is None


@dataclass(frozen=True)
class RQ1Record:
    """One formula/construction row in the frozen RQ1 artifact."""

    schema_version: str
    benchmark_schema_version: str
    formula_id: str
    formula: str
    family: str
    source: str
    parameters_json: str
    tree_shape: str
    categories_json: str
    rationale: str
    n_atoms: int
    ast_nodes: int
    distinct_subformulae: int
    ast_depth: int
    temporal_depth: int
    dfa_states: int
    dfa_guarded_transitions: int
    construction: str
    status: CharacterizationStatus
    applicable: bool
    failure_reason: str = ""
    final_language_equivalent: bool | None = None
    prefix_sound: bool | None = None
    exact_online_equivalent: bool | None = None
    explored_product_states: int | None = None
    certificate_formula: str = ""
    certificate_complete: bool | None = None
    certificate_product_states: int | None = None
    bounded_horizon: int | None = None
    bounded_islands: int | None = None
    bounded_exact_head_states: int | None = None
    progression_states: int | None = None
    progression_roots: int | None = None
    progression_closure: int | None = None
    progression_input_subformulae: int | None = None
    language_witness_json: str = ""
    prefix_witness_json: str = ""
    online_witness_json: str = ""

    def flat_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row


def _symbols(atoms: tuple[str, ...]) -> tuple[tuple[frozenset[str], Observation], ...]:
    return tuple(
        (
            frozenset(atom for atom, value in zip(atoms, values) if value),
            {atom: value for atom, value in zip(atoms, values)},
        )
        for values in product((False, True), repeat=len(atoms))
    )


def _dfa_online(dfa: DFA, state: int) -> Verdict:
    if state in dfa.trap_states:
        return Verdict.VIOLATE
    if state in dfa.accepting_sinks:
        return Verdict.SATISFY
    return Verdict.UNDECIDED


def _final_verdict(accepting: bool) -> Verdict:
    return Verdict.SATISFY if accepting else Verdict.VIOLATE


def _witness_json(witness: SemanticWitness | EquivalenceWitness | None) -> str:
    if witness is None:
        return ""
    if isinstance(witness, EquivalenceWitness):
        witness = SemanticWitness(witness.trace, witness.rulerunner, witness.dfa)
    return witness.flat_json()


def _analyze_progression(dfa: DFA, progression: ProgressionDFA) -> _ProductAnalysis:
    atoms = tuple(sorted(set(dfa.atoms) | set(progression.atoms)))
    alphabet = _symbols(atoms)
    initial = (progression.initial, dfa.initial)
    queue = deque([(initial, ())])
    seen = {initial}
    language_witness: SemanticWitness | None = None
    prefix_witness: SemanticWitness | None = None
    online_witness: SemanticWitness | None = None

    p_empty = progression.initial in progression.accepting
    d_empty = dfa.initial in dfa.accepting
    if p_empty != d_empty:
        language_witness = SemanticWitness(
            (), _final_verdict(p_empty), _final_verdict(d_empty)
        )

    while queue:
        (p_state, d_state), trace = queue.popleft()
        for true_atoms, obs in alphabet:
            p_symbol = progression.symbol(p_state, obs)
            next_p = progression.trans[p_state][p_symbol]
            next_d = dfa.step(d_state, obs)
            next_trace = trace + (true_atoms,)
            p_final = _final_verdict(next_p in progression.accepting)
            d_final = _final_verdict(next_d in dfa.accepting)
            p_online = (
                Verdict.VIOLATE
                if next_p in progression.trap_states
                else (
                    Verdict.SATISFY
                    if next_p in progression.accepting_sinks
                    else Verdict.UNDECIDED
                )
            )
            d_online = _dfa_online(dfa, next_d)

            if language_witness is None and p_final is not d_final:
                language_witness = SemanticWitness(next_trace, p_final, d_final)
            if online_witness is None and p_online is not d_online:
                online_witness = SemanticWitness(next_trace, p_online, d_online)
            if (
                prefix_witness is None
                and p_online is not Verdict.UNDECIDED
                and p_online is not d_online
            ):
                prefix_witness = SemanticWitness(next_trace, p_online, d_online)

            successor = (next_p, next_d)
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, next_trace))

    return _ProductAnalysis(
        explored_product_states=len(seen),
        language_witness=language_witness,
        prefix_witness=prefix_witness,
        online_witness=online_witness,
    )


def _analyze_bounded(
    dfa: DFA,
    eventized: EventizedFormula,
) -> tuple[_ProductAnalysis, int]:
    """Exactly compare the delayed pipeline and both label modes to ``dfa``."""
    head = BoundedExtrapolationCILP(
        eventized,
        torch.device("cpu"),
        max_atoms=DEFAULT_RESOURCE_BUDGETS.max_atoms_for_exact_certification,
    )
    atoms = tuple(sorted(set(dfa.atoms) | set(eventized.atoms)))
    alphabet = _symbols(atoms)
    pipeline_alphabet = tuple(cell for cell, _ in _symbols(eventized.atoms))
    pipeline_symbol_index = {
        cell: index for index, cell in enumerate(pipeline_alphabet)
    }
    initial_pipeline = head._initial_state()
    initial = (initial_pipeline, dfa.initial)
    queue = deque([(initial, ())])
    seen = {initial}
    language_witness: SemanticWitness | None = None
    prefix_witness: SemanticWitness | None = None
    default_online_witness: SemanticWitness | None = None
    exact_online_witness: SemanticWitness | None = None

    p_empty = initial_pipeline in head.accepting
    d_empty = dfa.initial in dfa.accepting
    if p_empty != d_empty:
        language_witness = SemanticWitness(
            (), _final_verdict(p_empty), _final_verdict(d_empty)
        )

    while queue:
        (pipeline_state, d_state), trace = queue.popleft()
        pipeline_successors = head.transitions[pipeline_state]
        for true_atoms, obs in alphabet:
            local_atoms = frozenset(atom for atom in eventized.atoms if obs.get(atom))
            next_pipeline = pipeline_successors[pipeline_symbol_index[local_atoms]]
            next_d = dfa.step(d_state, obs)
            next_trace = trace + (true_atoms,)
            pipeline_final = _final_verdict(next_pipeline in head.accepting)
            dfa_final = _final_verdict(next_d in dfa.accepting)
            default_online = next_pipeline.decided or Verdict.UNDECIDED
            exact_online = (
                Verdict.SATISFY
                if next_pipeline in head.accepting_sinks
                else (
                    Verdict.VIOLATE
                    if next_pipeline in head.traps
                    else Verdict.UNDECIDED
                )
            )
            dfa_online = _dfa_online(dfa, next_d)

            if language_witness is None and pipeline_final is not dfa_final:
                language_witness = SemanticWitness(
                    next_trace, pipeline_final, dfa_final
                )
            if (
                prefix_witness is None
                and default_online is not Verdict.UNDECIDED
                and default_online is not dfa_online
            ):
                prefix_witness = SemanticWitness(next_trace, default_online, dfa_online)
            if default_online_witness is None and default_online is not dfa_online:
                default_online_witness = SemanticWitness(
                    next_trace, default_online, dfa_online
                )
            if exact_online_witness is None and exact_online is not dfa_online:
                exact_online_witness = SemanticWitness(
                    next_trace, exact_online, dfa_online
                )

            successor = (next_pipeline, next_d)
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, next_trace))

    return (
        _ProductAnalysis(
            explored_product_states=len(seen),
            language_witness=language_witness,
            prefix_witness=prefix_witness,
            online_witness=default_online_witness,
            exact_online_witness=exact_online_witness,
        ),
        head.n_states,
    )


def _base_record(case: RQ1Case, construction: str) -> dict[str, Any]:
    structure = characterize_formula(case.benchmark, include_dfa=True)
    assert structure.dfa_states is not None
    assert structure.dfa_transitions is not None
    flat = structure.flat_dict()
    return {
        "schema_version": RQ1_SCHEMA_VERSION,
        "benchmark_schema_version": RESULT_SCHEMA_VERSION,
        "formula_id": case.benchmark.formula_id,
        "formula": case.benchmark.formula,
        "family": case.benchmark.family,
        "source": case.benchmark.source,
        "parameters_json": flat["parameters"],
        "tree_shape": case.benchmark.tree_shape,
        "categories_json": json.dumps(case.categories),
        "rationale": case.rationale,
        "n_atoms": structure.n_atoms,
        "ast_nodes": structure.ast_nodes,
        "distinct_subformulae": structure.distinct_subformulae,
        "ast_depth": structure.ast_depth,
        "temporal_depth": structure.temporal_depth,
        "dfa_states": structure.dfa_states,
        "dfa_guarded_transitions": structure.dfa_transitions,
        "construction": construction,
    }


def characterize_case(case: RQ1Case) -> tuple[RQ1Record, ...]:
    """Return the four exact construction records for one corpus case."""
    formula = case.benchmark.formula
    dfa = compile_ltlf(formula)
    records: list[RQ1Record] = []

    original = certify_rule_runner(
        formula,
        max_atoms=DEFAULT_RESOURCE_BUDGETS.max_atoms_for_exact_certification,
    )
    records.append(
        RQ1Record(
            **_base_record(case, "original"),
            status=CharacterizationStatus.CHARACTERIZED,
            applicable=original.language_equivalent and original.prefix_sound,
            final_language_equivalent=original.language_equivalent,
            prefix_sound=original.prefix_sound,
            exact_online_equivalent=original.online_equivalent,
            explored_product_states=original.explored_product_states,
            certificate_formula=formula,
            certificate_complete=original.complete,
            certificate_product_states=original.explored_product_states,
            language_witness_json=_witness_json(original.language_witness),
            prefix_witness_json=_witness_json(original.unsound_prefix_witness),
            online_witness_json=_witness_json(original.online_witness),
        )
    )

    eventized = eventize_bounded_islands(formula)
    skeleton = certify_rule_runner(
        eventized.skeleton,
        max_atoms=DEFAULT_RESOURCE_BUDGETS.max_atoms_for_exact_certification,
    )
    bounded_admitted = skeleton.language_equivalent and skeleton.prefix_sound
    bounded_common = {
        "certificate_formula": eventized.skeleton,
        "certificate_complete": skeleton.complete,
        "certificate_product_states": skeleton.explored_product_states,
        "bounded_horizon": eventized.horizon,
        "bounded_islands": len(eventized.islands),
    }
    if not bounded_admitted:
        witness = skeleton.language_witness or skeleton.unsound_prefix_witness
        for construction in ("bounded_default", "bounded_exact_online"):
            records.append(
                RQ1Record(
                    **_base_record(case, construction),
                    status=CharacterizationStatus.REJECTED,
                    applicable=False,
                    failure_reason="uncertified original-RuleRunner skeleton",
                    **bounded_common,
                    language_witness_json=_witness_json(witness),
                )
            )
    else:
        try:
            bounded, head_states = _analyze_bounded(dfa, eventized)
        except ExtrapolationLimitExceeded as exc:
            for construction in ("bounded_default", "bounded_exact_online"):
                records.append(
                    RQ1Record(
                        **_base_record(case, construction),
                        status=CharacterizationStatus.RESOURCE_LIMITED,
                        applicable=construction == "bounded_default",
                        failure_reason=str(exc),
                        **bounded_common,
                    )
                )
        else:
            records.append(
                RQ1Record(
                    **_base_record(case, "bounded_default"),
                    status=CharacterizationStatus.CHARACTERIZED,
                    applicable=True,
                    final_language_equivalent=bounded.language_equivalent,
                    prefix_sound=bounded.prefix_sound,
                    exact_online_equivalent=bounded.online_equivalent,
                    explored_product_states=bounded.explored_product_states,
                    bounded_exact_head_states=head_states,
                    **bounded_common,
                    language_witness_json=_witness_json(bounded.language_witness),
                    prefix_witness_json=_witness_json(bounded.prefix_witness),
                    online_witness_json=_witness_json(bounded.online_witness),
                )
            )
            records.append(
                RQ1Record(
                    **_base_record(case, "bounded_exact_online"),
                    status=CharacterizationStatus.CHARACTERIZED,
                    applicable=True,
                    final_language_equivalent=bounded.language_equivalent,
                    prefix_sound=(
                        bounded.language_equivalent and bounded.exact_head_equivalent
                    ),
                    exact_online_equivalent=bounded.exact_head_equivalent,
                    explored_product_states=bounded.explored_product_states,
                    bounded_exact_head_states=head_states,
                    **bounded_common,
                    language_witness_json=_witness_json(bounded.language_witness),
                    online_witness_json=_witness_json(bounded.exact_online_witness),
                )
            )

    try:
        progression = build_progression_dfa(formula)
        progression_analysis = _analyze_progression(dfa, progression)
    except (ValueError, MemoryError) as exc:
        records.append(
            RQ1Record(
                **_base_record(case, "progression"),
                status=CharacterizationStatus.RESOURCE_LIMITED,
                applicable=False,
                failure_reason=str(exc),
            )
        )
    else:
        records.append(
            RQ1Record(
                **_base_record(case, "progression"),
                status=CharacterizationStatus.CHARACTERIZED,
                applicable=True,
                final_language_equivalent=progression_analysis.language_equivalent,
                prefix_sound=progression_analysis.prefix_sound,
                exact_online_equivalent=progression_analysis.online_equivalent,
                explored_product_states=progression_analysis.explored_product_states,
                progression_states=progression.n_states,
                progression_roots=progression.n_roots,
                progression_closure=progression.n_closure,
                progression_input_subformulae=progression.n_input_sub,
                language_witness_json=_witness_json(
                    progression_analysis.language_witness
                ),
                prefix_witness_json=_witness_json(progression_analysis.prefix_witness),
                online_witness_json=_witness_json(progression_analysis.online_witness),
            )
        )

    assert tuple(record.construction for record in records) == RQ1_CONSTRUCTIONS
    return tuple(records)


def characterize_corpus() -> tuple[RQ1Record, ...]:
    """Characterize the complete corpus in stable formula/construction order."""
    return tuple(record for case in RQ1_CORPUS for record in characterize_case(case))


def corpus_category_counts() -> dict[str, int]:
    counts = Counter(category for case in RQ1_CORPUS for category in case.categories)
    return dict(sorted(counts.items()))
