"""Cached exact certificates for original-RuleRunner skeletons.

``certify_rule_runner`` is exact but expensive: it compiles the canonical DFA
with MONA and explores the synchronous product over ``2^|P|``.  Paying that at
every ``compile()`` would put a DFA construction inside the compile-time cost of
the very construction that claims to avoid one, biasing any measurement of the
bounded-event monitor against itself.

The certificates are therefore an *offline artifact*: they are computed once by
``python -m src.monitors.rulerunner.certificates --refresh``, stored in
``certificates.json`` next to this module, and looked up at compile time.  A
normal monitor compilation never modifies that artifact; only the explicit
``--refresh`` maintenance command writes it.

Staleness is the obvious risk of a checked-in safety claim, so every record
stores a fingerprint of the rule system it certifies.  Rebuilding that rule
system is cheap (no MONA, no product exploration), so the fingerprint is
verified on every lookup and a certificate that no longer matches the current
``rules.py`` is treated as absent rather than trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.monitors.base import Verdict
from src.monitors.rulerunner.bounded import eventize_bounded_islands
from src.monitors.rulerunner.equivalence import (
    EquivalenceWitness,
    RuleRunnerEquivalenceResult,
    certify_rule_runner,
)
from src.monitors.rulerunner.parse_tree import parse
from src.monitors.rulerunner.rules import build_rules

CERTIFICATE_PATH = Path(__file__).with_name("certificates.json")

#: Original formulas whose skeletons the checked-in artifact covers.  Extend
#: this list and re-run ``--refresh`` when a new bounded formula is used.
DEFAULT_FORMULAS: tuple[str, ...] = (
    "a",
    "X a",
    "WX a",
    "X X a",
    "G X a",
    "F (a & b)",
    "F (a & X b)",
    "F (a & X X b)",
    "G (a -> X b)",
    "G (a | X b)",
    "a U (b & X c)",
    "(X a) & X (X a)",
    "G (a -> X (b | WX c))",
    "G (a -> (b | X b | X X b))",
)


class MissingCertificate(LookupError):
    """No fresh cached certificate exists and recomputation was not allowed."""


@dataclass(frozen=True)
class Certificate:
    """A stored :class:`RuleRunnerEquivalenceResult`.

    The field names mirror the live result so that both can be consumed
    interchangeably by callers and error messages.
    """

    formula: str
    fingerprint: str
    atoms: tuple[str, ...]
    explored_product_states: int
    language_witness: EquivalenceWitness | None
    unsound_prefix_witness: EquivalenceWitness | None
    online_witness: EquivalenceWitness | None

    #: Stored certificates are always the result of a complete exploration;
    #: resource-bounded runs are never persisted.
    complete: bool = True

    @property
    def language_equivalent(self) -> bool:
        return self.language_witness is None

    @property
    def prefix_sound(self) -> bool:
        return self.unsound_prefix_witness is None

    @property
    def online_equivalent(self) -> bool:
        return self.online_witness is None


def rule_system_fingerprint(formula: str) -> str:
    """Fingerprint the compiled rule system of ``formula``.

    Cheap by design: it builds the rule system only, so a lookup never pays the
    certification cost it is meant to avoid.
    """
    rules = build_rules(parse(formula))

    def dump_literal(literal: object) -> str:
        return f"{'~' if literal.negated else ''}{literal.name}"  # type: ignore[attr-defined]

    def dump_rule(rule: object) -> str:
        body = ",".join(sorted(dump_literal(lit) for lit in rule.body))  # type: ignore[attr-defined]
        return f"{body}=>{dump_literal(rule.head)}"  # type: ignore[attr-defined]

    payload = "\n".join(
        (
            rules.root_key,
            ";".join(sorted(rules.atoms)),
            ";".join(sorted(dump_literal(lit) for lit in rules.initial_state)),
            ";".join(sorted(dump_rule(rule) for rule in rules.eval_rules)),
            ";".join(sorted(dump_rule(rule) for rule in rules.react_rules)),
        )
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _witness_to_json(witness: EquivalenceWitness | None) -> dict | None:
    if witness is None:
        return None
    return {
        "trace": [sorted(cell) for cell in witness.trace],
        "rulerunner": witness.rulerunner.name,
        "dfa": witness.dfa.name,
    }


def _witness_from_json(payload: dict | None) -> EquivalenceWitness | None:
    if payload is None:
        return None
    return EquivalenceWitness(
        trace=tuple(frozenset(cell) for cell in payload["trace"]),
        rulerunner=Verdict[payload["rulerunner"]],
        dfa=Verdict[payload["dfa"]],
    )


def certificate_from_result(
    result: RuleRunnerEquivalenceResult, fingerprint: str
) -> Certificate:
    if not result.complete:
        raise ValueError(
            f"Refusing to store an incomplete certificate for {result.formula!r}."
        )
    return Certificate(
        formula=result.formula,
        fingerprint=fingerprint,
        atoms=result.atoms,
        explored_product_states=result.explored_product_states,
        language_witness=result.language_witness,
        unsound_prefix_witness=result.unsound_prefix_witness,
        online_witness=result.online_witness,
    )


def _to_json(certificate: Certificate) -> dict:
    return {
        "formula": certificate.formula,
        "fingerprint": certificate.fingerprint,
        "atoms": list(certificate.atoms),
        "explored_product_states": certificate.explored_product_states,
        "language_witness": _witness_to_json(certificate.language_witness),
        "unsound_prefix_witness": _witness_to_json(
            certificate.unsound_prefix_witness
        ),
        "online_witness": _witness_to_json(certificate.online_witness),
    }


def _from_json(payload: dict) -> Certificate:
    return Certificate(
        formula=payload["formula"],
        fingerprint=payload["fingerprint"],
        atoms=tuple(payload["atoms"]),
        explored_product_states=payload["explored_product_states"],
        language_witness=_witness_from_json(payload["language_witness"]),
        unsound_prefix_witness=_witness_from_json(
            payload["unsound_prefix_witness"]
        ),
        online_witness=_witness_from_json(payload["online_witness"]),
    )


@lru_cache(maxsize=1)
def _store() -> dict[str, Certificate]:
    if not CERTIFICATE_PATH.exists():
        return {}
    payload = json.loads(CERTIFICATE_PATH.read_text())
    return {
        entry["formula"]: _from_json(entry)
        for entry in payload.get("certificates", [])
    }


def lookup(formula: str) -> Certificate | None:
    """Return the stored certificate for ``formula`` if it is still current."""
    certificate = _store().get(formula)
    if certificate is None:
        return None
    if certificate.fingerprint != rule_system_fingerprint(formula):
        return None  # rules.py changed; the stored claim no longer applies
    return certificate


def ensure_certificate(
    formula: str, mode: str = "auto"
) -> tuple[Certificate, str]:
    """Obtain a certificate for ``formula`` and report where it came from.

    ``mode`` selects the policy:

    * ``"cached"`` — use the artifact only.  Timing experiments must use this
      so that no compile-time measurement hides a DFA construction.
    * ``"auto"`` (default) — use the artifact when it is current, otherwise
      certify for this compilation without modifying the artifact.
    * ``"recompute"`` — always certify now; the artifact is not consulted.

    The second element of the result is ``"cache"`` or ``"computed"``.
    """
    if mode not in ("auto", "cached", "recompute"):
        raise ValueError(f"unknown certificate mode {mode!r}")

    if mode != "recompute":
        certificate = lookup(formula)
        if certificate is not None:
            return certificate, "cache"
        if mode == "cached":
            raise MissingCertificate(
                f"No current certificate for skeleton {formula!r}. Add the "
                f"formula to DEFAULT_FORMULAS and run "
                f"`python -m src.monitors.rulerunner.certificates --refresh`, "
                f"or pass certificate='auto' to certify at compile time."
            )

    result = certify_rule_runner(formula)
    certificate = certificate_from_result(
        result, rule_system_fingerprint(formula)
    )
    return certificate, "computed"


def _persist(certificates: dict[str, Certificate]) -> None:
    payload = {
        "note": (
            "Generated by `python -m src.monitors.rulerunner.certificates "
            "--refresh`. Each record certifies one original-RuleRunner "
            "skeleton against its canonical DFA; `fingerprint` pins the "
            "rule system the claim was checked against."
        ),
        "certificates": [
            _to_json(certificates[formula]) for formula in sorted(certificates)
        ],
    }
    CERTIFICATE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    _store.cache_clear()


def refresh(formulas: tuple[str, ...] = DEFAULT_FORMULAS) -> dict[str, Certificate]:
    """Recertify the skeletons of ``formulas`` and rewrite the artifact."""
    certificates: dict[str, Certificate] = {}
    for formula in formulas:
        skeleton = eventize_bounded_islands(formula).skeleton
        if skeleton in certificates:
            continue
        result = certify_rule_runner(skeleton)
        certificates[skeleton] = certificate_from_result(
            result, rule_system_fingerprint(skeleton)
        )
    _persist(certificates)
    return certificates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recertify every skeleton and rewrite certificates.json",
    )
    parser.add_argument(
        "formulas",
        nargs="*",
        help="extra original formulas to include (skeletons are derived)",
    )
    args = parser.parse_args()
    if not args.refresh:
        parser.error("nothing to do; pass --refresh")

    certificates = refresh(DEFAULT_FORMULAS + tuple(args.formulas))
    for skeleton in sorted(certificates):
        certificate = certificates[skeleton]
        flags = (
            f"language={certificate.language_equivalent} "
            f"prefix_sound={certificate.prefix_sound} "
            f"online={certificate.online_equivalent}"
        )
        print(f"{skeleton!r}: {flags} ({certificate.explored_product_states} states)")


if __name__ == "__main__":
    main()
