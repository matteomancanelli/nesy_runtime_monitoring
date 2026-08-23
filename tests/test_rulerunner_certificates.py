"""The checked-in certificate artifact must stay exact and current.

A stored certificate is a safety claim: the bounded-event monitors admit a
skeleton because the artifact says the original RuleRunner is language
equivalent and prefix sound on it.  These tests re-derive every stored record
from scratch, so the artifact cannot silently drift away from ``rules.py``.
"""

from __future__ import annotations

import pytest

from src.monitors.rulerunner.bounded import eventize_bounded_islands
from src.monitors.rulerunner.certificates import (
    DEFAULT_FORMULAS,
    Certificate,
    MissingCertificate,
    certificate_from_result,
    ensure_certificate,
    lookup,
    rule_system_fingerprint,
)
from src.monitors.rulerunner.equivalence import certify_rule_runner


def _skeletons() -> list[str]:
    seen: list[str] = []
    for formula in DEFAULT_FORMULAS:
        skeleton = eventize_bounded_islands(formula).skeleton
        if skeleton not in seen:
            seen.append(skeleton)
    return seen


@pytest.mark.parametrize("skeleton", _skeletons())
def test_stored_certificate_matches_a_fresh_exact_check(skeleton: str) -> None:
    stored = lookup(skeleton)
    assert stored is not None, (
        f"{skeleton!r} is missing from certificates.json; run "
        "`python -m src.monitors.rulerunner.certificates --refresh`"
    )
    fresh = certificate_from_result(
        certify_rule_runner(skeleton), rule_system_fingerprint(skeleton)
    )
    assert stored == fresh


def test_fingerprint_mismatch_invalidates_a_stored_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule-system change must void the claim rather than be trusted."""
    import src.monitors.rulerunner.certificates as certificates

    skeleton = _skeletons()[0]
    assert lookup(skeleton) is not None

    monkeypatch.setattr(
        certificates, "rule_system_fingerprint", lambda formula: "stale"
    )
    assert certificates.lookup(skeleton) is None
    with pytest.raises(MissingCertificate):
        certificates.ensure_certificate(skeleton, "cached")


def test_cached_mode_never_recomputes(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.monitors.rulerunner.certificates as certificates

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached mode certified at runtime")

    monkeypatch.setattr(certificates, "certify_rule_runner", forbidden)
    certificate, source = certificates.ensure_certificate(
        _skeletons()[0], "cached"
    )
    assert source == "cache"
    assert isinstance(certificate, Certificate)


def test_recompute_mode_reports_its_source() -> None:
    certificate, source = ensure_certificate("F a", "recompute")
    assert source == "computed"
    assert certificate.language_equivalent


def test_auto_mode_does_not_modify_the_offline_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the explicit refresh command may write certificates.json."""
    import src.monitors.rulerunner.certificates as certificates

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("auto mode persisted a compile-time certificate")

    assert certificates.lookup("F a") is None
    monkeypatch.setattr(certificates, "_persist", forbidden)
    certificate, source = certificates.ensure_certificate("F a", "auto")
    assert source == "computed"
    assert certificate.language_equivalent
