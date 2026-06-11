"""Unit tests for the strict schemas, extractor parsing, and pipeline validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.semantic_command_extractor import (
    SemanticExtractorInvalidResponseError,
    _parse_message_content,
)
from app.semantic_command_pipeline import (
    DispatchOutcome,
    MixedIntentOutcome,
    SuggestionOutcome,
    _normalize_and_validate_changes,
)
from app.semantic_command_schemas import SemanticChanges, SemanticCommand, SemanticTarget


# ── Schema strictness (extra="forbid") ────────────────────────────────────────

def test_schema_rejects_unknown_top_level_key():
    with pytest.raises(ValidationError):
        SemanticCommand.model_validate({"intent": "update_application", "bogus": 1})


def test_schema_rejects_unknown_changes_key():
    with pytest.raises(ValidationError):
        SemanticCommand.model_validate(
            {"intent": "update_application", "changes": {"company": "X"}}
        )  # company is identity, not a change → unknown key in changes


def test_schema_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        SemanticCommand.model_validate({"intent": "frobnicate"})


def test_schema_rejects_company_in_changes():
    with pytest.raises(ValidationError):
        SemanticChanges.model_validate({"company": "Acme"})


def test_schema_accepts_minimal_unsupported():
    cmd = SemanticCommand.model_validate({"intent": "unsupported"})
    assert cmd.intent == "unsupported"
    assert cmd.note is None


def test_schema_coerces_null_target_and_changes():
    # Models routinely emit explicit nulls for "no target / no changes".
    cmd = SemanticCommand.model_validate(
        {"intent": "unsupported", "target": None, "changes": None, "note": None}
    )
    assert isinstance(cmd.target, SemanticTarget)
    assert isinstance(cmd.changes, SemanticChanges)
    assert not cmd.changes.has_any_field()


# ── Extractor JSON parsing ────────────────────────────────────────────────────

def test_parse_message_content_rejects_invalid_json():
    with pytest.raises(SemanticExtractorInvalidResponseError):
        _parse_message_content({"message": {"content": "not json"}})


def test_parse_message_content_rejects_missing_message():
    with pytest.raises(SemanticExtractorInvalidResponseError):
        _parse_message_content({})


def test_parse_message_content_accepts_object():
    out = _parse_message_content({"message": {"content": '{"intent": "unsupported"}'}})
    assert out == {"intent": "unsupported"}


# ── Alias normalization + enum validation ─────────────────────────────────────

def test_normalize_aliases_canonicalizes():
    raw = SemanticChanges(
        priority="high",
        location_mode="onsite",
        employment_types=["fulltime", "part time"],
        status="in touch",
        current_stages=["tailored", "networked"],
    )
    result = _normalize_and_validate_changes(raw)
    assert not result.invalid
    assert result.changes.priority == "HIGH"
    assert result.changes.location_mode == "on-site"
    assert result.changes.employment_types == ["Full Time", "Part Time"]
    assert result.changes.status == "in_touch"
    assert result.changes.current_stages == ["Tailored", "Networked"]


def test_normalize_flags_invalid_enum():
    result = _normalize_and_validate_changes(SemanticChanges(priority="urgent"))
    assert result.invalid
    assert result.changes.priority is None


# ── Pipeline intent guards (pure, no DB needed) ───────────────────────────────

def test_create_with_note_is_mixed_intent():
    from app.semantic_command_pipeline import _handle_create
    cmd = SemanticCommand(
        intent="create_application",
        target=SemanticTarget(company="Acme", role="AI Engineer"),
        note="hi",
    )
    assert isinstance(_handle_create(cmd), MixedIntentOutcome)


def test_create_requires_company():
    from app.semantic_command_pipeline import _handle_create
    cmd = SemanticCommand(intent="create_application", target=SemanticTarget(role="AI Engineer"))
    assert isinstance(_handle_create(cmd), SuggestionOutcome)


def test_create_happy_path_builds_payload():
    from app.semantic_command_pipeline import _handle_create
    cmd = SemanticCommand(
        intent="create_application",
        target=SemanticTarget(company="Acme", role="AI Engineer"),
        changes=SemanticChanges(status="applied", priority="high"),
    )
    out = _handle_create(cmd)
    assert isinstance(out, DispatchOutcome)
    assert out.payload.operation == "create_draft"
    assert out.payload.changes.company == "Acme"
    assert out.payload.changes.role == "AI Engineer"
    assert out.payload.changes.status == "applied"
    assert out.payload.changes.priority == "HIGH"


def test_unsupported_offers_generic_examples():
    from app.semantic_command_pipeline import _handle_unsupported
    out = _handle_unsupported(SemanticCommand(intent="unsupported"))
    assert isinstance(out, SuggestionOutcome)
    assert "set priority as medium" in out.suggested_phrasings
