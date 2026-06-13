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


# ── Small-model output salvage (llama3.2:3b mis-routing) ──────────────────────

def test_lift_misplaced_note_from_changes():
    """A hallucinated changes.note is lifted to the top-level note (not rejected)."""
    from app.semantic_command_extractor import _lift_misplaced_note
    raw = {
        "intent": "append_note",
        "target": {"company": "Acme"},
        "changes": {"note": "2-3 years required", "comments": None},
        "note": None,
    }
    repaired = _lift_misplaced_note(raw)
    assert "note" not in repaired["changes"]
    assert repaired["note"] == "2-3 years required"
    # And the repaired payload now validates cleanly.
    cmd = SemanticCommand.model_validate(repaired)
    assert cmd.note == "2-3 years required"


def test_lift_misplaced_note_does_not_overwrite_existing():
    from app.semantic_command_extractor import _lift_misplaced_note
    raw = {
        "intent": "append_note",
        "changes": {"note": "from changes"},
        "note": "real note",
    }
    repaired = _lift_misplaced_note(raw)
    assert repaired["note"] == "real note"
    assert "note" not in repaired["changes"]


def test_salvage_note_from_comments_only():
    """append_note with prose mis-routed into comments is recovered as the note."""
    from app.semantic_command_pipeline import _salvage_note_from_comments
    cmd = SemanticCommand(
        intent="append_note",
        changes=SemanticChanges(comments="2-3 years required", employment_types=[], current_stages=[]),
    )
    note, residual = _salvage_note_from_comments(cmd)
    assert note == "2-3 years required"
    assert residual.comments is None


def test_salvage_note_skips_when_real_field_present():
    """A genuine field alongside comments is NOT salvaged (stays mixed-intent)."""
    from app.semantic_command_pipeline import _salvage_note_from_comments
    cmd = SemanticCommand(
        intent="append_note",
        changes=SemanticChanges(comments="some prose", priority="HIGH"),
    )
    note, residual = _salvage_note_from_comments(cmd)
    assert note is None
    assert residual.comments == "some prose"


def test_reconcile_location_employment_mixup_folds_into_employment():
    """location_mode='full-time' (an employment type) is moved out of location."""
    result = _normalize_and_validate_changes(SemanticChanges(location_mode="full-time"))
    assert not result.invalid
    assert result.changes.location_mode is None
    assert result.changes.employment_types == ["Full Time"]


def test_reconcile_location_employment_mixup_drops_duplicate():
    """When employment_types already has it, the bad location is just dropped."""
    result = _normalize_and_validate_changes(
        SemanticChanges(location_mode="fulltime", employment_types=["Full Time"])
    )
    assert not result.invalid
    assert result.changes.location_mode is None
    assert result.changes.employment_types == ["Full Time"]


def test_reconcile_leaves_real_location_untouched():
    result = _normalize_and_validate_changes(SemanticChanges(location_mode="onsite"))
    assert not result.invalid
    assert result.changes.location_mode == "on-site"
    assert result.changes.employment_types is None


# ── Extractor context block (selected-application injection) ───────────────────

def test_context_block_injects_selected_application():
    from app.semantic_command_extractor import _build_user_message
    ctx = {"active_application": {"id": 1, "company": "Kody Technolab", "role": "AI Engineer"}}
    msg = _build_user_message("add a note saying recruiter replied", ctx)
    assert "Selected application: Kody Technolab — AI Engineer" in msg
    assert "Command: add a note saying recruiter replied" in msg


def test_context_block_handles_no_selection():
    from app.semantic_command_extractor import _build_user_message
    msg = _build_user_message("set priority high", None)
    assert "Selected application: none" in msg
    assert "Draft active: no" in msg


def test_context_block_omits_known_applications_dump():
    """The noisy full applications list must never be sent to the model."""
    from app.semantic_command_extractor import _build_user_message
    ctx = {
        "active_application": None,
        "known_applications": [{"company": "Globex"}, {"company": "Initech"}],
    }
    msg = _build_user_message("bump devon", ctx)
    assert "Globex" not in msg
    assert "Initech" not in msg
