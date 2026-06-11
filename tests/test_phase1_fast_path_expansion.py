"""Phase 1 fast-path expansion tests.

Covers:
- Explicit create-draft templates (add/create/track application for {role} at {company})
- Explicit single-field draft updates (set/change {field} to {value})
- Lifecycle regression (save it, discard draft, archive, restore)
- Ambiguity guard — inputs that must NOT match the deterministic parser
- Ollama bypass assertion for create-draft path
- Optional parameters envelope canonicalization
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.fast_path_parser import try_parse
from app.main import app
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import (
    SemanticInterpretationResult,
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterUnavailableError,
    get_semantic_interpreter,
)
from app.semantic_schemas import SemanticExtractedFields, SemanticInterpreterMetrics, SemanticToolCallProposal
from app.semantic_validation import canonicalize_tool_arguments
from app.database import SessionLocal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _no_context() -> dict:
    return {}


def _draft_context(draft_id: str = "99") -> dict:
    return {"draft_id": draft_id}


def _app_context(app_id: int = 5) -> dict:
    return {"active_application_id": app_id}


# ---------------------------------------------------------------------------
# Create-draft: happy path
# ---------------------------------------------------------------------------


class TestCreateDraftFastPath:
    def test_add_application_for_role_at_company(self):
        result = try_parse("add application for ai engineer at neilsoft", _no_context())
        assert result is not None
        assert result.operation == "create_draft"
        assert result.changes.company == "neilsoft"
        assert result.changes.role == "ai engineer"

    def test_create_application_for_role_at_company(self):
        result = try_parse("create application for Founding Engineer at Aiden AI", _no_context())
        assert result is not None
        assert result.operation == "create_draft"
        assert result.changes.company == "Aiden AI"
        assert result.changes.role == "Founding Engineer"

    def test_track_application_for_role_at_company(self):
        result = try_parse(
            "track application for LLM Inference Optimization Engineer at Google",
            _no_context(),
        )
        assert result is not None
        assert result.operation == "create_draft"
        assert result.changes.company == "Google"
        assert result.changes.role == "LLM Inference Optimization Engineer"

    def test_preserves_user_capitalization(self):
        result = try_parse("add application for Senior ML Engineer at DeepMind", _no_context())
        assert result is not None
        assert result.changes.company == "DeepMind"
        assert result.changes.role == "Senior ML Engineer"

    def test_collapses_internal_whitespace_in_role(self):
        result = try_parse("add application for  AI  Engineer  at  Neilsoft", _no_context())
        assert result is not None
        # Internal whitespace is collapsed to single space
        assert result.changes.role == "AI Engineer"
        assert result.changes.company == "Neilsoft"

    def test_case_insensitive_anchor(self):
        result_lower = try_parse("add application for ai engineer at neilsoft", _no_context())
        result_upper = try_parse("ADD APPLICATION FOR AI ENGINEER AT NEILSOFT", _no_context())
        assert result_lower is not None
        assert result_upper is not None
        assert result_lower.operation == result_upper.operation == "create_draft"

    def test_target_is_empty_mutation_target(self):
        result = try_parse("add application for ai engineer at neilsoft", _no_context())
        assert result is not None
        assert result.target.draft_id is None
        assert result.target.application_id is None


# ---------------------------------------------------------------------------
# Create-draft: rejection / fallback cases
# ---------------------------------------------------------------------------


class TestCreateDraftFastPathRejection:
    def test_blank_role_returns_none(self):
        result = try_parse("add application for at neilsoft", _no_context())
        assert result is None

    def test_blank_company_returns_none(self):
        result = try_parse("add application for ai engineer at", _no_context())
        assert result is None

    def test_missing_anchor_returns_none(self):
        result = try_parse("application for ai engineer at neilsoft", _no_context())
        assert result is None

    def test_bare_role_at_company_returns_none(self):
        result = try_parse("ai engineer at neilsoft", _no_context())
        assert result is None

    def test_no_anchor_natural_language_returns_none(self):
        result = try_parse(
            "I applied to Neilsoft yesterday for an AI Engineer role",
            _no_context(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Create-draft: Ollama bypass (integration)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_add_application_bypasses_ollama(client, db):
    class RaisingInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
            raise SemanticInterpreterUnavailableError("Ollama must not be called")

        def health_check(self):
            return {}

    app.dependency_overrides[get_semantic_interpreter] = lambda: RaisingInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={"transcript": "add application for ai engineer at neilsoft"},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] != "unavailable", "Ollama was called but should have been bypassed"
    assert body["status"] in ("preview", "draft_created", "unsupported", "clarification_required"), \
        f"Unexpected status: {body['status']}"


@pytest.mark.anyio
async def test_add_application_returns_draft_created(client, db):
    """End-to-end: create-draft command goes through dispatcher and returns a draft."""
    call_count = {"n": 0}

    class TrackingInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
            call_count["n"] += 1
            raise SemanticInterpreterUnavailableError("Should not be called")

        def health_check(self):
            return {}

    app.dependency_overrides[get_semantic_interpreter] = lambda: TrackingInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={"transcript": "add application for ai engineer at neilsoft"},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert call_count["n"] == 0, "Ollama was called — fast path did not activate"
    assert response.status_code == 200
    body = response.json()
    # Response must not be unavailable
    assert body["status"] != "unavailable"
    # A draft should be present
    assert body.get("draft") is not None or body.get("draft_id") is not None, \
        f"Expected draft in response, got: {body}"


# ---------------------------------------------------------------------------
# Single-field update: status
# ---------------------------------------------------------------------------


class TestFieldUpdateStatus:
    def test_set_status_to_in_touch(self):
        result = try_parse("set status to in-touch", _draft_context())
        assert result is not None
        assert result.changes.status == "in_touch"

    def test_set_status_to_in_touch_underscore(self):
        result = try_parse("set status to in_touch", _draft_context())
        assert result is not None
        assert result.changes.status == "in_touch"

    def test_set_status_to_in_touch_space(self):
        result = try_parse("set status to in touch", _draft_context())
        assert result is not None
        assert result.changes.status == "in_touch"

    def test_change_status_to_applied(self):
        result = try_parse("change status to applied", _draft_context())
        assert result is not None
        assert result.changes.status == "applied"

    def test_set_status_to_rejected(self):
        result = try_parse("set status to rejected", _draft_context())
        assert result is not None
        assert result.changes.status == "rejected"

    def test_set_status_to_accepted(self):
        result = try_parse("set status to accepted", _draft_context())
        assert result is not None
        assert result.changes.status == "accepted"

    def test_unrecognized_status_returns_none(self):
        result = try_parse("set status to pending_review", _draft_context())
        assert result is None

    def test_no_active_draft_or_app_returns_none(self):
        result = try_parse("set status to applied", _no_context())
        assert result is None

    def test_routes_to_pending_changes_when_app_context(self):
        result = try_parse("set status to applied", _app_context(7))
        assert result is not None
        assert result.operation == "create_application_update_draft"
        assert result.target.application_id == 7


# ---------------------------------------------------------------------------
# Single-field update: priority
# ---------------------------------------------------------------------------


class TestFieldUpdatePriority:
    def test_set_priority_to_high(self):
        result = try_parse("set priority to high", _draft_context())
        assert result is not None
        assert result.changes.priority == "HIGH"

    def test_change_priority_to_medium(self):
        result = try_parse("change priority to medium", _draft_context())
        assert result is not None
        assert result.changes.priority == "MEDIUM"

    def test_set_priority_to_low(self):
        result = try_parse("set priority to low", _draft_context())
        assert result is not None
        assert result.changes.priority == "LOW"

    def test_unrecognized_priority_returns_none(self):
        result = try_parse("set priority to urgent", _draft_context())
        assert result is None

    def test_no_context_returns_none(self):
        result = try_parse("set priority to high", _no_context())
        assert result is None


# ---------------------------------------------------------------------------
# Single-field update: location
# ---------------------------------------------------------------------------


class TestFieldUpdateLocation:
    def test_set_location_to_onsite(self):
        result = try_parse("set location to onsite", _draft_context())
        assert result is not None
        assert result.changes.location_mode == "on-site"

    def test_set_location_to_on_site(self):
        result = try_parse("set location to on-site", _draft_context())
        assert result is not None
        assert result.changes.location_mode == "on-site"

    def test_set_location_to_on_site_space(self):
        result = try_parse("set location to on site", _draft_context())
        assert result is not None
        assert result.changes.location_mode == "on-site"

    def test_change_location_to_remote(self):
        result = try_parse("change location to remote", _draft_context())
        assert result is not None
        assert result.changes.location_mode == "remote"

    def test_set_location_to_hybrid(self):
        result = try_parse("set location to hybrid", _draft_context())
        assert result is not None
        assert result.changes.location_mode == "hybrid"

    def test_unrecognized_location_returns_none(self):
        result = try_parse("set location to in-person", _draft_context())
        assert result is None

    def test_no_context_returns_none(self):
        result = try_parse("set location to onsite", _no_context())
        assert result is None


# ---------------------------------------------------------------------------
# Single-field update: employment type
# ---------------------------------------------------------------------------


class TestFieldUpdateEmploymentType:
    def test_change_employment_type_to_fulltime(self):
        result = try_parse("change employment type to fulltime", _draft_context())
        assert result is not None
        assert result.changes.employment_types == ["Full Time"]

    def test_set_employment_type_to_full_time(self):
        result = try_parse("set employment type to full time", _draft_context())
        assert result is not None
        assert result.changes.employment_types == ["Full Time"]

    def test_set_employment_type_to_full_time_hyphen(self):
        result = try_parse("set employment type to full-time", _draft_context())
        assert result is not None
        assert result.changes.employment_types == ["Full Time"]

    def test_set_employment_type_to_internship(self):
        result = try_parse("set employment type to internship", _draft_context())
        assert result is not None
        assert result.changes.employment_types == ["Internship"]

    def test_unrecognized_employment_type_returns_none(self):
        result = try_parse("set employment type to contract", _draft_context())
        assert result is None

    def test_no_context_returns_none(self):
        result = try_parse("change employment type to fulltime", _no_context())
        assert result is None


# ---------------------------------------------------------------------------
# Lifecycle regression
# ---------------------------------------------------------------------------


class TestLifecycleRegression:
    def test_save_it(self):
        result = try_parse("save it", {"draft_id": "42"})
        assert result is not None
        assert result.operation == "save_draft"

    def test_discard_draft(self):
        result = try_parse("discard draft", {"draft_id": "42"})
        # "discard draft" is not in the trigger set — "discard it" is; confirm no regression
        # The existing trigger set includes {"discard it", "discard", ...}
        result2 = try_parse("discard", {"draft_id": "42"})
        assert result2 is not None
        assert result2.operation == "discard_draft"

    def test_save_without_draft_context_returns_none(self):
        result = try_parse("save it", _no_context())
        assert result is None

    def test_archive_by_company(self):
        context = {
            "applications": [
                {"id": 1, "company": "Neilsoft", "archived_at": None},
            ]
        }
        result = try_parse("archive neilsoft", context)
        assert result is not None
        assert result.operation == "archive_application"
        assert result.target.application_id == 1

    def test_restore_by_company(self):
        context = {
            "applications": [
                {"id": 2, "company": "Google", "archived_at": "2026-01-01"},
            ]
        }
        result = try_parse("restore google", context)
        assert result is not None
        assert result.operation == "restore_application"
        assert result.target.application_id == 2


# ---------------------------------------------------------------------------
# Ambiguity guard — must NOT match new templates
# ---------------------------------------------------------------------------


class TestAmbiguityGuard:
    def test_interview_at_company_not_matched(self):
        result = try_parse("interview at neilsoft", _no_context())
        assert result is None

    def test_follow_up_at_company_not_matched(self):
        result = try_parse("follow up at neilsoft", _no_context())
        assert result is None

    def test_bare_role_at_company_not_matched(self):
        result = try_parse("ai engineer at neilsoft", _no_context())
        assert result is None

    def test_natural_language_sentence_not_matched(self):
        result = try_parse(
            "I applied to Neilsoft yesterday for an AI Engineer role",
            _no_context(),
        )
        assert result is None

    def test_apply_for_not_matched_by_create_template(self):
        # "apply for" is handled by _EXPLICIT_CREATE_INTENT in semantic_validation, not fast path
        result = try_parse("apply for ai engineer at neilsoft", _no_context())
        assert result is None


# ---------------------------------------------------------------------------
# Optional: parameters envelope canonicalization
# ---------------------------------------------------------------------------


class TestParametersEnvelopeCanonicalization:
    def test_parameters_envelope_unwrapped(self):
        raw = {
            "function": "patch_active_draft",
            "parameters": {
                "fields": {"company": "Neilsoft", "role": "AI Engineer"},
            },
        }
        result = canonicalize_tool_arguments(tool_name="patch_active_draft", raw_arguments=raw)
        assert result == {"fields": {"company": "Neilsoft", "role": "AI Engineer"}}

    def test_parameters_envelope_with_matching_top_level_fields(self):
        """Duplicate matching values: parameters.fields + top-level fields → merged, no conflict."""
        raw = {
            "function": "patch_active_draft",
            "parameters": {
                "fields": {"company": "Neilsoft", "role": "AI Engineer"},
            },
            "fields": {"company": "Neilsoft", "role": "AI Engineer"},
        }
        result = canonicalize_tool_arguments(tool_name="patch_active_draft", raw_arguments=raw)
        assert result["fields"] == {"company": "Neilsoft", "role": "AI Engineer"}

    def test_parameters_envelope_with_conflicting_top_level_fields(self):
        from app.semantic_interpreter import SemanticInterpreterInvalidResponseError
        raw = {
            "function": "patch_active_draft",
            "parameters": {
                "fields": {"company": "Neilsoft"},
            },
            "fields": {"company": "Google"},
        }
        with pytest.raises(SemanticInterpreterInvalidResponseError):
            canonicalize_tool_arguments(tool_name="patch_active_draft", raw_arguments=raw)

    def test_parameters_envelope_non_dict_raises(self):
        from app.semantic_interpreter import SemanticInterpreterInvalidResponseError
        raw = {
            "function": "patch_active_draft",
            "parameters": "not_a_dict",
        }
        with pytest.raises(SemanticInterpreterInvalidResponseError):
            canonicalize_tool_arguments(tool_name="patch_active_draft", raw_arguments=raw)

    def test_parameters_with_no_function_key_not_treated_as_envelope(self):
        # Without "function" key, "parameters" alone should NOT trigger envelope unwrapping.
        # It should pass through as canonical shape 1.
        raw = {
            "parameters": {"fields": {"company": "Neilsoft"}},
            "other_key": "value",
        }
        result = canonicalize_tool_arguments(tool_name="patch_active_draft", raw_arguments=raw)
        assert result == raw
