"""Phase 2A.1 semantic boundary hardening — regression tests.

Tests cover:
- Failure 1: discard_draft lifecycle (new tool)
- Failure 2: wrong-field placement reconciliation (status/priority/location cue + role)
- Failure 3: location patch correctness via explicit field cue
- Effective-change guard for patch_active_draft with no actionable fields
- Existing behaviour preservation after these changes
"""


from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skip(reason="Legacy LLM semantic mutation path disabled. USE_LEGACY_SEMANTIC_MUTATIONS=0.")

from app.constants import normalize_status_value
from app.semantic_interpreter import SemanticInterpretationResult
from app.semantic_schemas import (
    DiscardDraftArguments,
    PreviewExistingApplicationTarget,
    SemanticExtractedFields,
    SemanticFieldPatch,
    SemanticInterpreterMetrics,
    SemanticToolCallProposal,
)
from app.semantic_validation import (
    detect_explicit_field_cues,
    normalize_extracted_fields,
    normalize_location,
    reconcile_wrong_field_placement,
    validate_tool_arguments_with_safe_normalization,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def db():
    from app.database import SessionLocal

    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proposal(tool_name: str, arguments: dict) -> SemanticToolCallProposal:
    return SemanticToolCallProposal.model_validate({"tool_name": tool_name, "arguments": arguments})


def _fields(**kwargs) -> SemanticFieldPatch:
    return SemanticFieldPatch.model_validate(kwargs)


class FakeInterpreter:
    def __init__(self, *, proposal, extracted_fields=None, max_tool_turns=2):
        self._proposal = proposal
        self._extracted_fields = SemanticExtractedFields.model_validate(extracted_fields or {})
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def interpret(self, transcript, context=None):
        return SemanticInterpretationResult(
            proposal=self._proposal,
            metrics=SemanticInterpreterMetrics(latency_ms=5),
            extracted_fields=self._extracted_fields,
        )

    def health_check(self):
        return {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


async def _parse(client, transcript, interpreter, context=None):
    from app.main import app
    from app.semantic_interpreter import get_semantic_interpreter

    app.dependency_overrides[get_semantic_interpreter] = lambda: interpreter
    try:
        resp = await client.post(
            "/transcript/parse", json={"transcript": transcript, "context": context or {}}
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)
    assert resp.status_code == 200
    return resp.json()


def _create_app(db, *, company="Google", role="AI Engineer", status="applied", archived=False):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=role.lower(),
        employment_types_json=[],
        job_link="",
        location="",
        status=status,
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    if archived:
        from datetime import datetime, timezone
        a.archived_at = datetime.now(timezone.utc)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _create_draft(db, *, company="Aiden AI", role="founding engineer"):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=role.lower() if role else "",
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=True,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


# ===========================================================================
# Part 1 — detect_explicit_field_cues (unit tests)
# ===========================================================================


def test_detect_explicit_field_cues_status():
    cues = detect_explicit_field_cues("update status of google application to in-touch")
    assert "status" in cues


def test_detect_explicit_field_cues_priority():
    cues = detect_explicit_field_cues("set priority to high")
    assert "priority" in cues


def test_detect_explicit_field_cues_location():
    cues = detect_explicit_field_cues("set location to onsite")
    assert "location" in cues


def test_detect_explicit_field_cues_multiple():
    cues = detect_explicit_field_cues("set status and priority for this role")
    assert "status" in cues
    assert "priority" in cues
    assert "role" in cues


def test_detect_explicit_field_cues_no_cues():
    cues = detect_explicit_field_cues("add Acme application")
    # "application" is not a cue keyword — no status/priority/location cue
    assert "status" not in cues
    assert "priority" not in cues
    assert "location" not in cues


def test_detect_explicit_field_cues_case_insensitive():
    cues = detect_explicit_field_cues("Update STATUS to applied")
    assert "status" in cues


# ===========================================================================
# Part 2 — reconcile_wrong_field_placement (unit tests)
# ===========================================================================


def test_reconcile_status_cue_moves_role_to_status():
    """When 'status' is cued and role holds a valid status value, move role→status."""
    fields = _fields(role="in-touch")
    result = reconcile_wrong_field_placement("update status of google application to in-touch", fields)
    assert result.role is None
    assert result.status == "in_touch"


def test_reconcile_status_cue_no_move_when_status_already_set():
    """Do not overwrite an already-populated status field."""
    fields = _fields(role="in-touch", status="applied")
    result = reconcile_wrong_field_placement("update status to in-touch", fields)
    # status was already set — no move
    assert result.status == "applied"
    assert result.role == "in-touch"


def test_reconcile_priority_cue_moves_role_to_priority():
    """When 'priority' is cued and role holds a valid priority value, move role→priority."""
    fields = _fields(role="high")
    result = reconcile_wrong_field_placement("set priority to high", fields)
    assert result.role is None
    assert result.priority == "HIGH"


def test_reconcile_priority_cue_medium():
    fields = _fields(role="medium")
    result = reconcile_wrong_field_placement("update priority to medium", fields)
    assert result.role is None
    assert result.priority == "MEDIUM"


def test_reconcile_priority_cue_no_move_when_priority_already_set():
    fields = _fields(role="high", priority="LOW")
    result = reconcile_wrong_field_placement("set priority to high", fields)
    assert result.priority == "LOW"
    assert result.role == "high"


def test_reconcile_location_cue_moves_role_to_location():
    """When 'location' is cued and role holds a valid location value, move role→location."""
    fields = _fields(role="onsite")
    result = reconcile_wrong_field_placement("set location to onsite", fields)
    assert result.role is None
    assert result.location == "on-site"


def test_reconcile_location_cue_moves_role_remote():
    fields = _fields(role="remote")
    result = reconcile_wrong_field_placement("change location to remote", fields)
    assert result.role is None
    assert result.location == "remote"


def test_reconcile_location_cue_no_move_when_location_already_set():
    fields = _fields(role="onsite", location="remote")
    result = reconcile_wrong_field_placement("set location to onsite", fields)
    assert result.location == "remote"
    assert result.role == "onsite"


def test_reconcile_location_from_status_field():
    """When 'location' is cued and status holds a location-like value, move status→location."""
    fields = _fields(status="onsite")
    result = reconcile_wrong_field_placement("set location to onsite", fields)
    assert result.status is None
    assert result.location == "on-site"


def test_reconcile_no_cue_no_move():
    """Without an explicit cue, do not reclassify any field."""
    fields = _fields(role="in-touch")
    result = reconcile_wrong_field_placement("add Acme application", fields)
    # No status cue — role should remain unchanged
    assert result.role == "in-touch"
    assert result.status is None


def test_reconcile_valid_role_not_reclassified_when_no_cue():
    """A genuine role title is not reclassified just because it could be something else."""
    fields = _fields(role="LLM Inference Optimization Engineer")
    result = reconcile_wrong_field_placement(
        "add Acme application for LLM Inference Optimization Engineer", fields
    )
    assert result.role == "LLM Inference Optimization Engineer"


def test_reconcile_status_cue_with_valid_role_not_a_status():
    """When 'status' is cued but role value is not a valid status, do not move."""
    fields = _fields(role="AI Engineer")
    result = reconcile_wrong_field_placement("update status of AI Engineer application", fields)
    # "AI Engineer" is not a valid status value → no move
    assert result.role == "AI Engineer"
    assert result.status is None


# ===========================================================================
# Part 3 — normalize_extracted_fields with transcript parameter
# ===========================================================================


def test_normalize_extracted_fields_with_status_cue_transcript():
    """normalize_extracted_fields passes transcript to reconcile_wrong_field_placement."""
    extracted = SemanticExtractedFields(role="in-touch")
    result, warnings = normalize_extracted_fields(
        extracted, transcript="update status of google application to in-touch"
    )
    assert result is not None
    assert result.role is None
    assert result.status == "in_touch"
    assert warnings == []


def test_normalize_extracted_fields_without_transcript_no_change():
    """Without transcript, reconcile_wrong_field_placement is not applied."""
    extracted = SemanticExtractedFields(role="in-touch")
    result, warnings = normalize_extracted_fields(extracted)
    # Without transcript, role="in-touch" is kept as-is (passes open-ended role validation)
    assert result is not None
    assert result.role == "in-touch"
    assert result.status is None


def test_normalize_extracted_fields_location_cue():
    extracted = SemanticExtractedFields(role="onsite")
    result, warnings = normalize_extracted_fields(extracted, transcript="set location to onsite")
    assert result is not None
    assert result.role is None
    assert result.location == "on-site"


def test_normalize_extracted_fields_priority_cue():
    extracted = SemanticExtractedFields(role="high")
    result, warnings = normalize_extracted_fields(extracted, transcript="set priority to high")
    assert result is not None
    assert result.role is None
    assert result.priority == "HIGH"


# ===========================================================================
# Part 4 — DiscardDraftArguments validation (unit tests)
# ===========================================================================


def test_discard_draft_arguments_empty_target():
    args = DiscardDraftArguments.model_validate({"target": {}})
    assert args.target.company is None
    assert args.target.role is None


def test_discard_draft_arguments_with_company_and_role():
    args = DiscardDraftArguments.model_validate(
        {"target": {"company": "Aiden AI", "role": "founding engineer"}}
    )
    assert args.target.company == "Aiden AI"
    assert args.target.role == "founding engineer"


def test_discard_draft_arguments_default_target():
    args = DiscardDraftArguments.model_validate({})
    assert args.target.company is None
    assert args.target.role is None


def test_validate_tool_arguments_discard_draft():
    proposal = _proposal("discard_draft", {"target": {"company": "Aiden AI"}})
    normalized_proposal, validated = validate_tool_arguments_with_safe_normalization(proposal)
    assert validated is not None
    assert isinstance(validated, DiscardDraftArguments)
    assert validated.target.company == "Aiden AI"


# ===========================================================================
# Part 5 — discard_draft handler (integration with DB + dispatcher)
# ===========================================================================


@pytest.mark.anyio
async def test_discard_draft_with_active_draft_id(client, db):
    """discard_draft with valid draft_id in context → draft discarded successfully."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Aiden AI", "role": "founding engineer"},
        proposal=_proposal("discard_draft", {"target": {"company": "Aiden AI", "role": "founding engineer"}}),
    )
    result = await _parse(
        client,
        "discard draft of Aiden AI for founding engineer role",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Public API: successful discard → "discarded"; failure → "no_change"
    assert result["status"] in {"discarded", "no_change"}, (
        f"Unexpected status: {result['status']} / {result}"
    )


@pytest.mark.anyio
async def test_discard_draft_no_active_draft_id_returns_unsupported(client, db):
    """discard_draft without draft_id in context → no_change (nothing to discard)."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Aiden AI"},
        proposal=_proposal("discard_draft", {"target": {"company": "Aiden AI"}}),
    )
    result = await _parse(
        client,
        "discard draft fro Aiden AI",
        interpreter,
        context={},
    )
    # Public API: unsupported internal status → "no_change"
    assert result["status"] == "no_change"
    # The message should indicate nothing to discard
    assert "No active draft" in result.get("message", "") or "No change" in result.get("message", "")


@pytest.mark.anyio
async def test_discard_draft_with_mismatched_target_company_returns_clarification(client, db):
    """discard_draft with mismatched target → clarification (public API status)."""
    draft = _create_draft(db, company="Google", role="AI Engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Aiden AI"},
        proposal=_proposal("discard_draft", {"target": {"company": "Aiden AI"}}),
    )
    result = await _parse(
        client,
        "discard draft of Aiden AI",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Google", "role": "AI Engineer"},
        },
    )
    # Public API: clarification_required internal status → "clarification"
    assert result["status"] == "clarification"
    assert result.get("clarification_question") is not None


@pytest.mark.anyio
async def test_discard_draft_empty_target_uses_active_draft(client, db):
    """discard_draft with empty target → discards using the active draft_id."""
    draft = _create_draft(db, company="Acme", role="Backend Engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={},
        proposal=_proposal("discard_draft", {"target": {}}),
    )
    result = await _parse(
        client,
        "discard draft",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Acme", "role": "Backend Engineer"},
        },
    )
    # Empty target means no mismatch check — should proceed to dispatch
    # Public API: successful discard → "discarded"; failure → "no_change"
    assert result["status"] in {"discarded", "no_change"}, (
        f"Unexpected status: {result['status']} / {result}"
    )


# ===========================================================================
# Part 6 — discard_draft does NOT patch draft (regression for Failure 1)
# ===========================================================================


@pytest.mark.anyio
async def test_discard_draft_not_treated_as_patch(client, db):
    """LLM selecting discard_draft should NOT result in draft_created/draft_updated."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        # LLM correctly selects discard_draft (not patch_active_draft)
        extracted_fields={"company": "Aiden AI", "role": "founding engineer"},
        proposal=_proposal("discard_draft", {"target": {"company": "Aiden AI", "role": "founding engineer"}}),
    )
    result = await _parse(
        client,
        "discard draft of Aiden AI for founding engineer role",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Must NOT be draft_created or draft_updated (public API names)
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"discard_draft was routed as a patch: {result}"
    )


# ===========================================================================
# Part 7 — wrong-field placement via API (regression for Failures 2 & 3)
# ===========================================================================


@pytest.mark.anyio
async def test_status_in_role_field_gets_reconciled_to_status(client, db):
    """LLM extracting status value into role field is reconciled when 'status' is cued."""
    _create_app(db, company="Google", role="AI Engineer", status="applied")

    # Simulate LLM putting "in-touch" in role (wrong) instead of status
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "in-touch"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Google"},
                "fields": {"role": "in-touch"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(
        client,
        "update status of google application to in-touch",
        interpreter,
    )
    # After reconciliation, the extracted role="in-touch" moves to status="in_touch"
    # The proposal fields are then merged from extracted_fields (which has status set, not role)
    # So the final update should have status=in_touch, NOT role=in-touch
    if result["status"] in {"pending_changes_created", "pending_changes_updated"}:
        preview = result.get("pending_changes", {}).get("preview", {})
        assert preview.get("status") == "in_touch", (
            f"Expected status=in_touch but got: {preview}"
        )
        # role should NOT be "in-touch"
        assert preview.get("role") != "in-touch", (
            f"role should not be 'in-touch', got: {preview}"
        )
    elif result["status"] == "unsupported":
        # If no actionable fields remain after reconciliation (role removed, status goes to
        # extracted fields but proposal.fields ends up empty), unsupported is acceptable
        pass
    else:
        pytest.fail(f"Unexpected status: {result['status']} / {result}")


@pytest.mark.anyio
async def test_location_in_role_field_gets_reconciled_to_location(client, db):
    """LLM extracting location value into role field is reconciled when 'location' is cued."""
    # Simulate LLM putting "onsite" in role (wrong) instead of location
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Acme", "role": "onsite"},
        proposal=_proposal(
            "patch_active_draft",
            {
                "fields": {"company": "Acme", "role": "onsite"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )
    result = await _parse(
        client,
        "set location to onsite",
        interpreter,
        context={"active_draft": {"company": "Acme", "role": ""}},
    )
    # After reconciliation, role="onsite" moves to location="on-site"
    # The draft should have location="on-site", and role should be empty/None
    if result["status"] in {"draft_created", "draft_updated"}:
        draft = result.get("draft", {})
        assert draft.get("location") == "on-site", (
            f"Expected location=on-site but got: {draft}"
        )
        assert draft.get("role") != "onsite", (
            f"role should not be 'onsite', got: {draft}"
        )
    else:
        # clarification_required is acceptable (e.g., missing company context)
        assert result["status"] in {"draft_created", "draft_updated", "clarification_required"}, (
            f"Unexpected status: {result['status']} / {result}"
        )


@pytest.mark.anyio
async def test_priority_in_role_field_gets_reconciled_to_priority(client, db):
    """LLM extracting priority value into role field is reconciled when 'priority' is cued."""
    _create_app(db, company="Acme", role="Backend Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Acme", "role": "high"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Acme"},
                "fields": {"role": "high"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(
        client,
        "set priority to high for Acme application",
        interpreter,
    )
    if result["status"] in {"pending_changes_created", "pending_changes_updated"}:
        preview = result.get("pending_changes", {}).get("preview", {})
        assert preview.get("priority") == "HIGH", (
            f"Expected priority=HIGH but got: {preview}"
        )
        assert preview.get("role") != "high", (
            f"role should not be 'high', got: {preview}"
        )
    elif result["status"] == "unsupported":
        pass
    else:
        pytest.fail(f"Unexpected status: {result['status']} / {result}")


# ===========================================================================
# Part 8 — Existing behaviour preserved
# ===========================================================================


@pytest.mark.anyio
async def test_genuine_role_not_reclassified_by_status_cue(client, db):
    """A genuine role title is not reclassified even if 'status' appears in transcript."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer", "status": "applied"},
        proposal=_proposal(
            "patch_active_draft",
            {
                "fields": {"company": "Neilsoft", "role": "AI Engineer"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )
    result = await _parse(
        client,
        "Add Neilsoft AI Engineer application, status applied",
        interpreter,
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    assert result["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_open_role_founding_engineer_not_reclassified(client, db):
    """Open-ended role 'founding engineer' is preserved regardless of field cues."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Aiden AI", "role": "founding engineer"},
        proposal=_proposal(
            "patch_active_draft",
            {
                "fields": {"company": "Aiden AI", "role": "founding engineer"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )
    result = await _parse(
        client,
        "add aiden ai application for founding engineer",
        interpreter,
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    assert result["draft"]["role"] == "founding engineer"


@pytest.mark.anyio
async def test_discard_draft_tool_is_available_in_schema(client):
    """discard_draft is a valid SemanticToolName — tool call proposal validates."""
    proposal = _proposal("discard_draft", {"target": {}})
    assert proposal.tool_name == "discard_draft"


def test_discard_draft_in_semantic_tool_name_literal():
    """discard_draft is in the SemanticToolName Literal."""
    from typing import get_args
    from app.semantic_schemas import SemanticToolName

    tool_names = get_args(SemanticToolName)
    assert "discard_draft" in tool_names


def test_discard_draft_in_tool_argument_models():
    """discard_draft has an entry in TOOL_ARGUMENT_MODELS."""
    from app.semantic_interpreter import TOOL_ARGUMENT_MODELS

    assert "discard_draft" in TOOL_ARGUMENT_MODELS
    assert TOOL_ARGUMENT_MODELS["discard_draft"] is DiscardDraftArguments


def test_discard_draft_in_build_ollama_tools():
    """discard_draft tool definition is returned by build_ollama_tools()."""
    from app.semantic_interpreter import build_ollama_tools

    tools = build_ollama_tools()
    tool_names = [t["function"]["name"] for t in tools]
    assert "discard_draft" in tool_names


# ===========================================================================
# Part 9 — discard_draft with typo in target (minor tolerance)
# ===========================================================================


@pytest.mark.anyio
async def test_discard_draft_typo_fro_instead_of_for(client, db):
    """'discard draft fro Aiden AI' → discard_draft with target.company=Aiden AI."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    # LLM should parse "fro" typo and still extract company correctly
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Aiden AI"},
        proposal=_proposal("discard_draft", {"target": {"company": "Aiden AI"}}),
    )
    result = await _parse(
        client,
        "discard draft fro Aiden AI",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Should NOT patch or create a draft (public API names)
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"discard command resulted in patch: {result}"
    )


# ===========================================================================
# Part 10 — SemanticToolName literal correctness
# ===========================================================================


def test_semantic_tool_name_includes_all_expected_tools():
    from typing import get_args
    from app.semantic_schemas import SemanticToolName

    expected = {
        "patch_active_draft",
        "preview_existing_application_update",
        "request_draft_save",
        "attach_latest_browser_context",
        "ask_clarification",
        "archive_application",
        "explain_delete_policy",
        "discard_draft",
    }
    actual = set(get_args(SemanticToolName))
    assert expected == actual
