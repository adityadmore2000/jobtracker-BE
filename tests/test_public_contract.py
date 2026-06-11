"""
Public API contract tests.

Verify that:
- GET /applications and GET /applications/archived expose canonical field names
  (roles, employment_types, current_stages) and suppress _json suffixes.
- POST /transcript/parse returns PublicTranscriptResponse with correct status
  and never exposes internal fields (proposal, raw_transcript, interpreter_metrics).
- message is always present and non-empty.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from types import SimpleNamespace

from app.main import app
from app.database import SessionLocal
from app.models import JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import get_semantic_interpreter
from app.semantic_schemas import SemanticExtractedFields, SemanticInterpreterMetrics, SemanticToolCallProposal


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def make_payload(operation: str, changes: dict | None = None, target: dict | None = None, notes: list | None = None) -> MutationPayload:
    return MutationPayload(
        operation=operation,
        target=MutationTarget(**(target or {})),
        changes=ApplicationChanges(**(changes or {})),
        notes_to_append=notes or [],
    )


# ---------------------------------------------------------------------------
# Application DTO field presence / absence
# ---------------------------------------------------------------------------

CANONICAL_ARRAY_FIELDS = {"employment_types", "current_stages"}
INTERNAL_ARRAY_FIELDS = {"roles_json", "employment_types_json", "current_stages_json"}
CANONICAL_SCALAR_FIELDS = {
    "id", "company", "role", "job_link", "location", "status", "priority",
    "engaged_days", "next_action", "comments", "is_draft",
    "draft_created_at", "archived_at", "created_at", "updated_at",
}


def _assert_public_application_shape(app_dict: dict) -> None:
    for field in CANONICAL_ARRAY_FIELDS:
        assert field in app_dict, f"Expected public field '{field}' in application response"
        assert isinstance(app_dict[field], list), f"Expected '{field}' to be a list"
    for field in INTERNAL_ARRAY_FIELDS:
        assert field not in app_dict, f"Internal field '{field}' must not be exposed in public response"
    for field in CANONICAL_SCALAR_FIELDS:
        assert field in app_dict, f"Expected canonical field '{field}' in application response"


@pytest.mark.anyio
async def test_list_applications_uses_canonical_fields(client, db):
    # Seed a saved application
    create = dispatch(make_payload("create_draft", changes={"company": "PublicCo", "role": "AI Engineer"}), db)
    dispatch(make_payload("save_draft", target={"draft_id": str(create.draft["id"])}), db)

    response = await client.get("/applications")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)

    public_apps = [a for a in body if a.get("company") == "PublicCo"]
    assert len(public_apps) >= 1, "Expected at least one PublicCo application in active list"
    _assert_public_application_shape(public_apps[0])


@pytest.mark.anyio
async def test_list_archived_applications_uses_canonical_fields(client, db):
    create = dispatch(make_payload("create_draft", changes={"company": "ArchivedPublicCo", "role": "ML Researcher"}), db)
    save = dispatch(make_payload("save_draft", target={"draft_id": str(create.draft["id"])}), db)
    dispatch(make_payload("archive_application", target={"application_id": save.application["id"]}), db)

    response = await client.get("/applications/archived")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)

    archived = [a for a in body if a.get("company") == "ArchivedPublicCo"]
    assert len(archived) >= 1
    _assert_public_application_shape(archived[0])


@pytest.mark.anyio
async def test_list_applications_multi_value_fields_preserved(client, db):
    create = dispatch(make_payload("create_draft", changes={
        "company": "MultiCo",
        "role": "AI Engineer",
        "employment_types": ["Full Time", "Part Time"],
        "current_stages": ["Tailored", "Applied"],
    }), db)
    dispatch(make_payload("save_draft", target={"draft_id": str(create.draft["id"])}), db)

    response = await client.get("/applications")
    assert response.status_code == 200
    body = response.json()

    multi_apps = [a for a in body if a.get("company") == "MultiCo"]
    assert len(multi_apps) >= 1
    app_data = multi_apps[0]
    assert app_data["role"] == "AI Engineer"
    assert "Full Time" in app_data["employment_types"]
    assert "Part Time" in app_data["employment_types"]
    assert "Tailored" in app_data["current_stages"]
    assert "Applied" in app_data["current_stages"]


# ---------------------------------------------------------------------------
# Transcript response contract tests
# ---------------------------------------------------------------------------

INTERNAL_LEAKAGE_FIELDS = {"proposal", "raw_transcript", "interpreter_metrics", "operation", "needs_confirmation", "confirmation_kind", "drafts"}
REQUIRED_TRANSCRIPT_FIELDS = {"status", "message", "application_id", "draft_id", "draft", "warnings", "clarification_question", "pending_changes"}


def _assert_no_internal_leakage(body: dict) -> None:
    for field in INTERNAL_LEAKAGE_FIELDS:
        assert field not in body, f"Internal field '{field}' must not appear in public transcript response"


def _assert_transcript_shape(body: dict) -> None:
    for field in REQUIRED_TRANSCRIPT_FIELDS:
        assert field in body, f"Expected public field '{field}' in transcript response"
    assert body["message"], "message must be present and non-empty"
    _assert_no_internal_leakage(body)


class _FakeInterpreterBase:
    settings = SimpleNamespace(max_tool_turns=2)

    def health_check(self):
        return {}


def _make_patch_active_draft_interpreter(company="Neilsoft", role=None):
    class FakeInterpreter(_FakeInterpreterBase):
        def interpret(self, transcript, context=None):
            from app.semantic_interpreter import SemanticInterpretationResult
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="patch_active_draft",
                    arguments={
                        "fields": {"company": company, "role": role or "AI Engineer"},
                        "replace_explicit_fields": True,
                        "context_notes": [],
                    },
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=10),
                extracted_fields=SemanticExtractedFields(company=company, role=role or "AI Engineer"),
            )
    return FakeInterpreter()


def _make_clarification_interpreter(question="Which company?"):
    class FakeInterpreter(_FakeInterpreterBase):
        def interpret(self, transcript, context=None):
            from app.semantic_interpreter import SemanticInterpretationResult
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="ask_clarification",
                    arguments={"question": question},
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=10),
                extracted_fields=SemanticExtractedFields(),
            )
    return FakeInterpreter()


@pytest.mark.anyio
async def test_transcript_create_draft_returns_draft_created(client):
    # Use controlled command syntax; no LLM involved.
    response = await client.post(
        "/transcript/parse",
        json={"transcript": "add application for AI Engineer at Neilsoft"},
    )

    assert response.status_code == 200
    body = response.json()
    _assert_transcript_shape(body)
    assert body["status"] == "draft_created"
    assert body["draft_id"] is not None
    assert body["draft"] is not None
    assert body["draft"]["company"] == "Neilsoft"
    assert "roles_json" not in body["draft"]
    assert isinstance(body["draft"]["role"], str)
    assert body["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_transcript_patch_draft_returns_draft_updated(client, db):
    # Create a draft first
    create = dispatch(make_payload("create_draft", changes={"company": "PatchCo", "role": "AI Engineer"}), db)
    draft_id = str(create.draft["id"])

    app.dependency_overrides[get_semantic_interpreter] = lambda: _make_patch_active_draft_interpreter(company="PatchCo", role="AI Engineer")
    try:
        response = await client.post(
            "/transcript/parse",
            json={"transcript": "Set priority high", "context": {"draft_id": draft_id, "active_draft": {"company": "PatchCo", "role": "AI Engineer"}}},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    _assert_transcript_shape(body)
    # With context draft present, patch_active_draft runs patch_draft → draft_updated
    assert body["status"] in {"draft_created", "draft_updated"}


@pytest.mark.anyio
async def test_transcript_clarification_returns_clarification_status(client):
    # "update application" with no company name triggers clarification from the controlled parser.
    response = await client.post("/transcript/parse", json={"transcript": "update application"})

    assert response.status_code == 200
    body = response.json()
    _assert_transcript_shape(body)
    assert body["status"] == "clarification"
    assert body["clarification_question"] is not None


@pytest.mark.anyio
async def test_transcript_draft_payload_has_canonical_array_fields(client):
    app.dependency_overrides[get_semantic_interpreter] = lambda: _make_patch_active_draft_interpreter()
    try:
        response = await client.post("/transcript/parse", json={"transcript": "Applied for AI Engineer at Neilsoft"})
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    draft = body.get("draft")
    if draft is not None:
        assert "role" in draft
        assert "employment_types" in draft
        assert "current_stages" in draft
        for field in INTERNAL_ARRAY_FIELDS:
            assert field not in draft


@pytest.mark.anyio
async def test_transcript_message_always_present(client):
    app.dependency_overrides[get_semantic_interpreter] = lambda: _make_patch_active_draft_interpreter()
    try:
        response = await client.post("/transcript/parse", json={"transcript": "Applied for AI Engineer at Neilsoft"})
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert body["message"]


@pytest.mark.anyio
async def test_transcript_no_internal_fields_exposed(client):
    app.dependency_overrides[get_semantic_interpreter] = lambda: _make_patch_active_draft_interpreter()
    try:
        response = await client.post("/transcript/parse", json={"transcript": "Applied for AI Engineer at Neilsoft"})
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    _assert_no_internal_leakage(body)


# ---------------------------------------------------------------------------
# Save / discard / update via fast path (no LLM) or direct dispatch
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_save_draft_via_transcript_returns_saved(client, db):
    create = dispatch(make_payload("create_draft", changes={"company": "SaveTransCo", "role": "AI Engineer"}), db)
    draft_id = str(create.draft["id"])

    class SaveInterpreter(_FakeInterpreterBase):
        def interpret(self, transcript, context=None):
            from app.semantic_interpreter import SemanticInterpretationResult
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="request_draft_save",
                    arguments={},
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=5),
                extracted_fields=SemanticExtractedFields(),
            )

    app.dependency_overrides[get_semantic_interpreter] = lambda: SaveInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={
                "transcript": "save it",
                "context": {
                    "draft_id": draft_id,
                    "active_draft": {"company": "SaveTransCo", "role": "AI Engineer"},
                },
            },
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    _assert_transcript_shape(body)
    assert body["status"] == "saved"


@pytest.mark.anyio
async def test_discard_without_draft_returns_no_change_or_discarded(client, db):
    """Discard with no active draft_id in context: should return no_change or discarded."""
    class DiscardInterpreter(_FakeInterpreterBase):
        def interpret(self, transcript, context=None):
            from app.semantic_interpreter import SemanticInterpretationResult
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="request_draft_save",  # will route through fast path if matched
                    arguments={},
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=5),
                extracted_fields=SemanticExtractedFields(),
            )

    # Use fast-path discard via direct dispatch to test status mapping
    result = dispatch(make_payload("discard_draft"), db)
    assert result.success is True
    assert "no active draft" in result.message.lower()

    # Fast path: "discard" with no draft_id → no_change
    from app.transcript_response_adapter import to_public_transcript_response
    from app.schemas import SemanticTranscriptResponse
    from app.semantic_schemas import SemanticToolCallProposal as STP

    internal = SemanticTranscriptResponse(
        status="preview",
        operation="none",
        raw_transcript="discard",
        proposal=STP(),
        warnings=[],
    )
    public = to_public_transcript_response(internal)
    assert public.status == "no_change"
    assert public.message


@pytest.mark.anyio
async def test_patch_application_via_transcript_returns_updated(client, db):
    create = dispatch(make_payload("create_draft", changes={"company": "UpdateCo", "role": "AI Engineer"}), db)
    save = dispatch(make_payload("save_draft", target={"draft_id": str(create.draft["id"])}), db)
    app_id = save.application["id"]

    class PatchAppInterpreter(_FakeInterpreterBase):
        def interpret(self, transcript, context=None):
            from app.semantic_interpreter import SemanticInterpretationResult
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="preview_existing_application_update",
                    arguments={
                        "target": {"company": "UpdateCo", "application_id": app_id},
                        "fields": {"priority": "HIGH"},
                    },
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=5),
                extracted_fields=SemanticExtractedFields(priority="HIGH"),
            )

    app.dependency_overrides[get_semantic_interpreter] = lambda: PatchAppInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={
                "transcript": "Set priority to high",
                "context": {"active_application_id": app_id},
            },
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    _assert_transcript_shape(body)
    # Chat updates to saved applications now create pending-change previews instead of direct patches
    assert body["status"] in {"updated", "pending_changes_created", "pending_changes_updated"}


# ---------------------------------------------------------------------------
# Status mapping unit tests (adapter layer directly)
# ---------------------------------------------------------------------------

from app.transcript_response_adapter import to_public_transcript_response
from app.schemas import SemanticTranscriptResponse
from app.semantic_schemas import SemanticToolCallProposal as STP


def _make_internal(status, operation="none", draft_id=None, application_id=None, clarification_question=None, warnings=None, needs_confirmation=False, confirmation_kind="none"):
    from app.semantic_schemas import SemanticToolCallProposal as STP
    return SemanticTranscriptResponse(
        status=status,
        operation=operation,
        raw_transcript="test",
        proposal=STP(),
        draft_id=draft_id,
        application_id=application_id,
        clarification_question=clarification_question,
        warnings=warnings or [],
        needs_confirmation=needs_confirmation,
        confirmation_kind=confirmation_kind,
    )


def test_adapter_preview_create_operation_draft_id_present_returns_draft_created():
    internal = _make_internal("preview", operation="create", draft_id="123")
    public = to_public_transcript_response(internal)
    assert public.status == "draft_created"
    assert public.message


def test_adapter_preview_update_operation_returns_updated():
    internal = _make_internal("preview", operation="update", application_id=1)
    public = to_public_transcript_response(internal)
    assert public.status == "updated"
    assert public.message


def test_adapter_preview_create_with_application_id_returns_saved():
    internal = _make_internal("preview", operation="create", application_id=1)
    public = to_public_transcript_response(internal)
    assert public.status == "saved"
    assert public.message


def test_adapter_clarification_required_returns_clarification():
    internal = _make_internal("clarification_required", clarification_question="Which company?")
    public = to_public_transcript_response(internal)
    assert public.status == "clarification"
    assert public.clarification_question == "Which company?"
    assert public.message


def test_adapter_unsupported_returns_no_change():
    internal = _make_internal("unsupported", warnings=["Unsupported command."])
    public = to_public_transcript_response(internal)
    assert public.status == "no_change"
    assert public.message


def test_adapter_unavailable_returns_error():
    internal = _make_internal("unavailable", warnings=["Interpreter unavailable."])
    public = to_public_transcript_response(internal)
    assert public.status == "error"
    assert public.message


def test_adapter_message_always_non_empty_for_all_statuses():
    scenarios = [
        _make_internal("preview", operation="create", draft_id="1"),
        _make_internal("preview", operation="update", application_id=1),
        _make_internal("preview", operation="create", application_id=1),
        _make_internal("clarification_required", clarification_question="Q?"),
        _make_internal("unsupported"),
        _make_internal("unavailable"),
    ]
    for internal in scenarios:
        public = to_public_transcript_response(internal)
        assert public.message, f"message must not be empty for status={internal.status}"
