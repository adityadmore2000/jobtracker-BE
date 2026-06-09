from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.fast_path_parser import try_parse
from app.main import app
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import SemanticInterpretationResult, SemanticInterpreterUnavailableError, get_semantic_interpreter
from app.semantic_schemas import SemanticExtractedFields, SemanticInterpreterMetrics, SemanticToolCallProposal
from app.database import SessionLocal


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


def make_draft_context(draft_id: str) -> dict:
    return {"draft_id": draft_id}


def make_app_context(app_id: int) -> dict:
    return {"active_application_id": app_id}


def test_save_it_produces_save_draft_payload():
    result = try_parse("save it", {"draft_id": "42"})
    assert result is not None
    assert result.operation == "save_draft"
    assert result.target.draft_id == "42"


def test_discard_it_produces_discard_draft_payload():
    result = try_parse("discard it", {"draft_id": "42"})
    assert result is not None
    assert result.operation == "discard_draft"
    assert result.target.draft_id == "42"


def test_priority_high_with_draft_context():
    result = try_parse("priority high", {"draft_id": "10"})
    assert result is not None
    assert result.operation == "patch_draft"
    assert result.target.draft_id == "10"
    assert result.changes.priority == "HIGH"


def test_priority_high_with_application_context():
    result = try_parse("priority high", {"active_application_id": 5})
    assert result is not None
    assert result.operation == "patch_application"
    assert result.target.application_id == 5
    assert result.changes.priority == "HIGH"


def test_priority_high_no_context_returns_none():
    result = try_parse("priority high", {})
    assert result is None


def test_onsite_produces_correct_location_mode():
    result = try_parse("onsite", {"draft_id": "1"})
    assert result is not None
    assert result.changes.location_mode == "onsite"


def test_on_site_variant_recognized():
    result = try_parse("on site", {"draft_id": "1"})
    assert result is not None
    assert result.changes.location_mode == "onsite"


def test_mark_applied_produces_status_applied():
    result = try_parse("mark applied", {"draft_id": "1"})
    assert result is not None
    assert result.changes.status == "Applied"


def test_already_applied_recognized():
    result = try_parse("already applied", {"draft_id": "1"})
    assert result is not None
    assert result.changes.status == "Applied"


def test_unrecognized_transcript_returns_none():
    result = try_parse("Neilsoft AI Engineer medium priority", {"draft_id": "1"})
    assert result is None


def test_fast_path_is_case_insensitive():
    result_upper = try_parse("PRIORITY HIGH", {"draft_id": "1"})
    result_lower = try_parse("priority high", {"draft_id": "1"})
    assert result_upper is not None
    assert result_lower is not None
    assert result_upper.changes.priority == result_lower.changes.priority


def test_fast_path_strips_whitespace():
    result = try_parse("  save it  ", {"draft_id": "42"})
    assert result is not None
    assert result.operation == "save_draft"


def test_fast_path_does_not_guess_target():
    result = try_parse("priority high", {})
    assert result is None


@pytest.mark.anyio
async def test_save_it_transcript_bypasses_ollama(client, db):
    create_result = dispatch(
        MutationPayload(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Neilsoft", role="AI Engineer"),
        ),
        db,
    )
    assert create_result.success is True
    draft_id = str(create_result.draft["id"])

    class RaisingInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
            raise SemanticInterpreterUnavailableError("Ollama should not be called")

        def health_check(self):
            return {}

    app.dependency_overrides[get_semantic_interpreter] = lambda: RaisingInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={
                "transcript": "save it",
                "context": {"draft_id": draft_id},
            },
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] != "unavailable", "Ollama was called but should have been bypassed"


@pytest.mark.anyio
async def test_unrecognized_transcript_still_calls_ollama(client):
    call_count = {"count": 0}

    class TrackingInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
            call_count["count"] += 1
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="patch_active_draft",
                    arguments={
                        "fields": {"company": "Neilsoft", "roles": ["AI Engineer"]},
                        "replace_explicit_fields": True,
                        "context_notes": [],
                    },
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=10),
                extracted_fields=SemanticExtractedFields(company="Neilsoft", roles=["AI Engineer"]),
            )

        def health_check(self):
            return {}

    app.dependency_overrides[get_semantic_interpreter] = lambda: TrackingInterpreter()
    try:
        await client.post(
            "/transcript/parse",
            json={"transcript": "Neilsoft sathi AI Engineer application add kar"},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert call_count["count"] >= 1, "Ollama interpreter was not called for an unrecognized transcript"
