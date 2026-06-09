import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import get_semantic_interpreter
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


def make_payload(operation: str, changes: dict | None = None, target: dict | None = None, notes: list | None = None) -> MutationPayload:
    return MutationPayload(
        operation=operation,
        target=MutationTarget(**(target or {})),
        changes=ApplicationChanges(**(changes or {})),
        notes_to_append=notes or [],
    )


def test_dispatch_rejects_unknown_operation(db):
    payload = make_payload("delete_everything")
    result = dispatch(payload, db)
    assert result.success is False
    assert "Unknown operation" in result.message


def test_dispatch_rejects_invalid_priority_enum(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "priority": "ultra"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "priority" in result.message.lower()


def test_dispatch_rejects_invalid_location_mode_enum(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "location_mode": "in-person"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "location_mode" in result.message.lower()


def test_dispatch_rejects_patch_application_without_application_id(db):
    payload = make_payload("patch_application", changes={"priority": "HIGH"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "application_id" in result.message.lower()


def test_dispatch_create_draft_returns_preview(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "role": "AI Engineer"})
    result = dispatch(payload, db)
    assert result.success is True
    assert result.draft is not None
    assert result.draft["company"] == "Neilsoft"


def test_dispatch_ask_clarification_returns_question(db):
    payload = make_payload("ask_clarification", notes=["Which company should I use?"])
    result = dispatch(payload, db)
    assert result.success is True
    assert result.clarification_question == "Which company should I use?"


@pytest.mark.anyio
async def test_existing_transcript_parse_behavior_unchanged(client):
    from types import SimpleNamespace
    from app.semantic_schemas import SemanticInterpreterMetrics, SemanticToolCallProposal
    from app.semantic_interpreter import SemanticInterpretationResult

    class FakeInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
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

    app.dependency_overrides[get_semantic_interpreter] = lambda: FakeInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={"transcript": "Add Neilsoft AI Engineer application"},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "operation" in body
    assert "draft" in body
    assert body["status"] == "preview"
    assert body["draft"]["company"] == "Neilsoft"
    assert body["draft"]["roles_json"] == ["AI Engineer"]
