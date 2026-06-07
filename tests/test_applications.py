import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import AsrCompanyCorrectionEvent, CanonicalCompany, CompanyAlias
from app.semantic_interpreter import (
    SemanticInterpretationResult,
    SemanticInterpreterUnavailableError,
    get_semantic_interpreter,
)
from app.semantic_schemas import SemanticInterpreterMetrics, SemanticToolCallProposal


REALISTIC_RECORD = {
    "company": "Bootcoding Pvt. LTD",
    "roles_json": ["AI Engineer"],
    "employment_types_json": ["Internship"],
    "job_link": "https://example.com/job",
    "location": "onsite",
    "status": "applied",
    "current_stages_json": ["Tailored", "Applied", "Networked"],
    "priority": "MEDIUM",
    "engaged_days": 1,
    "next_action": "Check whether request was accepted",
    "comments": "Connected with one person; awaiting acceptance",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


async def create_record(client, payload=None):
    response = await client.post("/applications", json=payload or REALISTIC_RECORD)
    assert response.status_code == 201
    return response.json()


async def create_candidate(client, payload):
    response = await client.post("/applications/create-candidate", json=payload)
    assert response.status_code == 200
    return response.json()


async def confirm_candidate(client, payload):
    response = await client.post("/applications/confirm-company", json=payload)
    assert response.status_code == 201
    return response.json()


async def create_browser_context(client, payload=None):
    response = await client.post(
        "/browser-context",
        json=payload
        or {
            "url": "https://example.com/job",
            "page_title": "AI Engineer - Example Company",
        },
    )
    assert response.status_code == 201
    return response.json()["context"]


class FakeInterpreter:
    def __init__(self, *, proposal=None, error=None, metrics=None, health=None):
        self.proposal = proposal
        self.error = error
        self.metrics = metrics or SemanticInterpreterMetrics(latency_ms=12)
        self.health = health or {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}
        self.calls: list[dict[str, object | None]] = []

    def interpret(self, transcript: str, context=None):
        self.calls.append({"transcript": transcript, "context": context})
        if self.error is not None:
            raise self.error
        return SemanticInterpretationResult(proposal=self.proposal, metrics=self.metrics)

    def health_check(self):
        return self.health


def proposal(**overrides):
    payload = {
        "tool_name": "ask_clarification",
        "arguments": {"question": "Could you clarify what to update?"},
    }
    payload.update(overrides)
    return SemanticToolCallProposal.model_validate(payload)


async def parse_transcript(client, transcript, interpreter, context=None):
    app.dependency_overrides[get_semantic_interpreter] = lambda: interpreter
    try:
        response = await client.post("/transcript/parse", json={"transcript": transcript, "context": context})
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)
    assert response.status_code == 200
    return response.json()


def export_correction_events() -> list[dict[str, object | None]]:
    with SessionLocal() as db:
        events = db.query(AsrCompanyCorrectionEvent).order_by(AsrCompanyCorrectionEvent.id.asc()).all()
        return [
            {
                "id": event.id,
                "raw_transcript": event.raw_transcript,
                "original_extracted_company_name": event.original_extracted_company_name,
                "confirmed_company_name": event.confirmed_company_name,
                "canonical_company_id": event.canonical_company_id,
                "application_id": event.application_id,
                "alias_created": event.alias_created,
                "audio_reference": event.audio_reference,
            }
            for event in events
        ]


def assert_bootcoding_current_stages(record):
    assert record["current_stages_json"] == ["Tailored", "Applied", "Networked"]


@pytest.mark.anyio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_local_frontend_origins_are_allowed_for_cors_preflight(client):
    for origin in ["http://localhost:3000", "http://127.0.0.1:3000"]:
        response = await client.options(
            "/applications",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.anyio
async def test_chrome_extension_origin_is_allowed_for_cors_preflight(client):
    origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    response = await client.options(
        "/browser-context",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.anyio
async def test_post_valid_browser_context(client):
    context = await create_browser_context(client)

    assert context["id"] == 1
    assert context["url"] == "https://example.com/job"
    assert context["page_title"] == "AI Engineer - Example Company"
    assert context["captured_at"]


@pytest.mark.anyio
async def test_get_latest_browser_context(client):
    created = await create_browser_context(client)
    response = await client.get("/browser-context/latest")

    assert response.status_code == 200
    assert response.json() == {"context": created}


@pytest.mark.anyio
async def test_latest_browser_context_ordering(client):
    first = await create_browser_context(client, {"url": "https://example.com/first", "page_title": "First"})
    second = await create_browser_context(client, {"url": "https://example.com/second", "page_title": "Second"})
    response = await client.get("/browser-context/latest")

    assert response.status_code == 200
    assert response.json()["context"]["id"] == second["id"]
    assert response.json()["context"]["id"] != first["id"]


@pytest.mark.anyio
async def test_browser_context_invalid_url_rejection(client):
    response = await client.post("/browser-context", json={"url": "not a url", "page_title": "Invalid"})

    assert response.status_code == 422


@pytest.mark.anyio
async def test_browser_context_rejects_non_http_url_schemes(client):
    for url in ["chrome://extensions", "file:///tmp/job.html", "about:blank"]:
        response = await client.post("/browser-context", json={"url": url, "page_title": "Unsupported"})

        assert response.status_code == 422


@pytest.mark.anyio
async def test_browser_context_empty_state_response(client):
    response = await client.get("/browser-context/latest")

    assert response.status_code == 200
    assert response.json() == {"context": None}


@pytest.mark.anyio
async def test_captured_context_does_not_modify_job_applications(client):
    created_application = await create_record(client)
    await create_browser_context(client, {"url": "https://example.com/context", "page_title": "Context Only"})

    fetched_application = await client.get(f"/applications/{created_application['id']}")
    listed_applications = await client.get("/applications")

    assert fetched_application.status_code == 200
    assert fetched_application.json() == created_application
    assert listed_applications.status_code == 200
    assert listed_applications.json() == [created_application]


@pytest.mark.anyio
async def test_captured_context_does_not_infer_application_fields(client):
    await create_browser_context(client, {"url": "https://example.com/job", "page_title": "AI Engineer - Example Company"})
    response = await client.get("/applications")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio
async def test_browser_context_page_title_may_be_blank(client):
    context = await create_browser_context(client, {"url": "https://example.com/job", "page_title": ""})

    assert context["url"] == "https://example.com/job"
    assert context["page_title"] == ""


@pytest.mark.anyio
async def test_get_latest_browser_context_response_shape_is_consistent(client):
    empty_response = await client.get("/browser-context/latest")
    assert empty_response.status_code == 200
    assert set(empty_response.json().keys()) == {"context"}

    await create_browser_context(client)
    populated_response = await client.get("/browser-context/latest")
    assert populated_response.status_code == 200
    assert set(populated_response.json().keys()) == {"context"}
    assert set(populated_response.json()["context"].keys()) == {"id", "url", "page_title", "captured_at"}


@pytest.mark.anyio
async def test_semantic_interpreter_health_endpoint(client):
    interpreter = FakeInterpreter(health={"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"})
    app.dependency_overrides[get_semantic_interpreter] = lambda: interpreter
    try:
        response = await client.get("/semantic-interpreter/health")
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_returns_preview_data(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "roles": ["AI Engineer"], "employment_types": ["full-time"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Add AI Engineer role for Neilsoft", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["operation"] == "create"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == ["AI Engineer"]
    assert parsed["draft"]["employment_types_json"] == ["Full Time"]


@pytest.mark.anyio
async def test_parse_transcript_partial_draft_accepts_company_only_phrase(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "I have a requirement. I want to add an application neilsoft", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["operation"] == "create"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == []
    assert parsed["clarification_question"] is None


@pytest.mark.anyio
async def test_parse_transcript_partial_draft_accepts_add_neilsoft_application(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Add a Neilsoft application", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"


@pytest.mark.anyio
async def test_parse_transcript_partial_draft_accepts_hinglish_company_only(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Neilsoft sathi application add kar", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_uses_active_context_when_needed(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"priority": "high"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(
        client,
        "Make it high priority",
        interpreter,
        context={
            "active_application": {"application_id": created["id"]},
            "active_draft": {
                "company": "Neilsoft",
                "roles": ["RAG Engineer"],
                "employment_types": ["Full Time"],
                "job_link": "",
                "location": "onsite",
                "status": "",
                "current_stages": ["Applied"],
                "priority": "LOW",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
            },
        },
    )

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == ["RAG Engineer"]
    assert parsed["draft"]["priority"] == "HIGH"
    assert parsed["draft"]["employment_types_json"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"
    assert parsed["draft"]["current_stages_json"] == ["Applied"]
    assert parsed["confirmation_kind"] == "context"
    assert parsed["needs_confirmation"] is True


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_preserves_company_for_role_follow_up(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(
        client,
        "AI Engineer role",
        interpreter,
        context={
            "active_draft": {
                "company": "Neilsoft",
                "roles": [],
                "employment_types": [],
                "job_link": "",
                "location": "",
                "status": "",
                "current_stages": [],
                "priority": "",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
            }
        },
    )

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == ["AI Engineer"]


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_preserves_company_for_type_and_location_follow_up(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"employment_types": ["Full Time"], "location": "onsite"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(
        client,
        "fulltime ani onsite",
        interpreter,
        context={
            "active_draft": {
                "company": "Neilsoft",
                "roles": ["AI Engineer"],
                "employment_types": [],
                "job_link": "",
                "location": "",
                "status": "",
                "current_stages": [],
                "priority": "",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
            }
        },
    )

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == ["AI Engineer"]
    assert parsed["draft"]["employment_types_json"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_preserves_prior_fields_for_stage_follow_up(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"current_stages": ["Applied"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(
        client,
        "Applied stage thev",
        interpreter,
        context={
            "active_draft": {
                "company": "Neilsoft",
                "roles": ["AI Engineer"],
                "employment_types": ["Full Time"],
                "job_link": "",
                "location": "onsite",
                "status": "",
                "current_stages": [],
                "priority": "",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
            }
        },
    )

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == ["AI Engineer"]
    assert parsed["draft"]["employment_types_json"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"
    assert parsed["draft"]["current_stages_json"] == ["Applied"]


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_requires_company_context(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"priority": "HIGH"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Make it high priority", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "Which company should I use?"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_for_status_change(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "status": "Interested"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"status": "Rejected"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Mark Neilsoft as rejected", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["operation"] == "update"
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["status"] == "Rejected"


@pytest.mark.anyio
async def test_parse_transcript_endpoints_do_not_persist_applications(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Bootcoding", "roles": ["AI Engineer"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )
    await parse_transcript(client, "Add a Bootcoding AI Engineer application.", interpreter)
    response = await client.get("/applications")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio
async def test_parse_transcript_existing_company_alias_resolves_to_canonical_preview(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
        },
    )
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Crew Trim Labs", "roles": ["AI Engineer"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Track Crew Trim Labs for AI Engineer", interpreter)

    assert parsed["draft"]["company"] == "Krutrim Labs"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_resolves_single_row(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "roles_json": ["AI Engineer"]})
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "roles_json": ["GET"]})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Rockwell Automation", "role": "AI Engineer"},
                "fields": {"priority": "HIGH"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Set Rockwell Automation AI Engineer priority to high", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["operation"] == "update"
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_multiple_company_matches_return_clarification(client):
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "roles_json": ["AI Engineer"]})
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "roles_json": ["GET"]})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Rockwell Automation"},
                "fields": {"priority": "HIGH"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Set Rockwell Automation priority to high", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "Multiple applications match this company. Specify the role."


@pytest.mark.anyio
async def test_parse_transcript_unknown_company_update_creates_no_row(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Unknown Co"},
                "fields": {"status": "Rejected"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Mark Unknown Co as rejected", interpreter)

    assert parsed["status"] == "clarification_required"
    assert 'Application for company "Unknown Co" was not found.' in parsed["warnings"]

    listed = await client.get("/applications")
    assert listed.status_code == 200
    assert listed.json() == []


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_replaces_comments_free_text(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "comments": "Already applied"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"comments": "recruiter la udya ping karaycha", "next_action": "follow up with HR"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Update Neilsoft notes and next action", interpreter)

    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["comments"] == "recruiter la udya ping karaycha"
    assert parsed["draft"]["next_action"] == "follow up with HR"


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_warns_when_full_time_is_not_used_as_status(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {
                    "company": "Neilsoft",
                    "roles": ["AI Engineer"],
                    "employment_types": ["Full Time"],
                    "current_stages": ["Applied"],
                    "status": "full time",
                },
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "yeah for neilsoft, i'm applying for AI Engineer role for full time role", interpreter)

    assert parsed["status"] == "preview"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["roles_json"] == ["AI Engineer"]
    assert parsed["draft"]["employment_types_json"] == ["Full Time"]
    assert parsed["draft"]["current_stages_json"] == ["Applied"]
    assert parsed["draft"]["status"] == ""
    assert 'Interpreted "full time" as Employment Type, not Status.' in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_unsupported_status_is_rejected(client):
    await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"status": "Interviewing"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Mark Neilsoft as interviewing", interpreter)

    assert parsed["status"] == "unsupported"
    assert "Unsupported status value." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_unsupported_priority_is_rejected(client):
    await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"priority": "URGENT"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Set Neilsoft priority to urgent", interpreter)

    assert parsed["status"] == "unsupported"
    assert "Unsupported priority value." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_attach_latest_browser_context_returns_clarification(client):
    await create_browser_context(client, {"url": "https://example.com/job", "page_title": "AI Engineer - Example Company"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="attach_latest_browser_context",
            arguments={},
        )
    )

    parsed = await parse_transcript(
        client,
        "Use the current tab",
        interpreter,
        context={
            "active_draft": {
                "company": "Neilsoft",
                "roles": ["AI Engineer"],
                "employment_types": [],
                "job_link": "",
                "location": "",
                "status": "",
                "current_stages": [],
                "priority": "",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
            }
        },
    )

    assert parsed["status"] == "clarification_required"
    assert "Latest browser context:" in parsed["clarification_question"]


@pytest.mark.anyio
async def test_parse_transcript_attach_latest_browser_context_requires_active_draft(client):
    await create_browser_context(client, {"url": "https://example.com/job", "page_title": "AI Engineer - Example Company"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="attach_latest_browser_context",
            arguments={},
        )
    )

    parsed = await parse_transcript(client, "use current link", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "There is no active draft to attach the current link to."


@pytest.mark.anyio
async def test_parse_transcript_request_draft_save_is_non_persistent(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="request_draft_save",
            arguments={},
        )
    )

    parsed = await parse_transcript(client, "Save this draft", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "There is no active draft to save."


@pytest.mark.anyio
async def test_parse_transcript_request_draft_save_returns_preview_when_active_draft_exists(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="request_draft_save",
            arguments={},
        )
    )

    parsed = await parse_transcript(
        client,
        "save it",
        interpreter,
        context={
            "active_draft": {
                "company": "Neilsoft",
                "roles": ["AI Engineer"],
                "employment_types": ["Full Time"],
                "job_link": "",
                "location": "onsite",
                "status": "",
                "current_stages": ["Applied"],
                "priority": "",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
            }
        },
    )

    assert parsed["status"] == "preview"
    assert parsed["operation"] == "create"
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["needs_confirmation"] is True
    assert "Use the existing Save action" in parsed["warnings"][0]


@pytest.mark.anyio
async def test_parse_transcript_ask_clarification_passes_question_through(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="ask_clarification",
            arguments={"question": "Which role should I use for Neilsoft?"},
        )
    )

    parsed = await parse_transcript(client, "Neilsoft role unclear", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "Which role should I use for Neilsoft?"


@pytest.mark.anyio
async def test_parse_transcript_bounded_recent_context_is_sent_to_interpreter(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "roles": ["AI Engineer"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    await parse_transcript(
        client,
        "Add Neilsoft for AI Engineer",
        interpreter,
        context={
            "active_application": {"application_id": 12},
            "active_draft": {
                "company": "Neilsoft",
                "roles": ["RAG Engineer"],
                "employment_types": [],
                "job_link": "",
                "location": "",
                "status": "",
                "current_stages": [],
                "priority": "",
                "engaged_days": None,
                "next_action": "",
                "comments": "",
            },
            "recent_actions": ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        },
    )

    assert interpreter.calls[0]["context"]["recent_actions"] == ["a5", "a6", "a7"]


@pytest.mark.anyio
async def test_parse_transcript_make_it_high_priority_without_active_draft_or_selected_row_requires_persisted_target(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="ask_clarification",
            arguments={"question": "Which company's application do you mean?"},
        )
    )

    parsed = await parse_transcript(client, "Make it high priority", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "Which company's application do you mean?"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_requires_explicitly_selected_persisted_row_for_application_id_target(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "priority": "LOW"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"application_id": created["id"]},
                "fields": {"priority": "HIGH"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Make it high priority", interpreter)

    assert parsed["status"] == "clarification_required"
    assert parsed["clarification_question"] == "Which company's application do you mean?"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_accepts_explicitly_selected_persisted_row(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "priority": "LOW"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"application_id": created["id"]},
                "fields": {"priority": "HIGH"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(
        client,
        "Make it high priority",
        interpreter,
        context={"active_application": {"application_id": created["id"]}},
    )

    assert parsed["status"] == "preview"
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_ollama_unavailable_returns_recoverable_error(client):
    interpreter = FakeInterpreter(error=SemanticInterpreterUnavailableError("Local language interpreter is unavailable. No tracker changes were saved."))

    parsed = await parse_transcript(client, "Add Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] == "unavailable"
    assert parsed["warnings"] == ["Local language interpreter is unavailable. No tracker changes were saved."]


@pytest.mark.anyio
async def test_create_application(client):
    created = await create_record(client)
    assert created["company"] == "Bootcoding Pvt. LTD"
    assert_bootcoding_current_stages(created)


@pytest.mark.anyio
async def test_list_applications(client):
    await create_record(client)
    response = await client.get("/applications")
    assert response.status_code == 200
    listed = response.json()
    assert len(listed) == 1
    assert_bootcoding_current_stages(listed[0])


@pytest.mark.anyio
async def test_fetch_application_by_id(client):
    created = await create_record(client)
    response = await client.get(f"/applications/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert_bootcoding_current_stages(response.json())


@pytest.mark.anyio
async def test_update_application(client):
    created = await create_record(client)
    response = await client.patch(f"/applications/{created['id']}", json={"status": "interviewing"})
    assert response.status_code == 200
    assert response.json()["status"] == "interviewing"


@pytest.mark.anyio
async def test_update_status_preserves_current_stage(client):
    created = await create_record(
        client,
        REALISTIC_RECORD
        | {
            "status": "applied",
            "current_stages_json": ["Tailored", "Applied", "Networked"],
        },
    )

    updated = await client.patch(f"/applications/{created['id']}", json={"status": "interview"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "interview"
    assert_bootcoding_current_stages(updated.json())

    fetched = await client.get(f"/applications/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "interview"
    assert_bootcoding_current_stages(fetched.json())


@pytest.mark.anyio
async def test_update_status_accepts_custom_string(client):
    created = await create_record(client, REALISTIC_RECORD | {"status": "applied"})

    updated = await client.patch(
        f"/applications/{created['id']}",
        json={"status": "waiting for recruiter response"},
    )

    assert updated.status_code == 200
    assert updated.json()["status"] == "waiting for recruiter response"


@pytest.mark.anyio
async def test_delete_application(client):
    created = await create_record(client)
    response = await client.delete(f"/applications/{created['id']}")
    assert response.status_code == 204
    missing = await client.get(f"/applications/{created['id']}")
    assert missing.status_code == 404


@pytest.mark.anyio
async def test_delete_application_preserves_asr_correction_history_and_nulls_application_id(client):
    created = await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
            "audio_reference": "audio-ref-delete-test",
        },
    )

    delete_response = await client.delete(f"/applications/{created['id']}")
    assert delete_response.status_code == 204

    missing = await client.get(f"/applications/{created['id']}")
    assert missing.status_code == 404

    with SessionLocal() as db:
        remaining_events = db.query(AsrCompanyCorrectionEvent).all()
        assert len(remaining_events) == 1
        event = remaining_events[0]
        assert event.application_id is None
        assert event.raw_transcript == "Add Crew Trim Labs for an AI Engineer role."
        assert event.original_extracted_company_name == "Crew Trim Labs"
        assert event.confirmed_company_name == "Krutrim Labs"
        assert event.alias_created is True
        assert event.audio_reference == "audio-ref-delete-test"

    exported_events = export_correction_events()
    assert len(exported_events) == 1
    assert exported_events[0]["application_id"] is None
    assert exported_events[0]["audio_reference"] == "audio-ref-delete-test"


@pytest.mark.anyio
async def test_delete_application_without_correction_events_still_works(client):
    created = await create_record(client)

    response = await client.delete(f"/applications/{created['id']}")

    assert response.status_code == 204
    missing = await client.get(f"/applications/{created['id']}")
    assert missing.status_code == 404


@pytest.mark.anyio
async def test_invalid_role_rejection(client):
    payload = REALISTIC_RECORD | {"roles_json": ["Backend Wizard"]}
    response = await client.post("/applications", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_invalid_type_rejection(client):
    payload = REALISTIC_RECORD | {"employment_types_json": ["Contract"]}
    response = await client.post("/applications", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_invalid_location_rejection(client):
    payload = REALISTIC_RECORD | {"location": "mars"}
    response = await client.post("/applications", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_invalid_current_stage_rejection(client):
    payload = REALISTIC_RECORD | {"current_stages_json": ["Ghosted"]}
    response = await client.post("/applications", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_invalid_priority_rejection(client):
    payload = REALISTIC_RECORD | {"priority": "URGENT"}
    response = await client.post("/applications", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_multiple_role_values_persist_correctly(client):
    payload = REALISTIC_RECORD | {"roles_json": ["AI Engineer", "LLM Engineer", "RAG Engineer"]}
    created = await create_record(client, payload)
    assert created["roles_json"] == ["AI Engineer", "LLM Engineer", "RAG Engineer"]


@pytest.mark.anyio
async def test_multiple_type_values_persist_correctly(client):
    payload = REALISTIC_RECORD | {"employment_types_json": ["Internship", "Part Time"]}
    created = await create_record(client, payload)
    assert created["employment_types_json"] == ["Internship", "Part Time"]


@pytest.mark.anyio
async def test_multiple_current_stage_values_persist_correctly(client):
    created = await create_record(client)
    assert created["current_stages_json"] == ["Tailored", "Applied", "Networked"]


@pytest.mark.anyio
async def test_current_stage_remains_exactly_what_user_submitted(client):
    payload = REALISTIC_RECORD | {"current_stages_json": ["Networked", "Tailored"]}
    created = await create_record(client, payload)
    assert created["current_stages_json"] == ["Networked", "Tailored"]


@pytest.mark.anyio
async def test_changing_status_does_not_change_current_stage(client):
    created = await create_record(client)
    response = await client.patch(f"/applications/{created['id']}", json={"status": "followed up"})
    assert response.status_code == 200
    assert_bootcoding_current_stages(response.json())


@pytest.mark.anyio
async def test_changing_comments_does_not_change_current_stage(client):
    created = await create_record(client)
    response = await client.patch(f"/applications/{created['id']}", json={"comments": "User-entered new note"})
    assert response.status_code == 200
    body = response.json()
    assert body["comments"] == "User-entered new note"
    assert_bootcoding_current_stages(body)


@pytest.mark.anyio
async def test_current_stage_survives_fetch_list_status_and_comments_updates(client):
    created = await create_record(client)
    assert_bootcoding_current_stages(created)

    fetched = await client.get(f"/applications/{created['id']}")
    assert fetched.status_code == 200
    assert_bootcoding_current_stages(fetched.json())

    listed = await client.get("/applications")
    assert listed.status_code == 200
    assert_bootcoding_current_stages(listed.json()[0])

    status_updated = await client.patch(f"/applications/{created['id']}", json={"status": "custom status"})
    assert status_updated.status_code == 200
    assert_bootcoding_current_stages(status_updated.json())

    comments_updated = await client.patch(f"/applications/{created['id']}", json={"comments": "Manual comment only"})
    assert comments_updated.status_code == 200
    assert_bootcoding_current_stages(comments_updated.json())


@pytest.mark.anyio
async def test_next_action_remains_exactly_what_user_submitted(client):
    created = await create_record(client)
    next_action = "Send a short follow-up on Friday"
    response = await client.patch(f"/applications/{created['id']}", json={"next_action": next_action})
    assert response.status_code == 200
    assert response.json()["next_action"] == next_action


@pytest.mark.anyio
async def test_engaged_days_remains_exactly_what_user_submitted(client):
    created = await create_record(client)
    response = await client.patch(f"/applications/{created['id']}", json={"engaged_days": 7})
    assert response.status_code == 200
    assert response.json()["engaged_days"] == 7


@pytest.mark.anyio
async def test_new_company_create_request_returns_confirmation_required(client):
    response = await create_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
        },
    )

    assert response["status"] == "confirmation_required"
    assert response["requires_confirmation"] is True
    assert response["candidate"]["company"] == "Crew Trim Labs"

    listed = await client.get("/applications")
    assert listed.status_code == 200
    assert listed.json() == []


@pytest.mark.anyio
async def test_confirming_unchanged_new_company_name_creates_application(client):
    created = await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Crew Trim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
            "audio_reference": None,
        },
    )

    assert created["company"] == "Crew Trim Labs"

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    assert "Crew Trim Labs" in hotwords.json()["hotwords"]


@pytest.mark.anyio
async def test_correcting_new_company_name_uses_confirmed_canonical_name(client):
    created = await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
            "audio_reference": "session-1/chunk-1",
        },
    )

    assert created["company"] == "Krutrim Labs"

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()
    assert "Krutrim Labs" in body["hotwords"]
    assert "Crew Trim Labs" in body["hotwords"]


@pytest.mark.anyio
async def test_changed_asr_company_name_becomes_alias_when_meaningfully_different(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
            "audio_reference": None,
        },
    )

    response = await create_candidate(
        client,
        {
            "company": "crew trim labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Internship"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
        },
    )

    assert response["status"] == "created"
    assert response["application"]["company"] == "Krutrim Labs"


@pytest.mark.anyio
async def test_existing_company_create_path_does_not_trigger_confirmation_popup(client):
    await confirm_candidate(
        client,
        {
            "company": "Analytics Vidhya",
            "confirmed_company_name": "Analytics Vidhya",
            "roles_json": ["Generative AI Engineer"],
            "employment_types_json": ["Internship"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Analytics Vidhya for a Generative AI Engineer internship.",
            "original_extracted_company_name": "Analytics Vidhya",
        },
    )

    response = await create_candidate(
        client,
        {
            "company": " analytics   vidhya ",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
        },
    )

    assert response["status"] == "created"
    assert response["application"]["company"] == "Analytics Vidhya"


@pytest.mark.anyio
async def test_alias_lookup_resolves_to_canonical_company(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
        },
    )

    response = await create_candidate(
        client,
        {
            "company": "Crew-Trim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Part Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
        },
    )

    assert response["status"] == "created"
    assert response["application"]["company"] == "Krutrim Labs"


@pytest.mark.anyio
async def test_hotword_list_is_deduplicated_and_bounded(client):
    for index in range(1, 120):
        await confirm_candidate(
            client,
            {
                "company": f"Company {index}",
                "confirmed_company_name": f"Company {index}",
                "roles_json": ["AI Engineer"],
                "employment_types_json": ["Full Time"],
                "job_link": "",
                "location": "",
                "status": "",
                "current_stages_json": [],
                "priority": "",
                "engaged_days": 0,
                "next_action": "",
                "comments": "",
                "raw_transcript": f"Add Company {index} for an AI Engineer role.",
                "original_extracted_company_name": f"Company {index}",
            },
        )

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()
    assert body["limit"] == 100
    assert len(body["hotwords"]) == 100
    assert len({value.lower() for value in body["hotwords"]}) == len(body["hotwords"])


@pytest.mark.anyio
async def test_hotword_list_prefers_canonical_name_and_ignores_duplicate_aliases(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
        },
    )

    await confirm_candidate(
        client,
        {
            "company": "crew   trim   labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Internship"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add crew trim labs for an AI Engineer role.",
            "original_extracted_company_name": "crew   trim   labs",
        },
    )

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()["hotwords"]
    assert body.index("Krutrim Labs") < body.index("Crew Trim Labs")
    assert sum(1 for value in body if value.lower().replace(" ", "") == "crewtrimlabs") == 1


@pytest.mark.anyio
async def test_hotword_list_ignores_blank_values_and_preserves_static_vocabulary(client):
    await confirm_candidate(
        client,
        {
            "company": "Whitespace Co",
            "confirmed_company_name": "Whitespace Co",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Whitespace Co for an AI Engineer role.",
            "original_extracted_company_name": "   ",
        },
    )

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()["hotwords"]
    assert "" not in body
    assert "AI Engineer" in body
    assert "next action" in body


@pytest.mark.anyio
async def test_case_and_punctuation_only_confirmation_change_does_not_create_redundant_alias(client):
    await confirm_candidate(
        client,
        {
            "company": "Krutrim Labs",
            "confirmed_company_name": "Krutrim Labs!",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Krutrim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Krutrim Labs",
        },
    )

    with SessionLocal() as db:
        aliases = db.query(CompanyAlias).all()
        assert aliases == []
        correction_event = db.query(AsrCompanyCorrectionEvent).one()
        assert correction_event.alias_created is False


@pytest.mark.anyio
async def test_correction_event_is_persisted(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "roles_json": ["AI Engineer"],
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
            "audio_reference": "audio-ref-123",
        },
    )

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    assert "Crew Trim Labs" in hotwords.json()["hotwords"]

    applications = await client.get("/applications")
    assert applications.status_code == 200
    assert applications.json()[0]["company"] == "Krutrim Labs"

    with SessionLocal() as db:
        correction_events = db.query(AsrCompanyCorrectionEvent).all()
        assert len(correction_events) == 1
        event = correction_events[0]
        assert event.raw_transcript == "Add Crew Trim Labs for an AI Engineer role."
        assert event.original_extracted_company_name == "Crew Trim Labs"
        assert event.confirmed_company_name == "Krutrim Labs"
        assert event.alias_created is True
        assert event.audio_reference == "audio-ref-123"

        canonical_companies = db.query(CanonicalCompany).all()
        aliases = db.query(CompanyAlias).all()
        assert len(canonical_companies) == 1
        assert canonical_companies[0].canonical_name == "Krutrim Labs"
        assert len(aliases) == 1
        assert aliases[0].alias_text == "Crew Trim Labs"


@pytest.mark.anyio
async def test_missing_application_returns_404(client):
    response = await client.get("/applications/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Application not found"}
