from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.company_resolution import detect_explicit_known_companies
from app.database import SessionLocal
from app.main import app
from app.models import AsrCompanyCorrectionEvent, CanonicalCompany, CompanyAlias, JobApplication
from app.semantic_validation import (
    CLARIFICATION_CONFLICTING_COMPANY,
    CLARIFICATION_MISSING_COMPANY,
    CLARIFICATION_MULTIPLE_EXPLICIT_COMPANIES,
    normalize_patch_active_draft_argument_shape,
    normalize_semantic_field_patch_argument_shape,
    normalize_role_title,
)
from app.semantic_interpreter import (
    SemanticInterpretationResult,
    SemanticInterpreterUnavailableError,
    get_semantic_interpreter,
)
from app.semantic_schemas import SemanticInterpreterMetrics, SemanticToolCallProposal
from app.semantic_schemas import SemanticExtractedFields


REALISTIC_RECORD = {
    "company": "Bootcoding Pvt. LTD",
    "role": "AI Engineer",
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
    def __init__(self, *, proposal=None, extracted_fields=None, error=None, metrics=None, health=None, max_tool_turns=2):
        self.proposal = proposal
        self.extracted_fields = SemanticExtractedFields.model_validate(extracted_fields or {})
        self.error = error
        self.metrics = metrics or SemanticInterpreterMetrics(latency_ms=12)
        self.health = health or {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)
        self.calls: list[dict[str, object | None]] = []

    def interpret(self, transcript: str, context=None):
        self.calls.append({"transcript": transcript, "context": context})
        if self.error is not None:
            raise self.error
        return SemanticInterpretationResult(
            proposal=self.proposal,
            metrics=self.metrics,
            extracted_fields=self.extracted_fields,
        )

    def health_check(self):
        return self.health


class SequencedFakeInterpreter(FakeInterpreter):
    def __init__(self, proposals, *, extracted_fields=None):
        super().__init__(proposal=proposals[0] if proposals else None, extracted_fields=extracted_fields)
        self._proposals = list(proposals)

    def interpret(self, transcript: str, context=None):
        self.calls.append({"transcript": transcript, "context": context})
        if not self._proposals:
            raise AssertionError("No more proposals configured.")
        proposal_value = self._proposals.pop(0)
        return SemanticInterpretationResult(
            proposal=proposal_value,
            metrics=self.metrics,
            extracted_fields=self.extracted_fields,
        )


def proposal(**overrides):
    payload = {
        "tool_name": "ask_clarification",
        "arguments": {"question": "Could you clarify what to update?"},
    }
    payload.update(overrides)
    return SemanticToolCallProposal.model_validate(payload)


def register_known_company(canonical_name: str, alias_text: str | None = None) -> None:
    with SessionLocal() as db:
        canonical_company = CanonicalCompany(canonical_name=canonical_name)
        db.add(canonical_company)
        db.flush()
        if alias_text is not None:
            db.add(CompanyAlias(canonical_company_id=canonical_company.id, alias_text=alias_text))
        db.commit()


def test_normalize_patch_active_draft_argument_shape_promotes_scalar_roles_to_role():
    repaired = normalize_patch_active_draft_argument_shape(
        proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": "AI Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    assert repaired.arguments["fields"]["role"] == "AI Engineer"
    assert "roles" not in repaired.arguments["fields"]


def test_normalize_patch_active_draft_argument_shape_preserves_role_scalar():
    repaired = normalize_patch_active_draft_argument_shape(
        proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "role": "AI Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    assert repaired.arguments["fields"]["role"] == "AI Engineer"
    assert "roles" not in repaired.arguments["fields"]


def test_normalize_patch_active_draft_argument_shape_rejects_conflicting_role_and_roles_values():
    # roles:["ML Engineer"] and role:"AI Engineer" conflict → normalize_semantic_field_patch_argument_shape returns None
    # → normalize_patch_active_draft_argument_shape returns the original proposal unchanged
    unrepaired = normalize_patch_active_draft_argument_shape(
        proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "role": "AI Engineer", "roles": ["ML Engineer"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    # When normalization fails, the original proposal is returned unchanged
    assert unrepaired.arguments["fields"]["role"] == "AI Engineer"
    assert unrepaired.arguments["fields"]["roles"] == ["ML Engineer"]


def test_normalize_role_title_trims_surrounding_whitespace():
    assert normalize_role_title(" AI Engineer ") == "AI Engineer"


def test_normalize_role_title_collapse_repeated_whitespace():
    assert normalize_role_title("AI   Engineer") == "AI Engineer"


def test_normalize_role_title_preserves_applied_ai_engineer():
    assert normalize_role_title("Applied AI Engineer") == "Applied AI Engineer"


def test_normalize_role_title_preserves_intern_role_title():
    assert normalize_role_title("AI Engineer Intern") == "AI Engineer Intern"


def test_normalize_role_title_preserves_generative_ai_engineer():
    assert normalize_role_title("Generative AI Engineer") == "Generative AI Engineer"


def test_normalize_role_title_preserves_computer_vision_engineer():
    assert normalize_role_title("Computer Vision Engineer") == "Computer Vision Engineer"


def test_normalize_role_title_accepts_open_ended_role_titles():
    assert normalize_role_title("Applied AI Engineer") == "Applied AI Engineer"


def test_normalize_role_title_rejects_blank_role_value():
    assert normalize_role_title("   ") is None


def test_normalize_role_title_accepts_unknown_but_non_blank_role_title():
    assert normalize_role_title("Unknown Ninja Role") == "Unknown Ninja Role"


def test_normalize_semantic_field_patch_argument_shape_converts_fulltime_to_full_time():
    normalized = normalize_semantic_field_patch_argument_shape({"type": "fulltime"})

    assert normalized == {"employment_types": ["Full Time"]}


def test_normalize_semantic_field_patch_argument_shape_converts_on_site_to_onsite():
    normalized = normalize_semantic_field_patch_argument_shape({"location": "on site"})

    assert normalized == {"location": "onsite"}


def test_normalize_semantic_field_patch_argument_shape_converts_wfh_to_remote():
    normalized = normalize_semantic_field_patch_argument_shape({"location": "wfh"})

    assert normalized == {"location": "remote"}


def test_normalize_semantic_field_patch_argument_shape_converts_high_to_uppercase_priority():
    normalized = normalize_semantic_field_patch_argument_shape({"priority": "high"})

    assert normalized == {"priority": "HIGH"}


def test_normalize_semantic_field_patch_argument_shape_converts_applied_to_canonical_status():
    normalized = normalize_semantic_field_patch_argument_shape({"status": "applied"})

    assert normalized == {"status": "applied"}


def test_normalize_semantic_field_patch_argument_shape_promotes_current_stage_scalar_to_array():
    normalized = normalize_semantic_field_patch_argument_shape({"current_stage": "Applied"})

    assert normalized == {"current_stages": ["Applied"]}


def test_normalize_semantic_field_patch_argument_shape_preserves_multiple_stages():
    normalized = normalize_semantic_field_patch_argument_shape({"stage": ["Applied", "Engaged"]})

    assert normalized == {"current_stages": ["Applied", "Engaged"]}


def test_normalize_semantic_field_patch_argument_shape_trims_comments_and_next_action():
    normalized = normalize_semantic_field_patch_argument_shape(
        {"comments": "  referral received  ", "next_action": "  follow up tomorrow  "}
    )

    assert normalized == {"comments": "referral received", "next_action": "follow up tomorrow"}


def test_normalize_semantic_field_patch_argument_shape_rejects_conflicting_alias_and_canonical_values():
    normalized = normalize_semantic_field_patch_argument_shape(
        {"employment_types": ["Full Time"], "type": "Internship"}
    )

    assert normalized is None


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
    # Accepts both the public DTO name and the internal schema name.
    stages = record.get("current_stages") or record.get("current_stages_json")
    assert stages == ["Tailored", "Applied", "Networked"]


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
    assert fetched_application.json()["id"] == created_application["id"]
    assert listed_applications.status_code == 200
    listed = listed_applications.json()
    assert len(listed) == 1
    assert listed[0]["id"] == created_application["id"]
    assert listed[0]["company"] == created_application["company"]


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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]


@pytest.mark.anyio
async def test_parse_transcript_partial_draft_accepts_company_only_phrase(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "I have a requirement. I want to add an application neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == ""
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"


def test_detect_explicit_known_companies_matches_canonical_with_punctuation(db_session):
    register_known_company("Rockwell Automation")

    detected = detect_explicit_known_companies(db_session, "Update Rockwell Automation, priority to high.")

    assert detected == ["Rockwell Automation"]


def test_detect_explicit_known_companies_resolves_alias_and_casefolds(db_session):
    register_known_company("Krutrim Labs", alias_text="Crew Trim Labs")

    detected = detect_explicit_known_companies(db_session, "track crew-trim labs for AI Engineer")

    assert detected == ["Krutrim Labs"]


def test_detect_explicit_known_companies_prefers_longest_phrase_first(db_session):
    register_known_company("Automation")
    register_known_company("Rockwell Automation")

    detected = detect_explicit_known_companies(db_session, "Set Rockwell Automation priority to high")

    assert detected == ["Rockwell Automation"]


@pytest.mark.anyio
async def test_parse_transcript_known_company_reconciles_missing_company_for_patch_active_draft(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "AI Engineer role for Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert interpreter.calls[0]["context"]["explicit_known_companies"] == ["Neilsoft"]


@pytest.mark.anyio
async def test_parse_transcript_conversational_extraction_safely_merges_missing_role(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )

    parsed = await parse_transcript(
        client,
        "I have previously worked at this company. It's called Neilsoft. I'd like to track AI Engineer application for this position.",
        interpreter,
    )

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert interpreter.calls[0]["context"]["explicit_known_companies"] == ["Neilsoft"]


@pytest.mark.anyio
async def test_parse_transcript_role_alias_shape_is_repaired_for_varied_word_order(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "role": "AI Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_scalar_roles_shape_is_repaired(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": "AI Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "At Neilsoft, role is AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_role_at_neilsoft_for_ai_engineer(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_ai_engineer_role_for_neilsoft(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "AI Engineer role for Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_at_neilsoft_role_is_ai_engineer(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "At Neilsoft, role is AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_track_ai_engineer_opening_at_neilsoft(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )

    parsed = await parse_transcript(client, "Track an AI Engineer opening at Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_applied_ai_engineer_role_for_neilsoft(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "Applied AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": ["Applied AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Applied AI Engineer role for Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "Applied AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_ai_engineer_intern_role_for_neilsoft(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer Intern"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft", "roles": ["AI Engineer Intern"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "AI Engineer Intern role for Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer Intern"


@pytest.mark.anyio
async def test_parse_transcript_representative_patch_normalizes_multiple_fields(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={
            "company": "Neilsoft",
            "role": "AI Engineer",
            "employment_types": ["fulltime"],
            "location": "onsite",
            "priority": "high",
        },
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {
                    "company": "Neilsoft",
                    "role": "AI Engineer",
                    "type": "fulltime",
                    "location": "on site",
                    "priority": "high priority",
                },
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Neilsoft sathi AI Engineer role, fulltime onsite, high priority", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_reported_false_conflict_sentence_uses_authoritative_extracted_fields(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={
            "company": "Neilsoft",
            "role": "AI Engineer",
            "employment_types": ["Full Time"],
            "location": "onsite",
        },
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {
                    "company": "Neilsoft",
                    "roles": ["AI Engineer"],
                    "employment_types": ["full time"],
                    "location": "on site",
                },
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )

    parsed = await parse_transcript(
        client,
        "I'd like to track an application for Neilsoft, the role is for AI Engineer, the type is full time, the location is onsite.",
        interpreter,
    )

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"


@pytest.mark.anyio
async def test_parse_transcript_multi_word_company_role_variant(client):
    register_known_company("Rockwell Automation")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Rockwell Automation", "role": "ML Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Rockwell Automation", "roles": ["ML Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Role at Rockwell Automation for ML Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Rockwell Automation"
    assert parsed["draft"]["role"] == "ML Engineer"


@pytest.mark.anyio
async def test_parse_transcript_filler_is_ignored_during_extraction_merge(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )

    parsed = await parse_transcript(
        client,
        "I previously worked at this company, actually. Anyway, it is called Neilsoft. Please track an AI Engineer application.",
        interpreter,
    )

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_remote_internship_aliases_normalize_for_draft(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "employment_types": ["internship"], "location": "remote"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "employment_type": "intern", "location": "wfh"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Neilsoft application is remote internship", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["employment_types"] == ["Internship"]
    assert parsed["draft"]["location"] == "remote"


@pytest.mark.anyio
async def test_parse_transcript_remote_fulltime_role_extraction_is_preserved(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={
            "company": "Neilsoft",
            "role": "AI Engineer",
            "employment_types": ["Full Time"],
            "location": "remote",
        },
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "Neilsoft"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )

    parsed = await parse_transcript(
        client,
        "I want to track a remote fulltime AI Engineer application for Neilsoft",
        interpreter,
    )

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "remote"


@pytest.mark.anyio
async def test_parse_transcript_equivalent_selected_values_do_not_conflict(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={
            "company": "Neilsoft",
            "role": "AI Engineer",
            "employment_types": ["Full Time"],
            "location": "onsite",
        },
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {
                    "company": "Neilsoft",
                    "roles": ["AI Engineer"],
                    "employment_types": ["full-time"],
                    "location": "on-site",
                },
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"


@pytest.mark.anyio
async def test_parse_transcript_legacy_roles_array_dropped_when_role_scalar_present(client):
    """When LLM sends both 'role' (scalar) and stale 'roles' (array) with different values,
    normalization drops 'roles' and the authoritative extracted scalar role wins."""
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "role": "AI Engineer", "roles": ["ML Engineer"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_extracted_company_conflict_fails_safely_without_db_write(client):
    register_known_company("Neilsoft")
    register_known_company("Rockwell Automation")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Rockwell Automation"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )

    parsed = await parse_transcript(client, "Track AI Engineer application for Neilsoft", interpreter)

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == CLARIFICATION_CONFLICTING_COMPANY

    listed = await client.get("/applications")
    assert listed.status_code == 200
    assert listed.json() == []


@pytest.mark.anyio
async def test_parse_transcript_real_field_conflict_fails_safely_without_db_write(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer", "location": "onsite"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "roles": ["ML Engineer"], "location": "remote"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )

    parsed = await parse_transcript(client, "Track AI Engineer application for Neilsoft onsite", interpreter)

    assert parsed["status"] == "no_change"
    assert "Extracted fields conflicted with selected tool arguments. No tracker changes were saved." in parsed["warnings"]

    listed = await client.get("/applications")
    assert listed.status_code == 200
    assert listed.json() == []


@pytest.mark.anyio
async def test_parse_transcript_non_string_role_rejected(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "roles": [{"title": "AI Engineer"}]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] == "no_change"
    assert "Local language interpreter returned invalid tool arguments. No tracker changes were saved." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_blank_role_rejected(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "roles": ["   "]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] == "no_change"
    assert "Local language interpreter returned invalid tool arguments. No tracker changes were saved." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_unsupported_role_alias_field_still_fails_safely(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "designation": "AI Engineer"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] == "no_change"
    assert "Local language interpreter returned invalid tool arguments. No tracker changes were saved." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_known_company_retry_recovers_missing_company_clarification(client):
    register_known_company("Neilsoft")
    interpreter = SequencedFakeInterpreter(
        [
            proposal(tool_name="ask_clarification", arguments={"question": CLARIFICATION_MISSING_COMPANY}),
            proposal(
                tool_name="patch_active_draft",
                arguments={"fields": {"roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
            ),
        ]
    )

    parsed = await parse_transcript(client, "AI Engineer role for Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert len(interpreter.calls) == 2
    assert interpreter.calls[1]["context"]["explicit_company_retry_hint"] is not None


@pytest.mark.anyio
async def test_parse_transcript_schema_repair_retry_recovers_invalid_first_shape(client):
    register_known_company("Neilsoft")
    interpreter = SequencedFakeInterpreter(
        [
            proposal(
                tool_name="patch_active_draft",
                arguments={
                    "fields": {"company": "Neilsoft", "role": {"bad": "shape"}},
                    "replace_explicit_fields": True,
                    "context_notes": [],
                },
            ),
            proposal(
                tool_name="patch_active_draft",
                arguments={
                    "fields": {"company": "Neilsoft", "roles": ["AI Engineer"]},
                    "replace_explicit_fields": True,
                    "context_notes": [],
                },
            ),
        ]
    )

    parsed = await parse_transcript(client, "Role at Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert len(interpreter.calls) == 2
    assert interpreter.calls[1]["context"]["schema_repair_retry_hint"] is not None


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
                "role": "RAG Engineer",
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "RAG Engineer"
    assert parsed["draft"]["priority"] == "HIGH"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"
    assert parsed["draft"]["current_stages"] == ["Applied"]


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_preserves_company_for_role_follow_up(client):
    register_known_company("Rockwell Automation")
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
                "role": "",
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"


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
                "role": "AI Engineer",
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_normalizes_current_stage_alias_with_active_draft(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"current_stage": "Applied"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(
        client,
        "Set current stage Applied",
        interpreter,
        context={
            "active_draft": {
                "company": "Neilsoft",
                "role": "AI Engineer",
                "employment_types": ["Full Time"],
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["current_stages"] == ["Applied"]


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
                "role": "AI Engineer",
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["location"] == "onsite"
    assert parsed["draft"]["current_stages"] == ["Applied"]


@pytest.mark.anyio
async def test_parse_transcript_patch_active_draft_requires_company_context(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"priority": "HIGH"}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "Make it high priority", interpreter)

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == "Which company should I use?"


@pytest.mark.anyio
async def test_parse_transcript_missing_company_does_not_use_recent_history_for_persisted_update(client):
    register_known_company("Neilsoft")
    await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "priority": "LOW"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={"target": {}, "fields": {"priority": "HIGH"}, "replace_explicit_fields": True},
        )
    )

    parsed = await parse_transcript(
        client,
        "Update priority to high",
        interpreter,
        context={"recent_actions": [{"company": "Neilsoft"}], "active_company": "Neilsoft"},
    )

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == "Which company's application do you mean?"


@pytest.mark.anyio
async def test_parse_transcript_known_company_reconciles_missing_company_for_persisted_update(client):
    register_known_company("Neilsoft")
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "priority": "LOW"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={"target": {}, "fields": {"priority": "HIGH"}, "replace_explicit_fields": True},
        )
    )

    parsed = await parse_transcript(client, "Update Neilsoft priority to high", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_for_status_change(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "status": "in_touch"})
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["status"] == "rejected"


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
            "role": "AI Engineer",
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
async def test_parse_transcript_known_company_conflict_returns_safe_clarification(client):
    register_known_company("Neilsoft")
    register_known_company("Rockwell Automation")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Rockwell Automation", "roles": ["AI Engineer"]},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "AI Engineer role for Neilsoft", interpreter)

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == CLARIFICATION_CONFLICTING_COMPANY


@pytest.mark.anyio
async def test_parse_transcript_multiple_explicit_companies_return_safe_clarification(client):
    register_known_company("Neilsoft")
    register_known_company("Rockwell Automation")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "AI Engineer role for Neilsoft and Rockwell Automation", interpreter)

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == CLARIFICATION_MULTIPLE_EXPLICIT_COMPANIES


@pytest.mark.anyio
async def test_parse_transcript_unknown_company_follows_existing_safe_behavior(client):
    interpreter = FakeInterpreter(
        extracted_fields={"role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"roles": ["AI Engineer"]}, "replace_explicit_fields": True, "context_notes": []},
        )
    )

    parsed = await parse_transcript(client, "AI Engineer role for NewStartup Labs", interpreter)

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == CLARIFICATION_MISSING_COMPANY


@pytest.mark.anyio
async def test_parse_transcript_unknown_new_company_still_supported_via_extraction(client):
    interpreter = FakeInterpreter(
        extracted_fields={"company": "NewStartup Labs", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={"fields": {"company": "NewStartup Labs"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )

    parsed = await parse_transcript(client, "Track an AI Engineer application at NewStartup Labs", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "NewStartup Labs"
    assert parsed["draft"]["role"] == "AI Engineer"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_resolves_single_row(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "role": "AI Engineer"})
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "role": "GET"})
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_multiple_company_matches_return_clarification(client):
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "role": "AI Engineer"})
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation", "role": "GET"})
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

    assert parsed["status"] == "clarification"
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

    assert parsed["status"] == "clarification"
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
async def test_parse_transcript_preview_existing_application_update_normalizes_status_alias(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "status": ""})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"status": "applied"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "For Neilsoft set status applied", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["status"] == "applied"


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_normalizes_priority_alias(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "priority": "LOW"})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"priority": "high"},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Set Neilsoft priority high", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_persisted_target_company_conflict_remains_safe(client):
    await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft"})
    await create_record(client, REALISTIC_RECORD | {"company": "Rockwell Automation"})
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "priority": "HIGH"},
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Rockwell Automation"},
                "fields": {"priority": "HIGH"},
                "replace_explicit_fields": True,
            },
        ),
    )

    parsed = await parse_transcript(client, "Set Neilsoft priority high", interpreter)

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == CLARIFICATION_CONFLICTING_COMPANY


@pytest.mark.anyio
async def test_parse_transcript_preview_existing_application_update_preserves_trimmed_note_text(client):
    created = await create_record(client, REALISTIC_RECORD | {"company": "Neilsoft", "comments": ""})
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="preview_existing_application_update",
            arguments={
                "target": {"company": "Neilsoft"},
                "fields": {"comments": "  referral received  "},
                "replace_explicit_fields": True,
            },
        )
    )

    parsed = await parse_transcript(client, "Add note for Neilsoft saying referral received", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["comments"] == "referral received"


@pytest.mark.anyio
async def test_parse_transcript_invented_pass2_patch_field_is_discarded_when_not_extracted(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "roles": ["AI Engineer"], "priority": "HIGH"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )

    parsed = await parse_transcript(client, "Track an AI Engineer application at Neilsoft", interpreter)

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["priority"] == ""


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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["draft"]["company"] == "Neilsoft"
    assert parsed["draft"]["role"] == "AI Engineer"
    assert parsed["draft"]["employment_types"] == ["Full Time"]
    assert parsed["draft"]["current_stages"] == ["Applied"]
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

    assert parsed["status"] == "no_change"
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

    assert parsed["status"] == "no_change"
    assert "Unsupported priority value." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_contract_type_alias_is_rejected_without_db_write(client):
    register_known_company("Neilsoft")
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="patch_active_draft",
            arguments={
                "fields": {"company": "Neilsoft", "type": "contract"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        )
    )

    parsed = await parse_transcript(client, "Neilsoft application is contract", interpreter)

    assert parsed["status"] == "no_change"
    assert "Unsupported employment type value." in parsed["warnings"]

    listed = await client.get("/applications")
    assert listed.status_code == 200
    assert listed.json() == []


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
                "role": "AI Engineer",
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

    assert parsed["status"] == "clarification"
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

    assert parsed["status"] == "clarification"
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

    assert parsed["status"] == "clarification"
    assert parsed["clarification_question"] == "There is no active draft to save."


@pytest.mark.anyio
async def test_parse_transcript_request_draft_save_without_draft_id_returns_error(client):
    """save command with active_draft context but no draft_id: no DB row → save fails."""
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
                "role": "AI Engineer",
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

    # Without a draft_id, there is no DB row to save.
    # The dispatcher returns failure → response is unsupported or clarification.
    assert parsed["status"] in {"no_change", "clarification"}


@pytest.mark.anyio
async def test_parse_transcript_ask_clarification_passes_question_through(client):
    interpreter = FakeInterpreter(
        proposal=proposal(
            tool_name="ask_clarification",
            arguments={"question": "Which role should I use for Neilsoft?"},
        )
    )

    parsed = await parse_transcript(client, "Neilsoft role unclear", interpreter)

    assert parsed["status"] == "clarification"
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
                "role": "RAG Engineer",
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

    assert parsed["status"] == "clarification"
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

    assert parsed["status"] == "clarification"
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

    assert parsed["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert parsed["application_id"] == created["id"]
    assert parsed["draft"]["priority"] == "HIGH"


@pytest.mark.anyio
async def test_parse_transcript_ollama_unavailable_returns_recoverable_error(client):
    interpreter = FakeInterpreter(error=SemanticInterpreterUnavailableError("Local language interpreter is unavailable. No tracker changes were saved."))

    parsed = await parse_transcript(client, "Add Neilsoft for AI Engineer", interpreter)

    assert parsed["status"] == "error"
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
    response = await client.patch(f"/applications/{created['id']}", json={"status": "applied"})
    assert response.status_code == 200
    assert response.json()["status"] == "applied"


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

    updated = await client.patch(f"/applications/{created['id']}", json={"status": "applied"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "applied"
    assert_bootcoding_current_stages(updated.json())

    fetched = await client.get(f"/applications/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "applied"
    assert_bootcoding_current_stages(fetched.json())


@pytest.mark.anyio
async def test_update_status_rejects_unknown_string(client):
    """Direct PATCH must reject custom/unknown status values — controlled enum only."""
    created = await create_record(client, REALISTIC_RECORD | {"status": "applied"})

    updated = await client.patch(
        f"/applications/{created['id']}",
        json={"status": "waiting for recruiter response"},
    )

    assert updated.status_code == 422


@pytest.mark.anyio
async def test_delete_application(client):
    created = await create_record(client)
    response = await client.delete(f"/applications/{created['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["requires_confirmation"] is True
    assert body["confirmation_kind"] == "archive"
    # Row is NOT deleted — use POST /archive to actually archive
    still_there = await client.get(f"/applications/{created['id']}")
    assert still_there.status_code == 200


@pytest.mark.anyio
async def test_delete_application_preserves_asr_correction_history_and_nulls_application_id(client):
    created = await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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
    assert delete_response.status_code == 200
    assert delete_response.json()["requires_confirmation"] is True

    # Hard delete via archive + direct DB delete to test correction event behavior
    archive_response = await client.post(f"/applications/{created['id']}/archive")
    assert archive_response.status_code == 200

    # Now actually hard-delete via DB to test ASR event behavior
    with SessionLocal() as db:
        app_obj = db.get(JobApplication, created["id"])
        db.delete(app_obj)
        db.commit()

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
        canonical_companies = db.query(CanonicalCompany).all()
        aliases = db.query(CompanyAlias).all()
        assert len(canonical_companies) == 1
        assert canonical_companies[0].canonical_name == "Krutrim Labs"
        assert len(aliases) == 1
        assert aliases[0].alias_text == "Crew Trim Labs"

    exported_events = export_correction_events()
    assert len(exported_events) == 1
    assert exported_events[0]["application_id"] is None
    assert exported_events[0]["audio_reference"] == "audio-ref-delete-test"

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()["hotwords"]
    assert "Krutrim Labs" in body
    assert "Crew Trim Labs" not in body


@pytest.mark.anyio
async def test_delete_application_without_correction_events_still_works(client):
    created = await create_record(client)

    response = await client.delete(f"/applications/{created['id']}")

    assert response.status_code == 200
    assert response.json()["requires_confirmation"] is True
    # Row not deleted — still accessible
    still_there = await client.get(f"/applications/{created['id']}")
    assert still_there.status_code == 200


@pytest.mark.anyio
async def test_open_ended_role_titles_are_accepted(client):
    payload = REALISTIC_RECORD | {"role": "Backend Wizard"}
    created = await create_record(client, payload)
    assert created["role"] == "Backend Wizard"


@pytest.mark.anyio
async def test_arbitrary_role_string_is_accepted(client):
    payload = REALISTIC_RECORD | {"role": "Galactic Overlord Engineer"}
    created = await create_record(client, payload)
    assert created["role"] == "Galactic Overlord Engineer"


@pytest.mark.anyio
async def test_blank_role_string_is_stored_as_empty(client):
    payload = REALISTIC_RECORD | {"role": ""}
    created = await create_record(client, payload)
    assert created["role"] == ""


@pytest.mark.anyio
async def test_role_with_word_role_in_title_is_stored_correctly(client):
    payload = REALISTIC_RECORD | {"role": "Platform Role"}
    created = await create_record(client, payload)
    assert created["role"] == "Platform Role"


@pytest.mark.anyio
async def test_blank_role_normalizes_to_empty_string(client):
    payload = REALISTIC_RECORD | {"role": "   "}
    created = await create_record(client, payload)
    assert created["role"] == ""


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
async def test_role_value_persists_correctly(client):
    payload = REALISTIC_RECORD | {"role": "AI Engineer"}
    created = await create_record(client, payload)
    assert created["role"] == "AI Engineer"


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
    response = await client.patch(f"/applications/{created['id']}", json={"status": "rejected"})
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

    status_updated = await client.patch(f"/applications/{created['id']}", json={"status": "accepted"})
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
            "role": "AI Engineer",
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
            "role": "AI Engineer",
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
            "role": "AI Engineer",
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
    assert "Crew Trim Labs" not in body["hotwords"]


@pytest.mark.anyio
async def test_changed_asr_company_name_becomes_alias_when_meaningfully_different(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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
            "role": "ML Engineer",
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
            "role": "Generative AI Engineer",
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
            "role": "AI Engineer",
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


def _candidate_payload(company: str, roles: list[str]) -> dict:
    return {
        "company": company,
        "role": roles[0] if roles else "",
        "employment_types_json": [],
        "job_link": "",
        "location": "",
        "status": "",
        "current_stages_json": [],
        "priority": "",
        "engaged_days": 0,
        "next_action": "",
        "comments": "",
    }


@pytest.mark.anyio
async def test_create_duplicate_company_and_role_is_rejected(client):
    register_known_company("Rockwell")
    first = await create_candidate(client, _candidate_payload("Rockwell", ["AI Engineer"]))
    assert first["status"] == "created"

    response = await client.post("/applications/create-candidate", json=_candidate_payload("Rockwell", ["AI Engineer"]))
    assert response.status_code == 409
    assert response.json()["detail"] == "Application for Rockwell — AI Engineer already exists."

    listed = await client.get("/applications")
    assert len([row for row in listed.json() if row["company"] == "Rockwell"]) == 1


@pytest.mark.anyio
async def test_create_same_company_different_role_is_allowed(client):
    register_known_company("Rockwell")
    first = await create_candidate(client, _candidate_payload("Rockwell", ["AI Intern"]))
    assert first["status"] == "created"

    second = await create_candidate(client, _candidate_payload("Rockwell", ["GET Program"]))
    assert second["status"] == "created"

    listed = await client.get("/applications")
    rockwell_rows = [row for row in listed.json() if row["company"] == "Rockwell"]
    assert len(rockwell_rows) == 2


@pytest.mark.anyio
async def test_create_duplicate_is_case_insensitive(client):
    register_known_company("Rockwell")
    first = await create_candidate(client, _candidate_payload("Rockwell", ["ai engineer"]))
    assert first["status"] == "created"

    response = await client.post("/applications/create-candidate", json=_candidate_payload("Rockwell", ["AI Engineer"]))
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

    listed = await client.get("/applications")
    assert len([row for row in listed.json() if row["company"] == "Rockwell"]) == 1


@pytest.mark.anyio
async def test_alias_lookup_resolves_to_canonical_company(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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
            "role": "ML Engineer",
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
                "role": "AI Engineer",
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
async def test_hotword_list_returns_canonical_name_and_excludes_aliases(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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
            "role": "ML Engineer",
            "employment_types_json": ["Internship"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add crew trim labs for an ML Engineer role.",
            "original_extracted_company_name": "crew   trim   labs",
        },
    )

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()["hotwords"]
    assert "Krutrim Labs" in body
    assert "Crew Trim Labs" not in body
    assert "crew   trim   labs" not in body


@pytest.mark.anyio
async def test_hotword_list_retains_canonical_company_after_last_application_is_deleted(client):
    created = await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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

    # Archive the application (soft delete) — canonical company still exists
    archived = await client.post(f"/applications/{created['id']}/archive")
    assert archived.status_code == 200

    hotwords = await client.get("/asr/hotwords")
    assert hotwords.status_code == 200
    body = hotwords.json()["hotwords"]
    assert "Krutrim Labs" in body
    assert "Crew Trim Labs" not in body


@pytest.mark.anyio
async def test_hotword_list_ignores_blank_values_and_preserves_static_vocabulary(client):
    await confirm_candidate(
        client,
        {
            "company": "Whitespace Co",
            "confirmed_company_name": "Whitespace Co",
            "role": "AI Engineer",
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
async def test_alias_hygiene_ignores_blank_normalized_identical_and_duplicate_aliases(client):
    await confirm_candidate(
        client,
        {
            "company": "Krutrim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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
            "original_extracted_company_name": "   ",
        },
    )

    await confirm_candidate(
        client,
        {
            "company": "Krutrim Labs",
            "confirmed_company_name": "Krutrim Labs!",
            "role": "ML Engineer",
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Krutrim Labs for an ML Engineer role.",
            "original_extracted_company_name": "Krutrim Labs",
        },
    )

    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "Platform Engineer",
            "employment_types_json": ["Full Time"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add Crew Trim Labs for a Platform Engineer role.",
            "original_extracted_company_name": "Crew Trim Labs",
        },
    )

    await confirm_candidate(
        client,
        {
            "company": "crew   trim   labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "Data Science",
            "employment_types_json": ["Internship"],
            "job_link": "",
            "location": "",
            "status": "",
            "current_stages_json": [],
            "priority": "",
            "engaged_days": 0,
            "next_action": "",
            "comments": "",
            "raw_transcript": "Add crew trim labs for a Data Science role.",
            "original_extracted_company_name": "crew   trim   labs",
        },
    )

    with SessionLocal() as db:
        aliases = db.query(CompanyAlias).order_by(CompanyAlias.alias_text.asc()).all()
        assert [alias.alias_text for alias in aliases] == ["Crew Trim Labs"]
        correction_events = db.query(AsrCompanyCorrectionEvent).order_by(AsrCompanyCorrectionEvent.id.asc()).all()
        assert [event.alias_created for event in correction_events] == [False, False, True, False]


@pytest.mark.anyio
async def test_correction_event_is_persisted(client):
    await confirm_candidate(
        client,
        {
            "company": "Crew Trim Labs",
            "confirmed_company_name": "Krutrim Labs",
            "role": "AI Engineer",
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
    body = hotwords.json()["hotwords"]
    assert "Krutrim Labs" in body
    assert "Crew Trim Labs" not in body

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
