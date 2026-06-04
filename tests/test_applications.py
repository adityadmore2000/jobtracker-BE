import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///./test_job_tracker.db"
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import Base, engine
from app.main import app


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


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


async def create_record(client, payload=None):
    response = await client.post("/applications", json=payload or REALISTIC_RECORD)
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


async def parse_transcript(client, transcript, path="/transcript/parse"):
    response = await client.post(path, json={"transcript": transcript})
    assert response.status_code == 200
    return response.json()


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
async def test_parse_transcript_basic_add(client):
    parsed = await parse_transcript(
        client,
        """
        Add a Bootcoding AI Engineer internship.
        Use the current link.
        It is onsite.
        Set priority to medium.
        Add Tailored and Applied stages.
        """,
    )

    patch = parsed["patch"]
    assert parsed["intent"] == "ADD_APPLICATION"
    assert patch["company"] == "Bootcoding"
    assert patch["roles_add"] == ["AI Engineer"]
    assert patch["employment_types_add"] == ["Internship"]
    assert patch["use_latest_browser_url"] is True
    assert patch["location"] == "onsite"
    assert patch["priority"] == "MEDIUM"
    assert patch["current_stages_add"] == ["Tailored", "Applied"]
    assert patch["next_action"] is None
    assert patch["comments_append"] is None


@pytest.mark.anyio
async def test_parse_transcript_arbitrary_order(client):
    parsed = await parse_transcript(
        client,
        "Set priority to high. Remote role. Company Gruve. Full time. Add LLM Engineer role.",
    )

    patch = parsed["patch"]
    assert patch["company"] == "Gruve"
    assert patch["priority"] == "HIGH"
    assert patch["location"] == "remote"
    assert patch["employment_types_add"] == ["Full Time"]
    assert patch["roles_add"] == ["LLM Engineer"]


@pytest.mark.anyio
async def test_parse_transcript_explicit_comment_only(client):
    parsed = await parse_transcript(client, "Update Bootcoding. Add a comment saying one LinkedIn request is pending.")

    patch = parsed["patch"]
    assert patch["comments_append"] == "one LinkedIn request is pending"
    assert patch["current_stages_add"] == []
    assert patch["next_action"] is None


@pytest.mark.anyio
async def test_parse_transcript_explicit_next_action(client):
    parsed = await parse_transcript(client, "Next action check request status in two days.")

    assert parsed["patch"]["next_action"] == "check request status in two days"


@pytest.mark.anyio
async def test_parse_transcript_future_action_phrases(client):
    should_action = await parse_transcript(client, "I should continue engaging before reaching out.")
    need_action = await parse_transcript(client, "I need to check the recruiter response tomorrow.")
    next_step_action = await parse_transcript(client, "My next step is to send a follow-up.")

    assert should_action["patch"]["next_action"] == "continue engaging before reaching out"
    assert need_action["patch"]["next_action"] == "check the recruiter response tomorrow"
    assert next_step_action["patch"]["next_action"] == "send a follow-up"


@pytest.mark.anyio
async def test_parse_transcript_does_not_infer_next_action(client):
    parsed = await parse_transcript(client, "Add a comment saying one request is pending.")

    assert parsed["patch"]["next_action"] is None


@pytest.mark.anyio
async def test_parse_transcript_does_not_infer_current_stage_from_comment(client):
    parsed = await parse_transcript(client, "Add a comment saying one LinkedIn request is pending.")

    assert parsed["patch"]["current_stages_add"] == []


@pytest.mark.anyio
async def test_parse_transcript_ignores_geographic_places_and_unmatched_narrative(client):
    parsed = await parse_transcript(
        client,
        "Add an Analytics Vidhya Generative AI Engineer internship. The role is onsite in Haryana near Pune and Bengaluru.",
    )

    patch = parsed["patch"]
    assert patch["company"] == "Analytics Vidhya"
    assert patch["location"] == "onsite"
    assert patch["comments_append"] is None
    assert patch["comments_replace"] is None
    assert patch["next_action"] is None
    assert patch["job_link"] is None
    assert patch["priority"] is None
    assert patch["engaged_days"] is None


@pytest.mark.anyio
async def test_parse_transcript_explicit_stage_addition(client):
    parsed = await parse_transcript(client, "Add Networked stage.")

    assert parsed["patch"]["current_stages_add"] == ["Networked"]


@pytest.mark.anyio
async def test_parse_transcript_explicit_stage_removal(client):
    parsed = await parse_transcript(client, "Remove Networked stage.", "/transcript/parse-correction")

    assert parsed["intent"] == "PATCH_ACTIVE_DRAFT"
    assert parsed["patch"]["current_stages_remove"] == ["Networked"]


@pytest.mark.anyio
async def test_parse_transcript_status_independence_for_applied_stage(client):
    parsed = await parse_transcript(client, "Add Applied stage.")

    assert parsed["patch"]["current_stages_add"] == ["Applied"]
    assert parsed["patch"]["status"] is None


@pytest.mark.anyio
async def test_parse_transcript_status_only_when_explicit(client):
    parsed = await parse_transcript(client, "Set status to Applied.")

    assert parsed["patch"]["status"] == "Applied"
    assert parsed["patch"]["current_stages_add"] == []


@pytest.mark.anyio
async def test_parse_transcript_reported_analytics_vidhya_example(client):
    parsed = await parse_transcript(
        client,
        """
        Add an Analytics Vidhya Generative AI Engineer internship.
        It is onsite in Haryana.
        Status is Applied.
        Current stages are Tailored and Engaged.
        I should Continue engaging for a few more days before reaching out to relevant employees for referrals.
        """,
    )

    patch = parsed["patch"]
    assert parsed["intent"] == "ADD_APPLICATION"
    assert patch["company"] == "Analytics Vidhya"
    assert patch["roles_add"] == ["Generative AI Engineer"]
    assert patch["employment_types_add"] == ["Internship"]
    assert patch["location"] == "onsite"
    assert patch["status"] == "Applied"
    assert patch["current_stages_add"] == ["Tailored", "Engaged"]
    assert patch["next_action"] == "Continue engaging for a few more days before reaching out to relevant employees for referrals"
    assert patch["comments_append"] is None
    assert patch["priority"] is None
    assert patch["engaged_days"] is None
    assert patch["job_link"] is None
    assert patch["use_latest_browser_url"] is False


@pytest.mark.anyio
async def test_parse_narrative_patch_does_not_return_unsupported_warning(client):
    parsed = await parse_transcript(
        client,
        """
        Applied for the Generative AI Engineer internship at Analytics Vidhya.
        The role is on-site.
        I have already tailored my application and started engaging with their posts.
        I should continue engaging for a few more days.
        """,
    )

    patch = parsed["patch"]
    assert parsed["intent"] == "ADD_APPLICATION"
    assert "No supported command was detected." not in parsed["warnings"]
    assert patch["company"] == "Analytics Vidhya"
    assert patch["roles_add"] == ["Generative AI Engineer"]
    assert patch["employment_types_add"] == ["Internship"]
    assert patch["location"] == "onsite"
    assert patch["status"] == "Applied"
    assert patch["current_stages_add"] == ["Tailored", "Engaged"]
    assert patch["next_action"] is not None


@pytest.mark.anyio
async def test_parse_partial_narrative_patch_does_not_return_unsupported_warning(client):
    parsed = await parse_transcript(client, "Applied for an internship at Analytics Vidhya.")

    patch = parsed["patch"]
    assert parsed["intent"] == "ADD_APPLICATION"
    assert patch["company"] == "Analytics Vidhya"
    assert patch["employment_types_add"] == ["Internship"]
    assert patch["status"] == "Applied"
    assert "No supported command was detected." not in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_truly_unsupported_input_returns_unknown_warning(client):
    parsed = await parse_transcript(client, "This looks interesting.")
    patch = parsed["patch"]

    assert parsed["intent"] == "UNKNOWN"
    assert patch["company"] is None
    assert patch["roles_add"] == []
    assert patch["roles_remove"] == []
    assert patch["employment_types_add"] == []
    assert patch["employment_types_remove"] == []
    assert patch["job_link"] is None
    assert patch["use_latest_browser_url"] is False
    assert patch["location"] is None
    assert patch["status"] is None
    assert patch["current_stages_add"] == []
    assert patch["current_stages_remove"] == []
    assert patch["priority"] is None
    assert patch["engaged_days"] is None
    assert patch["next_action"] is None
    assert patch["comments_append"] is None
    assert patch["comments_replace"] is None
    assert "No supported command was detected." in parsed["warnings"]


@pytest.mark.anyio
async def test_parse_transcript_engaged_days_explicit_only(client):
    explicit = await parse_transcript(client, "Engaged days 3.")
    implicit = await parse_transcript(client, "Applied three days ago.")

    assert explicit["patch"]["engaged_days"] == 3
    assert implicit["patch"]["engaged_days"] is None


@pytest.mark.anyio
async def test_parse_correction_returns_patch_only_values(client):
    parsed = await parse_transcript(
        client,
        "Remove Agentic AI Engineer tag. Add Networked. Add a comment saying one request is pending. Use current link.",
        "/transcript/parse-correction",
    )

    patch = parsed["patch"]
    assert parsed["intent"] == "PATCH_ACTIVE_DRAFT"
    assert patch["roles_remove"] == ["Agentic AI Engineer"]
    assert patch["roles_add"] == []
    assert patch["current_stages_add"] == ["Networked"]
    assert patch["comments_append"] == "one request is pending"
    assert patch["use_latest_browser_url"] is True


@pytest.mark.anyio
async def test_parse_transcript_endpoints_do_not_persist_applications(client):
    await parse_transcript(client, "Add a Bootcoding AI Engineer internship.")
    response = await client.get("/applications")

    assert response.status_code == 200
    assert response.json() == []


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
async def test_missing_application_returns_404(client):
    response = await client.get("/applications/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Application not found"}
