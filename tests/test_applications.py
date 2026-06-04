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
