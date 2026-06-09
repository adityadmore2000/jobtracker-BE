import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget


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


def _create_draft(db, company: str = "PatchCo", roles: list[str] | None = None) -> int:
    payload = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company=company, roles=roles or ["AI Engineer"]),
    )
    result = dispatch(payload, db)
    assert result.success
    return result.draft["id"]


@pytest.mark.anyio
async def test_patch_draft_returns_public_dto(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"priority": "HIGH"})
    assert response.status_code == 200
    data = response.json()
    assert data["priority"] == "HIGH"
    assert data["is_draft"] is True
    assert data["id"] == draft_id


@pytest.mark.anyio
async def test_patch_draft_persists_to_db(client, db):
    from app.models import JobApplication
    draft_id = _create_draft(db, company="Neilsoft")
    response = await client.patch(f"/drafts/{draft_id}", json={"priority": "MEDIUM", "location": "remote"})
    assert response.status_code == 200
    row = db.get(JobApplication, draft_id)
    db.refresh(row)
    assert row.priority == "MEDIUM"
    assert row.location == "remote"
    assert row.is_draft is True


@pytest.mark.anyio
async def test_patch_draft_multi_role_preserved(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"roles": ["AI Engineer", "RAG Engineer"]})
    assert response.status_code == 200
    data = response.json()
    assert data["roles"] == ["AI Engineer", "RAG Engineer"]


@pytest.mark.anyio
async def test_patch_draft_unknown_role_accepted(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"roles": ["LLM Inference Optimization Engineer"]})
    assert response.status_code == 200
    assert response.json()["roles"] == ["LLM Inference Optimization Engineer"]


@pytest.mark.anyio
async def test_patch_draft_invalid_status_rejected(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"status": "nonexistent_status"})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_patch_draft_open_fields(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(
        f"/drafts/{draft_id}",
        json={"next_action": "Follow up Monday", "comments": "Referral pending", "engaged_days": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["next_action"] == "Follow up Monday"
    assert data["comments"] == "Referral pending"
    assert data["engaged_days"] == 3


@pytest.mark.anyio
async def test_patch_missing_draft_returns_404(client, db):
    response = await client.patch("/drafts/999999", json={"priority": "HIGH"})
    assert response.status_code == 404


@pytest.mark.anyio
async def test_patch_draft_endpoint_rejects_saved_application(client, db):
    # Create a draft then save it — patching via /drafts/{id} should reject
    from app.models import JobApplication
    draft_id = _create_draft(db, company="SavedCo")
    save_payload = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=str(draft_id)),
        changes=ApplicationChanges(),
    )
    dispatch(save_payload, db)
    row = db.get(JobApplication, draft_id)
    db.refresh(row)
    assert row.is_draft is False

    response = await client.patch(f"/drafts/{draft_id}", json={"priority": "HIGH"})
    assert response.status_code == 404


@pytest.mark.anyio
async def test_save_draft_via_endpoint_promotes_to_application(client, db):
    draft_id = _create_draft(db, company="SaveEndpointCo")
    response = await client.post(f"/drafts/{draft_id}/save")
    assert response.status_code == 200
    data = response.json()
    assert data["is_draft"] is False
    assert data["company"] == "SaveEndpointCo"

    # Verify visible in applications list
    list_response = await client.get("/applications")
    ids = [r["id"] for r in list_response.json()]
    assert draft_id in ids


@pytest.mark.anyio
async def test_discard_draft_via_endpoint_deletes_row(client, db):
    from app.models import JobApplication
    draft_id = _create_draft(db, company="DiscardEndpointCo")
    response = await client.post(f"/drafts/{draft_id}/discard")
    assert response.status_code == 204

    row = db.get(JobApplication, draft_id)
    assert row is None


@pytest.mark.anyio
async def test_save_missing_draft_returns_404(client, db):
    response = await client.post("/drafts/999999/save")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_discard_missing_draft_returns_404(client, db):
    response = await client.post("/drafts/999999/discard")
    assert response.status_code == 404
