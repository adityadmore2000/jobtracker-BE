"""Tests for Phase 1.5A closure: one company + one normalized role = one mutable application."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.main import app
from app.models import Company, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.role_resolution import normalize_role_name, find_application_by_company_role


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 1. Role normalization unit tests
# ---------------------------------------------------------------------------


def test_normalize_role_trims_whitespace():
    assert normalize_role_name("  AI Engineer  ") == "ai engineer"


def test_normalize_role_collapses_internal_whitespace():
    assert normalize_role_name("AI   ENGINEER") == "ai engineer"


def test_normalize_role_casefolds():
    assert normalize_role_name("AI Engineer") == "ai engineer"


def test_normalize_role_all_variants_equal():
    assert normalize_role_name("AI Engineer") == normalize_role_name(" ai engineer ")
    assert normalize_role_name("AI Engineer") == normalize_role_name("AI   ENGINEER")


def test_normalize_role_preserves_hyphen():
    """Hyphens are not stripped — AI-ML Engineer != AI ML Engineer."""
    assert normalize_role_name("AI-ML Engineer") != normalize_role_name("AI ML Engineer")


def test_normalize_role_empty_string():
    assert normalize_role_name("") == ""
    assert normalize_role_name("   ") == ""


def test_normalize_role_unknown_open_ended_roles_accepted():
    roles = [
        "LLM Inference Optimization Engineer",
        "Conversational AI Systems Engineer",
        "Founding Applied AI Engineer",
    ]
    for role in roles:
        result = normalize_role_name(role)
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# 2. DB uniqueness constraint
# ---------------------------------------------------------------------------


def test_db_rejects_second_row_same_company_normalized_role(db):
    from app.company_resolution import get_or_create_company
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()

    app1 = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(app1)
    db.commit()

    app2 = JobApplication(
        company_id=company.id,
        role=" ai engineer ",
        normalized_role=normalize_role_name(" ai engineer "),
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(app2)
    with pytest.raises(IntegrityError):
        db.flush()


def test_db_allows_same_company_different_normalized_roles(db):
    from app.company_resolution import get_or_create_company
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()

    for role in ["AI Engineer", "Computer Vision Engineer"]:
        a = JobApplication(
            company_id=company.id,
            role=role,
            normalized_role=normalize_role_name(role),
            employment_types_json=[],
            job_link="",
            location="",
            status="",
            current_stages_json=[],
            priority="",
            engaged_days=0,
            next_action="",
            comments="",
            is_draft=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(a)
    db.commit()

    count = db.query(JobApplication).filter(JobApplication.company_id == company.id).count()
    assert count == 2


# ---------------------------------------------------------------------------
# 3. Dispatcher: create_draft uniqueness + reapply matrix
# ---------------------------------------------------------------------------


def _make_create_draft(company: str, role: str, status: str = "") -> MutationPayload:
    return MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company=company, role=role, status=status or None),
    )


def test_duplicate_create_draft_returns_existing_draft(db):
    payload = _make_create_draft("Neilsoft", "AI Engineer")
    r1 = dispatch(payload, db)
    assert r1.success
    assert r1.operation == "create_draft"
    draft_id = r1.draft["id"]

    r2 = dispatch(payload, db)
    assert r2.success
    assert r2.operation == "draft_updated"
    assert r2.draft["id"] == draft_id

    count = db.query(JobApplication).filter(JobApplication.company_id == r1.draft["company_id"]).count()
    assert count == 1


def test_create_draft_different_role_creates_new_row(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    r2 = dispatch(_make_create_draft("Neilsoft", "Computer Vision Engineer"), db)

    assert r1.success and r2.success
    assert r1.draft["id"] != r2.draft["id"]
    count = db.query(JobApplication).count()
    assert count == 2


def test_reapply_to_rejected_row_sets_applied(db):
    # Create and save an application, then set it to rejected.
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    draft_id = str(r1.draft["id"])
    save = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=draft_id), changes=ApplicationChanges()),
        db,
    )
    app_id = save.application["id"]

    patch = dispatch(
        MutationPayload(
            operation="patch_application",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(status="rejected"),
        ),
        db,
    )
    assert patch.application["status"] == "rejected"

    # Now reapply.
    r2 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    assert r2.success
    assert r2.application["id"] == app_id
    assert r2.application["status"] == "applied"

    count = db.query(JobApplication).count()
    assert count == 1


def test_reapply_to_applied_row_is_noop(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    save = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    patch = dispatch(
        MutationPayload(
            operation="patch_application",
            target=MutationTarget(application_id=save.application["id"]),
            changes=ApplicationChanges(status="applied"),
        ),
        db,
    )
    assert patch.application["status"] == "applied"

    r2 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    assert r2.success
    assert r2.operation == "no_change"
    assert r2.application["id"] == save.application["id"]

    count = db.query(JobApplication).count()
    assert count == 1


def test_reapply_to_archived_row_restores_and_sets_applied(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    save = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    app_id = save.application["id"]

    dispatch(
        MutationPayload(operation="archive_application", target=MutationTarget(application_id=app_id), changes=ApplicationChanges()),
        db,
    )
    row = db.get(JobApplication, app_id)
    assert row.archived_at is not None

    r2 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    assert r2.success
    assert r2.application["id"] == app_id
    assert r2.application["status"] == "applied"
    assert r2.application.get("archived_at") is None

    count = db.query(JobApplication).count()
    assert count == 1


def test_reapply_to_accepted_row_returns_clarification(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    save = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    app_id = save.application["id"]

    dispatch(
        MutationPayload(
            operation="patch_application",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(status="accepted"),
        ),
        db,
    )

    r2 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    assert r2.success
    assert r2.clarification_question is not None
    assert "accepted" in r2.clarification_question.lower()

    # Row is not mutated.
    row = db.get(JobApplication, app_id)
    assert row.status == "accepted"


# ---------------------------------------------------------------------------
# 4. Dispatcher: save_draft collision
# ---------------------------------------------------------------------------


def test_db_constraint_prevents_duplicate_draft_and_saved_row(db):
    """The DB UNIQUE constraint is the ultimate guard — no second row with same company+role
    can exist at all, regardless of is_draft state."""
    from datetime import datetime, timezone
    from sqlalchemy import text as sa_text

    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    save1 = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    company_id = save1.application["company_id"]

    now = datetime.now(timezone.utc)
    with pytest.raises(IntegrityError):
        db.execute(
            sa_text(
                "INSERT INTO job_applications "
                "(company_id, role, normalized_role, employment_types_json, job_link, location, "
                "status, current_stages_json, priority, engaged_days, next_action, comments, "
                "is_draft, draft_created_at, archived_at, created_at, updated_at) "
                "VALUES (:cid, 'AI Engineer', 'ai engineer', '[]'::json, '', '', '', '[]'::json, '', 0, '', '', "
                "true, :now, null, :now, :now)"
            ),
            {"cid": company_id, "now": now},
        )
        db.flush()


# ---------------------------------------------------------------------------
# 5. PATCH /applications/{id} — collision returns 409
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_patch_application_role_collision_returns_409(client):
    # Create two applications for the same company with different roles.
    r1 = await client.post("/applications", json={
        "company": "Rockwell",
        "role": "AI Engineer",
        "employment_types_json": [],
        "job_link": "",
        "location": "",
        "status": "",
        "current_stages_json": [],
        "priority": "",
        "engaged_days": 0,
        "next_action": "",
        "comments": "",
    })
    assert r1.status_code == 201

    r2 = await client.post("/applications", json={
        "company": "Rockwell",
        "role": "GET",
        "employment_types_json": [],
        "job_link": "",
        "location": "",
        "status": "",
        "current_stages_json": [],
        "priority": "",
        "engaged_days": 0,
        "next_action": "",
        "comments": "",
    })
    assert r2.status_code == 201
    get_id = r2.json()["id"]

    # Try to rename GET → AI Engineer: should 409.
    patch = await client.patch(f"/applications/{get_id}", json={"role": "AI Engineer"})
    assert patch.status_code == 409
    assert "AI Engineer" in patch.json()["detail"]


@pytest.mark.anyio
async def test_patch_application_role_no_collision_succeeds(client):
    r1 = await client.post("/applications", json={
        "company": "Rockwell",
        "role": "AI Engineer",
        "employment_types_json": [],
        "job_link": "",
        "location": "",
        "status": "",
        "current_stages_json": [],
        "priority": "",
        "engaged_days": 0,
        "next_action": "",
        "comments": "",
    })
    assert r1.status_code == 201
    app_id = r1.json()["id"]

    patch = await client.patch(f"/applications/{app_id}", json={"role": "Computer Vision Engineer"})
    assert patch.status_code == 200
    assert patch.json()["role"] == "Computer Vision Engineer"


@pytest.mark.anyio
async def test_patch_application_same_role_no_collision(client):
    """Patching to the same (normalized) role is never a collision."""
    r1 = await client.post("/applications", json={
        "company": "Rockwell",
        "role": "AI Engineer",
        "employment_types_json": [],
        "job_link": "",
        "location": "",
        "status": "",
        "current_stages_json": [],
        "priority": "",
        "engaged_days": 0,
        "next_action": "",
        "comments": "",
    })
    assert r1.status_code == 201
    app_id = r1.json()["id"]

    patch = await client.patch(f"/applications/{app_id}", json={"role": "ai engineer"})
    assert patch.status_code == 200


# ---------------------------------------------------------------------------
# 6. find_application_by_company_role helper
# ---------------------------------------------------------------------------


def test_find_application_by_company_role_returns_existing(db):
    from app.company_resolution import get_or_create_company
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()

    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(app_row)
    db.commit()

    found = find_application_by_company_role(db, company_id=company.id, role="AI Engineer")
    assert found is not None
    assert found.id == app_row.id


def test_find_application_by_company_role_case_insensitive(db):
    from app.company_resolution import get_or_create_company
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()

    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(app_row)
    db.commit()

    assert find_application_by_company_role(db, company_id=company.id, role=" ai engineer ") is not None
    assert find_application_by_company_role(db, company_id=company.id, role="AI   ENGINEER") is not None


def test_find_application_by_company_role_returns_none_for_different_role(db):
    from app.company_resolution import get_or_create_company
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()

    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(app_row)
    db.commit()

    assert find_application_by_company_role(db, company_id=company.id, role="ML Engineer") is None


# ---------------------------------------------------------------------------
# 7. Normalization: hyphen variants are NOT merged
# ---------------------------------------------------------------------------


def test_ai_ml_engineer_vs_ai_ml_engineer_are_distinct(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI-ML Engineer"), db)
    r2 = dispatch(_make_create_draft("Neilsoft", "AI ML Engineer"), db)
    assert r1.success and r2.success
    assert r1.draft["id"] != r2.draft["id"]
    assert db.query(JobApplication).count() == 2


# ---------------------------------------------------------------------------
# 8. Regression: existing workflows still work
# ---------------------------------------------------------------------------


def test_create_draft_save_discard_regression(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    assert r1.success

    save = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    assert save.success
    assert save.application["is_draft"] is False


def test_archive_and_restore_regression(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    save = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    app_id = save.application["id"]

    arc = dispatch(
        MutationPayload(operation="archive_application", target=MutationTarget(application_id=app_id), changes=ApplicationChanges()),
        db,
    )
    assert arc.success

    res = dispatch(
        MutationPayload(operation="restore_application", target=MutationTarget(application_id=app_id), changes=ApplicationChanges()),
        db,
    )
    assert res.success
    assert res.application["archived_at"] is None


def test_normalized_role_set_on_patch_draft(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    draft_id = str(r1.draft["id"])

    patch = dispatch(
        MutationPayload(
            operation="patch_draft",
            target=MutationTarget(draft_id=draft_id),
            changes=ApplicationChanges(role="Computer Vision Engineer"),
        ),
        db,
    )
    assert patch.success
    row = db.get(JobApplication, r1.draft["id"])
    assert row.normalized_role == normalize_role_name("Computer Vision Engineer")


def test_different_company_same_role_is_allowed(db):
    r1 = dispatch(_make_create_draft("Neilsoft", "AI Engineer"), db)
    r2 = dispatch(_make_create_draft("Rockwell", "AI Engineer"), db)
    assert r1.success and r2.success
    assert r1.draft["id"] != r2.draft["id"]
    assert db.query(JobApplication).count() == 2
