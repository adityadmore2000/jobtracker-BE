"""
Tests for the controlled natural-language command parser.

Covers:
- Conversational prefix tolerance
- Every supported command family
- Unsupported transcript safety (no mutation, no LLM fallback)
- Draft flow (add → field setters → note → save)
- Saved-row flow (update application → field setters → apply)
- Note behavior (never overwrites structured fields)
- Clarification continuation
- Remove application
- LLM bypass assertion
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from types import SimpleNamespace

from app.fast_path_parser import (
    ClarificationNeeded,
    MutationPayload,
    ParseMiss,
    try_parse_v2,
)
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload as MP, MutationTarget
from app.database import SessionLocal
from app.main import app
from app.semantic_interpreter import SemanticInterpreterUnavailableError, get_semantic_interpreter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

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


class RaisingInterpreter:
    """Interpreter stub that raises if called — asserts deterministic path was taken."""
    settings = SimpleNamespace(max_tool_turns=2)

    def interpret(self, transcript, context=None):
        raise SemanticInterpreterUnavailableError("LLM must not be called for deterministic commands")

    def extract_fields(self, transcript, context=None):
        raise SemanticInterpreterUnavailableError("LLM must not be called for deterministic commands")

    def health_check(self):
        return {}


def override_raising():
    app.dependency_overrides[get_semantic_interpreter] = lambda: RaisingInterpreter()


def restore_interpreter():
    app.dependency_overrides.pop(get_semantic_interpreter, None)


# ─────────────────────────────────────────────────────────────────────────────
# Conversational prefix tolerance
# ─────────────────────────────────────────────────────────────────────────────

def test_please_add_application():
    result = try_parse_v2("please add application for AI Engineer at Neilsoft", {})
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_draft"
    assert result.changes.company == "Neilsoft"
    assert result.changes.role == "AI Engineer"


def test_do_me_a_favor_add():
    result = try_parse_v2("do me a favor, add application for AI Engineer role at Virtusa Software", {})
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_draft"
    assert result.changes.company == "Virtusa Software"
    assert result.changes.role == "AI Engineer"  # "role" suffix stripped


def test_can_you_set_priority():
    result = try_parse_v2("can you set priority as medium", {"draft_id": "10"})
    assert isinstance(result, MutationPayload)
    assert result.changes.priority == "MEDIUM"


def test_could_you_set_location():
    result = try_parse_v2("could you set location as onsite", {"draft_id": "5"})
    assert isinstance(result, MutationPayload)
    assert result.changes.location_mode == "on-site"


def test_okay_add_note():
    result = try_parse_v2(
        "okay, add a note saying recruiter replied",
        {"draft_id": "7"},
    )
    assert isinstance(result, MutationPayload)
    assert result.operation == "append_note"
    assert result.notes_to_append == ["recruiter replied"]


# ─────────────────────────────────────────────────────────────────────────────
# Unsupported transcript safety
# ─────────────────────────────────────────────────────────────────────────────

def test_unsupported_make_it_better():
    result = try_parse_v2("make it better", {"draft_id": "1"})
    assert isinstance(result, ParseMiss)


def test_unsupported_change_that_thing():
    result = try_parse_v2("change that thing", {"draft_id": "1"})
    assert isinstance(result, ParseMiss)


def test_unsupported_put_medium():
    result = try_parse_v2("put medium", {"draft_id": "1"})
    assert isinstance(result, ParseMiss)


def test_unsupported_bare_stage_words():
    result = try_parse_v2("networked tailored", {"draft_id": "1"})
    assert isinstance(result, ParseMiss)


@pytest.mark.anyio
async def test_unsupported_returns_safe_response(client):
    override_raising()
    try:
        resp = await client.post(
            "/transcript/parse",
            json={"transcript": "make it better", "context": {"draft_id": "999"}},
        )
    finally:
        restore_interpreter()
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unsupported"
    assert "supported command" in body["message"].lower()


@pytest.mark.anyio
async def test_unsupported_does_not_call_llm(client):
    """Verify that no mutation occurs and LLM is never reached for unmatched input."""
    override_raising()
    try:
        resp = await client.post(
            "/transcript/parse",
            json={"transcript": "change that thing", "context": {}},
        )
    finally:
        restore_interpreter()
    assert resp.status_code == 200
    assert resp.json()["status"] == "unsupported"


# ─────────────────────────────────────────────────────────────────────────────
# Command: add application
# ─────────────────────────────────────────────────────────────────────────────

def test_add_application_basic():
    result = try_parse_v2("add application for Software Engineer at Google", {})
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_draft"
    assert result.changes.company == "Google"
    assert result.changes.role == "Software Engineer"


def test_add_application_strips_trailing_role_word():
    result = try_parse_v2("add application for AI Engineer role at Neilsoft", {})
    assert isinstance(result, MutationPayload)
    assert result.changes.role == "AI Engineer"


def test_add_application_multiword_role():
    result = try_parse_v2("add application for LLM Inference Optimization Engineer at Aiden AI", {})
    assert isinstance(result, MutationPayload)
    assert result.changes.role == "LLM Inference Optimization Engineer"
    assert result.changes.company == "Aiden AI"


def test_add_application_multiword_company():
    result = try_parse_v2("please add application for Founding Engineer at Virtusa Software Solutions", {})
    assert isinstance(result, MutationPayload)
    assert result.changes.company == "Virtusa Software Solutions"


# ─────────────────────────────────────────────────────────────────────────────
# Command: set priority
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("set priority as low", "LOW"),
    ("set priority as medium", "MEDIUM"),
    ("set priority as high", "HIGH"),
    ("set priority as LOW", "LOW"),
    ("set priority as High", "HIGH"),
])
def test_set_priority_normalizes(raw, expected):
    result = try_parse_v2(raw, {"draft_id": "1"})
    assert isinstance(result, MutationPayload)
    assert result.changes.priority == expected


def test_set_priority_routes_to_create_application_update_draft():
    result = try_parse_v2("set priority as medium", {"active_application_id": 42})
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_application_update_draft"
    assert result.target.application_id == 42


def test_set_priority_invalid_value():
    result = try_parse_v2("set priority as extreme", {"draft_id": "1"})
    assert isinstance(result, ParseMiss)


# ─────────────────────────────────────────────────────────────────────────────
# Command: set location
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_loc", [
    ("set location as remote", "remote"),
    ("set location as hybrid", "hybrid"),
    ("set location as onsite", "on-site"),
    ("set location as on site", "on-site"),
    ("set location as on-site", "on-site"),
])
def test_set_location_normalizes(raw, expected_loc):
    result = try_parse_v2(raw, {"draft_id": "1"})
    assert isinstance(result, MutationPayload)
    assert result.changes.location_mode == expected_loc


# ─────────────────────────────────────────────────────────────────────────────
# Command: set employment type
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_et", [
    ("set employment type as fulltime", "Full Time"),
    ("set employment type as full time", "Full Time"),
    ("set employment type as full-time", "Full Time"),
    ("set employment type as internship", "Internship"),
    ("set employment type as part time", "Part Time"),
])
def test_set_employment_type_normalizes(raw, expected_et):
    result = try_parse_v2(raw, {"draft_id": "1"})
    assert isinstance(result, MutationPayload)
    assert result.changes.employment_types == [expected_et]


# ─────────────────────────────────────────────────────────────────────────────
# Command: set current stages
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_stages", [
    ("set current stages as tailored", ["Tailored"]),
    ("set current stage as tailored", ["Tailored"]),
    ("set current stages as tailored, networked", ["Tailored", "Networked"]),
    ("set current stages as tailored and engaged", ["Tailored", "Engaged"]),
    ("set current stages as tailored, networked, and engaged", ["Tailored", "Networked", "Engaged"]),
])
def test_set_current_stages_parses(raw, expected_stages):
    result = try_parse_v2(raw, {"draft_id": "1"})
    assert isinstance(result, MutationPayload)
    assert result.changes.current_stages == expected_stages


def test_set_current_stages_invalid_value():
    result = try_parse_v2("set current stages as foobar", {"draft_id": "1"})
    assert isinstance(result, ParseMiss)


# ─────────────────────────────────────────────────────────────────────────────
# Command: set role
# ─────────────────────────────────────────────────────────────────────────────

def test_set_role_basic():
    result = try_parse_v2("set role as AI Engineer", {"draft_id": "1"})
    assert isinstance(result, MutationPayload)
    assert result.changes.role == "AI Engineer"


def test_set_role_multiword():
    result = try_parse_v2("set role as LLM Inference Optimization Engineer", {"draft_id": "5"})
    assert isinstance(result, MutationPayload)
    assert result.changes.role == "LLM Inference Optimization Engineer"


# ─────────────────────────────────────────────────────────────────────────────
# Command: add note
# ─────────────────────────────────────────────────────────────────────────────

def test_add_note_with_draft_context():
    result = try_parse_v2("add a note saying recruiter replied", {"draft_id": "7"})
    assert isinstance(result, MutationPayload)
    assert result.operation == "append_note"
    assert result.target.draft_id == "7"
    assert result.notes_to_append == ["recruiter replied"]


def test_add_note_with_app_context():
    result = try_parse_v2("add a note saying recruiter replied", {"active_application_id": 42})
    assert isinstance(result, MutationPayload)
    assert result.operation == "append_note"
    assert result.target.application_id == 42
    assert result.notes_to_append == ["recruiter replied"]


def test_add_note_no_context_asks_clarification():
    result = try_parse_v2("add a note saying recruiter replied", {})
    assert isinstance(result, ClarificationNeeded)
    assert "application" in result.question.lower()


def test_add_note_for_company_single_match():
    apps = [{"id": 10, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None}]
    result = try_parse_v2(
        "add note for Neilsoft saying follow-up is pending",
        {"applications": apps},
    )
    assert isinstance(result, MutationPayload)
    assert result.operation == "append_note"
    assert result.target.application_id == 10
    assert result.notes_to_append == ["follow-up is pending"]


def test_add_note_for_company_multiple_roles_asks_clarification():
    apps = [
        {"id": 10, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None},
        {"id": 11, "company": "Neilsoft", "role": "Backend Engineer", "archived_at": None},
    ]
    result = try_parse_v2(
        "add note for Neilsoft saying follow-up pending",
        {"applications": apps},
    )
    assert isinstance(result, ClarificationNeeded)
    assert "Neilsoft" in result.question


def test_add_note_for_company_and_role():
    apps = [
        {"id": 10, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None},
        {"id": 11, "company": "Neilsoft", "role": "Backend Engineer", "archived_at": None},
    ]
    result = try_parse_v2(
        "add a note for Neilsoft for AI Engineer role saying recruiter replied",
        {"applications": apps},
    )
    assert isinstance(result, MutationPayload)
    assert result.target.application_id == 10
    assert result.notes_to_append == ["recruiter replied"]


# ─────────────────────────────────────────────────────────────────────────────
# Command: remove application
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_application_with_selected_app():
    result = try_parse_v2("remove application", {"active_application_id": 5})
    assert isinstance(result, MutationPayload)
    assert result.operation == "archive_application"
    assert result.target.application_id == 5


def test_remove_application_no_context_asks_clarification():
    result = try_parse_v2("remove application", {})
    assert isinstance(result, ClarificationNeeded)


def test_remove_application_for_company():
    apps = [{"id": 3, "company": "Google", "role": "SWE", "archived_at": None}]
    result = try_parse_v2("remove application for Google", {"applications": apps})
    assert isinstance(result, MutationPayload)
    assert result.operation == "archive_application"
    assert result.target.application_id == 3


def test_remove_application_ambiguous_company_asks_clarification():
    apps = [
        {"id": 3, "company": "Google", "role": "SWE", "archived_at": None},
        {"id": 4, "company": "Google", "role": "ML Engineer", "archived_at": None},
    ]
    result = try_parse_v2("remove application for Google", {"applications": apps})
    assert isinstance(result, ClarificationNeeded)


# ─────────────────────────────────────────────────────────────────────────────
# Command: update application
# ─────────────────────────────────────────────────────────────────────────────

def test_update_application_single_match():
    apps = [{"id": 7, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None}]
    result = try_parse_v2("update application for Neilsoft", {"applications": apps})
    assert isinstance(result, MutationPayload)
    assert result.operation == "set_active_application"
    assert result.target.application_id == 7


def test_update_application_multiple_roles_asks_clarification():
    apps = [
        {"id": 7, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None},
        {"id": 8, "company": "Neilsoft", "role": "Backend Engineer", "archived_at": None},
    ]
    result = try_parse_v2("update application for Neilsoft", {"applications": apps})
    assert isinstance(result, ClarificationNeeded)
    assert "Neilsoft" in result.question
    assert result.pending_command is not None
    assert result.pending_command["missing_field"] == "role"


def test_update_application_with_role_disambiguates():
    apps = [
        {"id": 7, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None},
        {"id": 8, "company": "Neilsoft", "role": "Backend Engineer", "archived_at": None},
    ]
    result = try_parse_v2(
        "update application for Neilsoft for AI Engineer role",
        {"applications": apps},
    )
    assert isinstance(result, MutationPayload)
    assert result.operation == "set_active_application"
    assert result.target.application_id == 7


# ─────────────────────────────────────────────────────────────────────────────
# Save / Discard lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def test_save_draft():
    result = try_parse_v2("save it", {"draft_id": "42"})
    assert isinstance(result, MutationPayload)
    assert result.operation == "save_draft"


def test_discard_draft():
    result = try_parse_v2("discard draft", {"draft_id": "42"})
    assert isinstance(result, MutationPayload)
    assert result.operation == "discard_draft"


# ─────────────────────────────────────────────────────────────────────────────
# Full draft flow (integration)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_draft_flow_end_to_end(client, db):
    override_raising()
    try:
        # 1. create draft
        resp = await client.post("/transcript/parse", json={
            "transcript": "please add application for AI Engineer role at Virtusa Software",
            "context": {},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "draft_created"
        assert body["draft"]["company"] == "Virtusa Software"
        assert body["draft"]["role"] == "AI Engineer"
        draft_id = body["draft_id"]

        ctx = {"draft_id": draft_id}

        # 2. set location
        resp = await client.post("/transcript/parse", json={
            "transcript": "set location as onsite",
            "context": ctx,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "draft_updated"
        assert resp.json()["draft"]["location"] == "on-site"

        # 3. set employment type
        resp = await client.post("/transcript/parse", json={
            "transcript": "set employment type as fulltime",
            "context": ctx,
        })
        assert resp.status_code == 200
        assert resp.json()["draft"]["employment_types"] == ["Full Time"]

        # 4. set priority
        resp = await client.post("/transcript/parse", json={
            "transcript": "set priority as high",
            "context": ctx,
        })
        assert resp.status_code == 200
        assert resp.json()["draft"]["priority"] == "HIGH"

        # 5. set current stages
        resp = await client.post("/transcript/parse", json={
            "transcript": "set current stages as tailored, networked",
            "context": ctx,
        })
        assert resp.status_code == 200
        draft_stages = resp.json()["draft"]["current_stages"]
        assert "Tailored" in draft_stages
        assert "Networked" in draft_stages

        # 6. add note
        resp = await client.post("/transcript/parse", json={
            "transcript": "add a note saying recruiter replied",
            "context": ctx,
        })
        assert resp.status_code == 200
        note_body = resp.json()
        assert note_body["status"] == "note_added"
        assert note_body["note"]["text"] == "recruiter replied"

        # 7. save
        resp = await client.post("/transcript/parse", json={
            "transcript": "save it",
            "context": ctx,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"

        # Verify saved application has expected fields
        import json as _json
        app_id = resp.json()["application_id"]
        assert app_id is not None
    finally:
        restore_interpreter()


# ─────────────────────────────────────────────────────────────────────────────
# Notes do not overwrite structured fields
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_note_does_not_corrupt_draft_role(client, db):
    # Create a draft first
    create_result = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Neilsoft", role="AI Engineer"),
        ),
        db,
    )
    assert create_result.success
    draft_id = str(create_result.draft["id"])

    override_raising()
    try:
        resp = await client.post("/transcript/parse", json={
            "transcript": "add a note saying I connected with a previous employer",
            "context": {"draft_id": draft_id},
        })
    finally:
        restore_interpreter()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "note_added"
    # Role must be unchanged
    assert body["note"]["text"] == "I connected with a previous employer"

    # Verify the draft role is still AI Engineer
    draft_resp = await client.get(f"/drafts/{draft_id}")
    if draft_resp.status_code == 200:
        draft_data = draft_resp.json()
        assert draft_data["role"] == "AI Engineer"
        assert "employer" not in draft_data.get("role", "").lower()


@pytest.mark.anyio
async def test_note_does_not_create_application_change_draft(client, db):
    """Notes on saved apps must not produce ApplicationChangeDraft rows."""
    # Create and save an application
    create_result = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="NoteTestCo", role="QA Engineer"),
        ),
        db,
    )
    assert create_result.success
    draft_id = str(create_result.draft["id"])
    save_result = dispatch(
        MP(
            operation="save_draft",
            target=MutationTarget(draft_id=draft_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert save_result.success
    app_id = save_result.application["id"]

    override_raising()
    try:
        resp = await client.post("/transcript/parse", json={
            "transcript": "add a note saying follow-up pending",
            "context": {"active_application_id": app_id},
        })
    finally:
        restore_interpreter()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "note_added"
    assert body["pending_changes"] is None  # no change draft created


# ─────────────────────────────────────────────────────────────────────────────
# Saved-row flow (integration)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_saved_row_update_flow(client, db):
    # Create and save an application
    create_result = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Neilsoft", role="AI Engineer"),
        ),
        db,
    )
    draft_id = str(create_result.draft["id"])
    save_result = dispatch(
        MP(
            operation="save_draft",
            target=MutationTarget(draft_id=draft_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    app_id = save_result.application["id"]

    apps_list = [{"id": app_id, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None}]

    override_raising()
    try:
        # 1. update application (sets active context)
        resp = await client.post("/transcript/parse", json={
            "transcript": "update application for Neilsoft for AI Engineer role",
            "context": {"applications": apps_list},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "context_updated"
        assert body["application_id"] == app_id

        ctx = {"active_application_id": app_id}

        # 2. set priority → creates pending changes
        resp = await client.post("/transcript/parse", json={
            "transcript": "set priority as medium",
            "context": ctx,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "pending_changes_created"
        change_draft_id = body["pending_changes"]["id"]

        ctx_with_cd = dict(ctx)
        ctx_with_cd["change_draft_id"] = change_draft_id

        # 3. set stages → updates pending changes
        resp = await client.post("/transcript/parse", json={
            "transcript": "set current stages as networked, engaged",
            "context": ctx,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] in ("pending_changes_created", "pending_changes_updated")

        # 4. apply changes
        resp = await client.post("/transcript/parse", json={
            "transcript": "apply changes",
            "context": ctx_with_cd,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "changes_applied"
    finally:
        restore_interpreter()


# ─────────────────────────────────────────────────────────────────────────────
# Clarification continuation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_update_application_ambiguous_then_clarify(client, db):
    """When update application for {company} has multiple roles, ask for clarification."""
    create1 = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Neilsoft", role="AI Engineer"),
        ),
        db,
    )
    dispatch(
        MP(
            operation="save_draft",
            target=MutationTarget(draft_id=str(create1.draft["id"])),
            changes=ApplicationChanges(),
        ),
        db,
    )
    create2 = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Neilsoft", role="Backend Engineer"),
        ),
        db,
    )
    dispatch(
        MP(
            operation="save_draft",
            target=MutationTarget(draft_id=str(create2.draft["id"])),
            changes=ApplicationChanges(),
        ),
        db,
    )

    apps_list = [
        {"id": create1.draft["id"], "company": "Neilsoft", "role": "AI Engineer", "archived_at": None},
        {"id": create2.draft["id"], "company": "Neilsoft", "role": "Backend Engineer", "archived_at": None},
    ]

    override_raising()
    try:
        resp = await client.post("/transcript/parse", json={
            "transcript": "update application for Neilsoft",
            "context": {"applications": apps_list},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "clarification"
        assert "Neilsoft" in body["clarification_question"]
        assert body["pending_command"] is not None
        assert body["pending_command"]["missing_field"] == "role"
    finally:
        restore_interpreter()


# ─────────────────────────────────────────────────────────────────────────────
# LLM bypass: all supported deterministic commands skip the LLM
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_all_deterministic_commands_bypass_llm(client, db):
    """Verify deterministic commands never reach the LLM interpreter."""
    create_result = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="BypassTestCo", role="Engineer"),
        ),
        db,
    )
    draft_id = str(create_result.draft["id"])

    commands = [
        ("set priority as high", {"draft_id": draft_id}),
        ("set location as remote", {"draft_id": draft_id}),
        ("set employment type as fulltime", {"draft_id": draft_id}),
        ("set current stages as tailored", {"draft_id": draft_id}),
        ("set role as Senior Engineer", {"draft_id": draft_id}),
        ("add a note saying hello", {"draft_id": draft_id}),
        ("save it", {"draft_id": draft_id}),
    ]

    override_raising()
    try:
        for transcript, ctx in commands[:-1]:  # all except save (need draft to exist)
            resp = await client.post("/transcript/parse", json={
                "transcript": transcript,
                "context": ctx,
            })
            assert resp.status_code == 200, f"Failed on: {transcript!r}"
            assert resp.json()["status"] not in ("error", "unsupported"), (
                f"Unexpected status for {transcript!r}: {resp.json()['status']}"
            )
        # Now save
        resp = await client.post("/transcript/parse", json={
            "transcript": "save it",
            "context": {"draft_id": draft_id},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
    finally:
        restore_interpreter()


# ─────────────────────────────────────────────────────────────────────────────
# Regression: ensure existing tests still work
# ─────────────────────────────────────────────────────────────────────────────

def test_save_it_still_works():
    result = try_parse_v2("save it", {"draft_id": "42"})
    assert isinstance(result, MutationPayload)
    assert result.operation == "save_draft"


def test_discard_it_still_works():
    result = try_parse_v2("discard it", {"draft_id": "42"})
    assert isinstance(result, MutationPayload)
    assert result.operation == "discard_draft"


def test_priority_high_shorthand_still_works():
    result = try_parse_v2("priority high", {"draft_id": "10"})
    assert isinstance(result, MutationPayload)
    assert result.changes.priority == "HIGH"


def test_onsite_shorthand_still_works():
    result = try_parse_v2("onsite", {"draft_id": "1"})
    assert isinstance(result, MutationPayload)
    assert result.changes.location_mode == "on-site"


# ─────────────────────────────────────────────────────────────────────────────
# Explicit-target setter: set {field} {of|for} {company} [for {role} role] to {value}
# ─────────────────────────────────────────────────────────────────────────────

def _apps_neilsoft_single():
    return [{"id": 20, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None}]


def _apps_neilsoft_two_roles():
    return [
        {"id": 20, "company": "Neilsoft", "role": "AI Engineer", "archived_at": None},
        {"id": 21, "company": "Neilsoft", "role": "Backend Engineer", "archived_at": None},
    ]


# ── Parser unit tests ─────────────────────────────────────────────────────────

def test_explicit_setter_priority_of_company():
    result = try_parse_v2(
        "set priority of neilsoft to medium",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_application_update_draft"
    assert result.target.application_id == 20
    assert result.changes.priority == "MEDIUM"


def test_explicit_setter_priority_for_company():
    result = try_parse_v2(
        "set priority for Neilsoft to high",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_application_update_draft"
    assert result.target.application_id == 20
    assert result.changes.priority == "HIGH"


def test_explicit_setter_location_of_company():
    result = try_parse_v2(
        "set location of Neilsoft to onsite",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, MutationPayload)
    assert result.changes.location_mode == "on-site"


def test_explicit_setter_employment_type_for_company():
    result = try_parse_v2(
        "set employment type for Neilsoft to full-time",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, MutationPayload)
    assert result.changes.employment_types == ["Full Time"]


def test_explicit_setter_stages_of_company():
    result = try_parse_v2(
        "set current stages of Neilsoft to tailored, networked",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, MutationPayload)
    assert result.changes.current_stages == ["Tailored", "Networked"]


def test_explicit_setter_stages_of_company_with_role():
    result = try_parse_v2(
        "set current stages of Neilsoft for AI Engineer role to tailored, networked",
        {"applications": _apps_neilsoft_two_roles()},
    )
    assert isinstance(result, MutationPayload)
    assert result.target.application_id == 20
    assert result.changes.current_stages == ["Tailored", "Networked"]


def test_explicit_setter_status_of_company_with_role():
    apps = _apps_neilsoft_two_roles()
    result = try_parse_v2(
        "set status of Neilsoft for AI Engineer role to applied",
        {"applications": apps},
    )
    assert isinstance(result, MutationPayload)
    assert result.target.application_id == 20
    assert result.changes.status == "applied"


def test_explicit_setter_role_for_company():
    result = try_parse_v2(
        "set role for Neilsoft to Senior Engineer",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, MutationPayload)
    assert result.changes.role == "Senior Engineer"


# ── Ambiguity tests ───────────────────────────────────────────────────────────

def test_explicit_setter_ambiguous_company_asks_clarification():
    result = try_parse_v2(
        "set priority of Neilsoft to medium",
        {"applications": _apps_neilsoft_two_roles()},
    )
    assert isinstance(result, ClarificationNeeded)
    assert "Neilsoft" in result.question
    assert result.pending_command["missing_field"] == "role"


def test_explicit_setter_company_not_found_returns_clarification():
    result = try_parse_v2(
        "set priority of UnknownCorp to medium",
        {"applications": _apps_neilsoft_single()},
    )
    assert isinstance(result, ClarificationNeeded)
    assert "UnknownCorp" in result.question


def test_explicit_setter_no_applications_in_context_returns_clarification():
    result = try_parse_v2(
        "set priority of Neilsoft to medium",
        {},
    )
    assert isinstance(result, ClarificationNeeded)


# ── Safety: unsupported fuzzy forms must not match ────────────────────────────

def test_explicit_setter_fuzzy_make_neilsoft_medium_is_unsupported():
    result = try_parse_v2("make neilsoft medium", {"applications": _apps_neilsoft_single()})
    assert isinstance(result, ParseMiss)


def test_explicit_setter_fuzzy_put_medium_for_neilsoft_is_unsupported():
    result = try_parse_v2("put medium for neilsoft", {"applications": _apps_neilsoft_single()})
    assert isinstance(result, ParseMiss)


# ── Regression: contextual setter still works with selected saved app ─────────

def test_contextual_setter_with_active_application_id_still_works():
    result = try_parse_v2("set priority as medium", {"active_application_id": 42})
    assert isinstance(result, MutationPayload)
    assert result.operation == "create_application_update_draft"
    assert result.target.application_id == 42
    assert result.changes.priority == "MEDIUM"


# ── Integration: explicit-target setter creates ApplicationChangeDraft ────────

@pytest.mark.anyio
async def test_explicit_setter_creates_application_change_draft(client, db):
    """set priority of {company} to {value} → creates ApplicationChangeDraft, no LLM."""
    create_result = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="ExplicitSetterCo", role="AI Engineer"),
        ),
        db,
    )
    assert create_result.success
    draft_id = str(create_result.draft["id"])
    save_result = dispatch(
        MP(
            operation="save_draft",
            target=MutationTarget(draft_id=draft_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert save_result.success
    app_id = save_result.application["id"]

    apps_list = [{"id": app_id, "company": "ExplicitSetterCo", "role": "AI Engineer", "archived_at": None}]

    override_raising()
    try:
        resp = await client.post("/transcript/parse", json={
            "transcript": "set priority of ExplicitSetterCo to high",
            "context": {"applications": apps_list},
        })
    finally:
        restore_interpreter()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("pending_changes_created", "pending_changes_updated"), body
    assert body["pending_changes"] is not None
    assert body["pending_changes"]["target_application_id"] == app_id
    # Verify the saved row is unchanged before Apply
    from app.models import JobApplication
    with SessionLocal() as check_db:
        row = check_db.get(JobApplication, app_id)
        assert row.priority != "HIGH"


@pytest.mark.anyio
async def test_explicit_setter_with_role_disambiguates(client, db):
    """set current stages of {company} for {role} role to {stages} → correct target."""
    create1 = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="MultiRoleCo", role="AI Engineer"),
        ),
        db,
    )
    dispatch(
        MP(operation="save_draft", target=MutationTarget(draft_id=str(create1.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    create2 = dispatch(
        MP(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="MultiRoleCo", role="Backend Engineer"),
        ),
        db,
    )
    dispatch(
        MP(operation="save_draft", target=MutationTarget(draft_id=str(create2.draft["id"])), changes=ApplicationChanges()),
        db,
    )
    app1_id = create1.draft["id"]

    apps_list = [
        {"id": app1_id, "company": "MultiRoleCo", "role": "AI Engineer", "archived_at": None},
        {"id": create2.draft["id"], "company": "MultiRoleCo", "role": "Backend Engineer", "archived_at": None},
    ]

    override_raising()
    try:
        resp = await client.post("/transcript/parse", json={
            "transcript": "set current stages of MultiRoleCo for AI Engineer role to tailored, networked",
            "context": {"applications": apps_list},
        })
    finally:
        restore_interpreter()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("pending_changes_created", "pending_changes_updated"), body
    assert body["pending_changes"]["target_application_id"] == app1_id


@pytest.mark.anyio
async def test_explicit_setter_ambiguous_company_returns_clarification_via_http(client, db):
    """set priority of {company} to {value} with multiple roles → clarification."""
    create1 = dispatch(
        MP(operation="create_draft", target=MutationTarget(), changes=ApplicationChanges(company="AmbigCo", role="AI Engineer")), db,
    )
    dispatch(MP(operation="save_draft", target=MutationTarget(draft_id=str(create1.draft["id"])), changes=ApplicationChanges()), db)
    create2 = dispatch(
        MP(operation="create_draft", target=MutationTarget(), changes=ApplicationChanges(company="AmbigCo", role="Backend Engineer")), db,
    )
    dispatch(MP(operation="save_draft", target=MutationTarget(draft_id=str(create2.draft["id"])), changes=ApplicationChanges()), db)

    apps_list = [
        {"id": create1.draft["id"], "company": "AmbigCo", "role": "AI Engineer", "archived_at": None},
        {"id": create2.draft["id"], "company": "AmbigCo", "role": "Backend Engineer", "archived_at": None},
    ]

    override_raising()
    try:
        resp = await client.post("/transcript/parse", json={
            "transcript": "set priority of AmbigCo to medium",
            "context": {"applications": apps_list},
        })
    finally:
        restore_interpreter()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "clarification"
    assert "AmbigCo" in body["clarification_question"]
    assert body["pending_command"]["missing_field"] == "role"
