"""
Regression tests for conversational-prefix tolerance in the deterministic
create-draft parser.

Design: the parser uses .search() to find the command anchor anywhere in the
normalized transcript, so harmless prefixes before the anchor are ignored
without adding phrase-specific patterns.
"""

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.fast_path_parser import try_parse
from app.main import app
from app.models import Company, JobApplication
from app.role_resolution import normalize_role_name


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _cleanup(db, company_name: str, role: str) -> None:
    co = db.query(Company).filter_by(name=company_name).first()
    if co is None:
        return
    rows = (
        db.query(JobApplication)
        .filter(
            JobApplication.company_id == co.id,
            JobApplication.normalized_role == normalize_role_name(role),
        )
        .all()
    )
    for r in rows:
        db.delete(r)
    db.commit()


def _parse(client, transcript: str) -> dict:
    resp = client.post("/transcript/parse", json={"transcript": transcript})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Parser unit tests — no DB, no HTTP
# ---------------------------------------------------------------------------

class TestCreateDraftParserMatches:
    """Must match and produce create_draft with correct role/company."""

    def _assert_match(self, transcript: str, expected_role_lower: str, expected_company_lower: str):
        result = try_parse(transcript, {})
        assert result is not None, f"Expected match, got None for: {transcript!r}"
        assert result.operation == "create_draft"
        assert result.changes.role.lower() == expected_role_lower
        assert result.changes.company.lower() == expected_company_lower

    def test_bare_add(self):
        self._assert_match(
            "add application for AI Engineer role at virtusa software",
            "ai engineer role", "virtusa software",
        )

    def test_please_prefix(self):
        self._assert_match(
            "please add application for AI Engineer role at virtusa software",
            "ai engineer role", "virtusa software",
        )

    def test_do_me_a_favor_prefix(self):
        self._assert_match(
            "do me a favor, add application for AI Engineer role at virtusa software",
            "ai engineer role", "virtusa software",
        )

    def test_hey_can_you_prefix(self):
        self._assert_match(
            "hey, can you add application for AI Engineer role at virtusa software",
            "ai engineer role", "virtusa software",
        )

    def test_could_you_please_track(self):
        self._assert_match(
            "could you please track application for Founding Engineer at Aiden AI",
            "founding engineer", "aiden ai",
        )

    def test_please_create_anchor(self):
        self._assert_match(
            "please create application for LLM Engineer at Google",
            "llm engineer", "google",
        )


class TestCreateDraftParserNonMatches:
    """Must NOT match — these are ambiguous or lack the required anchor."""

    def _assert_no_match(self, transcript: str):
        result = try_parse(transcript, {})
        assert result is None or result.operation != "create_draft", (
            f"Expected no create_draft match, got {result} for: {transcript!r}"
        )

    def test_thinking_about_role(self):
        self._assert_no_match("I was thinking about an AI Engineer role at virtusa software")

    def test_interview_at(self):
        self._assert_no_match("interview at virtusa software")

    def test_follow_up(self):
        self._assert_no_match("follow up with virtusa software")

    def test_role_at_company_bare(self):
        self._assert_no_match("AI Engineer at virtusa software")

    def test_application_for_without_anchor(self):
        # "application for X at Y" lacks the add/create/track verb
        self._assert_no_match("application for AI Engineer at virtusa software")


# ---------------------------------------------------------------------------
# Integration tests — HTTP endpoint + DB + Ollama bypass
# ---------------------------------------------------------------------------

def test_do_me_a_favor_prefix_creates_persisted_draft(client, db):
    company, role = "Virtusa Software", "AI Engineer role"
    _cleanup(db, company, role)

    data = _parse(client, "do me a favor, add application for AI Engineer role at Virtusa Software")

    assert data["status"] == "draft_created", f"status={data['status']!r} message={data['message']!r}"
    assert data["draft"] is not None
    assert data["draft_id"] is not None
    assert data["draft"]["id"] != 0
    assert data["draft"]["id"] == int(data["draft_id"])

    row = db.get(JobApplication, data["draft"]["id"])
    assert row is not None
    assert row.is_draft is True

    _cleanup(db, company, role)


def test_please_prefix_creates_persisted_draft(client, db):
    company, role = "Virtusa Software", "AI Engineer role"
    _cleanup(db, company, role)

    data = _parse(client, "please add application for AI Engineer role at Virtusa Software")

    assert data["status"] == "draft_created"
    assert data["draft"]["id"] != 0
    assert data["draft"]["id"] == int(data["draft_id"])

    _cleanup(db, company, role)


def test_hey_can_you_create_prefix(client, db):
    company, role = "Aiden AI", "Founding Engineer"
    _cleanup(db, company, role)

    data = _parse(client, "hey, can you create application for Founding Engineer at Aiden AI")

    assert data["status"] == "draft_created"
    assert data["draft"]["id"] != 0
    assert data["draft"]["id"] == int(data["draft_id"])

    _cleanup(db, company, role)


def test_could_you_please_track_prefix(client, db):
    company, role = "Google", "LLM Engineer"
    _cleanup(db, company, role)

    data = _parse(client, "could you please track application for LLM Engineer at Google")

    assert data["status"] == "draft_created"
    assert data["draft"]["id"] != 0
    assert data["draft"]["id"] == int(data["draft_id"])

    _cleanup(db, company, role)


def test_prefixed_create_bypasses_ollama(client, db, monkeypatch):
    company, role = "Virtusa Software", "AI Engineer role"
    _cleanup(db, company, role)

    interpreter_calls = []
    from app import semantic_interpreter as si

    def mock_get():
        class MockInterpreter:
            def interpret(self, *a, **kw):
                interpreter_calls.append("interpret")
                raise AssertionError("Ollama must not be called for prefixed create commands")

            def extract_fields(self, *a, **kw):
                interpreter_calls.append("extract_fields")
                raise AssertionError("Ollama must not be called for prefixed create commands")

            @property
            def settings(self):
                return type("S", (), {"max_tool_turns": 2})()

        return MockInterpreter()

    monkeypatch.setattr(si, "get_semantic_interpreter", mock_get)

    data = _parse(client, "do me a favor, add application for AI Engineer role at Virtusa Software")

    assert interpreter_calls == [], f"Ollama was called: {interpreter_calls}"
    assert data["status"] == "draft_created"

    _cleanup(db, company, role)
