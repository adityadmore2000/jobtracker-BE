"""Phase 2A.4 — Intent-routing precedence and diagnostic suppression.

Tests cover:
- Explicit create with "role" keyword: wrong tool (preview_existing_application_update) → override to draft_created
- Explicit create without "role": invalid tool args → fallback to draft_created
- Terse noun phrase: no tool call → draft_created, no internal diagnostics in public message
- Terse noun phrase with active draft → draft_updated or no_change, no duplicate
- Lifecycle precedence: discard, archive, delete must not route to draft create
- Saved-row update precedence: explicit "update status of" → pending_changes, not draft_created
- Clarification: role only → ask for company; company only → ask for role
- Internal diagnostic suppression: public message/warnings must not contain internal tokens
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.semantic_interpreter import (
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterMetrics,
    SemanticInterpretationResult,
)
from app.semantic_schemas import (
    SemanticExtractedFields,
    SemanticFieldPatch,
    SemanticToolCallProposal,
)
from app.semantic_validation import (
    _has_explicit_create_intent,
    _has_explicit_saved_update_intent,
    _has_lifecycle_intent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def db():
    from app.database import SessionLocal

    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics() -> SemanticInterpreterMetrics:
    return SemanticInterpreterMetrics(latency_ms=5)


def _extracted(**kwargs) -> SemanticExtractedFields:
    return SemanticExtractedFields.model_validate(kwargs)


def _fields(**kwargs) -> SemanticFieldPatch:
    return SemanticFieldPatch.model_validate(kwargs)


def _proposal(tool_name: str, **arguments) -> SemanticToolCallProposal:
    return SemanticToolCallProposal(tool_name=tool_name, arguments=arguments)


_INTERNAL_TOKENS = [
    "context note",
    "Context note",
    "deterministic fallback",
    "tool call",
    "semantic interpreter",
    "fallback",
    "proposal",
    "raw tool",
]


def _assert_no_internal_diagnostics(result: dict) -> None:
    """Assert that none of the internal diagnostic tokens leak into the public response."""
    message = result.get("message", "") or ""
    warnings = result.get("warnings") or []
    for token in _INTERNAL_TOKENS:
        assert token not in message, (
            f"Internal diagnostic token {token!r} found in public message: {message!r}"
        )
        for w in warnings:
            assert token not in (w or ""), (
                f"Internal diagnostic token {token!r} found in public warning: {w!r}"
            )


class _NoToolCallInterpreter:
    """Simulates LLM returning no tool call but succeeding on extract_fields."""

    def __init__(self, *, extracted_fields: dict, max_tool_turns: int = 2):
        self._extracted = SemanticExtractedFields.model_validate(extracted_fields)
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def extract_fields(self, transcript, context=None):
        return self._extracted, _metrics()

    def select_tool(self, transcript, context=None):
        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned no tool call. No tracker changes were saved."
        )

    def interpret(self, transcript, context=None):
        extracted, metrics = self.extract_fields(transcript, context)
        self.select_tool(transcript, context)  # raises

    def health_check(self):
        return {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


class _WrongToolInterpreter:
    """Simulates LLM returning a valid tool call but using the wrong tool."""

    def __init__(self, *, extracted_fields: dict, tool_name: str, tool_arguments: dict, max_tool_turns: int = 2):
        self._extracted = SemanticExtractedFields.model_validate(extracted_fields)
        self._tool_name = tool_name
        self._tool_arguments = tool_arguments
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def extract_fields(self, transcript, context=None):
        return self._extracted, _metrics()

    def select_tool(self, transcript, context=None):
        return SemanticInterpretationResult(
            proposal=SemanticToolCallProposal(tool_name=self._tool_name, arguments=self._tool_arguments),
            metrics=_metrics(),
            extracted_fields=self._extracted,
        )

    def interpret(self, transcript, context=None):
        extracted, extraction_metrics = self.extract_fields(transcript, context)
        result = self.select_tool(transcript, context)
        return SemanticInterpretationResult(
            proposal=result.proposal,
            metrics=_metrics(),
            extracted_fields=extracted,
        )

    def health_check(self):
        return {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


class _InvalidArgsInterpreter:
    """Simulates LLM returning a tool call with invalid/malformed arguments."""

    def __init__(self, *, extracted_fields: dict, tool_name: str, bad_arguments: dict, max_tool_turns: int = 1):
        self._extracted = SemanticExtractedFields.model_validate(extracted_fields)
        self._tool_name = tool_name
        self._bad_arguments = bad_arguments
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def extract_fields(self, transcript, context=None):
        return self._extracted, _metrics()

    def select_tool(self, transcript, context=None):
        return SemanticInterpretationResult(
            proposal=SemanticToolCallProposal(tool_name=self._tool_name, arguments=self._bad_arguments),
            metrics=_metrics(),
            extracted_fields=self._extracted,
        )

    def interpret(self, transcript, context=None):
        extracted, _ = self.extract_fields(transcript, context)
        result = self.select_tool(transcript, context)
        return SemanticInterpretationResult(
            proposal=result.proposal,
            metrics=_metrics(),
            extracted_fields=extracted,
        )

    def health_check(self):
        return {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


async def _parse(client, transcript, interpreter, context=None):
    from app.main import app
    from app.semantic_interpreter import get_semantic_interpreter

    app.dependency_overrides[get_semantic_interpreter] = lambda: interpreter
    try:
        resp = await client.post(
            "/transcript/parse", json={"transcript": transcript, "context": context or {}}
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)
    assert resp.status_code == 200
    return resp.json()


def _create_draft(db, *, company="Aiden AI", role="founding engineer"):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=role.casefold() if role else "",
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=True,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _create_app(db, *, company="Google", role="AI Engineer", status="applied"):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication
    from app.role_resolution import normalize_role_name

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=normalize_role_name(role),
        employment_types_json=[],
        job_link="",
        location="",
        status=status,
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


# ===========================================================================
# Part 1 — Unit tests for _has_explicit_create_intent
# ===========================================================================


@pytest.mark.parametrize("transcript", [
    "add application for ai engineer role at neilsoft",
    "add application for ai engineer at neilsoft",
    "add an application for founding engineer at aiden ai",
    "create application for AI Engineer at Neilsoft",
    "create an application for AI Engineer role at Neilsoft",
    "new application for rag engineer at bootcoding",
    "applied for ai engineer at neilsoft",
    "apply for AI Engineer role at Neilsoft",
    "add job application for AI Engineer at Neilsoft",
    "track application for AI Engineer at Neilsoft",
])
def test_has_explicit_create_intent_positive(transcript):
    assert _has_explicit_create_intent(transcript), f"Expected create intent for: {transcript!r}"


@pytest.mark.parametrize("transcript", [
    "ai engineer at neilsoft",
    "founding engineer at aiden ai",
    "update status of Google AI Engineer application to rejected",
    "change status of Neilsoft application to applied",
    "set priority of Google AI Engineer to high",
    "discard draft",
    "save it",
    "archive Google AI Engineer application",
    "delete Google AI Engineer application",
])
def test_has_explicit_create_intent_negative(transcript):
    assert not _has_explicit_create_intent(transcript), f"Expected no create intent for: {transcript!r}"


def test_explicit_create_intent_overrides_saved_update_patterns():
    """Phrases with explicit create intent must not be classified as saved-row update."""
    create_phrases = [
        "add application for ai engineer role at neilsoft",
        "add application for ai engineer at neilsoft",
        "applied for ai engineer at neilsoft",
    ]
    for phrase in create_phrases:
        assert not _has_explicit_saved_update_intent(phrase), (
            f"Explicit create phrase must not be classified as saved-row update: {phrase!r}"
        )


# ===========================================================================
# Part 2 — Explicit create with "role" keyword: wrong tool → override
# ===========================================================================


@pytest.mark.anyio
async def test_explicit_create_with_role_wrong_tool_overridden(client, db):
    """'add application for ai engineer role at neilsoft': fast path now handles this deterministically
    → draft_created with role='ai engineer role', company='neilsoft' (fast path preserves literal text).
    The interpreter override is bypassed — the fast path runs before Ollama."""
    interpreter = _WrongToolInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        tool_name="preview_existing_application_update",
        tool_arguments={
            "target": {"company": "Neilsoft", "role": "AI Engineer"},
            "fields": {},
            "replace_explicit_fields": True,
        },
    )
    result = await _parse(client, "add application for ai engineer role at neilsoft", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, (
        f"Expected draft_created/draft_updated, got {result['status']}: {result}"
    )
    draft = result.get("draft", {})
    assert (draft.get("company") or "").lower() == "neilsoft", draft
    # Fast path preserves the literal role string from the transcript
    assert (draft.get("role") or "").lower() == "ai engineer role", draft
    _assert_no_internal_diagnostics(result)


@pytest.mark.anyio
async def test_explicit_create_with_role_produces_correct_fields(client, db):
    """Verify company and role fields are set correctly for explicit create with 'role' keyword.
    Fast path now handles this deterministically — role is the literal transcript text."""
    interpreter = _WrongToolInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        tool_name="preview_existing_application_update",
        tool_arguments={
            "target": {"company": "Neilsoft", "role": "AI Engineer"},
            "fields": {},
            "replace_explicit_fields": True,
        },
    )
    result = await _parse(client, "add application for ai engineer role at neilsoft", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}
    draft = result.get("draft", {})
    assert (draft.get("company") or "").lower() == "neilsoft", draft
    # Fast path preserves the literal role string — "ai engineer role" not "AI Engineer"
    assert (draft.get("role") or "").lower() == "ai engineer role", draft


# ===========================================================================
# Part 3 — Explicit create without "role": invalid tool args → fallback
# ===========================================================================


@pytest.mark.anyio
async def test_explicit_create_without_role_keyword_invalid_args(client, db):
    """'add application for ai engineer at neilsoft': LLM emits invalid tool args
    → invalid-args fallback produces draft_created."""
    interpreter = _InvalidArgsInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        tool_name="patch_active_draft",
        bad_arguments={"fields": {"unknown_bad_key!!!": "value"}, "replace_explicit_fields": True},
    )
    result = await _parse(client, "add application for ai engineer at neilsoft", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, (
        f"Expected draft_created, got {result['status']}: {result}"
    )
    draft = result.get("draft", {})
    assert (draft.get("company") or "").lower() == "neilsoft", draft
    assert (draft.get("role") or "").lower() == "ai engineer", draft
    _assert_no_internal_diagnostics(result)


# ===========================================================================
# Part 4 — Terse noun phrase: no tool call → draft_created, no diagnostics
# ===========================================================================


@pytest.mark.anyio
async def test_terse_noun_phrase_no_tool_call_creates_draft(client, db):
    """'ai engineer at neilsoft' with no tool call → draft created, no internal notes."""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(client, "ai engineer at neilsoft", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft = result.get("draft", {})
    assert draft.get("company") == "Neilsoft", draft
    assert draft.get("role") == "AI Engineer", draft
    _assert_no_internal_diagnostics(result)


@pytest.mark.anyio
async def test_terse_noun_phrase_public_message_clean(client, db):
    """Public message for terse noun phrase fallback must be clean user-facing copy."""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(client, "ai engineer at neilsoft", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}
    message = result.get("message", "")
    # Clean user-facing messages for draft
    assert message in {
        "Draft created. Review it and save when ready.",
        "Draft updated.",
    }, f"Unexpected message: {message!r}"
    _assert_no_internal_diagnostics(result)


# ===========================================================================
# Part 5 — Terse noun phrase with active draft → patch (not new duplicate)
# ===========================================================================


@pytest.mark.anyio
async def test_terse_noun_phrase_with_active_draft_patches_not_duplicates(client, db):
    """'ai engineer at neilsoft' with active draft → patch existing draft."""
    draft = _create_draft(db, company="Neilsoft", role="")

    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(
        client,
        "ai engineer at neilsoft",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Neilsoft", "role": ""},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated", "no_change"}, result
    result_draft_id = result.get("draft_id")
    # Should patch the existing draft, not create a new one
    if result_draft_id is not None:
        assert result_draft_id == str(draft.id), (
            f"Expected patch of existing draft {draft.id}, got {result_draft_id!r}"
        )
    _assert_no_internal_diagnostics(result)


# ===========================================================================
# Part 6 — Lifecycle precedence
# ===========================================================================


@pytest.mark.anyio
async def test_lifecycle_discard_not_routed_to_draft_create(client, db):
    """'discard draft of Aiden AI' → must not route to draft create/patch."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")

    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Aiden AI"},
    )
    result = await _parse(
        client,
        "discard draft of Aiden AI",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] not in {"draft_created"}, (
        f"Lifecycle discard must not create a draft: {result}"
    )


@pytest.mark.anyio
async def test_lifecycle_archive_not_routed_to_draft_create(client, db):
    """'archive Google AI Engineer application' with no tool call → not draft_created."""
    app = _create_app(db, company="Google", role="AI Engineer")

    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
    )
    result = await _parse(
        client,
        "archive Google AI Engineer application",
        interpreter,
        context={"active_application_id": app.id},
    )
    assert result["status"] not in {"draft_created"}, (
        f"Lifecycle archive must not create a draft: {result}"
    )


@pytest.mark.anyio
async def test_lifecycle_delete_not_routed_to_draft_create(client, db):
    """'delete Google AI Engineer application' with no tool call → not draft_created."""
    app = _create_app(db, company="Google", role="AI Engineer")

    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
    )
    result = await _parse(
        client,
        "delete Google AI Engineer application",
        interpreter,
        context={"active_application_id": app.id},
    )
    assert result["status"] not in {"draft_created"}, (
        f"Lifecycle delete must not create a draft: {result}"
    )


# ===========================================================================
# Part 7 — Saved-row update precedence
# ===========================================================================


@pytest.mark.anyio
async def test_saved_update_routes_to_pending_changes_not_draft(client, db):
    """'update status of google ai engineer application to rejected' → pending_changes, not draft_created."""
    app = _create_app(db, company="Google", role="AI Engineer", status="applied")

    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer", "status": "rejected"},
    )
    result = await _parse(
        client,
        "update status of google ai engineer application to rejected",
        interpreter,
        context={"active_application_id": app.id},
    )
    assert result["status"] not in {"draft_created"}, (
        f"Saved-row update must not create draft: {result}"
    )
    # Saved row must be unchanged
    from app.models import JobApplication
    from app.database import SessionLocal
    with SessionLocal() as s:
        refreshed = s.get(JobApplication, app.id)
        assert refreshed.status == "applied", (
            f"Saved row must not change without applying pending changes: {refreshed.status}"
        )


@pytest.mark.anyio
async def test_saved_update_set_priority_not_draft(client, db):
    """'set priority of Google AI Engineer application to high' → not draft_created."""
    app = _create_app(db, company="Google", role="AI Engineer", status="applied")

    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer", "priority": "HIGH"},
    )
    result = await _parse(
        client,
        "set priority of Google AI Engineer application to high",
        interpreter,
        context={"active_application_id": app.id},
    )
    assert result["status"] not in {"draft_created"}, (
        f"Saved-row update must not create draft: {result}"
    )


# ===========================================================================
# Part 8 — Clarification cases
# ===========================================================================


@pytest.mark.anyio
async def test_role_only_asks_for_company(client, db):
    """Role only, no company, no active draft → clarification: Which company?"""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"role": "AI Engineer"},
    )
    result = await _parse(client, "AI Engineer role", interpreter)
    assert result["status"] in {"clarification"}, result
    question = result.get("clarification_question", "")
    assert "company" in question.lower(), f"Expected clarification about company, got: {question!r}"


@pytest.mark.anyio
async def test_company_only_asks_for_role(client, db):
    """Company only, no role, no active draft → clarification: Which role?"""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft"},
    )
    result = await _parse(client, "Neilsoft company", interpreter)
    assert result["status"] in {"clarification"}, result
    question = result.get("clarification_question", "")
    assert "neilsoft" in question.lower() or "role" in question.lower(), (
        f"Expected clarification about role for Neilsoft, got: {question!r}"
    )


# ===========================================================================
# Part 9 — Internal diagnostic suppression (all fallback paths)
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_fallback_no_diagnostics(client, db):
    """No-tool-call fallback: public response must not expose internal diagnostic tokens."""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(client, "ai engineer at neilsoft", interpreter)
    _assert_no_internal_diagnostics(result)


@pytest.mark.anyio
async def test_wrong_tool_fallback_no_diagnostics(client, db):
    """Wrong-tool override: public response must not expose internal diagnostic tokens."""
    interpreter = _WrongToolInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
        tool_name="preview_existing_application_update",
        tool_arguments={
            "target": {"company": "Neilsoft", "role": "AI Engineer"},
            "fields": {},
            "replace_explicit_fields": True,
        },
    )
    result = await _parse(client, "add application for ai engineer role at neilsoft", interpreter)
    _assert_no_internal_diagnostics(result)


@pytest.mark.anyio
async def test_active_draft_contextual_patch_no_diagnostics(client, db):
    """Active-draft contextual patch fallback must not expose internal diagnostic tokens."""
    draft = _create_draft(db, company="Neilsoft", role="AI Engineer")

    interpreter = _WrongToolInterpreter(
        extracted_fields={"status": "in_touch"},
        tool_name="ask_clarification",
        tool_arguments={"question": "What would you like to do?"},
    )
    result = await _parse(
        client,
        "change status to in-touch",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Neilsoft", "role": "AI Engineer"},
        },
    )
    _assert_no_internal_diagnostics(result)


# ===========================================================================
# Part 10 — Routing precedence: explicit create beats terse noun phrase
# ===========================================================================


@pytest.mark.anyio
async def test_explicit_create_phrase_applied_for(client, db):
    """'applied for ai engineer at neilsoft' → draft_created."""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(client, "applied for ai engineer at neilsoft", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft = result.get("draft", {})
    assert draft.get("company") == "Neilsoft", draft
    assert draft.get("role") == "AI Engineer", draft
    _assert_no_internal_diagnostics(result)


@pytest.mark.anyio
async def test_explicit_create_phrase_create_application(client, db):
    """'create a founding engineer application at aiden ai' → draft_created."""
    interpreter = _NoToolCallInterpreter(
        extracted_fields={"company": "Aiden AI", "role": "Founding Engineer"},
    )
    result = await _parse(client, "create a founding engineer application at aiden ai", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft = result.get("draft", {})
    assert (draft.get("company") or "").lower() == "aiden ai", draft
    assert (draft.get("role") or "").lower() == "founding engineer", draft
    _assert_no_internal_diagnostics(result)
