"""
Slow integration regression tests for indirect / paraphrased phrasings.

These call the REAL single-call extractor (``extract_semantic_command_once``)
directly — not through the HTTP route — and assert the correct intent and key
fields come back. They hit a live Ollama model (MODEL_WINNER = llama3.2:3b with
the improved system prompt), so they are slow and are marked ``slow`` to keep
them out of the fast suite:

    pytest -q -m "not slow"     # fast suite — excludes these
    pytest -q -m slow           # runs only these

If Ollama is not reachable, the tests skip rather than fail, so the marked
suite is still runnable in environments without a local model.

Schema reminder (app.semantic_command_schemas.SemanticCommand):
    intent ∈ {create_application, update_application, append_note,
              archive_application, unsupported}
    target.{company, role, application_id}
    changes.{status, priority, location_mode, ...}
    note, clarification, suggested_phrasings

Because status / priority / location updates to an existing row all share the
single ``update_application`` intent, these tests distinguish them by the
populated ``changes`` field rather than a dedicated intent.
"""
from __future__ import annotations

import httpx
import pytest

from app.semantic_command_extractor import (
    SemanticExtractorError,
    extract_semantic_command_once,
)
from app.semantic_interpreter import get_ollama_settings

pytestmark = pytest.mark.slow

# Mutating intents — used by the "must not mutate" informational check.
MUTATING_INTENTS = {
    "create_application",
    "update_application",
    "append_note",
    "archive_application",
}


@pytest.fixture(scope="module")
def settings():
    s = get_ollama_settings()
    # Probe Ollama once; skip the whole module if it is not available so the
    # slow suite degrades gracefully instead of erroring.
    try:
        resp = httpx.get(f"{s.base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
    except Exception:
        pytest.skip("Ollama is not reachable; skipping slow extractor tests.")
    if s.model not in names:
        pytest.skip(f"Model {s.model!r} is not pulled; skipping slow extractor tests.")
    return s


def _extract(settings, transcript: str):
    try:
        command, _metrics = extract_semantic_command_once(transcript, None, settings=settings)
    except SemanticExtractorError as exc:  # pragma: no cover - surfaced as failure
        pytest.fail(f"extractor raised on {transcript!r}: {exc}")
    return command


def _norm(value):
    return value.strip().lower() if isinstance(value, str) else value


# ─────────────────────────────────────────────────────────────────────────────
# Indirect create_application (>= 3)
# ─────────────────────────────────────────────────────────────────────────────

def test_indirect_create_track_phrasing(settings):
    cmd = _extract(settings, "i should track at Spotify for data scientist role")
    assert cmd.intent == "create_application"
    assert _norm(cmd.target.company) == "spotify"


def test_indirect_create_been_meaning_to(settings):
    cmd = _extract(settings, "been meaning to add the Netflix software engineer role")
    assert cmd.intent == "create_application"
    assert _norm(cmd.target.company) == "netflix"


def test_indirect_create_i_applied(settings):
    cmd = _extract(settings, "i think i applied to Google last week")
    assert cmd.intent == "create_application"
    assert _norm(cmd.target.company) == "google"
    assert _norm(cmd.changes.status) == "applied"


# ─────────────────────────────────────────────────────────────────────────────
# Indirect update_status (>= 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_indirect_status_rejected(settings):
    cmd = _extract(settings, "Spotify turned me down")
    assert cmd.intent == "update_application"
    assert _norm(cmd.target.company) == "spotify"
    assert _norm(cmd.changes.status) == "rejected"


def test_indirect_status_accepted(settings):
    cmd = _extract(settings, "got accepted at Netflix!")
    assert cmd.intent == "update_application"
    assert _norm(cmd.target.company) == "netflix"
    assert _norm(cmd.changes.status) == "accepted"


# ─────────────────────────────────────────────────────────────────────────────
# Indirect update_priority (>= 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_indirect_priority_bump(settings):
    cmd = _extract(settings, "the Spotify one should be higher priority")
    assert cmd.intent == "update_application"
    assert _norm(cmd.target.company) == "spotify"
    assert _norm(cmd.changes.priority) == "high"


# ─────────────────────────────────────────────────────────────────────────────
# Informational query must NOT produce a mutating intent (>= 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_informational_query_does_not_mutate(settings):
    cmd = _extract(settings, "how many applications do I have?")
    assert cmd.intent not in MUTATING_INTENTS
    assert cmd.changes.has_any_field() is False
    assert cmd.note is None


def test_informational_status_question_does_not_mutate(settings):
    cmd = _extract(settings, "what's the status of my Spotify application?")
    assert cmd.intent not in MUTATING_INTENTS


# ─────────────────────────────────────────────────────────────────────────────
# Incomplete command — no concrete identity/action → safe, non-mutating
# (the extractor must not hallucinate a target; the backend asks to clarify)
# ─────────────────────────────────────────────────────────────────────────────

def test_incomplete_command_is_safe(settings):
    cmd = _extract(settings, "do something")
    # No company/role and no actionable field: must stay non-mutating so the
    # deterministic backend can ask for clarification rather than guessing.
    assert cmd.intent == "unsupported"
    assert cmd.target.company is None
