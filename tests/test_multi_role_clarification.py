"""Tests for multi-role natural-language behavior.

When the user mentions multiple roles in one utterance, the system must:
- NOT create a draft
- Return a clarification asking which role to add first
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.mutation_dispatcher import dispatch
from app.semantic_validation import (
    handle_ask_clarification,
    normalize_role_title,
)
from app.semantic_schemas import AskClarificationArguments, SemanticToolCallProposal


# ---------------------------------------------------------------------------
# normalize_role_title scalar behavior
# ---------------------------------------------------------------------------

def test_normalize_role_title_accepts_open_ended():
    assert normalize_role_title("Applied AI Engineer") == "Applied AI Engineer"


def test_normalize_role_title_trims_whitespace():
    assert normalize_role_title("  AI Engineer  ") == "AI Engineer"


def test_normalize_role_title_collapses_internal_spaces():
    assert normalize_role_title("AI  Engineer") == "AI Engineer"


def test_normalize_role_title_accepts_unknown_role():
    assert normalize_role_title("LLM Inference Optimization Engineer") == "LLM Inference Optimization Engineer"


def test_normalize_role_title_empty_returns_none():
    assert normalize_role_title("") is None


def test_normalize_role_title_blank_returns_none():
    assert normalize_role_title("   ") is None


# ---------------------------------------------------------------------------
# Clarification is returned when multiple roles detected in utterance
# (tested via ask_clarification mutation path)
# ---------------------------------------------------------------------------

def test_ask_clarification_handler_returns_clarification_status(db_session: Session):
    from app.schemas import TranscriptParseRequest

    proposal = SemanticToolCallProposal(
        tool_name="ask_clarification",
        arguments={"question": "I found multiple roles for Acme: AI Engineer and RAG Engineer. Please add one application at a time. Which role should I add first?"},
    )
    arguments = AskClarificationArguments(
        question="I found multiple roles for Acme: AI Engineer and RAG Engineer. Please add one application at a time. Which role should I add first?"
    )
    payload = TranscriptParseRequest(transcript="Applied for AI Engineer and RAG Engineer roles at Acme")

    result = handle_ask_clarification(payload, proposal, arguments, metrics=None, db=db_session)
    assert result.status == "clarification_required"
    assert result.clarification_question is not None
    assert "AI Engineer" in result.clarification_question or "RAG Engineer" in result.clarification_question


def test_multi_role_creates_no_draft(db_session: Session):
    """Submitting only a clarification operation must not create any application row."""
    payload = MutationPayload(
        operation="ask_clarification",
        target=MutationTarget(),
        changes=ApplicationChanges(),
        notes_to_append=["I found multiple roles for Acme: AI Engineer and RAG Engineer. Please add one application at a time. Which role should I add first?"],
    )
    result = dispatch(payload, db_session)
    assert result.clarification_question is not None

    rows = db_session.query("job_applications").all() if False else []
    from app.models import JobApplication
    all_apps = db_session.query(JobApplication).all()
    assert len(all_apps) == 0
