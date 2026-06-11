"""Regression tests for blank-string sanitization at the LLM extraction boundary.

The llama3.2:3b runtime sometimes emits empty strings for optional fields it
cannot fill (e.g. comments="", next_action="", job_link="").  Pydantic's
extra="forbid" schema previously rejected the entire payload when a blank
string was emitted for a field that requires non-empty text, causing
"Local language interpreter returned invalid extracted fields."

After the fix, blank/whitespace-only optional scalars are silently dropped
(treated as absent) and blank entries are removed from optional string arrays
before model_validate is called.
"""

import json

import httpx
import pytest

from app import semantic_interpreter
from app.semantic_interpreter import (
    OllamaSemanticInterpreter,
    _sanitize_extracted_fields_dict,
)
from app.semantic_schemas import SemanticExtractedFields


# ---------------------------------------------------------------------------
# Unit tests: _sanitize_extracted_fields_dict
# ---------------------------------------------------------------------------


def test_sanitize_blank_comments_removed():
    raw = {"company": "Google", "role": "AI Engineer", "comments": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "comments" not in result
    assert result["company"] == "Google"
    assert result["role"] == "AI Engineer"


def test_sanitize_blank_next_action_removed():
    raw = {"company": "Google", "next_action": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "next_action" not in result


def test_sanitize_whitespace_only_next_action_removed():
    raw = {"company": "Google", "next_action": "   "}
    result = _sanitize_extracted_fields_dict(raw)
    assert "next_action" not in result


def test_sanitize_blank_job_link_removed():
    raw = {"company": "Google", "job_link": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "job_link" not in result


def test_sanitize_blank_location_removed():
    raw = {"company": "Google", "location": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "location" not in result


def test_sanitize_blank_status_removed():
    raw = {"company": "Google", "status": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "status" not in result


def test_sanitize_blank_priority_removed():
    raw = {"company": "Google", "priority": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "priority" not in result


def test_sanitize_blank_role_removed():
    raw = {"company": "Google", "role": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert "role" not in result


def test_sanitize_non_blank_values_preserved():
    raw = {
        "company": "Google",
        "role": "AI Engineer",
        "comments": "Looks interesting",
        "next_action": "Follow up",
        "job_link": "https://example.com",
        "location": "remote",
        "status": "applied",
        "priority": "HIGH",
    }
    result = _sanitize_extracted_fields_dict(raw)
    assert result == raw


def test_sanitize_strips_surrounding_whitespace_from_scalars():
    raw = {"company": "  Google  ", "role": "  AI Engineer  "}
    result = _sanitize_extracted_fields_dict(raw)
    assert result["company"] == "Google"
    assert result["role"] == "AI Engineer"


def test_sanitize_blank_entry_in_employment_types_removed():
    raw = {"company": "Google", "employment_types": ["Full Time", "", "  "]}
    result = _sanitize_extracted_fields_dict(raw)
    assert result["employment_types"] == ["Full Time"]


def test_sanitize_all_blank_employment_types_drops_key():
    raw = {"company": "Google", "employment_types": ["", "  "]}
    result = _sanitize_extracted_fields_dict(raw)
    assert "employment_types" not in result


def test_sanitize_blank_entry_in_current_stages_removed():
    raw = {"company": "Google", "current_stages": ["Applied", "", "Engaged"]}
    result = _sanitize_extracted_fields_dict(raw)
    assert result["current_stages"] == ["Applied", "Engaged"]


def test_sanitize_empty_list_employment_types_drops_key():
    raw = {"company": "Google", "employment_types": []}
    result = _sanitize_extracted_fields_dict(raw)
    assert "employment_types" not in result


def test_sanitize_engaged_days_integer_preserved():
    raw = {"company": "Google", "engaged_days": 5}
    result = _sanitize_extracted_fields_dict(raw)
    assert result["engaged_days"] == 5


def test_sanitize_engaged_days_zero_preserved():
    raw = {"company": "Google", "engaged_days": 0}
    result = _sanitize_extracted_fields_dict(raw)
    assert result["engaged_days"] == 0


def test_sanitize_none_values_dropped():
    raw = {"company": "Google", "role": None, "comments": None}
    result = _sanitize_extracted_fields_dict(raw)
    assert "role" not in result
    assert "comments" not in result


def test_sanitize_all_blank_produces_empty_dict():
    raw = {"comments": "", "next_action": "  ", "job_link": "", "location": "", "status": ""}
    result = _sanitize_extracted_fields_dict(raw)
    assert result == {}


# ---------------------------------------------------------------------------
# Integration: blank strings in LLM JSON must not fail extraction
# ---------------------------------------------------------------------------


def _make_extraction_response(content: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps(content)},
            "total_duration": 80,
            "load_duration": 10,
            "prompt_eval_duration": 20,
            "eval_duration": 30,
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )


def _make_selection_response(tool_name: str, arguments: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {
                "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}]
            },
            "total_duration": 100,
            "load_duration": 20,
            "prompt_eval_duration": 30,
            "eval_duration": 40,
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )


def test_blank_comments_in_llm_json_does_not_fail_extraction(monkeypatch: pytest.MonkeyPatch):
    """LLM emits comments="" — extraction must succeed and drop the blank field."""
    responses = [
        _make_extraction_response({"company": "Google", "role": "AI Engineer", "comments": ""}),
        _make_selection_response(
            "patch_active_draft",
            {"fields": {"company": "Google", "role": "AI Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    ]

    def fake_post(url, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret("add Google application for AI Engineer")

    assert result.extracted_fields.company == "Google"
    assert result.extracted_fields.role == "AI Engineer"
    assert result.extracted_fields.comments is None


def test_blank_next_action_in_llm_json_does_not_fail_extraction(monkeypatch: pytest.MonkeyPatch):
    """LLM emits next_action="" — extraction succeeds with next_action=None."""
    responses = [
        _make_extraction_response({"company": "Neilsoft", "role": "SWE", "next_action": ""}),
        _make_selection_response(
            "patch_active_draft",
            {"fields": {"company": "Neilsoft", "role": "SWE"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    ]

    def fake_post(url, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret("add Neilsoft application for SWE")

    assert result.extracted_fields.next_action is None


def test_blank_job_link_in_llm_json_does_not_fail_extraction(monkeypatch: pytest.MonkeyPatch):
    """LLM emits job_link="" — extraction succeeds with job_link=None."""
    responses = [
        _make_extraction_response({"company": "Acme", "role": "PM", "job_link": ""}),
        _make_selection_response(
            "patch_active_draft",
            {"fields": {"company": "Acme", "role": "PM"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    ]

    def fake_post(url, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret("add Acme application for PM")

    assert result.extracted_fields.job_link is None


def test_multiple_blank_optional_fields_in_llm_json_do_not_fail_extraction(monkeypatch: pytest.MonkeyPatch):
    """LLM emits multiple blank optional fields — all sanitized; extraction succeeds."""
    responses = [
        _make_extraction_response({
            "company": "StartupCo",
            "role": "founding engineer",
            "comments": "",
            "next_action": "  ",
            "job_link": "",
            "location": "",
            "status": "",
            "priority": "",
        }),
        _make_selection_response(
            "patch_active_draft",
            {"fields": {"company": "StartupCo", "role": "founding engineer"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    ]

    def fake_post(url, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret("add StartupCo for founding engineer")

    assert result.extracted_fields.company == "StartupCo"
    assert result.extracted_fields.role == "founding engineer"
    assert result.extracted_fields.comments is None
    assert result.extracted_fields.next_action is None
    assert result.extracted_fields.job_link is None
    assert result.extracted_fields.location is None
    assert result.extracted_fields.status is None
    assert result.extracted_fields.priority is None


def test_blank_entry_in_employment_types_does_not_fail_extraction(monkeypatch: pytest.MonkeyPatch):
    """LLM emits employment_types=["Full Time", ""] — blank entry removed; extraction succeeds."""
    responses = [
        _make_extraction_response({"company": "Google", "employment_types": ["Full Time", ""]}),
        _make_selection_response(
            "patch_active_draft",
            {"fields": {"company": "Google", "employment_types": ["Full Time"]}, "replace_explicit_fields": True, "context_notes": []},
        ),
    ]

    def fake_post(url, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret("add Google Full Time application")

    assert result.extracted_fields.employment_types == ["Full Time"]


def test_blank_controlled_optional_field_does_not_fail_extraction(monkeypatch: pytest.MonkeyPatch):
    """LLM emits status="" — blank controlled field treated as absent, not an error."""
    responses = [
        _make_extraction_response({"company": "Google", "role": "AI Engineer", "status": ""}),
        _make_selection_response(
            "patch_active_draft",
            {"fields": {"company": "Google", "role": "AI Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    ]

    def fake_post(url, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret("add Google AI Engineer")

    assert result.extracted_fields.status is None


# ---------------------------------------------------------------------------
# Schema-level: SemanticFieldPatch / SemanticExtractedFields validators
# ---------------------------------------------------------------------------


def test_semantic_extracted_fields_accepts_none_next_action():
    fields = SemanticExtractedFields(company="Google", next_action=None)
    assert fields.next_action is None


def test_semantic_extracted_fields_accepts_none_comments():
    fields = SemanticExtractedFields(company="Google", comments=None)
    assert fields.comments is None


def test_semantic_field_patch_blank_next_action_becomes_none():
    """SemanticFieldPatch normalizes blank next_action to None (not raises)."""
    from app.semantic_schemas import SemanticFieldPatch
    patch = SemanticFieldPatch(next_action="")
    assert patch.next_action is None


def test_semantic_field_patch_whitespace_only_comments_becomes_none():
    """SemanticFieldPatch normalizes whitespace-only comments to None (not raises)."""
    from app.semantic_schemas import SemanticFieldPatch
    patch = SemanticFieldPatch(comments="   ")
    assert patch.comments is None
