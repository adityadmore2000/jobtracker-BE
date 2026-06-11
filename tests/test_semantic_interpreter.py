import json
from pathlib import Path

import httpx
import pytest

from app import database_config, semantic_interpreter
from app.semantic_interpreter import (
    OllamaSemanticInterpreter,
    OllamaSettings,
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterUnavailableError,
    build_field_extraction_messages,
    build_ollama_messages,
    get_ollama_settings,
)


def build_success_response(tool_calls: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"tool_calls": tool_calls},
            "total_duration": 100,
            "load_duration": 20,
            "prompt_eval_duration": 30,
            "eval_duration": 40,
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )


def build_extraction_response(content: dict[str, object]) -> httpx.Response:
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


@pytest.fixture(autouse=True)
def reset_env_cache():
    database_config.reset_backend_environment_cache()
    yield
    database_config.reset_backend_environment_cache()


def test_ollama_request_uses_expected_endpoint_model_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_post(url, json, timeout):
        captured.append({"url": url, "json": json, "timeout": timeout})
        if "tools" not in json:
            return build_extraction_response({"company": "Neilsoft", "role": "AI Engineer"})
        return build_success_response(
            [
                {
                    "function": {
                        "name": "patch_active_draft",
                        "arguments": {
                            "fields": {"company": "Neilsoft", "role": "AI Engineer"},
                            "replace_explicit_fields": True,
                            "context_notes": [],
                        },
                    }
                }
            ]
        )

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    interpreter = OllamaSemanticInterpreter()
    result = interpreter.interpret(
        "Add AI Engineer role for Neilsoft",
        {"recent_actions": ["x"], "explicit_known_companies": ["Neilsoft"]},
    )

    assert result.proposal.tool_name == "patch_active_draft"
    assert result.extracted_fields.model_dump(exclude_none=True) == {"company": "Neilsoft", "role": "AI Engineer"}
    assert len(captured) == 2
    extraction_call, selection_call = captured
    assert extraction_call["url"] == "http://127.0.0.1:11434/api/chat"
    assert extraction_call["timeout"] == 20.0
    assert extraction_call["json"]["model"] == "llama3.2:3b"
    assert extraction_call["json"]["stream"] is False
    assert extraction_call["json"]["messages"][1]["content"]
    assert extraction_call["json"]["format"]["title"] == "SemanticExtractedFields"
    assert "Extract only information explicitly stated in the current utterance." in extraction_call["json"]["messages"][0]["content"]
    assert selection_call["url"] == "http://127.0.0.1:11434/api/chat"
    assert selection_call["timeout"] == 20.0
    assert selection_call["json"]["model"] == "llama3.2:3b"
    assert selection_call["json"]["stream"] is False
    assert selection_call["json"]["messages"][1]["content"]
    assert "I want to add an application Neilsoft" in selection_call["json"]["messages"][0]["content"]
    assert "AI Engineer role for Neilsoft" in selection_call["json"]["messages"][0]["content"]
    assert "Which company's application do you mean?" in selection_call["json"]["messages"][0]["content"]
    assert "Do not ask \"Which company should I use?\" when exactly one explicit company is already present." in selection_call["json"]["messages"][0]["content"]
    assert '"explicit_known_companies_in_current_utterance": ["Neilsoft"]' in selection_call["json"]["messages"][1]["content"]
    assert '"normalized_extracted_fields": {"company": "Neilsoft", "role": "AI Engineer"}' in selection_call["json"]["messages"][1]["content"]
    assert selection_call["json"]["tools"]
    assert {tool["function"]["name"] for tool in selection_call["json"]["tools"]} == {
        "patch_active_draft",
        "preview_existing_application_update",
        "request_draft_save",
        "attach_latest_browser_context",
        "ask_clarification",
        "archive_application",
        "explain_delete_policy",
        "discard_draft",
    }
    assert "format" not in selection_call["json"]


def test_build_ollama_messages_includes_retry_hint_and_explicit_companies() -> None:
    messages = build_ollama_messages(
        "AI Engineer role for Neilsoft",
        {
            "explicit_known_companies": ["Neilsoft"],
            "explicit_company_retry_hint": "Use Neilsoft instead of asking for company clarification.",
        },
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "For list-valued fields such as employment_types and current_stages, always emit JSON arrays." in messages[0]["content"]
    assert "Company and other fields may appear in any natural order." in messages[0]["content"]
    assert "Role at Neilsoft for AI Engineer" in messages[0]["content"]
    assert "Do not include connector words such as \"for\"" in messages[0]["content"]
    assert "Do not include label words like \"role\", \"priority\", or \"stage\" inside values" in messages[0]["content"]
    assert "Neilsoft sathi AI Engineer role, fulltime onsite, high priority" in messages[0]["content"]
    assert "Set Neilsoft current stage to Applied and Engaged" in messages[0]["content"]
    assert '"explicit_known_companies_in_current_utterance": ["Neilsoft"]' in messages[1]["content"]
    assert '"retry_hint": "Use Neilsoft instead of asking for company clarification."' in messages[1]["content"]


def test_build_field_extraction_messages_include_examples_and_retry_hint() -> None:
    messages = build_field_extraction_messages(
        "I have previously worked at this company. It's called Neilsoft. I'd like to track AI Engineer application for this position.",
        {
            "explicit_known_companies": ["Neilsoft"],
            "field_extraction_retry_hint": "Return only valid JSON matching the extraction schema.",
        },
    )

    assert len(messages) == 2
    assert "Do not select a tool." in messages[0]["content"]
    assert "Ignore conversational filler." in messages[0]["content"]
    assert "The role field is free-form open-ended text emitted as a single string (not an array)." in messages[0]["content"]
    assert "It's called Neilsoft. I'd like to track AI Engineer application for this position." in messages[0]["content"]
    assert '"explicit_known_companies_in_current_utterance": ["Neilsoft"]' in messages[1]["content"]
    assert '"field_extraction_retry_hint": "Return only valid JSON matching the extraction schema."' in messages[1]["content"]


def test_build_ollama_messages_includes_schema_repair_retry_hint() -> None:
    messages = build_ollama_messages(
        "Role at Neilsoft for AI Engineer",
        {"schema_repair_retry_hint": "Use fields.roles as a JSON array of strings."},
    )

    assert '"schema_repair_retry_hint": "Use fields.roles as a JSON array of strings."' in messages[1]["content"]


def test_field_extraction_retries_once_when_first_json_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        httpx.Response(
            200,
            json={"message": {"content": "not-json"}, "total_duration": 1},
            request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
        ),
        build_extraction_response({"company": "Neilsoft", "role": "AI Engineer"}),
        build_success_response(
            [
                {
                    "function": {
                        "name": "patch_active_draft",
                        "arguments": {
                            "fields": {"company": "Neilsoft", "role": "AI Engineer"},
                            "replace_explicit_fields": True,
                            "context_notes": [],
                        },
                    }
                }
            ]
        ),
    ]
    seen_payloads: list[dict[str, object]] = []

    def fake_post(url, json, timeout):
        seen_payloads.append(json)
        return responses.pop(0)

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    result = OllamaSemanticInterpreter().interpret(
        "I have previously worked at this company. It's called Neilsoft. I'd like to track AI Engineer application for this position.",
        {"explicit_known_companies": ["Neilsoft"]},
    )

    assert result.extracted_fields.model_dump(exclude_none=True) == {"company": "Neilsoft", "role": "AI Engineer"}
    assert len(seen_payloads) == 3
    assert '"field_extraction_retry_hint": "Your previous field-extraction JSON was invalid. Return only valid JSON matching the extraction schema. Do not add fields that were not explicitly stated."' in seen_payloads[1]["messages"][1]["content"]


def test_ollama_settings_are_loaded_from_backend_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OLLAMA_BASE_URL=http://127.0.0.1:22434",
                "OLLAMA_MODEL=llama3.2:3b",
                "OLLAMA_TIMEOUT_SECONDS=9",
                "OLLAMA_KEEP_ALIVE=5m",
                "OLLAMA_MAX_TOOL_TURNS=3",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(database_config, "get_backend_env_path", lambda: env_file)

    settings = get_ollama_settings()

    assert settings.base_url == "http://127.0.0.1:22434"
    assert settings.model == "llama3.2:3b"
    assert settings.timeout_seconds == 9.0
    assert settings.keep_alive == "5m"
    assert settings.max_tool_turns == 3


def test_os_environment_overrides_backend_dotenv_for_ollama_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OLLAMA_BASE_URL=http://127.0.0.1:22434",
                "OLLAMA_MODEL=wrong-model",
                "OLLAMA_TIMEOUT_SECONDS=9",
                "OLLAMA_KEEP_ALIVE=5m",
                "OLLAMA_MAX_TOOL_TURNS=7",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(database_config, "get_backend_env_path", lambda: env_file)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "10m")
    monkeypatch.setenv("OLLAMA_MAX_TOOL_TURNS", "2")

    settings = get_ollama_settings()

    assert settings.base_url == "http://127.0.0.1:11434"
    assert settings.model == "llama3.2:3b"
    assert settings.timeout_seconds == 20.0
    assert settings.keep_alive == "10m"
    assert settings.max_tool_turns == 2


def test_ollama_timeout_raises_recoverable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url, json, timeout):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    with pytest.raises(SemanticInterpreterUnavailableError):
        OllamaSemanticInterpreter().interpret("Add Neilsoft for AI Engineer")


def test_ollama_connection_failure_raises_recoverable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("failed", request=httpx.Request("POST", url))

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    with pytest.raises(SemanticInterpreterUnavailableError):
        OllamaSemanticInterpreter().interpret("Add Neilsoft for AI Engineer")


def test_missing_tool_call_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_interpreter.httpx, "post", lambda url, json, timeout: build_success_response([]))

    with pytest.raises(SemanticInterpreterInvalidResponseError):
        OllamaSemanticInterpreter().interpret("Add Neilsoft for AI Engineer")


def test_unknown_tool_call_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        semantic_interpreter.httpx,
        "post",
        lambda url, json, timeout: build_success_response([{"function": {"name": "delete_everything", "arguments": {}}}]),
    )

    with pytest.raises(SemanticInterpreterInvalidResponseError):
        OllamaSemanticInterpreter().interpret("Add Neilsoft for AI Engineer")


def test_malformed_tool_arguments_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        semantic_interpreter.httpx,
        "post",
        lambda url, json, timeout: build_success_response(
            [{"function": {"name": "patch_active_draft", "arguments": "not-json"}}]
        ),
    )

    with pytest.raises(SemanticInterpreterInvalidResponseError):
        OllamaSemanticInterpreter().interpret("Add Neilsoft for AI Engineer")


def test_multiple_tool_calls_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        semantic_interpreter.httpx,
        "post",
        lambda url, json, timeout: build_success_response(
            [
                {"function": {"name": "patch_active_draft", "arguments": {"fields": {"company": "Neilsoft"}}}},
                {"function": {"name": "request_draft_save", "arguments": {}}},
            ]
        ),
    )

    with pytest.raises(SemanticInterpreterInvalidResponseError):
        OllamaSemanticInterpreter().interpret("Add Neilsoft for AI Engineer")


def test_interpret_respects_ollama_max_tool_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_post(url, json, timeout):
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={"message": {"content": "not-json"}, "total_duration": 1},
            request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
        )

    monkeypatch.setattr(semantic_interpreter.httpx, "post", fake_post)

    settings = OllamaSettings(
        base_url="http://127.0.0.1:11434",
        model="llama3.2:3b",
        timeout_seconds=5.0,
        keep_alive="5m",
        max_tool_turns=1,
    )
    interpreter = OllamaSemanticInterpreter(settings)
    assert interpreter.settings.max_tool_turns == 1

    # Always-invalid JSON must terminate with a recoverable error, never an infinite loop.
    with pytest.raises(SemanticInterpreterInvalidResponseError):
        interpreter.interpret("Add Neilsoft", {})

    # A single interpret() stays bounded to its two passes (extract + select); invalid
    # extraction JSON triggers at most one internal retry and never exceeds two Ollama calls.
    assert call_count["n"] <= 2
