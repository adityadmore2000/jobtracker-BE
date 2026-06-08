from pathlib import Path

import httpx
import pytest

from app import database_config, semantic_interpreter
from app.semantic_interpreter import (
    OllamaSemanticInterpreter,
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterUnavailableError,
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


@pytest.fixture(autouse=True)
def reset_env_cache():
    database_config.reset_backend_environment_cache()
    yield
    database_config.reset_backend_environment_cache()


def test_ollama_request_uses_expected_endpoint_model_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return build_success_response(
            [
                {
                    "function": {
                        "name": "patch_active_draft",
                        "arguments": {
                            "fields": {"company": "Neilsoft", "roles": ["AI Engineer"]},
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
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 20.0
    assert captured["json"]["model"] == "llama3.2:3b"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][1]["content"]
    assert "I want to add an application Neilsoft" in captured["json"]["messages"][0]["content"]
    assert "AI Engineer role for Neilsoft" in captured["json"]["messages"][0]["content"]
    assert "Which company's application do you mean?" in captured["json"]["messages"][0]["content"]
    assert "Do not ask \"Which company should I use?\" when exactly one explicit company is already present." in captured["json"]["messages"][0]["content"]
    assert '"explicit_known_companies_in_current_utterance": ["Neilsoft"]' in captured["json"]["messages"][1]["content"]
    assert captured["json"]["tools"]
    assert {tool["function"]["name"] for tool in captured["json"]["tools"]} == {
        "patch_active_draft",
        "preview_existing_application_update",
        "request_draft_save",
        "attach_latest_browser_context",
        "ask_clarification",
    }
    assert "format" not in captured["json"]


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
    assert "For list-valued fields such as roles and employment_types, always emit JSON arrays." in messages[0]["content"]
    assert "Company and role may appear in any natural order." in messages[0]["content"]
    assert "Role at Neilsoft for AI Engineer" in messages[0]["content"]
    assert "Do not include connector words such as \"for\"" in messages[0]["content"]
    assert "Do not include the label word \"role\" inside the role value" in messages[0]["content"]
    assert '"explicit_known_companies_in_current_utterance": ["Neilsoft"]' in messages[1]["content"]
    assert '"retry_hint": "Use Neilsoft instead of asking for company clarification."' in messages[1]["content"]


def test_build_ollama_messages_includes_schema_repair_retry_hint() -> None:
    messages = build_ollama_messages(
        "Role at Neilsoft for AI Engineer",
        {"schema_repair_retry_hint": "Use fields.roles as a JSON array of strings."},
    )

    assert '"schema_repair_retry_hint": "Use fields.roles as a JSON array of strings."' in messages[1]["content"]


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
