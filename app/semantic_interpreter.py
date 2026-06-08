import json
import time
from dataclasses import dataclass

import httpx

from .database_config import get_optional_env_value
from .semantic_schemas import (
    AskClarificationArguments,
    AttachLatestBrowserContextArguments,
    PatchActiveDraftArguments,
    PreviewExistingApplicationUpdateArguments,
    RequestDraftSaveArguments,
    SemanticInterpreterMetrics,
    SemanticToolCallProposal,
)


class SemanticInterpreterError(RuntimeError):
    pass


class SemanticInterpreterUnavailableError(SemanticInterpreterError):
    pass


class SemanticInterpreterInvalidResponseError(SemanticInterpreterError):
    pass


TOOL_ARGUMENT_MODELS = {
    "patch_active_draft": PatchActiveDraftArguments,
    "preview_existing_application_update": PreviewExistingApplicationUpdateArguments,
    "request_draft_save": RequestDraftSaveArguments,
    "attach_latest_browser_context": AttachLatestBrowserContextArguments,
    "ask_clarification": AskClarificationArguments,
}


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    timeout_seconds: float
    keep_alive: str
    max_tool_turns: int


@dataclass(frozen=True)
class SemanticInterpretationResult:
    proposal: SemanticToolCallProposal
    metrics: SemanticInterpreterMetrics


def get_ollama_settings() -> OllamaSettings:
    base_url = get_optional_env_value("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
    model = get_optional_env_value("OLLAMA_MODEL") or "llama3.2:3b"
    timeout_raw = get_optional_env_value("OLLAMA_TIMEOUT_SECONDS") or "20"
    keep_alive = get_optional_env_value("OLLAMA_KEEP_ALIVE") or "10m"
    max_tool_turns_raw = get_optional_env_value("OLLAMA_MAX_TOOL_TURNS") or "2"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError("OLLAMA_TIMEOUT_SECONDS must be a number.") from exc
    try:
        max_tool_turns = int(max_tool_turns_raw)
    except ValueError as exc:
        raise RuntimeError("OLLAMA_MAX_TOOL_TURNS must be an integer.") from exc
    return OllamaSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_seconds=timeout_seconds,
        keep_alive=keep_alive,
        max_tool_turns=max_tool_turns,
    )


def build_ollama_messages(transcript: str, context: dict[str, object] | None) -> list[dict[str, str]]:
    normalized_context = context or {}
    explicit_known_companies = normalized_context.get("explicit_known_companies")
    retry_hint = normalized_context.get("explicit_company_retry_hint")
    schema_repair_retry_hint = normalized_context.get("schema_repair_retry_hint")
    return [
        {
            "role": "system",
            "content": (
                "You interpret transcript-style job tracker commands for a local job application tracker. "
                "Use exactly one tool call from the provided tools. Do not return prose. "
                "Do not call multiple tools. Do not invent facts. Do not guess missing company, role, next action, comments, or stages. "
                "Never persist data. Never access databases. Only prepare preview-safe arguments for the backend. "
                "Partial unsaved drafts are valid and should usually use patch_active_draft. "
                "Do not ask for missing draft fields immediately. Patch only fields explicitly mentioned by the user. "
                "A company-like name near words such as application, company, opening, job, role, apply, track, or add should usually be treated as company. "
                "A company name may appear anywhere in the user's sentence. "
                "When exactly one explicit known company is listed for the current utterance, treat it as the company mentioned by the user. "
                "Do not ask \"Which company should I use?\" when exactly one explicit company is already present. "
                "Unknown company names are valid during draft creation. "
                "Use preview_existing_application_update only for an already persisted application. "
                "That tool requires either an explicit company in the user utterance or an explicit persisted application_id selected in UI context. "
                "Do not infer a persisted-row target from active draft context, recent actions, or vague pronouns alone. "
                "Use request_draft_save only when there is an active unsaved draft. "
                "Use attach_latest_browser_context only for the active unsaved draft and never to target a persisted row. "
                "Use ask_clarification only when genuinely needed. "
                "Company and role may appear in any natural order. "
                "Extract the company into fields.company. "
                "Extract one or more roles into fields.roles. "
                "Extract only the role value. Do not include connector words such as \"for\". "
                "Do not include the label word \"role\" inside the role value unless it is genuinely part of the title. "
                "For list-valued fields such as roles and employment_types, always emit JSON arrays. "
                "Examples: "
                "\"I want to add an application Neilsoft\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Add a Neilsoft application\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft sathi application add kar\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"AI Engineer role for Neilsoft\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"roles\":[\"AI Engineer\"]},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Role at Neilsoft for AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"roles\":[\"AI Engineer\"]},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"At Neilsoft, role is AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"roles\":[\"AI Engineer\"]},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"For Neilsoft, set role to AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"roles\":[\"AI Engineer\"]},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft sathi AI Engineer role\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"roles\":[\"AI Engineer\"]},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"AI Engineer role\" with active unsaved draft -> patch_active_draft({\"fields\":{\"roles\":[\"AI Engineer\"]},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"fulltime ani onsite\" with active unsaved draft -> patch_active_draft({\"fields\":{\"employment_types\":[\"Full Time\"],\"location\":\"onsite\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft high priority kar\" -> preview_existing_application_update({\"target\":{\"company\":\"Neilsoft\"},\"fields\":{\"priority\":\"HIGH\"},\"replace_explicit_fields\":true}). "
                "\"Update Neilsoft priority to high\" -> preview_existing_application_update({\"target\":{\"company\":\"Neilsoft\"},\"fields\":{\"priority\":\"HIGH\"},\"replace_explicit_fields\":true}). "
                "\"Make it high priority\" without explicit company and without selected persisted row -> ask_clarification({\"question\":\"Which company's application do you mean?\"}). "
                "\"Add application\" without company and without active draft -> ask_clarification({\"question\":\"Which company should I use?\"}). "
                "\"save it\" with active draft -> request_draft_save({}). "
                "\"save it\" without active draft -> ask_clarification({\"question\":\"There is no active draft to save.\"})."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "transcript": transcript,
                    "session_context": normalized_context,
                    "explicit_known_companies_in_current_utterance": explicit_known_companies if isinstance(explicit_known_companies, list) else [],
                    "retry_hint": retry_hint if isinstance(retry_hint, str) else None,
                    "schema_repair_retry_hint": schema_repair_retry_hint if isinstance(schema_repair_retry_hint, str) else None,
                    "instruction": "Select exactly one backend tool call for this utterance.",
                },
                ensure_ascii=False,
            ),
        },
    ]


def build_ollama_tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "patch_active_draft",
                "description": (
                    "Use this when the user provides any partial or complete information for a new or currently unsaved application draft. "
                    "Partial drafts are valid. Do not ask for missing fields immediately. Patch only fields explicitly mentioned by the user. "
                    "A company name may appear anywhere in the sentence, and explicit known companies from the current utterance should be used as company when exactly one is present. "
                    "A company-like name near words such as application, company, opening, job, role, apply, track, or add should normally be treated as company. "
                    "Company and role may appear in any natural order. Extract company into fields.company and one or more roles into fields.roles. "
                    "Extract only the role value and do not include connector words like \"for\" or the label word \"role\" unless it is genuinely part of the title. "
                    "For list-valued fields such as roles and employment_types, always emit JSON arrays. "
                    "Unknown company names are valid during draft creation."
                ),
                "parameters": PatchActiveDraftArguments.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "preview_existing_application_update",
                "description": (
                    "Use this only for an already persisted application. "
                    "Require either explicit company from the user utterance or an explicitly selected persisted application_id from UI context. "
                    "When exactly one explicit known company is listed for the current utterance, use it as the company target. "
                    "Do not infer a persisted-row target from active draft context, recent actions, or vague pronouns alone."
                ),
                "parameters": PreviewExistingApplicationUpdateArguments.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_draft_save",
                "description": "Use this only when there is an active unsaved draft and the user wants to save it through the normal UI flow.",
                "parameters": RequestDraftSaveArguments.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "attach_latest_browser_context",
                "description": "Use this only to attach the latest captured browser link to the active unsaved draft. Never use it to target a persisted row.",
                "parameters": AttachLatestBrowserContextArguments.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_clarification",
                "description": (
                    "Use this only when clarification is genuinely needed. "
                    "Good examples: missing company for a new draft, missing persisted-row target for an update, no active draft to save, or multiple existing matches."
                ),
                "parameters": AskClarificationArguments.model_json_schema(),
            },
        },
    ]


def _parse_tool_arguments(raw_arguments: object) -> dict[str, object]:
    if isinstance(raw_arguments, str):
        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise SemanticInterpreterInvalidResponseError(
                "Local language interpreter returned malformed tool arguments. No tracker changes were saved."
            ) from exc
        if not isinstance(decoded, dict):
            raise SemanticInterpreterInvalidResponseError(
                "Local language interpreter returned malformed tool arguments. No tracker changes were saved."
            )
        return decoded
    if isinstance(raw_arguments, dict):
        return raw_arguments
    raise SemanticInterpreterInvalidResponseError(
        "Local language interpreter returned malformed tool arguments. No tracker changes were saved."
    )


def _extract_tool_call(payload: dict[str, object]) -> SemanticToolCallProposal:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise SemanticInterpreterInvalidResponseError("Local language interpreter returned no tool call. No tracker changes were saved.")

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) == 0:
        raise SemanticInterpreterInvalidResponseError("Local language interpreter returned no tool call. No tracker changes were saved.")
    if len(tool_calls) != 1:
        raise SemanticInterpreterInvalidResponseError("Local language interpreter returned multiple tool calls. No tracker changes were saved.")

    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise SemanticInterpreterInvalidResponseError("Local language interpreter returned an invalid tool call. No tracker changes were saved.")

    function_payload = tool_call.get("function")
    if not isinstance(function_payload, dict):
        raise SemanticInterpreterInvalidResponseError("Local language interpreter returned an invalid tool call. No tracker changes were saved.")

    name = function_payload.get("name")
    if not isinstance(name, str) or name not in TOOL_ARGUMENT_MODELS:
        raise SemanticInterpreterInvalidResponseError("Local language interpreter returned an unknown tool call. No tracker changes were saved.")

    arguments = _parse_tool_arguments(function_payload.get("arguments", {}))
    return SemanticToolCallProposal(tool_name=name, arguments=arguments)


class OllamaSemanticInterpreter:
    def __init__(self, settings: OllamaSettings | None = None) -> None:
        self.settings = settings or get_ollama_settings()

    def health_check(self) -> dict[str, str]:
        try:
            response = httpx.get(f"{self.settings.base_url}/api/tags", timeout=self.settings.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            if any(model.get("name") == self.settings.model for model in models):
                return {"status": "ok", "provider": "ollama", "model": self.settings.model, "mode": "tool_calling"}
        except Exception:
            pass
        return {"status": "unavailable", "provider": "ollama", "model": self.settings.model, "mode": "tool_calling"}

    def interpret(self, transcript: str, context: dict[str, object] | None = None) -> SemanticInterpretationResult:
        request_payload = {
            "model": self.settings.model,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "messages": build_ollama_messages(transcript, context),
            "tools": build_ollama_tools(),
            "options": {"temperature": 0},
        }

        started_at = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.settings.base_url}/api/chat",
                json=request_payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SemanticInterpreterUnavailableError("Local language interpreter timed out. No tracker changes were saved.") from exc
        except httpx.HTTPError as exc:
            raise SemanticInterpreterUnavailableError("Local language interpreter is unavailable. No tracker changes were saved.") from exc

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        payload = response.json()
        proposal = _extract_tool_call(payload)
        metrics = SemanticInterpreterMetrics(
            latency_ms=latency_ms,
            total_duration_ns=payload.get("total_duration"),
            load_duration_ns=payload.get("load_duration"),
            prompt_eval_duration_ns=payload.get("prompt_eval_duration"),
            eval_duration_ns=payload.get("eval_duration"),
        )
        return SemanticInterpretationResult(proposal=proposal, metrics=metrics)


def get_semantic_interpreter() -> OllamaSemanticInterpreter:
    return OllamaSemanticInterpreter()
