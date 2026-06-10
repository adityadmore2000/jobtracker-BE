import json
import logging
import time
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from .database_config import get_optional_env_value
from .semantic_schemas import (
    ArchiveApplicationArguments,
    AskClarificationArguments,
    AttachLatestBrowserContextArguments,
    ExplainDeletePolicyArguments,
    PatchActiveDraftArguments,
    PreviewExistingApplicationTarget,
    PreviewExistingApplicationUpdateArguments,
    RequestDraftSaveArguments,
    SemanticExtractedFields,
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
    "archive_application": ArchiveApplicationArguments,
    "explain_delete_policy": ExplainDeletePolicyArguments,
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
    extracted_fields: SemanticExtractedFields = field(default_factory=SemanticExtractedFields)


logger = logging.getLogger(__name__)


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
    normalized_extracted_fields = normalized_context.get("normalized_extracted_fields")
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
                "Use normalized_extracted_fields as the authoritative set of fields explicitly stated in the current utterance. "
                "Do not rewrite, paraphrase, or reinterpret extracted field values. "
                "Do not invent fields that are not present in normalized_extracted_fields. "
                "For a field-patch tool, it is acceptable to omit patch fields from your proposal because the backend will attach normalized extracted fields deterministically. "
                "Recent actions may provide context but must not authorize persisted-row mutation. "
                "Saved-row mutation requires explicit company in the current utterance or an explicitly selected persisted row ID. "
                "Use request_draft_save only when there is an active unsaved draft. "
                "Use attach_latest_browser_context only for the active unsaved draft and never to target a persisted row. "
                "Use ask_clarification only when genuinely needed. "
                "Company and other fields may appear in any natural order. "
                "Emit canonical field names such as company, role, employment_types, location, status, priority, current_stages, comments, and next_action. "
                "Extract the company into fields.company. "
                "Extract the role into fields.role as a single string (not an array). "
                "Role titles are open-ended free-form text. Extract any non-empty role title exactly as stated. "
                "Examples of valid roles: 'AI Engineer', 'founding engineer', 'LLM Inference Optimization Engineer', 'Head of Product'. "
                "These examples are illustrative only — any non-empty role string is valid. "
                "Do not reject a role because it is unfamiliar or not in any list. "
                "For list-valued fields such as employment_types and current_stages, always emit JSON arrays. "
                "The role field is a single scalar string, not an array. Do not wrap it in brackets. "
                "Extract only the value. Do not include connector words such as \"for\", \"set\", or \"to\" inside values. "
                "Do not include label words like \"role\", \"priority\", or \"stage\" inside values unless they are genuinely part of the value itself. "
                "Use only explicitly stated values. "
                "Examples: "
                "\"I want to add an application Neilsoft\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Add a Neilsoft application\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft sathi application add kar\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"AI Engineer role for Neilsoft\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Role at Neilsoft for AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"At Neilsoft, role is AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"For Neilsoft, set role to AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft sathi AI Engineer role\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft sathi AI Engineer role, fulltime onsite, high priority\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\",\"employment_types\":[\"Full Time\"],\"location\":\"onsite\",\"priority\":\"HIGH\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"For Neilsoft, set role to AI Engineer and status applied\" -> preview_existing_application_update({\"target\":{\"company\":\"Neilsoft\"},\"fields\":{\"role\":\"AI Engineer\",\"status\":\"Applied\"},\"replace_explicit_fields\":true}). "
                "\"Add Neilsoft as an internship application, remote, medium priority\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"employment_types\":[\"Internship\"],\"location\":\"remote\",\"priority\":\"MEDIUM\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Set Neilsoft current stage to Applied and Engaged\" -> preview_existing_application_update({\"target\":{\"company\":\"Neilsoft\"},\"fields\":{\"current_stages\":[\"Applied\",\"Engaged\"]},\"replace_explicit_fields\":true}). "
                "\"At Neilsoft, role is Applied AI Engineer\" -> patch_active_draft({\"fields\":{\"company\":\"Neilsoft\",\"role\":\"Applied AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"AI Engineer role\" with active unsaved draft -> patch_active_draft({\"fields\":{\"role\":\"AI Engineer\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"fulltime ani onsite\" with active unsaved draft -> patch_active_draft({\"fields\":{\"employment_types\":[\"Full Time\"],\"location\":\"onsite\"},\"replace_explicit_fields\":true,\"context_notes\":[]}). "
                "\"Neilsoft high priority kar\" -> preview_existing_application_update({\"target\":{\"company\":\"Neilsoft\"},\"fields\":{\"priority\":\"HIGH\"},\"replace_explicit_fields\":true}). "
                "\"Update Neilsoft priority to high\" -> preview_existing_application_update({\"target\":{\"company\":\"Neilsoft\"},\"fields\":{\"priority\":\"HIGH\"},\"replace_explicit_fields\":true}). "
                "\"Make it high priority\" without explicit company and without selected persisted row -> ask_clarification({\"question\":\"Which company's application do you mean?\"}). "
                "\"Add application\" without company and without active draft -> ask_clarification({\"question\":\"Which company should I use?\"}). "
                "\"save it\" with active draft -> request_draft_save({}). "
                "\"save it\" without active draft -> ask_clarification({\"question\":\"There is no active draft to save.\"}). "
                "\"archive Google AI Engineer application\" -> archive_application({\"target\":{\"company\":\"Google\",\"role\":\"AI Engineer\"}}). "
                "\"archieve Google application having AI Engineer role\" -> archive_application({\"target\":{\"company\":\"Google\",\"role\":\"AI Engineer\"}}). "
                "\"remove Neilsoft from active list\" -> archive_application({\"target\":{\"company\":\"Neilsoft\"}}). "
                "\"delete Google application for AI Engineer role\" -> explain_delete_policy({\"target\":{\"company\":\"Google\",\"role\":\"AI Engineer\"}}). "
                "\"permanently delete Neilsoft AI Engineer\" -> explain_delete_policy({\"target\":{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"}})."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "transcript": transcript,
                    "session_context": normalized_context,
                    "explicit_known_companies_in_current_utterance": explicit_known_companies if isinstance(explicit_known_companies, list) else [],
                    "normalized_extracted_fields": normalized_extracted_fields if isinstance(normalized_extracted_fields, dict) else {},
                    "retry_hint": retry_hint if isinstance(retry_hint, str) else None,
                    "schema_repair_retry_hint": schema_repair_retry_hint if isinstance(schema_repair_retry_hint, str) else None,
                    "instruction": "Select exactly one backend tool call for this utterance.",
                },
                ensure_ascii=False,
            ),
        },
    ]


def build_field_extraction_messages(transcript: str, context: dict[str, object] | None) -> list[dict[str, str]]:
    normalized_context = context or {}
    explicit_known_companies = normalized_context.get("explicit_known_companies")
    extraction_retry_hint = normalized_context.get("field_extraction_retry_hint")
    return [
        {
            "role": "system",
            "content": (
                "You extract explicitly stated job-tracker fields from the current utterance. "
                "Return only valid JSON matching the extraction schema. "
                "Do not select a tool. "
                "Do not infer missing values. "
                "Do not use recent history to invent fields. "
                "Do not rewrite free-form meaning. "
                "Ignore conversational filler. "
                "Company and field values may appear anywhere in natural language. "
                "The role field is free-form open-ended text emitted as a single string (not an array). "
                "Any non-empty role title is valid: 'founding engineer', 'LLM Inference Optimization Engineer', 'Head of Product' are all valid. "
                "Do not reject or omit a role because it is unfamiliar or not in any list. "
                "Emit arrays for employment_types and current_stages. "
                "Do not include filler or connector words inside extracted values. "
                "Extract only information explicitly stated in the current utterance. "
                "Examples: "
                "\"I have previously worked at this company. It's called Neilsoft. I'd like to track AI Engineer application for this position.\" -> "
                "{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\"}. "
                "\"Neilsoft sathi fulltime AI Engineer application onsite thev\" -> "
                "{\"company\":\"Neilsoft\",\"role\":\"AI Engineer\",\"employment_types\":[\"fulltime\"],\"location\":\"onsite\"}. "
                "\"Set priority high and current stages Applied and Engaged\" -> "
                "{\"priority\":\"high\",\"current_stages\":[\"Applied\",\"Engaged\"]}. "
                "\"Add a note saying I previously worked here\" -> "
                "{\"comments\":\"I previously worked here\"}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "transcript": transcript,
                    "explicit_known_companies_in_current_utterance": explicit_known_companies if isinstance(explicit_known_companies, list) else [],
                    "field_extraction_retry_hint": extraction_retry_hint if isinstance(extraction_retry_hint, str) else None,
                    "instruction": "Extract only explicitly stated tracker fields as JSON.",
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
                    "Company and other fields may appear in any natural order. Extract company into fields.company and the role into fields.role as a single string. "
                    "Emit canonical field names. For list-valued fields such as employment_types and current_stages, always emit JSON arrays. "
                    "The role field is a single scalar string — do not wrap it in an array. "
                    "Extract only the value and do not include connector words like \"for\" or label words like \"role\" unless they are genuinely part of the value. "
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
        {
            "type": "function",
            "function": {
                "name": "archive_application",
                "description": (
                    "Use this when the user wants to archive, soft-delete, hide, or remove an application from the active list. "
                    "Examples: 'archive Google AI Engineer application', 'archieve Google application having AI Engineer role' (minor typos are fine), "
                    "'move Neilsoft AI Engineer to archive', 'hide this application', 'remove from active list'. "
                    "Use target.company and target.role to identify which application to archive. "
                    "Do NOT use this for permanent deletion — use explain_delete_policy for that."
                ),
                "parameters": ArchiveApplicationArguments.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_delete_policy",
                "description": (
                    "Use this when the user wants to permanently delete or destroy an application. "
                    "Examples: 'delete Google AI Engineer application', 'permanently delete Neilsoft AI Engineer', 'remove it permanently'. "
                    "This tool explains the delete policy (archive first, then use the archived view). "
                    "It does NOT execute a permanent delete — that is UI-only."
                ),
                "parameters": ExplainDeletePolicyArguments.model_json_schema(),
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


def _extract_json_message_content(payload: dict[str, object]) -> dict[str, object]:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned invalid extracted fields. No tracker changes were saved."
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned invalid extracted fields. No tracker changes were saved."
        )
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned invalid extracted fields. No tracker changes were saved."
        ) from exc
    if not isinstance(decoded, dict):
        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned invalid extracted fields. No tracker changes were saved."
        )
    return decoded


def _combine_duration(first: int | None, second: int | None) -> int | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


def _merge_metrics(first: SemanticInterpreterMetrics, second: SemanticInterpreterMetrics) -> SemanticInterpreterMetrics:
    return SemanticInterpreterMetrics(
        latency_ms=first.latency_ms + second.latency_ms,
        total_duration_ns=_combine_duration(first.total_duration_ns, second.total_duration_ns),
        load_duration_ns=_combine_duration(first.load_duration_ns, second.load_duration_ns),
        prompt_eval_duration_ns=_combine_duration(first.prompt_eval_duration_ns, second.prompt_eval_duration_ns),
        eval_duration_ns=_combine_duration(first.eval_duration_ns, second.eval_duration_ns),
    )


def _safe_extracted_field_log(fields: dict[str, object]) -> dict[str, object]:
    safe_payload: dict[str, object] = {}
    for key, value in fields.items():
        if key in {"comments", "next_action"}:
            safe_payload[f"{key}_present"] = isinstance(value, str) and bool(value)
        else:
            safe_payload[key] = value
    return safe_payload


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

    def _chat(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        response_format: object | None = None,
    ) -> tuple[dict[str, object], SemanticInterpreterMetrics]:
        request_payload = {
            "model": self.settings.model,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "messages": messages,
            "options": {"temperature": 0},
        }
        if tools is not None:
            request_payload["tools"] = tools
        if response_format is not None:
            request_payload["format"] = response_format

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
        return payload, SemanticInterpreterMetrics(
            latency_ms=latency_ms,
            total_duration_ns=payload.get("total_duration"),
            load_duration_ns=payload.get("load_duration"),
            prompt_eval_duration_ns=payload.get("prompt_eval_duration"),
            eval_duration_ns=payload.get("eval_duration"),
        )

    def extract_fields(self, transcript: str, context: dict[str, object] | None = None) -> tuple[SemanticExtractedFields, SemanticInterpreterMetrics]:
        logger.info("semantic_field_extraction_start transcript_chars=%s", len(transcript))
        extraction_context = context or {}
        attempts = 0
        max_attempts = 2
        last_error: Exception | None = None

        while attempts < max_attempts:
            payload, metrics = self._chat(
                messages=build_field_extraction_messages(transcript, extraction_context),
                response_format=SemanticExtractedFields.model_json_schema(),
            )
            try:
                extracted_fields = SemanticExtractedFields.model_validate(_extract_json_message_content(payload))
            except (ValidationError, SemanticInterpreterInvalidResponseError) as exc:
                last_error = exc
                attempts += 1
                logger.warning(
                    "semantic_field_extraction_failure attempt=%s reason=%r",
                    attempts,
                    exc,
                )
                if attempts >= max_attempts:
                    break
                extraction_context = extraction_context | {
                    "field_extraction_retry_hint": (
                        "Your previous field-extraction JSON was invalid. "
                        "Return only valid JSON matching the extraction schema. "
                        "Do not add fields that were not explicitly stated."
                    )
                }
                logger.info("semantic_field_extraction_retry attempt=%s", attempts)
                continue

            logger.info(
                "semantic_field_extraction_success keys=%s fields=%r",
                sorted(extracted_fields.model_dump(exclude_none=True).keys()),
                _safe_extracted_field_log(extracted_fields.model_dump(exclude_none=True)),
            )
            return extracted_fields, metrics

        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned invalid extracted fields. No tracker changes were saved."
        ) from last_error

    def select_tool(self, transcript: str, context: dict[str, object] | None = None) -> SemanticInterpretationResult:
        logger.info("semantic_tool_selection_start transcript_chars=%s", len(transcript))
        payload, metrics = self._chat(
            messages=build_ollama_messages(transcript, context),
            tools=build_ollama_tools(),
        )
        proposal = _extract_tool_call(payload)
        logger.info("semantic_tool_selection_success tool=%s", proposal.tool_name)
        return SemanticInterpretationResult(proposal=proposal, metrics=metrics)

    def interpret(self, transcript: str, context: dict[str, object] | None = None) -> SemanticInterpretationResult:
        extracted_fields, extraction_metrics = self.extract_fields(transcript, context)
        selection_context = (context or {}) | {
            "normalized_extracted_fields": extracted_fields.model_dump(exclude_none=True),
        }
        selection_result = self.select_tool(transcript, selection_context)
        return SemanticInterpretationResult(
            proposal=selection_result.proposal,
            metrics=_merge_metrics(extraction_metrics, selection_result.metrics),
            extracted_fields=extracted_fields,
        )


def get_semantic_interpreter() -> OllamaSemanticInterpreter:
    return OllamaSemanticInterpreter()
