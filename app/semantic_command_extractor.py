"""
Single-call semantic command extractor.

This module makes EXACTLY ONE Ollama call per transcript and returns a strict
structured ``SemanticCommand`` envelope (or raises a typed error). It is fully
isolated from the legacy dual-output interpreter in semantic_interpreter.py:

  * No Ollama tool-calling — structured JSON output only (``format`` = JSON schema).
  * No ``extract_fields`` / ``select_tool`` / merge / fallback-routing pipeline.
  * The LLM understands language; the backend (pipeline.py) controls mutations.

The extractor's job is purely *understanding*: turn natural language into the
strict envelope. All routing, validation, alias normalization, target resolution
and dispatch happen deterministically downstream.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from .semantic_command_schemas import SemanticCommand
from .semantic_interpreter import OllamaSettings, get_ollama_settings

logger = logging.getLogger(__name__)


class SemanticExtractorError(RuntimeError):
    """Base class for all single-call extractor failures."""


class SemanticExtractorUnavailableError(SemanticExtractorError):
    """Ollama could not be reached or timed out."""


class SemanticExtractorInvalidResponseError(SemanticExtractorError):
    """Ollama returned malformed / non-conforming output."""


@dataclass(frozen=True)
class ExtractionMetrics:
    latency_ms: int


_SYSTEM_PROMPT = (
    "You convert one natural-language message from a job-application tracker into a single "
    "strict JSON object. You only understand and structure language. You never decide what to "
    "persist; a deterministic backend does that. Return ONLY the JSON object, no prose.\n"
    "\n"
    "Output schema (every key must be present; use null when not applicable):\n"
    "{\n"
    '  "intent": one of "create_application" | "update_application" | "append_note" | '
    '"archive_application" | "unsupported",\n'
    '  "target": {"company": str|null, "role": str|null, "application_id": int|null},\n'
    '  "changes": {"status": str|null, "priority": str|null, "location_mode": str|null, '
    '"employment_types": [str]|null, "current_stages": [str]|null, "job_link": str|null, '
    '"engaged_days": int|null, "next_action": str|null, "comments": str|null},\n'
    '  "note": str|null,\n'
    '  "clarification": str|null,\n'
    '  "suggested_phrasings": [str]|null\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- target carries identity only (company / role). NEVER put company or role inside changes.\n"
    "- changes carries mutable fields only.\n"
    "- A free-form note belongs ONLY in note. NEVER copy note text into role, comments, "
    "next_action, or any changes field.\n"
    "- Do not invent values. Only emit fields explicitly stated in the message.\n"
    "- Role is open-ended free text (a single string, never an array). Any non-empty role title "
    "is valid; do not reject unfamiliar roles.\n"
    "- employment_types and current_stages are always JSON arrays.\n"
    "- Keep field labels out of values (no 'role:', 'priority:', 'set', 'to' inside a value).\n"
    "\n"
    "Intent selection:\n"
    "- 'I applied for X role at Y' / 'I have applied for X at Y' / 'add application for X at Y' "
    "→ intent=create_application, target.company=Y, target.role=X, and (for 'applied' phrasing) "
    "changes.status='applied'.\n"
    "- 'make it onsite, full-time, priority high' / 'set priority to medium and location hybrid' "
    "→ intent=update_application with the stated changes; leave target null when the message does "
    "not name a company/role (the backend resolves context).\n"
    "- 'For <Company> <Role> application, set priority to medium, location hybrid' → "
    "intent=update_application, target.company/role set, changes populated.\n"
    "- 'add a note saying ...' / 'note that ...' / 'I connected with their recruiter, add that as a "
    "note' → intent=append_note, note=<the prose>, changes ALL null.\n"
    "- 'archive <Company> <Role>' / 'remove from active list' → intent=archive_application, "
    "target set, changes all null, note null.\n"
    "- If the message mixes a field update AND a note (e.g. 'set priority to medium and add a note "
    "saying recruiter replied'), still extract BOTH the changes and the note honestly into the same "
    "object. The backend will detect the conflict and refuse safely — do not drop either part.\n"
    "- If you cannot identify a concrete tracker action, use intent='unsupported' and, when you can "
    "guess the user's likely goal, fill suggested_phrasings with 1-3 short rephrasings such as "
    '"set priority of <Company> to medium".\n'
    "\n"
    "Value vocabulary (normalize toward these; the backend will canonicalize further):\n"
    "- priority: LOW | MEDIUM | HIGH\n"
    "- location_mode: remote | hybrid | on-site\n"
    "- employment_types items: Internship | Full Time | Part Time\n"
    "- status: in_touch | applied | accepted | rejected\n"
    "- current_stages items: Tailored | Applied | Networked | Engaged | COLD_MAIL | Followed up\n"
)


def _build_user_message(transcript: str, read_only_context: dict | None) -> str:
    """Compact, advisory context. IDs here are NEVER trusted; the backend verifies."""
    context = read_only_context or {}
    safe_context = {
        "active_draft": context.get("active_draft"),
        "active_application": context.get("active_application"),
        "known_applications": context.get("known_applications"),
    }
    return json.dumps(
        {
            "message": transcript,
            "context": safe_context,
            "instruction": (
                "Return one JSON object matching the schema for this message. "
                "Context is advisory only; do not invent identities it does not contain."
            ),
        },
        ensure_ascii=False,
    )


def _post_chat(settings: OllamaSettings, transcript: str, read_only_context: dict | None) -> tuple[dict, int]:
    request_payload = {
        "model": settings.model,
        "stream": False,
        "keep_alive": settings.keep_alive,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(transcript, read_only_context)},
        ],
    }
    started_at = time.perf_counter()
    try:
        response = httpx.post(
            f"{settings.base_url}/api/chat",
            json=request_payload,
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise SemanticExtractorUnavailableError("Semantic extractor timed out. No tracker changes were saved.") from exc
    except httpx.HTTPError as exc:
        raise SemanticExtractorUnavailableError("Semantic extractor is unavailable. No tracker changes were saved.") from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return response.json(), latency_ms


def _parse_message_content(payload: dict) -> dict:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise SemanticExtractorInvalidResponseError("Semantic extractor returned no message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise SemanticExtractorInvalidResponseError("Semantic extractor returned empty content.")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SemanticExtractorInvalidResponseError("Semantic extractor returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise SemanticExtractorInvalidResponseError("Semantic extractor returned a non-object JSON value.")
    return decoded


def extract_semantic_command_once(
    transcript: str,
    read_only_context: dict | None = None,
    *,
    settings: OllamaSettings | None = None,
) -> tuple[SemanticCommand, ExtractionMetrics]:
    """Make ONE Ollama JSON call and return a validated ``SemanticCommand``.

    Raises:
        SemanticExtractorUnavailableError — Ollama unreachable / timed out.
        SemanticExtractorInvalidResponseError — malformed JSON or schema violation.
    """
    resolved_settings = settings or get_ollama_settings()
    logger.info("semantic_single_extractor_invoked transcript=%r", transcript)

    payload, latency_ms = _post_chat(resolved_settings, transcript, read_only_context)
    raw = _parse_message_content(payload)
    logger.info("semantic_single_extractor_raw_output=%s", json.dumps(raw, ensure_ascii=False)[:2000])

    try:
        command = SemanticCommand.model_validate(raw)
    except ValidationError as exc:
        logger.warning("semantic_single_extractor_rejected reason=schema_invalid detail=%r", exc.errors())
        raise SemanticExtractorInvalidResponseError("Semantic extractor output did not match the schema.") from exc

    return command, ExtractionMetrics(latency_ms=latency_ms)
