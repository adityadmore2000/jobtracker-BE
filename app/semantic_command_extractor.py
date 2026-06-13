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
    "CRITICAL: intent MUST be exactly one of the five literals "
    "create_application | update_application | append_note | archive_application | unsupported. "
    "NEVER put a status, priority, or any other value in intent (e.g. intent is never 'applied', "
    "'accepted', 'rejected', or 'high'). A status like 'accepted' goes in changes.status, not intent.\n"
    "\n"
    "Understand intent from MEANING, not keywords. People phrase the same action many ways — direct "
    "('set status to rejected'), indirect ('they turned me down'), casual ('got the offer!'), or "
    "incomplete. Map the underlying action regardless of phrasing.\n"
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
    "- 'For <Company> <Role> application, set status to rejected' → intent=update_application, "
    "target.company=<Company>, target.role=<Role>, changes.status='rejected'. Any 'set status to "
    "<value>' is ALWAYS update_application, even when the status is 'rejected' and even when a role "
    "noun phrase like 'data scientist application' appears in the sentence.\n"
    "- 'add a note saying ...' / 'note that ...' / 'I connected with their recruiter, add that as a "
    "note' → intent=append_note, note=<the prose>, changes ALL null.\n"
    "- 'archive <Company> <Role>' / 'remove from active list' / 'I'm not pursuing them anymore' → "
    "intent=archive_application, target set, changes all null, note null. archive_application is ONLY "
    "for explicitly removing/hiding an application from the active list. A rejection or a 'rejected' "
    "status is NOT an archive — it is intent=update_application with changes.status='rejected'. "
    "'set status to rejected', 'mark <Company> as rejected', 'change status to rejected' → "
    "intent=update_application, changes.status='rejected' (NEVER archive_application). Only choose "
    "archive_application when the user literally says archive / remove from list / hide / take it off.\n"
    "- If the message mixes a field update AND a note (e.g. 'set priority to medium and add a note "
    "saying recruiter replied'), still extract BOTH the changes and the note honestly into the same "
    "object. The backend will detect the conflict and refuse safely — do not drop either part.\n"
    "- If you cannot identify a concrete tracker action, use intent='unsupported' and, when you can "
    "guess the user's likely goal, fill suggested_phrasings with 1-3 short rephrasings such as "
    '"set priority of <Company> to medium".\n'
    "\n"
    "Indirect / casual status & lifecycle phrasings (these are real commands, NOT 'unsupported'):\n"
    "- 'I applied to <Company>' / 'I think I applied to <Company> last week' / 'just applied at "
    "<Company>' → if the message reads like first-time tracking, intent=create_application with "
    "target.company set and changes.status='applied'; if it clearly refers to an application already "
    "being tracked, intent=update_application with changes.status='applied'. When unsure between the "
    "two for an 'I applied to <Company>' phrasing, prefer create_application.\n"
    "- '<Company> turned me down' / '<Company> rejected me' / 'didn't get <Company>' / 'no luck with "
    "<Company>' → intent=update_application, target.company=<Company>, changes.status='rejected'.\n"
    "- 'got an offer from <Company>' / 'got accepted at <Company>' / '<Company> accepted me' / "
    "'I'm in!' about a company → intent=update_application, target.company=<Company>, "
    "changes.status='accepted'.\n"
    "- '<Company> got back to me' / 'I'm in touch with <Company>' / 'heard back from <Company>' "
    "(stating it happened, not asking) → intent=update_application, target.company=<Company>, "
    "changes.status='in_touch'.\n"
    "- '<Company> is remote' / 'the <Company> role is remote by the way' → intent=update_application, "
    "target.company=<Company>, changes.location_mode='remote' (or hybrid / on-site).\n"
    "- 'the <Company> one should be higher priority' / 'bump <Company>' / 'make <Company> a priority' "
    "→ intent=update_application, target.company=<Company>, changes.priority='HIGH'.\n"
    "- 'been meaning to add the <Company> <Role> role' / 'should track <Company> for <Role>' / "
    "'add the <Company> <Role>' → intent=create_application, target.company=<Company>, "
    "target.role=<Role>. ALWAYS pull the company name out of such sentences into target.company.\n"
    "\n"
    "Questions are NOT commands. If the user is ASKING for information rather than telling you to "
    "change something — 'how many applications do I have?', \"what's the status of my <Company> "
    "application?\", 'show me my applications', 'have I heard back from <Company>?', 'which ones are "
    "high priority?' — return intent='unsupported' with all changes null and no note. Never emit a "
    "mutating intent for a question. A sentence ending in '?' or starting with how/what/which/show/"
    "list/have/did is almost always a question.\n"
    "\n"
    "Value vocabulary (normalize toward these; the backend will canonicalize further):\n"
    "- priority: LOW | MEDIUM | HIGH\n"
    "- location_mode: remote | hybrid | on-site\n"
    "- employment_types items: Internship | Full Time | Part Time\n"
    "- status: in_touch | applied | accepted | rejected\n"
    "- current_stages items: Tailored | Applied | Networked | Engaged | COLD_MAIL | Followed up\n"
    "\n"
    "CONTEXT RESOLUTION RULES\n"
    "You will receive a Context block before the Command. Use it as follows:\n"
    "1. If the Command mentions a company name ANYWHERE — even indirectly like 'the Spotify "
    "one' or 'bump Acme' — ALWAYS extract that company into target.company. A company named in "
    "the Command always overrides the selected application. Never put a company name in role.\n"
    "2. ONLY when the Command names no company at all (e.g. 'add a note', 'set priority high', "
    "'mark as applied') and the Context shows a selected application, copy that selected "
    "application's company into target.company.\n"
    "3. If the Command names no company AND the Context shows no selected application, leave "
    "target.company null (the backend will ask which application to target).\n"
    "4. Never infer the target from anything other than the current Command text or the "
    "Context block. There is no conversation history; do not invent identities.\n"
)


def _format_context_block(read_only_context: dict | None) -> str:
    """Render the selected-application fact as an explicit, unambiguous prefix.

    The model is TOLD what is selected rather than having to infer it. IDs are
    never exposed here — the backend re-resolves and verifies every target.
    """
    context = read_only_context or {}
    active_application = context.get("active_application") or {}
    active_draft = context.get("active_draft")

    company = active_application.get("company") if isinstance(active_application, dict) else None
    role = active_application.get("role") if isinstance(active_application, dict) else None

    if company:
        selected = f"{company} — {role}" if role else str(company)
    else:
        selected = "none"

    draft_active = "yes" if active_draft else "no"

    return (
        "Context:\n"
        f"- Selected application: {selected}\n"
        f"- Draft active: {draft_active}\n"
    )


def _build_user_message(transcript: str, read_only_context: dict | None) -> str:
    """Structured context prefix + the current command.

    The selected application is injected as an explicit fact (see
    ``_format_context_block``) so the model resolves the target from the current
    Context block, not from any implicit history. The noisy full
    ``known_applications`` list is intentionally NOT included — dumping every
    company into the prompt let short/ambiguous names pattern-match the wrong
    application. IDs are never exposed; the backend re-resolves and verifies.
    """
    context_block = _format_context_block(read_only_context)
    return f"{context_block}\nCommand: {transcript}"


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


def _lift_misplaced_note(raw: dict) -> dict:
    """Lift a hallucinated ``changes.note`` up to the top-level ``note`` field.

    Small models sometimes emit ``{"changes": {"note": "..."}}`` even though
    ``note`` is not a valid ``changes`` key. ``extra="forbid"`` would otherwise
    reject the entire payload (dropping the command). Since ``note`` is a
    first-class top-level field, lifting it is strictly safe: we only move it
    when the top-level ``note`` is absent/empty, and we never overwrite it.
    """
    changes = raw.get("changes")
    if not isinstance(changes, dict) or "note" not in changes:
        return raw

    misplaced = changes.get("note")
    repaired_changes = {k: v for k, v in changes.items() if k != "note"}
    repaired = {**raw, "changes": repaired_changes}
    if not raw.get("note") and isinstance(misplaced, str) and misplaced.strip():
        repaired["note"] = misplaced
    return repaired


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

    raw = _lift_misplaced_note(raw)

    try:
        command = SemanticCommand.model_validate(raw)
    except ValidationError as exc:
        logger.warning("semantic_single_extractor_rejected reason=schema_invalid detail=%r", exc.errors())
        raise SemanticExtractorInvalidResponseError("Semantic extractor output did not match the schema.") from exc

    return command, ExtractionMetrics(latency_ms=latency_ms)
