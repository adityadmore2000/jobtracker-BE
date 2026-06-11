from __future__ import annotations

import logging
import re
from typing import Literal

from sqlalchemy.orm import Session
from pydantic import ValidationError

from .company_resolution import detect_explicit_known_companies, get_application_matches_for_company, resolve_company_name
from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    STATUS_OPTIONS,
    EMPLOYMENT_TYPE_ALIASES,
    LOCATION_ALIASES,
    normalize_status_value,
)
from .models import BrowserContext, JobApplication
from .fast_path_parser import ClarificationNeeded, ParseMiss, try_parse, try_parse_v2
from .mutation_dispatcher import dispatch
from .mutation_schemas import ApplicationChanges, MutationPayload, MutationResult, MutationTarget
from .schemas import JobApplicationCreate, SemanticTranscriptResponse, TranscriptParseRequest
from .semantic_interpreter import (
    OllamaSemanticInterpreter,
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterUnavailableError,
)
from .semantic_schemas import (
    ArchiveApplicationArguments,
    AskClarificationArguments,
    DiscardDraftArguments,
    ExplainDeletePolicyArguments,
    PatchActiveDraftArguments,
    PreviewExistingApplicationTarget,
    PreviewExistingApplicationUpdateArguments,
    RequestDraftSaveArguments,
    SemanticExtractedFields,
    SemanticFieldPatch,
    SemanticToolCallProposal,
)

MAX_RECENT_ACTIONS = 3

CLARIFICATION_MISSING_COMPANY = "Which company should I use?"

# Lifecycle verbs that must never be absorbed by the active-draft contextual patch fallback.
# Matched case-insensitively against the whole transcript.
_LIFECYCLE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b(save|submit)\b', re.IGNORECASE),
    re.compile(r'\b(discard|cancel|drop|remove|delete)\b', re.IGNORECASE),
    re.compile(r'\b(archive|restore)\b', re.IGNORECASE),
    re.compile(r'\bpermanently\s+delete\b', re.IGNORECASE),
]


def _has_lifecycle_intent(transcript: str) -> bool:
    """Return True when the transcript expresses a draft or app lifecycle intent."""
    return any(p.search(transcript) for p in _LIFECYCLE_INTENT_PATTERNS)


# Explicit note-command anchors. Only exact command-phrase anchors are listed —
# no fuzzy free-text matching. Used solely to block the active-draft contextual
# patch fallback from synthesising patch_draft with hallucinated note text.
# Negative lookahead on `for` excludes "add note for [company]" which is a
# saved-application comment command that belongs to the LLM pipeline.
_NOTE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r'\badd\s+a?\s*note\b(?!\s+for\b)', re.IGNORECASE),
    re.compile(r'\bappend\s+a?\s*note\b', re.IGNORECASE),
    re.compile(r'\bnote\s+that\b', re.IGNORECASE),
]


def _has_note_intent(transcript: str) -> bool:
    """Return True when the transcript contains an explicit note-command anchor."""
    return any(p.search(transcript) for p in _NOTE_INTENT_PATTERNS)


# Explicit create-intent cues — generic verb/noun phrases that signal a new application.
# Must win over saved-row update detection.
_EXPLICIT_CREATE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r'\badd\s+(?:an?\s+)?application\b', re.IGNORECASE),
    re.compile(r'\bcreate\s+(?:an?\s+)?application\b', re.IGNORECASE),
    re.compile(r'\bnew\s+application\b', re.IGNORECASE),
    re.compile(r'\bapplied\s+for\b', re.IGNORECASE),
    re.compile(r'\bapply\s+for\b', re.IGNORECASE),
    re.compile(r'\badd\s+(?:a\s+)?job\s+application\b', re.IGNORECASE),
    re.compile(r'\btrack\s+(?:an?\s+)?application\b', re.IGNORECASE),
    re.compile(r'\btrack\s+(?:this|my)\b', re.IGNORECASE),
]


def _has_explicit_create_intent(transcript: str) -> bool:
    """Return True when the transcript explicitly requests creating a new application draft."""
    return any(p.search(transcript) for p in _EXPLICIT_CREATE_INTENT_PATTERNS)


# Cues that indicate the user is asking to update an already-persisted saved row,
# not create or patch a new draft.  Conservative list — avoid false positives.
# NOTE: these must NOT fire when explicit create intent is also present.
_SAVED_UPDATE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r'\bupdate\s+status\s+of\b', re.IGNORECASE),
    re.compile(r'\bchange\s+status\s+of\b', re.IGNORECASE),
    re.compile(r'\bset\s+(?:priority|status|location)\s+of\b', re.IGNORECASE),
    re.compile(r'\bupdate\s+(?:priority|location|role|employment|stage)\s+of\b', re.IGNORECASE),
]


def _has_explicit_saved_update_intent(transcript: str) -> bool:
    """Return True when the transcript clearly targets an existing saved row for update.

    Explicit create intent always overrides saved-update intent.
    """
    if _has_explicit_create_intent(transcript):
        return False
    return any(p.search(transcript) for p in _SAVED_UPDATE_INTENT_PATTERNS)


def _fields_can_create_or_patch_draft(fields: "SemanticFieldPatch") -> bool:
    """Return True when validated fields contain enough to meaningfully create or patch a draft."""
    return fields_have_values(fields, allow_company=True)


def build_transcript_response_from_mutation(
    mutation_result: MutationResult,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    *,
    metrics=None,
    warnings: list[str] | None = None,
    draft: JobApplicationCreate | None = None,
    draft_id: str | None = None,
    application_id: int | None = None,
    needs_confirmation: bool = False,
    confirmation_kind: str = "none",
) -> SemanticTranscriptResponse:
    if mutation_result.clarification_question:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=warnings or [],
            clarification_question=mutation_result.clarification_question,
            interpreter_metrics=metrics,
        )
    if not mutation_result.success:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=(warnings or []) + [mutation_result.message],
            interpreter_metrics=metrics,
        )
    operation_map = {
        "create_draft": "create",
        "draft_updated": "create",  # reused existing draft
        "patch_draft": "create",
        "save_draft": "create",
        "patch_application": "update",
        "updated": "update",        # reapply semantics updated a saved row
        "no_change": "none",        # truthful no-op from reapply
        "discard_draft": "none",
        "ask_clarification": "none",
        "append_note": "none",
        "archive_application": "update",
        "restore_application": "update",
        "create_application_update_draft": "pending_changes",
        "patch_application_update_draft": "pending_changes",
        "apply_application_update_draft": "update",
        "discard_application_update_draft": "none",
    }
    op = operation_map.get(mutation_result.operation, "none")
    effective_draft = draft
    effective_draft_dict: dict | None = None
    if mutation_result.draft:
        effective_draft_dict = mutation_result.draft
        if effective_draft is None:
            try:
                effective_draft = JobApplicationCreate.model_validate(mutation_result.draft)
            except Exception:
                effective_draft = None
    # Extract draft_id from mutation result (from DB row) or use the one passed in
    effective_draft_id = draft_id
    if effective_draft_id is None and mutation_result.draft and isinstance(mutation_result.draft.get("id"), int):
        effective_draft_id = str(mutation_result.draft["id"])
    # Extract application_id from mutation result when not explicitly provided
    effective_application_id = application_id
    if effective_application_id is None and mutation_result.application and isinstance(mutation_result.application.get("id"), int):
        effective_application_id = mutation_result.application["id"]
    return SemanticTranscriptResponse(
        status="preview",
        operation=op,
        raw_transcript=payload.transcript,
        proposal=proposal,
        draft=effective_draft,
        draft_dict=effective_draft_dict,
        draft_id=effective_draft_id,
        change_draft=mutation_result.change_draft,
        warnings=warnings or [],
        needs_confirmation=needs_confirmation,
        confirmation_kind=confirmation_kind,
        application_id=effective_application_id,
        interpreter_metrics=metrics,
    )
CLARIFICATION_MISSING_PERSISTED_TARGET = "Which company's application do you mean?"
CLARIFICATION_AMBIGUOUS_APPLICATION = "Multiple applications match this company. Specify the role."
CLARIFICATION_NO_ACTIVE_DRAFT = "There is no active draft to save."
CLARIFICATION_CONFLICTING_COMPANY = "I found conflicting company names. Which company should I use?"
CLARIFICATION_MULTIPLE_EXPLICIT_COMPANIES = "I found multiple company names. Which company should I use?"
CLARIFICATION_RETRY_EXHAUSTED = "I could not interpret that reliably. Please rephrase your request."
logger = logging.getLogger(__name__)

_MISSING = object()
_INVALID = object()
_CONFLICT = object()


def empty_proposal() -> SemanticToolCallProposal:
    return SemanticToolCallProposal()


def unsupported_response(
    payload: TranscriptParseRequest,
    warnings: list[str],
    *,
    proposal: SemanticToolCallProposal | None = None,
    metrics=None,
) -> SemanticTranscriptResponse:
    return SemanticTranscriptResponse(
        status="unsupported",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal or empty_proposal(),
        warnings=warnings,
        interpreter_metrics=metrics,
    )


def build_context_payload(payload: TranscriptParseRequest) -> dict[str, object]:
    raw_context = payload.context or {}
    recent_actions = raw_context.get("recent_actions")
    bounded_recent_actions = recent_actions[-MAX_RECENT_ACTIONS:] if isinstance(recent_actions, list) else []
    active_application = raw_context.get("active_application")
    active_draft = raw_context.get("active_draft")
    draft_id_raw = raw_context.get("draft_id")
    draft_id = str(draft_id_raw) if draft_id_raw is not None else None
    return {
        "active_application_id": (
            active_application.get("application_id")
            if isinstance(active_application, dict)
            else raw_context.get("active_application_id")
        ),
        "active_draft": active_draft if isinstance(active_draft, dict) else None,
        "active_company": raw_context.get("active_company"),
        "active_role": raw_context.get("active_role"),
        "recent_actions": bounded_recent_actions,
        "draft_id": draft_id,
    }


def build_interpreter_context(
    db: Session,
    payload: TranscriptParseRequest,
) -> tuple[dict[str, object], list[str]]:
    context = build_context_payload(payload)
    explicit_known_companies = detect_explicit_known_companies(db, payload.transcript)
    return (
        context | {"explicit_known_companies": explicit_known_companies},
        explicit_known_companies,
    )


def normalize_status(value: str | None) -> str | None:
    if value is None:
        return None
    return normalize_status_value(value)


def normalize_priority(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_lookup_text(value)
    if normalized.endswith(" priority"):
        normalized = normalized[: -len(" priority")].strip()
    priority_aliases = {
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }
    canonical_priority = priority_aliases.get(normalized)
    if canonical_priority is not None:
        return canonical_priority
    normalized = value.strip().upper()
    return normalized if normalized in ALLOWED_PRIORITIES else None


def _normalize_lookup_text(value: str) -> str:
    return " ".join(value.replace("-", " ").replace("_", " ").strip().casefold().split())


def normalize_role_title(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized if normalized else None


def normalize_employment_types(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized_values: list[str] = []
    for value in values:
        canonical_value = EMPLOYMENT_TYPE_ALIASES.get(_normalize_lookup_text(value))
        if canonical_value not in ALLOWED_EMPLOYMENT_TYPES:
            return None
        if canonical_value not in normalized_values:
            normalized_values.append(canonical_value)
    return normalized_values


def normalize_location(value: str | None) -> str | None:
    if value is None:
        return None
    canonical_value = LOCATION_ALIASES.get(_normalize_lookup_text(value))
    if canonical_value not in ALLOWED_LOCATIONS:
        return None
    return canonical_value


def normalize_stages(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized_values: list[str] = []
    for value in values:
        normalized = _normalize_lookup_text(value)
        matched = None
        for option in ALLOWED_CURRENT_STAGES:
            if _normalize_lookup_text(option) == normalized:
                matched = option
                break
        if matched is None:
            return None
        if matched not in normalized_values:
            normalized_values.append(matched)
    return normalized_values


def validate_fields(fields: SemanticFieldPatch, *, tool_name: str | None = None) -> tuple[SemanticFieldPatch | None, list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    role: str | None = None
    if fields.role is not None:
        normalized_role = normalize_role_title(fields.role)
        if not normalized_role:
            errors.append("Unsupported role value.")
            logger.warning(
                "semantic_role_validation_failed tool=%s raw_role_value=%r reason=%r",
                tool_name,
                fields.role,
                "blank_role_value",
            )
        else:
            role = normalized_role

    employment_types = normalize_employment_types(fields.employment_types)
    if fields.employment_types is not None and employment_types is None:
        errors.append("Unsupported employment type value.")

    status_alias_as_type = EMPLOYMENT_TYPE_ALIASES.get(_normalize_lookup_text(fields.status)) if fields.status is not None else None

    location = normalize_location(fields.location)
    if fields.location is not None and location is None:
        errors.append("Unsupported location value.")

    status = normalize_status(fields.status)
    if fields.status is not None and status is None:
        if status_alias_as_type is not None and employment_types is not None and status_alias_as_type in employment_types:
            warnings.append(f'Interpreted "{fields.status}" as Employment Type, not Status.')
        else:
            errors.append("Unsupported status value.")

    current_stages = normalize_stages(fields.current_stages)
    if fields.current_stages is not None and current_stages is None:
        errors.append("Unsupported current stage value.")

    priority = normalize_priority(fields.priority)
    if fields.priority is not None and priority is None:
        errors.append("Unsupported priority value.")

    if errors:
        return None, errors

    return (
        SemanticFieldPatch(
            company=fields.company,
            role=role,
            employment_types=employment_types,
            job_link=fields.job_link,
            location=location,
            status=status,
            current_stages=current_stages,
            priority=priority,
            engaged_days=fields.engaged_days,
            next_action=fields.next_action,
            comments=fields.comments,
        ),
        warnings,
    )


def _reconcile_controlled_field_misclassification(fields: SemanticFieldPatch) -> SemanticFieldPatch:
    """Move a controlled value to the correct field when it was unambiguously misclassified.

    Only repairs when:
    - The value is invalid for the extracted field
    - Valid for exactly one other controlled field
    - The destination field is unset (or list is empty)
    Does not overwrite an already-populated destination.
    """
    result = fields.model_copy()

    # Check for a single employment_type item that is actually a valid location
    if fields.employment_types is not None and len(fields.employment_types) == 1:
        candidate = fields.employment_types[0]
        candidate_normalized = _normalize_lookup_text(candidate)
        # Invalid as employment type?
        if candidate_normalized not in EMPLOYMENT_TYPE_ALIASES:
            # Valid as location?
            canonical_location = LOCATION_ALIASES.get(candidate_normalized)
            if canonical_location is not None and canonical_location in ALLOWED_LOCATIONS:
                # Destination (location) unset or empty?
                if fields.location is None:
                    logger.info(
                        "semantic_field_reconciliation field=employment_types candidate=%r -> location=%r",
                        candidate,
                        canonical_location,
                    )
                    result = result.model_copy(update={"employment_types": None, "location": canonical_location})

    # Check for a location value that is actually a valid employment type
    if fields.location is not None:
        loc_normalized = _normalize_lookup_text(fields.location)
        if loc_normalized not in LOCATION_ALIASES:
            canonical_employment_type = EMPLOYMENT_TYPE_ALIASES.get(loc_normalized)
            if canonical_employment_type is not None and canonical_employment_type in ALLOWED_EMPLOYMENT_TYPES:
                if fields.employment_types is None or fields.employment_types == []:
                    logger.info(
                        "semantic_field_reconciliation field=location candidate=%r -> employment_types=%r",
                        fields.location,
                        canonical_employment_type,
                    )
                    result = result.model_copy(update={"location": None, "employment_types": [canonical_employment_type]})

    return result


_FIELD_CUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bstatus\b', re.IGNORECASE), "status"),
    (re.compile(r'\bpriority\b', re.IGNORECASE), "priority"),
    (re.compile(r'\blocation\b', re.IGNORECASE), "location"),
    (re.compile(r'\bemployment\s+type\b|\bfull\s+time\b|\bpart\s+time\b|\binternship\b', re.IGNORECASE), "employment_types"),
    (re.compile(r'\bcurrent\s+stage\b|\bstage\b', re.IGNORECASE), "current_stages"),
    (re.compile(r'\brole\b', re.IGNORECASE), "role"),
    (re.compile(r'\bcompany\b', re.IGNORECASE), "company"),
    (re.compile(r'\bnext\s+action\b', re.IGNORECASE), "next_action"),
    (re.compile(r'\bcomments?\b|\bnotes?\b', re.IGNORECASE), "comments"),
]


def detect_explicit_field_cues(transcript: str) -> set[str]:
    """Return the set of field names explicitly cued by keyword in the transcript."""
    cues: set[str] = set()
    for pattern, field_name in _FIELD_CUE_PATTERNS:
        if pattern.search(transcript):
            cues.add(field_name)
    return cues


def reconcile_wrong_field_placement(transcript: str, fields: SemanticFieldPatch) -> SemanticFieldPatch:
    """Use explicit field cues from transcript to repair wrong-field LLM extractions.

    If the transcript says 'status' and the extracted role value normalizes as a valid
    status value, and no status is already set, move it from role to status.

    Only moves when:
    1. Transcript explicitly names the destination field
    2. Source field's value normalizes validly for the destination field
    3. Destination field is unset
    4. Move is unambiguous (source has only that value)
    """
    cues = detect_explicit_field_cues(transcript)
    result = fields

    # Rule: if "status" is cued and role contains a status-like value
    if "status" in cues and result.role is not None and result.status is None:
        candidate = result.role
        canonical_status = normalize_status_value(candidate) or normalize_status_value(_normalize_lookup_text(candidate))
        if canonical_status is not None:
            logger.info(
                "semantic_wrong_field_reconciliation source=role dest=status candidate=%r canonical=%r",
                candidate,
                canonical_status,
            )
            result = result.model_copy(update={"role": None, "status": canonical_status})

    # Rule: if "priority" is cued and role contains a priority-like value
    if "priority" in cues and result.role is not None and result.priority is None:
        candidate = result.role
        canonical_priority = normalize_priority(candidate)
        if canonical_priority is not None:
            logger.info(
                "semantic_wrong_field_reconciliation source=role dest=priority candidate=%r canonical=%r",
                candidate,
                canonical_priority,
            )
            result = result.model_copy(update={"role": None, "priority": canonical_priority})

    # Rule: if "location" is cued and role contains a location-like value
    if "location" in cues and result.role is not None and result.location is None:
        candidate = result.role
        canonical_location = normalize_location(candidate)
        if canonical_location is not None:
            logger.info(
                "semantic_wrong_field_reconciliation source=role dest=location candidate=%r canonical=%r",
                candidate,
                canonical_location,
            )
            result = result.model_copy(update={"role": None, "location": canonical_location})

    # Rule: if "location" is cued and status contains a location-like value
    if "location" in cues and result.status is not None and result.location is None:
        candidate = result.status
        canonical_location = normalize_location(candidate)
        if canonical_location is not None:
            logger.info(
                "semantic_wrong_field_reconciliation source=status dest=location candidate=%r canonical=%r",
                candidate,
                canonical_location,
            )
            result = result.model_copy(update={"status": None, "location": canonical_location})

    return result


def normalize_extracted_fields(
    fields: SemanticExtractedFields,
    *,
    transcript: str | None = None,
) -> tuple[SemanticFieldPatch | None, list[str]]:
    patch = SemanticFieldPatch.model_validate(fields.model_dump(exclude_none=True))
    reconciled = _reconcile_controlled_field_misclassification(patch)
    if transcript is not None:
        reconciled = reconcile_wrong_field_placement(transcript, reconciled)
    return validate_fields(reconciled, tool_name="semantic_field_extraction")


def describe_application(application: JobApplication) -> str:
    if application.role:
        return application.role
    return f"Application #{application.id}"


def build_ambiguous_update_question(company: str, matches: list[JobApplication]) -> str:
    role_list = ", ".join(f'"{app.role}"' for app in matches if app.role)
    if role_list:
        return f"Which {company} application do you mean? {role_list}"
    return CLARIFICATION_AMBIGUOUS_APPLICATION


def filter_matches_by_role(matches: list[JobApplication], role: str | None) -> list[JobApplication]:
    if role is None:
        return matches
    normalized_requested_role = normalize_role_title(role)
    if normalized_requested_role is None:
        return matches
    return [
        application
        for application in matches
        if normalize_role_title(application.role) is not None
        and normalize_role_title(application.role).casefold() == normalized_requested_role.casefold()
    ]


def resolve_existing_application_target(
    db: Session,
    target: PreviewExistingApplicationTarget,
    context: dict[str, object],
) -> tuple[JobApplication | None, list[str], str | None]:
    requested_role = normalize_role_title(target.role) if target.role else None
    selected_application_id = context.get("active_application_id")

    if target.application_id is not None:
        if not isinstance(selected_application_id, int) or selected_application_id != target.application_id:
            return None, [], CLARIFICATION_MISSING_PERSISTED_TARGET
        application = db.get(JobApplication, target.application_id)
        if application is None:
            return None, ["Referenced application was not found."], None
        if requested_role is not None:
            existing_role = normalize_role_title(application.role)
            if existing_role is None or existing_role.casefold() != requested_role.casefold():
                return None, ["Referenced application does not match the requested role."], None
        return application, [], None

    requested_company = target.company
    if not requested_company:
        return None, [], CLARIFICATION_MISSING_PERSISTED_TARGET

    resolved_company_name, _canonical_company = resolve_company_name(db, requested_company)
    if resolved_company_name is None:
        return None, [f'Application for company "{requested_company}" was not found.'], None

    matches = filter_matches_by_role(get_application_matches_for_company(db, resolved_company_name), requested_role)
    if len(matches) == 1:
        return matches[0], [], None
    if len(matches) > 1:
        return None, [], build_ambiguous_update_question(resolved_company_name, matches)
    return None, [f'Application for company "{resolved_company_name}" was not found.'], None


def build_draft_preview(
    base_draft: JobApplicationCreate,
    fields: SemanticFieldPatch,
    context: dict[str, object],
) -> tuple[JobApplicationCreate | None, list[str], str]:
    warnings: list[str] = []
    confirmation_kind = "none"
    active_draft = context.get("active_draft")
    used_context = active_draft is not None

    # Resolve roles_json: use [fields.role] if role explicitly provided, else fall back to base_draft
    if fields.role is not None:
        roles_json = [fields.role]
    else:
        roles_json = list(base_draft.roles_json)

    try:
        draft = JobApplicationCreate(
            company=fields.company if fields.company is not None else base_draft.company,
            role=roles_json[0] if roles_json else "",
            roles_json=roles_json,
            employment_types_json=(
                list(fields.employment_types) if fields.employment_types is not None else list(base_draft.employment_types_json)
            ),
            job_link=fields.job_link if fields.job_link is not None else base_draft.job_link,
            location=fields.location if fields.location is not None else base_draft.location,
            status=fields.status if fields.status is not None else base_draft.status,
            current_stages_json=(
                list(fields.current_stages) if fields.current_stages is not None else list(base_draft.current_stages_json)
            ),
            priority=fields.priority if fields.priority is not None else base_draft.priority,
            engaged_days=fields.engaged_days if fields.engaged_days is not None else base_draft.engaged_days,
            next_action=fields.next_action if fields.next_action is not None else base_draft.next_action,
            comments=fields.comments if fields.comments is not None else base_draft.comments,
        )
    except Exception as exc:
        return None, [str(exc)], confirmation_kind

    if used_context and draft != base_draft:
        warnings.append("Resolved this draft using the active draft context. Review before saving.")
        confirmation_kind = "context"
    return draft, warnings, confirmation_kind


def build_context_draft(context: dict[str, object]) -> JobApplicationCreate | None:
    active_draft = context.get("active_draft")
    if not isinstance(active_draft, dict):
        active_company = context.get("active_company")
        role_from_context = context.get("active_role")
        active_role = role_from_context if isinstance(role_from_context, str) else None
        company = active_company.strip() if isinstance(active_company, str) else ""
        normalized_active_role = normalize_role_title(active_role)
        roles_json = [normalized_active_role] if normalized_active_role else []
        if not company:
            return None
        return JobApplicationCreate(
            company=company,
            roles_json=roles_json,
            employment_types_json=[],
            job_link="",
            location="",
            status="",
            current_stages_json=[],
            priority="",
            engaged_days=0,
            next_action="",
            comments="",
        )

    company_value = active_draft.get("company")
    role_value = active_draft.get("role")
    employment_types_value = active_draft.get("employment_types")
    current_stages_value = active_draft.get("current_stages")

    # role is scalar string; normalize it and wrap in list for roles_json
    raw_role = role_value if isinstance(role_value, str) else None
    normalized_role = normalize_role_title(raw_role)
    roles_json = [normalized_role] if normalized_role else []

    employment_types = normalize_employment_types(employment_types_value if isinstance(employment_types_value, list) else None) or []
    current_stages = normalize_stages(current_stages_value if isinstance(current_stages_value, list) else None) or []
    location = normalize_location(active_draft.get("location")) if isinstance(active_draft.get("location"), str) else None
    status = normalize_status(active_draft.get("status")) if isinstance(active_draft.get("status"), str) else None
    priority = normalize_priority(active_draft.get("priority")) if isinstance(active_draft.get("priority"), str) else None
    engaged_days = active_draft.get("engaged_days")

    company = company_value.strip() if isinstance(company_value, str) else ""
    if not company:
        return None

    return JobApplicationCreate(
        company=company,
        roles_json=roles_json,
        employment_types_json=employment_types,
        job_link=active_draft.get("job_link") if isinstance(active_draft.get("job_link"), str) else "",
        location=location or "",
        status=status or "",
        current_stages_json=current_stages,
        priority=priority or "",
        engaged_days=engaged_days if isinstance(engaged_days, int) and engaged_days >= 0 else 0,
        next_action=active_draft.get("next_action") if isinstance(active_draft.get("next_action"), str) else "",
        comments=active_draft.get("comments") if isinstance(active_draft.get("comments"), str) else "",
    )


def build_existing_application_preview(application: JobApplication, fields: SemanticFieldPatch) -> JobApplicationCreate | None:
    # role is scalar on JobApplication; wrap in list for roles_json
    if fields.role is not None:
        roles_json = [fields.role]
    else:
        existing_role = application.role
        roles_json = [existing_role] if existing_role else []

    try:
        return JobApplicationCreate(
            company=application.company,
            roles_json=roles_json,
            employment_types_json=(
                list(fields.employment_types) if fields.employment_types is not None else list(application.employment_types_json)
            ),
            job_link=fields.job_link if fields.job_link is not None else application.job_link,
            location=fields.location if fields.location is not None else application.location,
            status=fields.status if fields.status is not None else application.status,
            current_stages_json=(
                list(fields.current_stages) if fields.current_stages is not None else list(application.current_stages_json)
            ),
            priority=fields.priority if fields.priority is not None else application.priority,
            engaged_days=fields.engaged_days if fields.engaged_days is not None else application.engaged_days,
            next_action=fields.next_action if fields.next_action is not None else application.next_action,
            comments=fields.comments if fields.comments is not None else application.comments,
        )
    except Exception:
        return None


def fields_have_values(fields: SemanticFieldPatch, *, allow_company: bool) -> bool:
    values = {
        "company": fields.company,
        "role": fields.role,
        "employment_types": fields.employment_types,
        "job_link": fields.job_link,
        "location": fields.location,
        "status": fields.status,
        "current_stages": fields.current_stages,
        "priority": fields.priority,
        "engaged_days": fields.engaged_days,
        "next_action": fields.next_action,
        "comments": fields.comments,
    }
    if not allow_company:
        values.pop("company")
    return any(value is not None and value != [] for value in values.values())


def handle_patch_active_draft(
    db: Session,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: PatchActiveDraftArguments,
    metrics,
) -> SemanticTranscriptResponse:
    logger.info(
        "semantic_tool_arguments_normalized tool=%s arguments=%r",
        proposal.tool_name,
        proposal.arguments,
    )
    validated_fields, warnings = validate_fields(arguments.fields, tool_name=proposal.tool_name)
    if validated_fields is None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=warnings,
            interpreter_metrics=metrics,
        )

    context = build_context_payload(payload)
    context_draft = build_context_draft(context)
    company = validated_fields.company
    confirmation_kind = "none"
    if company is None:
        if context_draft is not None and context_draft.company.strip():
            company = context_draft.company.strip()
            confirmation_kind = "context"
        else:
            return SemanticTranscriptResponse(
                status="clarification_required",
                operation="none",
                raw_transcript=payload.transcript,
                proposal=proposal,
                warnings=["Draft company is missing."],
                clarification_question=CLARIFICATION_MISSING_COMPANY,
                interpreter_metrics=metrics,
            )

    resolved_company_name, _canonical_company = resolve_company_name(db, company)
    normalized_fields = validated_fields.model_copy(
        update={"company": resolved_company_name or company if validated_fields.company is not None else validated_fields.company}
    )
    base_draft = context_draft or JobApplicationCreate(
        company=resolved_company_name or company,
        roles_json=[],
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
    )
    if resolved_company_name:
        base_draft = base_draft.model_copy(update={"company": resolved_company_name})

    draft, draft_warnings, draft_confirmation_kind = build_draft_preview(base_draft, normalized_fields, context)
    if draft is None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=draft_warnings,
            interpreter_metrics=metrics,
        )

    warnings.extend(draft_warnings)
    # context_notes are internal implementation metadata — never expose them to the user
    final_confirmation_kind = draft_confirmation_kind if draft_confirmation_kind != "none" else confirmation_kind

    incoming_draft_id = context.get("draft_id")
    if isinstance(incoming_draft_id, str) and incoming_draft_id:
        operation = "patch_draft"
        target = MutationTarget(draft_id=incoming_draft_id)
    else:
        operation = "create_draft"
        target = MutationTarget()

    # roles_json is a list[str] in JobApplicationCreate; extract scalar role for ApplicationChanges
    draft_role = draft.roles_json[0] if draft.roles_json else None

    mutation_payload = MutationPayload(
        operation=operation,
        target=target,
        changes=ApplicationChanges(
            company=draft.company,
            role=draft_role,
            status=draft.status or None,
            priority=draft.priority or None,
            location_mode=draft.location or None,
            job_link=draft.job_link or None,
            employment_types=list(draft.employment_types_json) if draft.employment_types_json else None,
            current_stages=list(draft.current_stages_json) if draft.current_stages_json else None,
        ),
    )
    mutation_result = dispatch(mutation_payload, db)
    # Truthful no-op: dispatcher signals that no field actually changed.
    if mutation_result.operation == "no_change":
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[mutation_result.message],
            interpreter_metrics=metrics,
        )
    effective_draft_id = str(mutation_result.draft["id"]) if mutation_result.draft and isinstance(mutation_result.draft.get("id"), int) else incoming_draft_id
    return build_transcript_response_from_mutation(
        mutation_result,
        payload,
        proposal,
        metrics=metrics,
        warnings=warnings,
        draft=draft,
        draft_id=effective_draft_id,
        needs_confirmation=final_confirmation_kind == "context",
        confirmation_kind=final_confirmation_kind,
    )


def handle_preview_existing_application_update(
    db: Session,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: PreviewExistingApplicationUpdateArguments,
    metrics,
) -> SemanticTranscriptResponse:
    context = build_context_payload(payload)
    logger.info(
        "semantic_tool_arguments_normalized tool=%s arguments=%r",
        proposal.tool_name,
        proposal.arguments,
    )
    validated_fields, warnings = validate_fields(arguments.fields, tool_name=proposal.tool_name)
    if validated_fields is None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=warnings,
            interpreter_metrics=metrics,
        )
    if not fields_have_values(validated_fields, allow_company=False):
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=["No supported command was detected."],
            interpreter_metrics=metrics,
        )

    application, target_warnings, clarification_question = resolve_existing_application_target(db, arguments.target, context)
    if clarification_question:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=target_warnings,
            clarification_question=clarification_question,
            interpreter_metrics=metrics,
        )
    if application is None:
        message = target_warnings[0] if target_warnings else "I could not find that application."
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[message],
            interpreter_metrics=metrics,
        )

    preview = build_existing_application_preview(application, validated_fields)
    if preview is None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=["Proposed update preview was invalid."],
            interpreter_metrics=metrics,
        )

    # Check: is there already a pending-changes draft for a *different* application?
    from .models import ApplicationChangeDraft as _ACD
    active_cd = db.query(_ACD).first()
    if active_cd is not None and active_cd.target_application_id != application.id:
        conflict_app = db.get(JobApplication, active_cd.target_application_id)
        conflict_label = f"{conflict_app.company} — {conflict_app.role}" if conflict_app else f"application #{active_cd.target_application_id}"
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=target_warnings,
            clarification_question=(
                f"You already have unsaved changes for {conflict_label}. "
                f"Apply or discard them before editing another application."
            ),
            interpreter_metrics=metrics,
        )

    mutation_payload = MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=application.id),
        changes=ApplicationChanges(
            status=validated_fields.status or None,
            priority=validated_fields.priority or None,
            location_mode=validated_fields.location or None,
            job_link=validated_fields.job_link or None,
            role=validated_fields.role or None,
            employment_types=list(validated_fields.employment_types) if validated_fields.employment_types else None,
            current_stages=list(validated_fields.current_stages) if validated_fields.current_stages else None,
            next_action=validated_fields.next_action or None,
            comments=validated_fields.comments or None,
            engaged_days=validated_fields.engaged_days if validated_fields.engaged_days is not None else None,
        ),
    )
    mutation_result = dispatch(mutation_payload, db)
    return build_transcript_response_from_mutation(
        mutation_result,
        payload,
        proposal,
        metrics=metrics,
        warnings=target_warnings,
        draft=preview,
        application_id=application.id,
    )


def handle_request_draft_save(
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    _arguments: RequestDraftSaveArguments,
    metrics,
    db: Session | None = None,
) -> SemanticTranscriptResponse:
    context = build_context_payload(payload)
    context_draft = build_context_draft(context)
    if context_draft is None:
        mutation_payload = MutationPayload(
            operation="ask_clarification",
            target=MutationTarget(),
            changes=ApplicationChanges(),
            notes_to_append=[CLARIFICATION_NO_ACTIVE_DRAFT],
        )
        mutation_result = dispatch(mutation_payload, db) if db else None
        if mutation_result:
            return build_transcript_response_from_mutation(
                mutation_result, payload, proposal, metrics=metrics, warnings=[]
            )
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[],
            clarification_question=CLARIFICATION_NO_ACTIVE_DRAFT,
            interpreter_metrics=metrics,
        )

    draft_id = context.get("draft_id")
    draft_id_str = str(draft_id) if draft_id is not None else None

    # roles_json is a list[str] in JobApplicationCreate; extract scalar role for ApplicationChanges
    context_draft_role = context_draft.roles_json[0] if context_draft.roles_json else None

    mutation_payload = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=draft_id_str),
        changes=ApplicationChanges(
            company=context_draft.company,
            role=context_draft_role,
        ),
    )
    if db is not None:
        mutation_result = dispatch(mutation_payload, db)
        if mutation_result.success and mutation_result.application:
            # Draft was saved: return a response that truthfully reflects saved state.
            # draft_id is cleared (no longer a draft), application_id is set.
            saved_app_id = mutation_result.application.get("id")
            return SemanticTranscriptResponse(
                status="preview",
                operation="create",
                raw_transcript=payload.transcript,
                proposal=proposal,
                draft=context_draft,
                draft_id=None,
                application_id=saved_app_id,
                warnings=[],
                needs_confirmation=False,
                interpreter_metrics=metrics,
            )
        # Save failed (e.g. no draft_id in context, or draft not found): surface error.
        return build_transcript_response_from_mutation(
            mutation_result, payload, proposal, metrics=metrics, warnings=[]
        )
    # db is None (test-only path without DB): return clarification that we cannot save.
    return SemanticTranscriptResponse(
        status="clarification_required",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal,
        warnings=[],
        clarification_question=CLARIFICATION_NO_ACTIVE_DRAFT,
        interpreter_metrics=metrics,
    )


def handle_attach_latest_browser_context(
    db: Session,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    metrics,
) -> SemanticTranscriptResponse:
    context_payload = build_context_payload(payload)
    if build_context_draft(context_payload) is None:
        question = "There is no active draft to attach the current link to."
        mutation_payload = MutationPayload(
            operation="ask_clarification",
            target=MutationTarget(),
            changes=ApplicationChanges(),
            notes_to_append=[question],
        )
        mutation_result = dispatch(mutation_payload, db)
        return build_transcript_response_from_mutation(
            mutation_result, payload, proposal, metrics=metrics, warnings=[]
        )

    browser_context = db.query(BrowserContext).order_by(BrowserContext.captured_at.desc(), BrowserContext.id.desc()).first()
    if browser_context is None:
        question = "Open a job page first, then try again."
        mutation_payload = MutationPayload(
            operation="ask_clarification",
            target=MutationTarget(),
            changes=ApplicationChanges(),
            notes_to_append=[question],
        )
        mutation_result = dispatch(mutation_payload, db)
        return build_transcript_response_from_mutation(
            mutation_result, payload, proposal, metrics=metrics, warnings=["No browser context is available."]
        )

    question = f'Latest browser context: "{browser_context.page_title}" at {browser_context.url}. Which tracker fields should I fill from it?'
    mutation_payload = MutationPayload(
        operation="ask_clarification",
        target=MutationTarget(),
        changes=ApplicationChanges(),
        notes_to_append=[question],
    )
    mutation_result = dispatch(mutation_payload, db)
    return build_transcript_response_from_mutation(
        mutation_result, payload, proposal, metrics=metrics, warnings=[]
    )


def handle_ask_clarification(
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: AskClarificationArguments,
    metrics,
    db: Session | None = None,
) -> SemanticTranscriptResponse:
    mutation_payload = MutationPayload(
        operation="ask_clarification",
        target=MutationTarget(),
        changes=ApplicationChanges(),
        notes_to_append=[arguments.question],
    )
    if db is not None:
        mutation_result = dispatch(mutation_payload, db)
        return build_transcript_response_from_mutation(
            mutation_result, payload, proposal, metrics=metrics, warnings=[]
        )
    return SemanticTranscriptResponse(
        status="clarification_required",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal,
        clarification_question=arguments.question,
        interpreter_metrics=metrics,
    )


def handle_archive_application(
    db: Session,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: ArchiveApplicationArguments,
    metrics,
) -> SemanticTranscriptResponse:
    context = build_context_payload(payload)
    application, target_warnings, clarification_question = resolve_existing_application_target(db, arguments.target, context)
    if clarification_question:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=target_warnings,
            clarification_question=clarification_question,
            interpreter_metrics=metrics,
        )
    if application is None:
        message = target_warnings[0] if target_warnings else "I could not find that application."
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[message],
            interpreter_metrics=metrics,
        )
    if application.archived_at is not None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=["This application is already archived."],
            interpreter_metrics=metrics,
        )

    mutation_payload = MutationPayload(
        operation="archive_application",
        target=MutationTarget(application_id=application.id),
        changes=ApplicationChanges(),
    )
    mutation_result = dispatch(mutation_payload, db)
    return build_transcript_response_from_mutation(
        mutation_result,
        payload,
        proposal,
        metrics=metrics,
        warnings=target_warnings,
        application_id=application.id,
    )


def handle_explain_delete_policy(
    db: Session,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: ExplainDeletePolicyArguments,
    metrics,
) -> SemanticTranscriptResponse:
    context = build_context_payload(payload)
    application, target_warnings, clarification_question = resolve_existing_application_target(db, arguments.target, context)
    if clarification_question:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=target_warnings,
            clarification_question=clarification_question,
            interpreter_metrics=metrics,
        )
    if application is None:
        message = target_warnings[0] if target_warnings else "I could not find that application."
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[message],
            interpreter_metrics=metrics,
        )

    if application.archived_at is not None:
        guidance = "This application is archived. Use Delete Permanently in the archived view to remove it irreversibly."
    else:
        guidance = "This application is active. Archive it first before deleting it permanently."

    question = guidance
    mutation_payload_obj = MutationPayload(
        operation="ask_clarification",
        target=MutationTarget(),
        changes=ApplicationChanges(),
        notes_to_append=[question],
    )
    mutation_result = dispatch(mutation_payload_obj, db)
    return build_transcript_response_from_mutation(
        mutation_result,
        payload,
        proposal,
        metrics=metrics,
        warnings=[],
    )


def handle_discard_draft(
    db: Session,
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: DiscardDraftArguments,
    metrics,
) -> SemanticTranscriptResponse:
    context = build_context_payload(payload)
    draft_id_raw = context.get("draft_id")

    if draft_id_raw is None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=["No active draft to discard."],
            interpreter_metrics=metrics,
        )

    draft_id = str(draft_id_raw)

    # If target hints are provided, validate they match the active draft
    if arguments.target.company or arguments.target.role:
        active_draft = context.get("active_draft")
        if isinstance(active_draft, dict):
            draft_company = (active_draft.get("company") or "").strip().casefold()
            draft_role = (active_draft.get("role") or "").strip().casefold()
            target_company = (arguments.target.company or "").strip().casefold()
            target_role = (arguments.target.role or "").strip().casefold()

            company_matches = (
                not target_company
                or draft_company == target_company
                or target_company in draft_company
                or draft_company in target_company
            )
            role_matches = (
                not target_role
                or draft_role == target_role
                or target_role in draft_role
                or draft_role in target_role
            )

            if not (company_matches and role_matches):
                active_company_label = active_draft.get("company") or "unknown"
                active_role_label = active_draft.get("role") or "unknown"
                return SemanticTranscriptResponse(
                    status="clarification_required",
                    operation="none",
                    raw_transcript=payload.transcript,
                    proposal=proposal,
                    warnings=["Target hints do not match the active draft."],
                    clarification_question=(
                        f"The active draft is for {active_company_label} — {active_role_label}. "
                        "Did you mean to discard that?"
                    ),
                    interpreter_metrics=metrics,
                )

    mutation_payload = MutationPayload(
        operation="discard_draft",
        target=MutationTarget(draft_id=draft_id),
        changes=ApplicationChanges(),
    )
    mutation_result = dispatch(mutation_payload, db)
    return build_transcript_response_from_mutation(
        mutation_result, payload, proposal, metrics=metrics, warnings=[]
    )


def _proposal_with_clarification(question: str) -> SemanticToolCallProposal:
    return SemanticToolCallProposal(tool_name="ask_clarification", arguments={"question": question})


def _normalize_optional_text_value(value: object) -> object:
    if not isinstance(value, str):
        return _INVALID
    stripped = value.strip()
    return stripped if stripped else _INVALID


def _normalize_company_value(value: object) -> object:
    if not isinstance(value, str):
        return _INVALID
    stripped = value.strip()
    return stripped if stripped else _INVALID


def _normalize_role_value(value: object) -> object:
    """Normalize a scalar role string. Accepts str; rejects everything else."""
    if isinstance(value, str):
        normalized = " ".join(value.strip().split())
        return normalized if normalized else _INVALID
    # Also accept a single-element list for backward compatibility with LLM output
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        normalized = " ".join(value[0].strip().split())
        return normalized if normalized else _INVALID
    return _INVALID


def _normalize_employment_type_values(value: object) -> object:
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        raw_values = value
    else:
        return _INVALID

    normalized_values: list[str] = []
    for raw_value in raw_values:
        stripped = " ".join(raw_value.strip().split())
        if not stripped:
            return _INVALID
        canonical_value = EMPLOYMENT_TYPE_ALIASES.get(_normalize_lookup_text(stripped), stripped)
        if canonical_value not in normalized_values:
            normalized_values.append(canonical_value)
    return normalized_values


def _normalize_stage_values(value: object) -> object:
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        raw_values = value
    else:
        return _INVALID

    normalized_values: list[str] = []
    for raw_value in raw_values:
        stripped = " ".join(raw_value.strip().split())
        if not stripped:
            return _INVALID
        normalized_lookup = _normalize_lookup_text(stripped)
        canonical_value = next(
            (option for option in ALLOWED_CURRENT_STAGES if _normalize_lookup_text(option) == normalized_lookup),
            stripped,
        )
        if canonical_value not in normalized_values:
            normalized_values.append(canonical_value)
    return normalized_values


def _normalize_status_value(value: object) -> object:
    if not isinstance(value, str):
        return _INVALID
    stripped = " ".join(value.strip().split())
    if not stripped:
        return _INVALID
    return normalize_status(stripped) or stripped


def _normalize_priority_value(value: object) -> object:
    if not isinstance(value, str):
        return _INVALID
    stripped = " ".join(value.strip().split())
    if not stripped:
        return _INVALID
    return normalize_priority(stripped) or stripped


def _normalize_location_value(value: object) -> object:
    if not isinstance(value, str):
        return _INVALID
    stripped = " ".join(value.strip().split())
    if not stripped:
        return _INVALID
    return normalize_location(stripped) or stripped


def _values_conflict(left: object, right: object) -> bool:
    return left != right


def _merge_field_aliases(
    normalized_fields: dict[str, object],
    canonical_key: str,
    alias_keys: tuple[str, ...],
    normalizer,
) -> object:
    present_keys = [key for key in (canonical_key, *alias_keys) if key in normalized_fields]
    if not present_keys:
        return _MISSING

    normalized_values: dict[str, object] = {}
    for key in present_keys:
        normalized_value = normalizer(normalized_fields[key])
        if normalized_value is _INVALID:
            return _INVALID
        normalized_values[key] = normalized_value

    merged_value = normalized_values[present_keys[0]]
    for key in present_keys[1:]:
        if _values_conflict(merged_value, normalized_values[key]):
            return _CONFLICT

    normalized_fields[canonical_key] = merged_value
    for alias_key in alias_keys:
        normalized_fields.pop(alias_key, None)
    return merged_value


def normalize_semantic_field_patch_argument_shape(fields: dict[str, object]) -> dict[str, object] | None:
    normalized_fields = dict(fields)

    # List-valued fields that need alias merging
    alias_merge_specs = (
        ("employment_types", ("employment_type", "type"), _normalize_employment_type_values),
        ("current_stages", ("current_stage", "stage"), _normalize_stage_values),
    )
    for canonical_key, alias_keys, normalizer in alias_merge_specs:
        merged = _merge_field_aliases(normalized_fields, canonical_key, alias_keys, normalizer)
        if merged is _INVALID or merged is _CONFLICT:
            return None

    # Handle "roles" key from LLM output: collapse to scalar "role"
    # LLMs trained on the old schema may still emit roles:[...] — accept a single-element array
    if "roles" in normalized_fields and "role" not in normalized_fields:
        roles_raw = normalized_fields.pop("roles")
        normalized_role = _normalize_role_value(roles_raw)
        if normalized_role is _INVALID:
            return None
        normalized_fields["role"] = normalized_role
    elif "roles" in normalized_fields and "role" in normalized_fields:
        # Both present — try to reconcile; if they conflict, reject
        roles_raw = normalized_fields.pop("roles")
        normalized_roles_as_role = _normalize_role_value(roles_raw)
        normalized_role = _normalize_role_value(normalized_fields["role"])
        if normalized_roles_as_role is _INVALID or normalized_role is _INVALID:
            return None
        if normalized_roles_as_role != normalized_role:
            return None
        normalized_fields["role"] = normalized_role

    scalar_normalizers = {
        "company": _normalize_company_value,
        "role": _normalize_role_value,
        "status": _normalize_status_value,
        "priority": _normalize_priority_value,
        "location": _normalize_location_value,
        "comments": _normalize_optional_text_value,
        "next_action": _normalize_optional_text_value,
    }
    for key, normalizer in scalar_normalizers.items():
        if key not in normalized_fields:
            continue
        normalized_value = normalizer(normalized_fields[key])
        if normalized_value is _INVALID:
            return None
        normalized_fields[key] = normalized_value

    return normalized_fields


def normalize_patch_active_draft_argument_shape(proposal: SemanticToolCallProposal) -> SemanticToolCallProposal:
    if proposal.tool_name not in {"patch_active_draft", "preview_existing_application_update"}:
        return proposal

    arguments = dict(proposal.arguments)
    fields = arguments.get("fields")
    if not isinstance(fields, dict):
        return proposal
    normalized_fields = normalize_semantic_field_patch_argument_shape(fields)
    if normalized_fields is None:
        return proposal
    arguments["fields"] = normalized_fields
    return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)


def canonicalize_tool_arguments(
    *,
    tool_name: str,
    raw_arguments: object,
) -> dict[str, object]:
    """Unwrap known LLM tool-call envelope variations into a canonical argument dict.

    Supported shapes:
      1. Already canonical: {"fields": {...}, ...}
      2. args envelope:     {"function": tool_name, "args": {"fields": {...}}}
      3. arguments envelope:{"name": tool_name, "arguments": {"fields": {...}}}
      4. Duplicate (envelope + canonical keys present with same values): keep canonical.

    Raises SemanticInterpreterInvalidResponseError when:
    - raw_arguments is not a dict
    - An envelope is present but wraps a non-dict payload
    - Conflicting values between envelope and top-level canonical keys
    """
    if not isinstance(raw_arguments, dict):
        raise SemanticInterpreterInvalidResponseError(
            f"Tool arguments must be an object, got {type(raw_arguments).__name__}."
        )

    args: dict[str, object] = dict(raw_arguments)

    # Detect which envelope variation is present
    has_args_envelope = "args" in args and isinstance(args.get("function"), str)
    has_arguments_envelope = "arguments" in args and isinstance(args.get("name"), str)
    # Legacy llama envelope: {"function": tool, "parameters": {...}, ...top-level fields...}
    has_parameters_envelope = (
        "parameters" in args
        and isinstance(args.get("function"), str)
        and not has_args_envelope
    )

    if has_args_envelope:
        function_val = args["function"]
        inner = args["args"]
        if not isinstance(inner, dict):
            raise SemanticInterpreterInvalidResponseError(
                f"Tool call envelope 'args' must be an object, got {type(inner).__name__}."
            )
        if function_val != tool_name:
            # Mismatched function name — still unwrap but log discrepancy; caller decides
            logger.warning(
                "canonicalize_tool_arguments_function_mismatch envelope_function=%r selected_tool=%r",
                function_val,
                tool_name,
            )
        # Build canonical from inner; check for conflicts with any top-level canonical keys
        envelope_keys = {"function", "args"}
        top_level_canonical = {k: v for k, v in args.items() if k not in envelope_keys}
        if top_level_canonical:
            # Shape 4: duplicate — verify they agree, then keep top-level
            for key, top_val in top_level_canonical.items():
                inner_val = inner.get(key)
                if inner_val is not None and inner_val != top_val:
                    raise SemanticInterpreterInvalidResponseError(
                        f"Conflicting values for '{key}' between envelope 'args' and top-level arguments."
                    )
            return top_level_canonical
        return dict(inner)

    if has_arguments_envelope:
        name_val = args["name"]
        inner = args["arguments"]
        if not isinstance(inner, dict):
            raise SemanticInterpreterInvalidResponseError(
                f"Tool call envelope 'arguments' must be an object, got {type(inner).__name__}."
            )
        if name_val != tool_name:
            logger.warning(
                "canonicalize_tool_arguments_name_mismatch envelope_name=%r selected_tool=%r",
                name_val,
                tool_name,
            )
        envelope_keys = {"name", "arguments"}
        top_level_canonical = {k: v for k, v in args.items() if k not in envelope_keys}
        if top_level_canonical:
            for key, top_val in top_level_canonical.items():
                inner_val = inner.get(key)
                if inner_val is not None and inner_val != top_val:
                    raise SemanticInterpreterInvalidResponseError(
                        f"Conflicting values for '{key}' between envelope 'arguments' and top-level arguments."
                    )
            return top_level_canonical
        return dict(inner)

    if has_parameters_envelope:
        function_val = args["function"]
        inner = args["parameters"]
        if not isinstance(inner, dict):
            raise SemanticInterpreterInvalidResponseError(
                f"Tool call envelope 'parameters' must be an object, got {type(inner).__name__}."
            )
        if function_val != tool_name:
            logger.warning(
                "canonicalize_tool_arguments_function_mismatch envelope_function=%r selected_tool=%r",
                function_val,
                tool_name,
            )
        envelope_keys = {"function", "parameters"}
        top_level_canonical = {k: v for k, v in args.items() if k not in envelope_keys}
        if top_level_canonical:
            # Shape 4 variant: duplicate — merge, checking for conflicts
            merged = dict(inner)
            for key, top_val in top_level_canonical.items():
                inner_val = inner.get(key)
                if inner_val is not None and inner_val != top_val:
                    raise SemanticInterpreterInvalidResponseError(
                        f"Conflicting values for '{key}' between envelope 'parameters' and top-level arguments."
                    )
                merged[key] = top_val
            return merged
        return dict(inner)

    # Shape 1: already canonical
    return args


def _safe_extracted_field_log(fields: dict[str, object]) -> dict[str, object]:
    safe_payload: dict[str, object] = {}
    for key, value in fields.items():
        if key in {"comments", "next_action"}:
            safe_payload[f"{key}_present"] = isinstance(value, str) and bool(value)
        else:
            safe_payload[key] = value
    return safe_payload


def _canonicalize_company_for_comparison(db: Session, value: str) -> str:
    resolved_company_name, _canonical_company = resolve_company_name(db, value)
    return resolved_company_name or value.strip()


def _canonicalize_selected_field_value(field_name: str, raw_value: object) -> object:
    normalized_fields = normalize_semantic_field_patch_argument_shape({field_name: raw_value})
    if normalized_fields is None or field_name not in normalized_fields:
        return _INVALID
    try:
        semantic_field_patch = SemanticFieldPatch.model_validate(normalized_fields)
    except ValidationError:
        return _INVALID
    validated_fields, _warnings = validate_fields(semantic_field_patch, tool_name="semantic_tool_selection")
    if validated_fields is None:
        return _INVALID
    canonical_payload = validated_fields.model_dump(exclude_none=True)
    return canonical_payload.get(field_name, _INVALID)


def _log_field_conflict(
    *,
    proposal: SemanticToolCallProposal,
    field_name: str,
    normalized_extracted_value: object,
    normalized_selected_tool_value: object,
) -> None:
    logger.warning(
        "semantic_extracted_fields_conflict tool=%s field_name=%s normalized_extracted_value=%r normalized_selected_tool_value=%r",
        proposal.tool_name,
        field_name,
        normalized_extracted_value,
        normalized_selected_tool_value,
    )


def merge_extracted_fields_into_proposal(
    db: Session,
    proposal: SemanticToolCallProposal,
    extracted_fields: SemanticFieldPatch,
) -> SemanticToolCallProposal | None:
    extracted_payload = extracted_fields.model_dump(exclude_none=True)
    if not extracted_payload:
        return proposal

    if proposal.tool_name == "patch_active_draft":
        # Canonicalize envelope before reading fields (handles wrapper shapes from LLM)
        try:
            canonical_args = canonicalize_tool_arguments(
                tool_name=proposal.tool_name,
                raw_arguments=proposal.arguments,
            )
        except SemanticInterpreterInvalidResponseError:
            canonical_args = dict(proposal.arguments)
        arguments = dict(canonical_args)
        raw_fields = arguments.get("fields")
        if raw_fields is None:
            fields = {}
        elif isinstance(raw_fields, dict):
            fields = dict(raw_fields)
        else:
            raise SemanticInterpreterInvalidResponseError(
                f"Tool fields must be an object, got {type(raw_fields).__name__}."
            )
        merged_fields = dict(extracted_payload)

        if "company" not in merged_fields:
            selected_company = fields.get("company")
            if isinstance(selected_company, str) and selected_company.strip():
                merged_fields["company"] = selected_company.strip()
        elif isinstance(fields.get("company"), str) and fields.get("company", "").strip():
            extracted_company = _canonicalize_company_for_comparison(db, str(merged_fields["company"]))
            selected_company = _canonicalize_company_for_comparison(db, str(fields["company"]))
            if extracted_company != selected_company:
                _log_field_conflict(
                    proposal=proposal,
                    field_name="company",
                    normalized_extracted_value=extracted_company,
                    normalized_selected_tool_value=selected_company,
                )
                return _proposal_with_clarification(CLARIFICATION_CONFLICTING_COMPANY)

        for field_name, extracted_value in extracted_payload.items():
            if field_name == "company" or field_name not in fields:
                continue
            canonical_selected_value = _canonicalize_selected_field_value(field_name, fields[field_name])
            if canonical_selected_value is _INVALID:
                continue
            if canonical_selected_value != extracted_value:
                _log_field_conflict(
                    proposal=proposal,
                    field_name=field_name,
                    normalized_extracted_value=extracted_value,
                    normalized_selected_tool_value=canonical_selected_value,
                )
                return None

        arguments["fields"] = merged_fields
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)

    if proposal.tool_name == "preview_existing_application_update":
        try:
            canonical_args = canonicalize_tool_arguments(
                tool_name=proposal.tool_name,
                raw_arguments=proposal.arguments,
            )
        except SemanticInterpreterInvalidResponseError:
            canonical_args = dict(proposal.arguments)
        arguments = dict(canonical_args)
        raw_target = arguments.get("target")
        raw_fields = arguments.get("fields")
        if raw_target is not None and not isinstance(raw_target, dict):
            raise SemanticInterpreterInvalidResponseError(
                f"Tool target must be an object, got {type(raw_target).__name__}."
            )
        if raw_fields is not None and not isinstance(raw_fields, dict):
            raise SemanticInterpreterInvalidResponseError(
                f"Tool fields must be an object, got {type(raw_fields).__name__}."
            )
        target = dict(raw_target) if isinstance(raw_target, dict) else {}
        fields = dict(raw_fields) if isinstance(raw_fields, dict) else {}
        merged_fields = {key: value for key, value in extracted_payload.items() if key != "company"}

        extracted_company = extracted_payload.get("company")
        if extracted_company is not None:
            existing_company = target.get("company")
            if existing_company is None:
                target["company"] = extracted_company
            else:
                canonical_extracted_company = _canonicalize_company_for_comparison(db, str(extracted_company))
                canonical_selected_company = _canonicalize_company_for_comparison(db, str(existing_company))
                if canonical_extracted_company != canonical_selected_company:
                    _log_field_conflict(
                        proposal=proposal,
                        field_name="company",
                        normalized_extracted_value=canonical_extracted_company,
                        normalized_selected_tool_value=canonical_selected_company,
                    )
                    return _proposal_with_clarification(CLARIFICATION_CONFLICTING_COMPANY)
        for field_name, extracted_value in merged_fields.items():
            if field_name not in fields:
                continue
            canonical_selected_value = _canonicalize_selected_field_value(field_name, fields[field_name])
            if canonical_selected_value is _INVALID:
                continue
            if canonical_selected_value != extracted_value:
                _log_field_conflict(
                    proposal=proposal,
                    field_name=field_name,
                    normalized_extracted_value=extracted_value,
                    normalized_selected_tool_value=canonical_selected_value,
                )
                return _proposal_with_clarification(CLARIFICATION_CONFLICTING_COMPANY)
        arguments["target"] = target
        arguments["fields"] = merged_fields
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)

    return proposal


def validate_tool_arguments_with_safe_normalization(proposal: SemanticToolCallProposal) -> tuple[SemanticToolCallProposal, object]:
    # Canonicalize envelope variations before strict Pydantic validation
    try:
        canonical_args = canonicalize_tool_arguments(
            tool_name=proposal.tool_name or "",
            raw_arguments=proposal.arguments,
        )
    except SemanticInterpreterInvalidResponseError as exc:
        logger.warning(
            "canonicalize_tool_arguments_failed tool=%s reason=%s",
            proposal.tool_name,
            exc,
        )
        return proposal, None
    if canonical_args is not proposal.arguments:
        proposal = SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=canonical_args)
    normalized_proposal = normalize_patch_active_draft_argument_shape(proposal)
    if normalized_proposal.tool_name == "patch_active_draft":
        return normalized_proposal, PatchActiveDraftArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "preview_existing_application_update":
        return normalized_proposal, PreviewExistingApplicationUpdateArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "request_draft_save":
        return normalized_proposal, RequestDraftSaveArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "ask_clarification":
        return normalized_proposal, AskClarificationArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "archive_application":
        return normalized_proposal, ArchiveApplicationArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "explain_delete_policy":
        return normalized_proposal, ExplainDeletePolicyArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "discard_draft":
        return normalized_proposal, DiscardDraftArguments.model_validate(normalized_proposal.arguments)
    return normalized_proposal, None


def _extract_proposed_company(proposal: SemanticToolCallProposal) -> str | None:
    if proposal.tool_name == "patch_active_draft":
        fields = proposal.arguments.get("fields")
        if isinstance(fields, dict):
            company = fields.get("company")
            return company if isinstance(company, str) else None
        return None
    if proposal.tool_name == "preview_existing_application_update":
        target = proposal.arguments.get("target")
        if isinstance(target, dict):
            company = target.get("company")
            return company if isinstance(company, str) else None
        return None
    return None


def _with_reconciled_company(proposal: SemanticToolCallProposal, company: str) -> SemanticToolCallProposal:
    arguments = dict(proposal.arguments)
    if proposal.tool_name == "patch_active_draft":
        fields = dict(arguments.get("fields") or {})
        fields["company"] = company
        arguments["fields"] = fields
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)
    if proposal.tool_name == "preview_existing_application_update":
        target = dict(arguments.get("target") or {})
        target["company"] = company
        arguments["target"] = target
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)
    return proposal


def reconcile_explicit_company_candidates(
    db: Session,
    proposal: SemanticToolCallProposal,
    explicit_known_companies: list[str],
) -> SemanticToolCallProposal:
    if len(explicit_known_companies) > 1:
        return _proposal_with_clarification(CLARIFICATION_MULTIPLE_EXPLICIT_COMPANIES)

    if len(explicit_known_companies) != 1:
        return proposal

    candidate = explicit_known_companies[0]
    if proposal.tool_name not in {"patch_active_draft", "preview_existing_application_update", "ask_clarification"}:
        return proposal

    proposed_company = _extract_proposed_company(proposal)
    if proposal.tool_name == "ask_clarification":
        question = proposal.arguments.get("question")
        if question == CLARIFICATION_MISSING_COMPANY:
            return proposal
        return proposal

    if proposed_company is None:
        return _with_reconciled_company(proposal, candidate)

    resolved_company_name, _canonical_company = resolve_company_name(db, proposed_company)
    if resolved_company_name == candidate:
        return _with_reconciled_company(proposal, candidate)
    if resolved_company_name is None and proposed_company.strip() == candidate:
        return _with_reconciled_company(proposal, candidate)
    if resolved_company_name is None:
        return _proposal_with_clarification(CLARIFICATION_CONFLICTING_COMPANY)
    return _proposal_with_clarification(CLARIFICATION_CONFLICTING_COMPANY)


def _fast_path_proposal() -> SemanticToolCallProposal:
    return SemanticToolCallProposal()


def resolve_no_tool_call_fallback(
    db: "Session",
    payload: "TranscriptParseRequest",
    extracted_fields: "SemanticFieldPatch",
    metrics,
) -> "SemanticTranscriptResponse | None":
    """Deterministic routing when the LLM returned no tool call or invalid tool arguments.

    Delegates to resolve_semantic_fallback with failure_kind="no_tool_call".
    """
    return resolve_semantic_fallback(
        db=db,
        payload=payload,
        extracted_fields=extracted_fields,
        metrics=metrics,
        failure_kind="no_tool_call",
    )


def resolve_semantic_fallback(
    *,
    db: "Session",
    payload: "TranscriptParseRequest",
    extracted_fields: "SemanticFieldPatch",
    metrics,
    failure_kind: Literal["no_tool_call", "invalid_tool_arguments", "unsupported_tool"] = "no_tool_call",
) -> SemanticTranscriptResponse | None:
    """Unified deterministic routing for all LLM failure modes.

    Routing precedence (applied consistently regardless of failure_kind):
    1. Lifecycle intent → None (caller must not absorb lifecycle commands).
    2. Explicit create intent + company + role → synthesise patch_active_draft.
    3. Explicit saved-row update intent → attempt preview_existing_application_update.
    4. Company + role present (no active draft) → synthesise patch_active_draft → create draft.
    5. Active draft exists + actionable non-company patch fields → patch active draft.
    6. Active draft exists + company or role → patch active draft (identity update).
    7. Company only, no active draft, no role → clarification: Which role?
    8. Role only, no company, no active draft → clarification: Which company?
    9. No actionable fields → None (caller emits no_change).
    """
    transcript = payload.transcript

    # Rule 1 — lifecycle commands must never be absorbed
    if _has_lifecycle_intent(transcript):
        return None

    has_company = bool(extracted_fields.company)
    has_role = bool(extracted_fields.role)

    # Rule 2 — explicit create intent: route straight to draft create/patch
    if _has_explicit_create_intent(transcript):
        if has_company and has_role:
            return _synthesise_patch_active_draft(db, payload, extracted_fields, metrics)
        if has_company and not has_role:
            question = f"Which role should I add for {extracted_fields.company}?"
            return SemanticTranscriptResponse(
                status="clarification_required",
                operation="none",
                raw_transcript=transcript,
                proposal=_proposal_with_clarification(question),
                warnings=[],
                clarification_question=question,
                interpreter_metrics=metrics,
            )
        if has_role and not has_company:
            return SemanticTranscriptResponse(
                status="clarification_required",
                operation="none",
                raw_transcript=transcript,
                proposal=_proposal_with_clarification(CLARIFICATION_MISSING_COMPANY),
                warnings=[],
                clarification_question=CLARIFICATION_MISSING_COMPANY,
                interpreter_metrics=metrics,
            )
        # Create intent but neither company nor role extracted
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=transcript,
            proposal=_proposal_with_clarification(CLARIFICATION_MISSING_COMPANY),
            warnings=[],
            clarification_question=CLARIFICATION_MISSING_COMPANY,
            interpreter_metrics=metrics,
        )

    # Rule 3 — explicit saved-row update intent
    if _has_explicit_saved_update_intent(transcript):
        if has_company:
            target = PreviewExistingApplicationTarget(
                company=extracted_fields.company,
                role=extracted_fields.role,
            )
            update_fields = extracted_fields.model_copy(update={"company": None})
            proposal = SemanticToolCallProposal(
                tool_name="preview_existing_application_update",
                arguments={
                    "target": target.model_dump(exclude_none=True),
                    "fields": update_fields.model_dump(exclude_none=True),
                    "replace_explicit_fields": True,
                },
            )
            try:
                proposal, validated_args = validate_tool_arguments_with_safe_normalization(proposal)
            except Exception:
                validated_args = None
            if validated_args is not None:
                return handle_preview_existing_application_update(db, payload, proposal, validated_args, metrics)
        return None

    context = build_context_payload(payload)
    active_draft = build_context_draft(context)
    has_active_draft = active_draft is not None
    has_non_identity_fields = fields_have_values(
        extracted_fields.model_copy(update={"company": None, "role": None}),
        allow_company=False,
    )

    # Rule 4 — company + role present, no active draft → create draft
    if has_company and has_role and not has_active_draft:
        return _synthesise_patch_active_draft(db, payload, extracted_fields, metrics)

    # Rule 5 — active draft exists + non-identity actionable fields → patch draft
    if has_active_draft and has_non_identity_fields:
        return _synthesise_patch_active_draft(db, payload, extracted_fields, metrics)

    # Rule 6 — active draft exists + company or role → patch draft (identity update)
    if has_active_draft and (has_company or has_role):
        return _synthesise_patch_active_draft(db, payload, extracted_fields, metrics)

    # Rule 7 — company only, no active draft, no role → ask for role
    if has_company and not has_role and not has_active_draft:
        question = f"Which role should I add for {extracted_fields.company}?"
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=transcript,
            proposal=_proposal_with_clarification(question),
            warnings=[],
            clarification_question=question,
            interpreter_metrics=metrics,
        )

    # Rule 8 — role only, no company, no active draft → ask for company
    if has_role and not has_company and not has_active_draft:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=transcript,
            proposal=_proposal_with_clarification(CLARIFICATION_MISSING_COMPANY),
            warnings=[],
            clarification_question=CLARIFICATION_MISSING_COMPANY,
            interpreter_metrics=metrics,
        )

    # Rule 9 — nothing actionable
    return None


def _synthesise_patch_active_draft(
    db: "Session",
    payload: "TranscriptParseRequest",
    extracted_fields: "SemanticFieldPatch",
    metrics,
) -> "SemanticTranscriptResponse":
    """Build and execute a synthetic patch_active_draft call from validated extracted fields."""
    fallback_proposal = SemanticToolCallProposal(
        tool_name="patch_active_draft",
        arguments={
            "fields": extracted_fields.model_dump(exclude_none=True),
            "replace_explicit_fields": True,
            "context_notes": [],  # no internal diagnostics in public output
        },
    )
    try:
        fallback_proposal, fallback_args = validate_tool_arguments_with_safe_normalization(fallback_proposal)
    except Exception:
        fallback_args = None
    if fallback_args is None:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=fallback_proposal,
            warnings=["No recognized tracker changes were found."],
            interpreter_metrics=metrics,
        )
    return handle_patch_active_draft(db, payload, fallback_proposal, fallback_args, metrics)


def interpret_transcript_command(
    db: Session,
    payload: TranscriptParseRequest,
    interpreter: OllamaSemanticInterpreter,
) -> SemanticTranscriptResponse:
    context, explicit_known_companies = build_interpreter_context(db, payload)

    fast_path_result = try_parse(payload.transcript, context)
    if fast_path_result is not None:
        mutation_result = dispatch(fast_path_result, db)
        proposal = _fast_path_proposal()
        return build_transcript_response_from_mutation(
            mutation_result,
            payload,
            proposal,
        )

    # Note-intent guard: block before the LLM pipeline runs.
    # The LLM has no note tool — it hallucinates note text into role/comments
    # and the active-draft fallback would corrupt the draft.  Reject early with
    # a safe, informative message rather than letting any LLM path fire.
    if _has_note_intent(payload.transcript):
        return unsupported_response(
            payload,
            ["Could not add that note safely. No tracker changes were saved."],
        )

    # OLLAMA_MAX_TOOL_TURNS caps how many times interpret() may run for one transcript
    # request (initial call plus any clarification/schema-repair retries). Default is 2.
    max_tool_turns = max(1, interpreter.settings.max_tool_turns)
    interpret_calls = 1
    try:
        interpretation = interpreter.interpret(payload.transcript, context)
    except SemanticInterpreterUnavailableError as exc:
        return SemanticTranscriptResponse(status="unavailable", operation="none", raw_transcript=payload.transcript, proposal=empty_proposal(), warnings=[str(exc)])
    except SemanticInterpreterInvalidResponseError as exc:
        # No tool call returned — attempt field-extraction-based fallback before giving up.
        # Try to extract fields independently; if that also fails, surface the original error.
        try:
            fallback_extracted, fallback_metrics = interpreter.extract_fields(payload.transcript, context)
        except (SemanticInterpreterUnavailableError, SemanticInterpreterInvalidResponseError):
            fallback_extracted = None
            fallback_metrics = None
        if fallback_extracted is not None:
            fallback_fields, _fw = normalize_extracted_fields(fallback_extracted, transcript=payload.transcript)
            if fallback_fields is not None and _fields_can_create_or_patch_draft(fallback_fields):
                logger.info(
                    "semantic_no_tool_call_fallback_attempt transcript=%r extracted=%r",
                    payload.transcript,
                    _safe_extracted_field_log(fallback_fields.model_dump(exclude_none=True)),
                )
                fallback_response = resolve_no_tool_call_fallback(db, payload, fallback_fields, fallback_metrics)
                if fallback_response is not None:
                    return fallback_response
        return unsupported_response(payload, ["No recognized tracker changes were found."])

    extracted_fields, extraction_warnings = normalize_extracted_fields(
        interpretation.extracted_fields, transcript=payload.transcript
    )
    if extracted_fields is None:
        logger.warning(
            "semantic_field_extraction_failure reason=%r fields=%r",
            extraction_warnings,
            _safe_extracted_field_log(interpretation.extracted_fields.model_dump(exclude_none=True)),
        )
        return unsupported_response(payload, extraction_warnings, metrics=interpretation.metrics)

    logger.info(
        "semantic_raw_ollama_tool_call tool=%s arguments=%r",
        interpretation.proposal.tool_name,
        interpretation.proposal.arguments,
    )
    try:
        merged_proposal = merge_extracted_fields_into_proposal(db, interpretation.proposal, extracted_fields)
    except SemanticInterpreterInvalidResponseError as exc:
        logger.warning("merge_extracted_fields_invalid_shape tool=%s reason=%s", interpretation.proposal.tool_name, exc)
        merged_proposal = None
    if merged_proposal is None:
        return unsupported_response(
            payload,
            ["Extracted fields conflicted with selected tool arguments. No tracker changes were saved."],
            metrics=interpretation.metrics,
        )
    proposal = reconcile_explicit_company_candidates(db, merged_proposal, explicit_known_companies)
    metrics = interpretation.metrics

    if (
        proposal.tool_name == "ask_clarification"
        and proposal.arguments.get("question") == CLARIFICATION_MISSING_COMPANY
        and len(explicit_known_companies) == 1
        and interpret_calls < max_tool_turns
    ):
        retry_context = context | {
            "explicit_company_retry_hint": (
                f'Exactly one explicit known company appears in the current utterance: "{explicit_known_companies[0]}". '
                "Do not ask which company to use. Use that company if the selected tool accepts a company field."
            ),
            "normalized_extracted_fields": extracted_fields.model_dump(exclude_none=True),
        }
        try:
            retry_interpretation = interpreter.interpret(payload.transcript, retry_context)
            interpret_calls += 1
        except (SemanticInterpreterUnavailableError, SemanticInterpreterInvalidResponseError):
            retry_interpretation = None
        if retry_interpretation is not None:
            retry_extracted_fields, retry_extraction_warnings = normalize_extracted_fields(
                retry_interpretation.extracted_fields, transcript=payload.transcript
            )
            if retry_extracted_fields is None:
                return unsupported_response(payload, retry_extraction_warnings, metrics=retry_interpretation.metrics)
            try:
                merged_retry_proposal = merge_extracted_fields_into_proposal(db, retry_interpretation.proposal, retry_extracted_fields)
            except SemanticInterpreterInvalidResponseError as exc:
                logger.warning("merge_extracted_fields_invalid_shape tool=%s reason=%s", retry_interpretation.proposal.tool_name, exc)
                merged_retry_proposal = None
            if merged_retry_proposal is None:
                return unsupported_response(
                    payload,
                    ["Extracted fields conflicted with selected tool arguments. No tracker changes were saved."],
                    metrics=retry_interpretation.metrics,
                )
            proposal = reconcile_explicit_company_candidates(db, merged_retry_proposal, explicit_known_companies)
            metrics = retry_interpretation.metrics

    try:
        proposal, validated_arguments = validate_tool_arguments_with_safe_normalization(proposal)
        if validated_arguments is not None:
            logger.info(
                "semantic_post_schema_repair_arguments tool=%s arguments=%r",
                proposal.tool_name,
                proposal.arguments,
            )
    except ValidationError as exc:
        logger.warning(
            "semantic_tool_argument_validation_failed tool=%s arguments=%r reason=%r",
            proposal.tool_name,
            proposal.arguments,
            exc.errors(),
        )
        validated_arguments = None
        if proposal.tool_name == "patch_active_draft" and interpret_calls < max_tool_turns:
            retry_context = context | {
                "schema_repair_retry_hint": (
                    "Your previous tool arguments were invalid. Use the existing patch_active_draft schema. "
                    "Put company in fields.company. Put the role as a single string in fields.role. "
                    "Do not use fields.roles — role is a scalar string, not an array."
                ),
                "normalized_extracted_fields": extracted_fields.model_dump(exclude_none=True),
            }
            try:
                retry_interpretation = interpreter.interpret(payload.transcript, retry_context)
                interpret_calls += 1
            except (SemanticInterpreterUnavailableError, SemanticInterpreterInvalidResponseError):
                retry_interpretation = None
            if retry_interpretation is not None:
                logger.info(
                    "semantic_raw_ollama_tool_call tool=%s arguments=%r",
                    retry_interpretation.proposal.tool_name,
                    retry_interpretation.proposal.arguments,
                )
                retry_extracted_fields, retry_extraction_warnings = normalize_extracted_fields(
                    retry_interpretation.extracted_fields, transcript=payload.transcript
                )
                if retry_extracted_fields is None:
                    return unsupported_response(payload, retry_extraction_warnings, metrics=retry_interpretation.metrics)
                try:
                    merged_retry_proposal = merge_extracted_fields_into_proposal(db, retry_interpretation.proposal, retry_extracted_fields)
                except SemanticInterpreterInvalidResponseError as exc:
                    logger.warning("merge_extracted_fields_invalid_shape tool=%s reason=%s", retry_interpretation.proposal.tool_name, exc)
                    merged_retry_proposal = None
                if merged_retry_proposal is None:
                    return unsupported_response(
                        payload,
                        ["Extracted fields conflicted with selected tool arguments. No tracker changes were saved."],
                        metrics=retry_interpretation.metrics,
                    )
                proposal = reconcile_explicit_company_candidates(db, merged_retry_proposal, explicit_known_companies)
                metrics = retry_interpretation.metrics
                try:
                    proposal, validated_arguments = validate_tool_arguments_with_safe_normalization(proposal)
                    logger.info(
                        "semantic_post_schema_repair_arguments tool=%s arguments=%r",
                        proposal.tool_name,
                        proposal.arguments,
                    )
                except ValidationError:
                    return unsupported_response(payload, ["Local language interpreter returned invalid tool arguments. No tracker changes were saved."])
        if validated_arguments is None:
            # Before giving up: try deterministic routing using extracted fields.
            # This recovers the case where explicit create intent was present but
            # the LLM emitted invalid tool arguments (e.g. malformed patch_active_draft).
            if extracted_fields is not None and _fields_can_create_or_patch_draft(extracted_fields):
                logger.info(
                    "semantic_invalid_tool_args_fallback_attempt tool=%s transcript=%r extracted=%r",
                    proposal.tool_name,
                    payload.transcript,
                    _safe_extracted_field_log(extracted_fields.model_dump(exclude_none=True)),
                )
                fallback_response = resolve_semantic_fallback(
                    db=db,
                    payload=payload,
                    extracted_fields=extracted_fields,
                    metrics=metrics,
                    failure_kind="invalid_tool_arguments",
                )
                if fallback_response is not None:
                    return fallback_response
            if interpret_calls >= max_tool_turns:
                # Retry budget exhausted: ask the user to rephrase instead of looping further.
                return SemanticTranscriptResponse(
                    status="clarification_required",
                    operation="none",
                    raw_transcript=payload.transcript,
                    proposal=_proposal_with_clarification(CLARIFICATION_RETRY_EXHAUSTED),
                    warnings=["Local language interpreter returned invalid tool arguments. No tracker changes were saved."],
                    clarification_question=CLARIFICATION_RETRY_EXHAUSTED,
                    interpreter_metrics=metrics,
                )
            return unsupported_response(payload, ["Local language interpreter returned invalid tool arguments. No tracker changes were saved."])

    # Canonicalization may return (proposal, None) without raising ValidationError
    # (e.g. conflicting envelope values). Only treat as invalid args for schema-validated
    # tools — unknown/passthrough tools (attach_latest_browser_context) return None
    # intentionally and must fall through to the contextual patch fallback below.
    _SCHEMA_VALIDATED_TOOLS = {
        "patch_active_draft",
        "preview_existing_application_update",
        "request_draft_save",
        "ask_clarification",
        "archive_application",
        "explain_delete_policy",
        "discard_draft",
    }
    if validated_arguments is None and proposal.tool_name in _SCHEMA_VALIDATED_TOOLS:
        if extracted_fields is not None and _fields_can_create_or_patch_draft(extracted_fields):
            fallback_response = resolve_semantic_fallback(
                db=db,
                payload=payload,
                extracted_fields=extracted_fields,
                metrics=metrics,
                failure_kind="invalid_tool_arguments",
            )
            if fallback_response is not None:
                return fallback_response
        return unsupported_response(payload, ["Local language interpreter returned invalid tool arguments. No tracker changes were saved."])

    # -----------------------------------------------------------------------
    # Active-draft contextual patch fallback
    # -----------------------------------------------------------------------
    # Intercept non-mutation tool outcomes (ask_clarification, or tools that
    # don't produce a draft/application mutation) when:
    #   1. An active new-application draft exists in context
    #   2. Validated extracted fields contain one or more actionable patch fields
    #   3. The transcript does not express a lifecycle intent (save/discard/archive…)
    #   4. The transcript does not explicitly target a known saved application
    # → synthesise a patch_active_draft call against the active draft instead.
    # This handles short follow-up commands like "change status to in-touch" or
    # "role is AI Engineer, change employment type to fulltime" after a draft exists.
    _FALLBACK_ELIGIBLE_TOOLS = {"ask_clarification", "attach_latest_browser_context"}
    if (
        proposal.tool_name in _FALLBACK_ELIGIBLE_TOOLS
        and not _has_lifecycle_intent(payload.transcript)
        and not explicit_known_companies  # saved-target exclusion
        and fields_have_values(extracted_fields, allow_company=False)
    ):
        context_for_fallback = build_context_payload(payload)
        fallback_draft = build_context_draft(context_for_fallback)
        if fallback_draft is not None:
            logger.info(
                "semantic_active_draft_contextual_patch_fallback transcript=%r extracted=%r",
                payload.transcript,
                extracted_fields.model_dump(exclude_none=True),
            )
            fallback_fields = dict(extracted_fields.model_dump(exclude_none=True))
            fallback_proposal = SemanticToolCallProposal(
                tool_name="patch_active_draft",
                arguments={
                    "fields": fallback_fields,
                    "replace_explicit_fields": True,
                    "context_notes": [],
                },
            )
            try:
                fallback_proposal, fallback_args = validate_tool_arguments_with_safe_normalization(fallback_proposal)
            except Exception:
                fallback_args = None
            if fallback_args is not None:
                return handle_patch_active_draft(db, payload, fallback_proposal, fallback_args, metrics)

    # -----------------------------------------------------------------------
    # Explicit create-intent precedence override
    # -----------------------------------------------------------------------
    # When the transcript has explicit create intent (e.g. "add application for AI Engineer
    # at Neilsoft") but the LLM selected preview_existing_application_update, override the
    # tool selection and route to draft create/patch instead.
    if (
        proposal.tool_name == "preview_existing_application_update"
        and not _has_lifecycle_intent(payload.transcript)
        and _has_explicit_create_intent(payload.transcript)
        and extracted_fields is not None
    ):
        logger.info(
            "semantic_explicit_create_intent_override tool=%s transcript=%r",
            proposal.tool_name,
            payload.transcript,
        )
        override_response = resolve_semantic_fallback(
            db=db,
            payload=payload,
            extracted_fields=extracted_fields,
            metrics=metrics,
            failure_kind="unsupported_tool",
        )
        if override_response is not None:
            return override_response

    if proposal.tool_name == "patch_active_draft":
        arguments = validated_arguments
        return handle_patch_active_draft(db, payload, proposal, arguments, metrics)
    if proposal.tool_name == "preview_existing_application_update":
        arguments = validated_arguments
        return handle_preview_existing_application_update(db, payload, proposal, arguments, metrics)
    if proposal.tool_name == "request_draft_save":
        arguments = validated_arguments
        return handle_request_draft_save(payload, proposal, arguments, metrics, db=db)
    if proposal.tool_name == "attach_latest_browser_context":
        return handle_attach_latest_browser_context(db, payload, proposal, metrics)
    if proposal.tool_name == "ask_clarification":
        arguments = validated_arguments
        return handle_ask_clarification(payload, proposal, arguments, metrics, db=db)
    if proposal.tool_name == "archive_application":
        arguments = validated_arguments
        return handle_archive_application(db, payload, proposal, arguments, metrics)
    if proposal.tool_name == "explain_delete_policy":
        arguments = validated_arguments
        return handle_explain_delete_policy(db, payload, proposal, arguments, metrics)
    if proposal.tool_name == "discard_draft":
        arguments = validated_arguments
        return handle_discard_draft(db, payload, proposal, arguments, metrics)

    return SemanticTranscriptResponse(
        status="unsupported",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal,
        warnings=["No supported command was detected."],
        interpreter_metrics=metrics,
    )
