import logging

from sqlalchemy.orm import Session
from pydantic import ValidationError

from .company_resolution import detect_explicit_known_companies, get_application_matches_for_company, resolve_company_name
from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    STATUS_OPTIONS,
)
from .models import BrowserContext, JobApplication
from .schemas import JobApplicationCreate, SemanticTranscriptResponse, TranscriptParseRequest
from .semantic_interpreter import (
    OllamaSemanticInterpreter,
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterUnavailableError,
)
from .semantic_schemas import (
    AskClarificationArguments,
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
    normalized = _normalize_lookup_text(value)
    for option in STATUS_OPTIONS:
        if _normalize_lookup_text(option) == normalized:
            return option
    return None


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


def normalize_role_title(value: str) -> str:
    return " ".join(value.strip().split())


EMPLOYMENT_TYPE_ALIASES = {
    "internship": "Internship",
    "intern": "Internship",
    "full time": "Full Time",
    "fulltime": "Full Time",
    "part time": "Part Time",
    "parttime": "Part Time",
}

LOCATION_ALIASES = {
    "remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
    "hybrid": "hybrid",
    "onsite": "onsite",
    "on site": "onsite",
    "on-site": "onsite",
}


def normalize_roles(values: list[str] | None, *, tool_name: str | None = None) -> list[str] | None:
    if values is None:
        return None
    normalized_values: list[str] = []
    for value in values:
        normalized_role_value = normalize_role_title(value)
        if not normalized_role_value:
            logger.warning(
                "semantic_role_validation_failed tool=%s raw_role_value=%r normalized_role_value=%r reason=%r",
                tool_name,
                value,
                normalized_role_value,
                "blank_role_value",
            )
            return None
        if normalized_role_value not in normalized_values:
            normalized_values.append(normalized_role_value)
    return normalized_values


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

    roles = normalize_roles(fields.roles, tool_name=tool_name)
    if fields.roles is not None and roles is None:
        errors.append("Unsupported role value.")

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
            roles=roles,
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


def normalize_extracted_fields(fields: SemanticExtractedFields) -> tuple[SemanticFieldPatch | None, list[str]]:
    return validate_fields(
        SemanticFieldPatch.model_validate(fields.model_dump(exclude_none=True)),
        tool_name="semantic_field_extraction",
    )


def describe_application(application: JobApplication) -> str:
    if application.roles_json:
        return ", ".join(application.roles_json)
    return f"Application #{application.id}"


def build_ambiguous_update_question(company: str, matches: list[JobApplication]) -> str:
    return CLARIFICATION_AMBIGUOUS_APPLICATION


def filter_matches_by_role(matches: list[JobApplication], role: str | None) -> list[JobApplication]:
    if role is None:
        return matches
    normalized_requested_role = normalize_role_title(role)
    return [
        application
        for application in matches
        if any(normalize_role_title(existing_role).casefold() == normalized_requested_role.casefold() for existing_role in application.roles_json)
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
        if requested_role and not any(
            normalize_role_title(existing_role).casefold() == requested_role.casefold() for existing_role in application.roles_json
        ):
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

    try:
        draft = JobApplicationCreate(
            company=fields.company if fields.company is not None else base_draft.company,
            roles_json=list(fields.roles) if fields.roles is not None else list(base_draft.roles_json),
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
        normalized_active_role = normalize_role_title(active_role) if active_role else None
        roles = [normalized_active_role] if normalized_active_role else []
        if not company:
            return None
        return JobApplicationCreate(
            company=company,
            roles_json=[role for role in roles if role],
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
    roles_value = active_draft.get("roles")
    employment_types_value = active_draft.get("employment_types")
    current_stages_value = active_draft.get("current_stages")

    roles = normalize_roles(roles_value if isinstance(roles_value, list) else None) or []
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
        roles_json=roles,
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
    try:
        return JobApplicationCreate(
            company=application.company,
            roles_json=list(fields.roles) if fields.roles is not None else list(application.roles_json),
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
        "roles": fields.roles,
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
    for note in arguments.context_notes:
        warnings.append(f"Context note: {note}")
    final_confirmation_kind = draft_confirmation_kind if draft_confirmation_kind != "none" else confirmation_kind

    return SemanticTranscriptResponse(
        status="preview",
        operation="create",
        raw_transcript=payload.transcript,
        proposal=proposal,
        draft=draft,
        warnings=warnings,
        needs_confirmation=final_confirmation_kind == "context",
        confirmation_kind=final_confirmation_kind,
        interpreter_metrics=metrics,
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
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=target_warnings,
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

    return SemanticTranscriptResponse(
        status="preview",
        operation="update",
        raw_transcript=payload.transcript,
        proposal=proposal,
        application_id=application.id,
        draft=preview,
        warnings=target_warnings,
        interpreter_metrics=metrics,
    )


def handle_request_draft_save(
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    _arguments: RequestDraftSaveArguments,
    metrics,
) -> SemanticTranscriptResponse:
    context = build_context_payload(payload)
    if build_context_draft(context) is None:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[],
            clarification_question=CLARIFICATION_NO_ACTIVE_DRAFT,
            interpreter_metrics=metrics,
        )

    return SemanticTranscriptResponse(
        status="preview",
        operation="create",
        raw_transcript=payload.transcript,
        proposal=proposal,
        draft=build_context_draft(context),
        warnings=["Use the existing Save action to persist this draft."],
        needs_confirmation=True,
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
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=[],
            clarification_question="There is no active draft to attach the current link to.",
            interpreter_metrics=metrics,
        )

    context = db.query(BrowserContext).order_by(BrowserContext.captured_at.desc(), BrowserContext.id.desc()).first()
    if context is None:
        return SemanticTranscriptResponse(
            status="clarification_required",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=proposal,
            warnings=["No browser context is available."],
            clarification_question="Open a job page first, then try again.",
            interpreter_metrics=metrics,
        )

    question = f'Latest browser context: "{context.page_title}" at {context.url}. Which tracker fields should I fill from it?'
    return SemanticTranscriptResponse(
        status="clarification_required",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal,
        clarification_question=question,
        interpreter_metrics=metrics,
    )


def handle_ask_clarification(
    payload: TranscriptParseRequest,
    proposal: SemanticToolCallProposal,
    arguments: AskClarificationArguments,
    metrics,
) -> SemanticTranscriptResponse:
    return SemanticTranscriptResponse(
        status="clarification_required",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal,
        clarification_question=arguments.question,
        interpreter_metrics=metrics,
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


def _normalize_role_values(value: object) -> object:
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        raw_values = value
    else:
        return _INVALID

    normalized_values: list[str] = []
    for raw_value in raw_values:
        normalized = normalize_role_title(raw_value)
        if not normalized:
            return _INVALID
        if normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


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

    alias_merge_specs = (
        ("roles", ("role",), _normalize_role_values),
        ("employment_types", ("employment_type", "type"), _normalize_employment_type_values),
        ("current_stages", ("current_stage", "stage"), _normalize_stage_values),
    )
    for canonical_key, alias_keys, normalizer in alias_merge_specs:
        merged = _merge_field_aliases(normalized_fields, canonical_key, alias_keys, normalizer)
        if merged is _INVALID or merged is _CONFLICT:
            return None

    scalar_normalizers = {
        "company": _normalize_company_value,
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
        arguments = dict(proposal.arguments)
        fields = dict(arguments.get("fields") or {})
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
        arguments = dict(proposal.arguments)
        target = dict(arguments.get("target") or {})
        fields = dict(arguments.get("fields") or {})
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
    normalized_proposal = normalize_patch_active_draft_argument_shape(proposal)
    if normalized_proposal.tool_name == "patch_active_draft":
        return normalized_proposal, PatchActiveDraftArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "preview_existing_application_update":
        return normalized_proposal, PreviewExistingApplicationUpdateArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "request_draft_save":
        return normalized_proposal, RequestDraftSaveArguments.model_validate(normalized_proposal.arguments)
    if normalized_proposal.tool_name == "ask_clarification":
        return normalized_proposal, AskClarificationArguments.model_validate(normalized_proposal.arguments)
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


def interpret_transcript_command(
    db: Session,
    payload: TranscriptParseRequest,
    interpreter: OllamaSemanticInterpreter,
) -> SemanticTranscriptResponse:
    context, explicit_known_companies = build_interpreter_context(db, payload)
    # OLLAMA_MAX_TOOL_TURNS caps how many times interpret() may run for one transcript
    # request (initial call plus any clarification/schema-repair retries). Default is 2.
    max_tool_turns = max(1, interpreter.settings.max_tool_turns)
    interpret_calls = 1
    try:
        interpretation = interpreter.interpret(payload.transcript, context)
    except SemanticInterpreterUnavailableError as exc:
        return SemanticTranscriptResponse(status="unavailable", operation="none", raw_transcript=payload.transcript, proposal=empty_proposal(), warnings=[str(exc)])
    except SemanticInterpreterInvalidResponseError as exc:
        return unsupported_response(payload, [str(exc)])

    extracted_fields, extraction_warnings = normalize_extracted_fields(interpretation.extracted_fields)
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
    merged_proposal = merge_extracted_fields_into_proposal(db, interpretation.proposal, extracted_fields)
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
            retry_extracted_fields, retry_extraction_warnings = normalize_extracted_fields(retry_interpretation.extracted_fields)
            if retry_extracted_fields is None:
                return unsupported_response(payload, retry_extraction_warnings, metrics=retry_interpretation.metrics)
            merged_retry_proposal = merge_extracted_fields_into_proposal(db, retry_interpretation.proposal, retry_extracted_fields)
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
                    "Put company in fields.company. Put one or more roles in fields.roles as a JSON array of strings. "
                    "Do not use fields.role unless you are mirroring the same value into fields.roles."
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
                retry_extracted_fields, retry_extraction_warnings = normalize_extracted_fields(retry_interpretation.extracted_fields)
                if retry_extracted_fields is None:
                    return unsupported_response(payload, retry_extraction_warnings, metrics=retry_interpretation.metrics)
                merged_retry_proposal = merge_extracted_fields_into_proposal(db, retry_interpretation.proposal, retry_extracted_fields)
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

    if proposal.tool_name == "patch_active_draft":
        arguments = validated_arguments
        return handle_patch_active_draft(db, payload, proposal, arguments, metrics)
    if proposal.tool_name == "preview_existing_application_update":
        arguments = validated_arguments
        return handle_preview_existing_application_update(db, payload, proposal, arguments, metrics)
    if proposal.tool_name == "request_draft_save":
        arguments = validated_arguments
        return handle_request_draft_save(payload, proposal, arguments, metrics)
    if proposal.tool_name == "attach_latest_browser_context":
        return handle_attach_latest_browser_context(db, payload, proposal, metrics)
    if proposal.tool_name == "ask_clarification":
        arguments = validated_arguments
        return handle_ask_clarification(payload, proposal, arguments, metrics)

    return SemanticTranscriptResponse(
        status="unsupported",
        operation="none",
        raw_transcript=payload.transcript,
        proposal=proposal,
        warnings=["No supported command was detected."],
        interpreter_metrics=metrics,
    )
