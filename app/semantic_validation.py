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
logger = logging.getLogger(__name__)


def empty_proposal() -> SemanticToolCallProposal:
    return SemanticToolCallProposal()


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
    normalized = value.strip().casefold()
    for option in STATUS_OPTIONS:
        if option.casefold() == normalized:
            return option
    return None


def normalize_priority(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return normalized if normalized in ALLOWED_PRIORITIES else None


def _normalize_lookup_text(value: str) -> str:
    return " ".join(value.replace("-", " ").replace("_", " ").strip().casefold().split())


def normalize_role_title(value: str) -> str:
    return " ".join(value.strip().split())


EMPLOYMENT_TYPE_ALIASES = {
    "internship": "Internship",
    "full time": "Full Time",
    "fulltime": "Full Time",
    "part time": "Part Time",
    "parttime": "Part Time",
}

LOCATION_ALIASES = {
    "remote": "remote",
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


def _normalize_role_alias_value(value: object) -> list[str] | None:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        normalized = [item.strip() for item in value if item.strip()]
        return normalized
    return None


def normalize_patch_active_draft_argument_shape(proposal: SemanticToolCallProposal) -> SemanticToolCallProposal:
    if proposal.tool_name != "patch_active_draft":
        return proposal

    arguments = dict(proposal.arguments)
    fields = arguments.get("fields")
    if not isinstance(fields, dict):
        return proposal

    normalized_fields = dict(fields)
    roles_present = "roles" in normalized_fields
    role_present = "role" in normalized_fields
    if not roles_present and not role_present:
        arguments["fields"] = normalized_fields
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)

    normalized_roles = _normalize_role_alias_value(normalized_fields.get("roles")) if roles_present else None
    normalized_role_alias = _normalize_role_alias_value(normalized_fields.get("role")) if role_present else None

    if roles_present and normalized_roles is None:
        return proposal
    if role_present and normalized_role_alias is None:
        return proposal

    if roles_present and role_present:
        if normalized_roles != normalized_role_alias:
            return proposal
        normalized_fields.pop("role", None)
        normalized_fields["roles"] = normalized_roles
        arguments["fields"] = normalized_fields
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)

    if role_present:
        normalized_fields.pop("role", None)
        normalized_fields["roles"] = normalized_role_alias
        arguments["fields"] = normalized_fields
        return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)

    normalized_fields["roles"] = normalized_roles
    arguments["fields"] = normalized_fields
    return SemanticToolCallProposal(tool_name=proposal.tool_name, arguments=arguments)


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
    try:
        interpretation = interpreter.interpret(payload.transcript, context)
    except SemanticInterpreterUnavailableError as exc:
        return SemanticTranscriptResponse(
            status="unavailable",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=empty_proposal(),
            warnings=[str(exc)],
        )
    except SemanticInterpreterInvalidResponseError as exc:
        return SemanticTranscriptResponse(
            status="unsupported",
            operation="none",
            raw_transcript=payload.transcript,
            proposal=empty_proposal(),
            warnings=[str(exc)],
        )

    logger.info(
        "semantic_raw_ollama_tool_call tool=%s arguments=%r",
        interpretation.proposal.tool_name,
        interpretation.proposal.arguments,
    )
    proposal = reconcile_explicit_company_candidates(db, interpretation.proposal, explicit_known_companies)
    metrics = interpretation.metrics

    if (
        proposal.tool_name == "ask_clarification"
        and proposal.arguments.get("question") == CLARIFICATION_MISSING_COMPANY
        and len(explicit_known_companies) == 1
    ):
        retry_context = context | {
            "explicit_company_retry_hint": (
                f'Exactly one explicit known company appears in the current utterance: "{explicit_known_companies[0]}". '
                "Do not ask which company to use. Use that company if the selected tool accepts a company field."
            )
        }
        try:
            retry_interpretation = interpreter.interpret(payload.transcript, retry_context)
        except (SemanticInterpreterUnavailableError, SemanticInterpreterInvalidResponseError):
            retry_interpretation = None
        if retry_interpretation is not None:
            proposal = reconcile_explicit_company_candidates(db, retry_interpretation.proposal, explicit_known_companies)
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
        if proposal.tool_name == "patch_active_draft":
            retry_context = context | {
                "schema_repair_retry_hint": (
                    "Your previous tool arguments were invalid. Use the existing patch_active_draft schema. "
                    "Put company in fields.company. Put one or more roles in fields.roles as a JSON array of strings. "
                    "Do not use fields.role unless you are mirroring the same value into fields.roles."
                )
            }
            try:
                retry_interpretation = interpreter.interpret(payload.transcript, retry_context)
            except (SemanticInterpreterUnavailableError, SemanticInterpreterInvalidResponseError):
                retry_interpretation = None
            if retry_interpretation is not None:
                logger.info(
                    "semantic_raw_ollama_tool_call tool=%s arguments=%r",
                    retry_interpretation.proposal.tool_name,
                    retry_interpretation.proposal.arguments,
                )
                proposal = reconcile_explicit_company_candidates(db, retry_interpretation.proposal, explicit_known_companies)
                metrics = retry_interpretation.metrics
                try:
                    proposal, validated_arguments = validate_tool_arguments_with_safe_normalization(proposal)
                    logger.info(
                        "semantic_post_schema_repair_arguments tool=%s arguments=%r",
                        proposal.tool_name,
                        proposal.arguments,
                    )
                except ValidationError:
                    return SemanticTranscriptResponse(
                        status="unsupported",
                        operation="none",
                        raw_transcript=payload.transcript,
                        proposal=empty_proposal(),
                        warnings=["Local language interpreter returned invalid tool arguments. No tracker changes were saved."],
                    )
        if validated_arguments is None:
            return SemanticTranscriptResponse(
                status="unsupported",
                operation="none",
                raw_transcript=payload.transcript,
                proposal=empty_proposal(),
                warnings=["Local language interpreter returned invalid tool arguments. No tracker changes were saved."],
            )

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
