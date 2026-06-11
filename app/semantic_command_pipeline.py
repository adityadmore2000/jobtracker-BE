"""
Deterministic pipeline that turns a strict ``SemanticCommand`` (produced by the
single-call extractor) into a safe outcome:

    raw SemanticCommand
      → blank-string sanitization
      → alias normalization
      → enum validation
      → intent-specific validation
      → deterministic target resolution (DB-verified)
      → MutationPayload (handed to the dispatcher)
        OR clarification (with pending_command)
        OR suggestion-only / unsupported (NO mutation)

The LLM understands language; THIS module controls mutations. Nothing here trusts
an LLM-provided ``application_id`` — every target is verified against the DB.

No mutation is ever partially applied: if any field is invalid the whole command
is downgraded to a clarification / suggestion outcome.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from .constants import (
    normalize_current_stage_value,
    normalize_employment_type_value,
    normalize_location_value,
    normalize_priority_value,
    normalize_status_value,
)
from .models import JobApplication
from .mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from .role_resolution import normalize_role_name
from .semantic_command_schemas import SemanticChanges, SemanticCommand

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline outcome types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DispatchOutcome:
    """A safe, fully-resolved mutation ready for the dispatcher."""
    payload: MutationPayload


@dataclass
class ClarificationOutcome:
    """A supported intent that needs one more piece of identity."""
    question: str
    pending_command: dict


@dataclass
class SuggestionOutcome:
    """No safe mutation could be produced. Optionally offers rephrasings."""
    message: str
    clarification_question: Optional[str] = None
    suggested_phrasings: list[str] = field(default_factory=list)


@dataclass
class MixedIntentOutcome:
    """Transcript contained both a field update and a note. No mutation."""
    message: str = (
        "I found both an application update and a note. "
        "Please update the fields and add the note in separate messages."
    )


PipelineOutcome = DispatchOutcome | ClarificationOutcome | SuggestionOutcome | MixedIntentOutcome


_GENERIC_EXAMPLES = [
    "set priority as medium",
    "set location as on-site",
    "add a note saying recruiter replied",
]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — sanitization
# ──────────────────────────────────────────────────────────────────────────────

def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = " ".join(value.split())
    return stripped or None


def _clean_list(values: Optional[list[str]]) -> Optional[list[str]]:
    if values is None:
        return None
    cleaned = [" ".join(v.split()) for v in values if isinstance(v, str) and v.strip()]
    return cleaned or None


# ──────────────────────────────────────────────────────────────────────────────
# Step 2+3 — alias normalization + enum validation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _NormalizedChanges:
    changes: ApplicationChanges
    invalid: list[str]  # human-readable descriptions of fields that failed


def _normalize_and_validate_changes(raw: SemanticChanges) -> _NormalizedChanges:
    invalid: list[str] = []
    out = ApplicationChanges()

    status = _clean_str(raw.status)
    if status is not None:
        canonical = normalize_status_value(status)
        if canonical is None:
            invalid.append(f"status={status!r}")
        else:
            out.status = canonical

    priority = _clean_str(raw.priority)
    if priority is not None:
        canonical = normalize_priority_value(priority)
        if canonical is None:
            invalid.append(f"priority={priority!r}")
        else:
            out.priority = canonical

    location = _clean_str(raw.location_mode)
    if location is not None:
        canonical = normalize_location_value(location)
        if canonical is None:
            invalid.append(f"location={location!r}")
        else:
            out.location_mode = canonical

    employment = _clean_list(raw.employment_types)
    if employment is not None:
        canonical_list = []
        for item in employment:
            canonical = normalize_employment_type_value(item)
            if canonical is None:
                invalid.append(f"employment_type={item!r}")
            else:
                canonical_list.append(canonical)
        if canonical_list and not invalid:
            out.employment_types = _dedupe(canonical_list)

    stages = _clean_list(raw.current_stages)
    if stages is not None:
        canonical_list = []
        for item in stages:
            canonical = normalize_current_stage_value(item)
            if canonical is None:
                invalid.append(f"current_stage={item!r}")
            else:
                canonical_list.append(canonical)
        if canonical_list and not invalid:
            out.current_stages = _dedupe(canonical_list)

    job_link = _clean_str(raw.job_link)
    if job_link is not None:
        out.job_link = job_link

    next_action = _clean_str(raw.next_action)
    if next_action is not None:
        out.next_action = next_action

    comments = _clean_str(raw.comments)
    if comments is not None:
        out.comments = comments

    if raw.engaged_days is not None:
        if isinstance(raw.engaged_days, int) and raw.engaged_days >= 0:
            out.engaged_days = raw.engaged_days
        else:
            invalid.append(f"engaged_days={raw.engaged_days!r}")

    return _NormalizedChanges(changes=out, invalid=invalid)


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def _changes_has_any(changes: ApplicationChanges) -> bool:
    return any(
        getattr(changes, f) is not None
        for f in (
            "status", "priority", "location_mode", "employment_types",
            "current_stages", "job_link", "engaged_days", "next_action", "comments",
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — deterministic target resolution (DB-verified)
# ──────────────────────────────────────────────────────────────────────────────

def _active_applications(db: Session) -> list[JobApplication]:
    return (
        db.query(JobApplication)
        .filter(JobApplication.is_draft == False)  # noqa: E712
        .filter(JobApplication.archived_at == None)  # noqa: E711
        .all()
    )


def _match_by_company(company: str, apps: list[JobApplication]) -> list[JobApplication]:
    target = " ".join(company.casefold().split())
    return [a for a in apps if " ".join((a.company or "").casefold().split()) == target]


def _match_by_company_role(company: str, role: str, apps: list[JobApplication]) -> list[JobApplication]:
    target_company = " ".join(company.casefold().split())
    target_role = normalize_role_name(role)
    return [
        a for a in apps
        if " ".join((a.company or "").casefold().split()) == target_company
        and (a.normalized_role or "") == target_role
    ]


@dataclass
class _ResolvedTarget:
    operation: Optional[str] = None
    target: Optional[MutationTarget] = None
    clarification: Optional[str] = None
    missing_field: Optional[str] = None  # "company" | "role"


def _resolve_update_target(
    cmd: SemanticCommand,
    context: dict,
    db: Session,
) -> _ResolvedTarget:
    """Resolve the application to update for intent=update_application.

    Precedence:
      explicit company (+role) in the transcript  →  resolve against DB
      else active draft in context                →  patch_draft
      else active application in context           →  create_application_update_draft
      else                                         →  clarification (which app?)
    """
    company = _clean_str(cmd.target.company)
    role = _clean_str(cmd.target.role)

    # Explicit target wins, but only if it resolves safely.
    if company:
        active = _active_applications(db)
        if role:
            matches = _match_by_company_role(company, role, active)
            if len(matches) == 1:
                return _ResolvedTarget(
                    operation="create_application_update_draft",
                    target=MutationTarget(application_id=matches[0].id),
                )
            if not matches:
                return _ResolvedTarget(clarification=f"No active application found for {company} — {role}.")
            return _ResolvedTarget(clarification=f"Multiple applications match {company} — {role}. Please clarify.")
        matches = _match_by_company(company, active)
        if len(matches) == 1:
            return _ResolvedTarget(
                operation="create_application_update_draft",
                target=MutationTarget(application_id=matches[0].id),
            )
        if len(matches) > 1:
            roles = ", ".join(a.role or "(no role)" for a in matches)
            return _ResolvedTarget(
                clarification=f"Which role at {company} should I update? ({roles})",
                missing_field="role",
            )
        return _ResolvedTarget(clarification=f"No active application found for {company}.")

    # No explicit target — fall back to selected context.
    draft_id = context.get("draft_id")
    if draft_id is not None:
        return _ResolvedTarget(operation="patch_draft", target=MutationTarget(draft_id=str(draft_id)))

    active_app_id = context.get("active_application_id")
    if active_app_id is not None:
        app = db.get(JobApplication, int(active_app_id))
        if app is not None and not app.is_draft and app.archived_at is None:
            return _ResolvedTarget(
                operation="create_application_update_draft",
                target=MutationTarget(application_id=app.id),
            )

    return _ResolvedTarget(
        clarification="Which application should I update?",
        missing_field="company",
    )


def _resolve_archive_target(cmd: SemanticCommand, context: dict, db: Session) -> _ResolvedTarget:
    company = _clean_str(cmd.target.company)
    role = _clean_str(cmd.target.role)
    active = _active_applications(db)

    if company:
        if role:
            matches = _match_by_company_role(company, role, active)
        else:
            matches = _match_by_company(company, active)
        if len(matches) == 1:
            return _ResolvedTarget(
                operation="archive_application",
                target=MutationTarget(application_id=matches[0].id),
            )
        if not matches:
            label = f"{company} — {role}" if role else company
            return _ResolvedTarget(clarification=f"No active application found for {label}.")
        roles = ", ".join(a.role or "(no role)" for a in matches)
        return _ResolvedTarget(
            clarification=f"Which role at {company} should I archive? ({roles})",
            missing_field="role",
        )

    active_app_id = context.get("active_application_id")
    if active_app_id is not None:
        app = db.get(JobApplication, int(active_app_id))
        if app is not None and not app.is_draft and app.archived_at is None:
            return _ResolvedTarget(operation="archive_application", target=MutationTarget(application_id=app.id))

    return _ResolvedTarget(clarification="Which application should I archive?", missing_field="company")


def _resolve_note_target(cmd: SemanticCommand, context: dict, db: Session) -> _ResolvedTarget:
    company = _clean_str(cmd.target.company)
    role = _clean_str(cmd.target.role)

    if company:
        active = _active_applications(db)
        if role:
            matches = _match_by_company_role(company, role, active)
            if len(matches) == 1:
                return _ResolvedTarget(operation="append_note", target=MutationTarget(application_id=matches[0].id))
            if not matches:
                return _ResolvedTarget(clarification=f"No active application found for {company} — {role}.")
        company_matches = _match_by_company(company, active)
        if len(company_matches) == 1:
            return _ResolvedTarget(operation="append_note", target=MutationTarget(application_id=company_matches[0].id))
        if len(company_matches) > 1:
            roles = ", ".join(a.role or "(no role)" for a in company_matches)
            return _ResolvedTarget(
                clarification=f"Which role at {company} should I add this note to? ({roles})",
                missing_field="role",
            )
        return _ResolvedTarget(clarification=f"No active application found for {company}.")

    active_app_id = context.get("active_application_id")
    if active_app_id is not None:
        app = db.get(JobApplication, int(active_app_id))
        if app is not None:
            return _ResolvedTarget(operation="append_note", target=MutationTarget(application_id=app.id))

    draft_id = context.get("draft_id")
    if draft_id is not None:
        return _ResolvedTarget(operation="append_note", target=MutationTarget(draft_id=str(draft_id)))

    return _ResolvedTarget(
        clarification="Which application should I add this note to?",
        missing_field="company",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Suggestion generation (Level 1 deterministic templates)
# ──────────────────────────────────────────────────────────────────────────────

_PRIORITY_WORD = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}


def _deterministic_suggestions(cmd: SemanticCommand) -> list[str]:
    """Build safe templated rephrasings from whatever partial meaning we have."""
    company = _clean_str(cmd.target.company)
    suggestions: list[str] = []
    norm = _normalize_and_validate_changes(cmd.changes)
    c = norm.changes

    if company:
        if c.priority:
            suggestions.append(f"set priority of {company} to {_PRIORITY_WORD.get(c.priority, c.priority.lower())}")
        if c.location_mode:
            suggestions.append(f"set location of {company} to {c.location_mode}")
        if c.status:
            suggestions.append(f"set status of {company} to {c.status}")
        if c.employment_types:
            suggestions.append(f"set employment type of {company} to {c.employment_types[0].lower()}")
    return _dedupe(suggestions)


# ──────────────────────────────────────────────────────────────────────────────
# Intent handlers
# ──────────────────────────────────────────────────────────────────────────────

def _handle_create(cmd: SemanticCommand) -> PipelineOutcome:
    company = _clean_str(cmd.target.company)
    role = _clean_str(cmd.target.role)
    if cmd.note:
        return MixedIntentOutcome()
    if not company:
        return SuggestionOutcome(
            message="I could not tell which company to create an application for.",
            suggested_phrasings=["add application for AI Engineer at Acme"],
        )
    norm = _normalize_and_validate_changes(cmd.changes)
    if norm.invalid:
        logger.info("semantic_single_extractor_rejected reason=invalid_fields fields=%s", norm.invalid)
        return SuggestionOutcome(
            message="I could not apply some of those values.",
            suggested_phrasings=_deterministic_suggestions(cmd) or list(_GENERIC_EXAMPLES),
        )
    changes = norm.changes
    changes.company = company
    if role:
        changes.role = role
    return DispatchOutcome(
        MutationPayload(operation="create_draft", target=MutationTarget(), changes=changes)
    )


def _handle_update(cmd: SemanticCommand, context: dict, db: Session) -> PipelineOutcome:
    if cmd.note:
        return MixedIntentOutcome()
    norm = _normalize_and_validate_changes(cmd.changes)
    if norm.invalid:
        logger.info("semantic_single_extractor_rejected reason=invalid_fields fields=%s", norm.invalid)
        return SuggestionOutcome(
            message="I am not sure which field you want to change.",
            clarification_question=_deterministic_suggestions(cmd)[0] if _deterministic_suggestions(cmd) else None,
            suggested_phrasings=_deterministic_suggestions(cmd) or list(_GENERIC_EXAMPLES),
        )
    if not _changes_has_any(norm.changes):
        # No concrete field to change → suggestion only (never guess).
        return SuggestionOutcome(
            message="I am not sure which field you want to change.",
            suggested_phrasings=_deterministic_suggestions(cmd) or list(_GENERIC_EXAMPLES),
        )

    resolved = _resolve_update_target(cmd, context, db)
    logger.info(
        "semantic_single_extractor_target_resolution result=%s",
        resolved.operation or f"clarification:{resolved.missing_field}",
    )
    if resolved.clarification:
        return ClarificationOutcome(
            question=resolved.clarification,
            pending_command=_pending_command_for_update(cmd, norm.changes, resolved.missing_field),
        )

    payload = MutationPayload(
        operation=resolved.operation,
        target=resolved.target,
        changes=norm.changes,
    )
    return DispatchOutcome(payload)


def _handle_append_note(cmd: SemanticCommand, context: dict, db: Session) -> PipelineOutcome:
    note = _clean_str(cmd.note)
    if not note:
        return SuggestionOutcome(
            message="I could not find any note text.",
            suggested_phrasings=["add a note saying recruiter replied"],
        )
    norm = _normalize_and_validate_changes(cmd.changes)
    if _changes_has_any(norm.changes) or norm.invalid:
        # Note intent must not carry field updates.
        return MixedIntentOutcome()

    resolved = _resolve_note_target(cmd, context, db)
    logger.info(
        "semantic_single_extractor_target_resolution result=%s",
        resolved.operation or f"clarification:{resolved.missing_field}",
    )
    if resolved.clarification:
        return ClarificationOutcome(
            question=resolved.clarification,
            pending_command=_pending_command_for_note(cmd, note, resolved.missing_field),
        )

    payload = MutationPayload(
        operation="append_note",
        target=resolved.target,
        changes=ApplicationChanges(),
        notes_to_append=[note],
    )
    return DispatchOutcome(payload)


def _handle_archive(cmd: SemanticCommand, context: dict, db: Session) -> PipelineOutcome:
    if cmd.note or _changes_has_any(_normalize_and_validate_changes(cmd.changes).changes):
        return SuggestionOutcome(
            message="Archiving cannot include field changes or a note. Please send them separately.",
            suggested_phrasings=["archive Acme AI Engineer"],
        )
    resolved = _resolve_archive_target(cmd, context, db)
    if resolved.clarification:
        return ClarificationOutcome(
            question=resolved.clarification,
            pending_command=_pending_command_for_archive(cmd, resolved.missing_field),
        )
    return DispatchOutcome(
        MutationPayload(
            operation="archive_application",
            target=resolved.target,
            changes=ApplicationChanges(),
        )
    )


def _handle_unsupported(cmd: SemanticCommand) -> PipelineOutcome:
    # Prefer model-proposed phrasings (validated downstream by the endpoint's
    # dry-run check) then deterministic templates, then generic examples.
    proposed = _clean_list(cmd.suggested_phrasings) or []
    deterministic = _deterministic_suggestions(cmd)
    suggestions = _dedupe([*proposed, *deterministic])
    if suggestions:
        return SuggestionOutcome(
            message="I am not sure which field you want to change.",
            clarification_question=(deterministic[0] if deterministic else None),
            suggested_phrasings=suggestions,
        )
    return SuggestionOutcome(
        message="I could not identify a tracker update.",
        suggested_phrasings=list(_GENERIC_EXAMPLES),
    )


# ──────────────────────────────────────────────────────────────────────────────
# pending_command builders (strict shape echoed to the frontend)
# ──────────────────────────────────────────────────────────────────────────────

def _changes_to_pending_dict(changes: ApplicationChanges) -> dict:
    return {k: v for k, v in changes.model_dump().items() if v is not None and k not in {"company", "role"}}


def _pending_command_for_update(cmd: SemanticCommand, changes: ApplicationChanges, missing: Optional[str]) -> dict:
    return {
        "operation": "update_application",
        "target": {
            "company": _clean_str(cmd.target.company),
            "role": _clean_str(cmd.target.role),
            "application_id": None,
        },
        "changes": _changes_to_pending_dict(changes),
        "note": None,
        "missing_field": missing or "company",
    }


def _pending_command_for_note(cmd: SemanticCommand, note: str, missing: Optional[str]) -> dict:
    return {
        "operation": "append_note",
        "target": {
            "company": _clean_str(cmd.target.company),
            "role": _clean_str(cmd.target.role),
            "application_id": None,
        },
        "changes": {},
        "note": note,
        "missing_field": missing or "company",
    }


def _pending_command_for_archive(cmd: SemanticCommand, missing: Optional[str]) -> dict:
    return {
        "operation": "archive_application",
        "target": {
            "company": _clean_str(cmd.target.company),
            "role": _clean_str(cmd.target.role),
            "application_id": None,
        },
        "changes": {},
        "note": None,
        "missing_field": missing or "company",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def resolve_semantic_command(cmd: SemanticCommand, context: dict, db: Session) -> PipelineOutcome:
    """Validate + resolve a SemanticCommand into a safe PipelineOutcome.

    Never raises for ordinary invalid input — always returns a safe outcome.
    """
    logger.info("semantic_single_extractor_validated intent=%s", cmd.intent)

    if cmd.intent == "create_application":
        return _handle_create(cmd)
    if cmd.intent == "update_application":
        return _handle_update(cmd, context, db)
    if cmd.intent == "append_note":
        return _handle_append_note(cmd, context, db)
    if cmd.intent == "archive_application":
        return _handle_archive(cmd, context, db)
    # unsupported
    return _handle_unsupported(cmd)
