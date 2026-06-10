import re
from typing import TYPE_CHECKING

from .constants import ALLOWED_PRIORITIES, ALLOWED_LOCATIONS
from .mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _resolve_patch_target(context: dict) -> tuple[str | None, MutationTarget | None]:
    draft_id = context.get("draft_id")
    if draft_id is not None:
        return "patch_draft", MutationTarget(draft_id=str(draft_id))
    app_id = context.get("active_application_id")
    if app_id is not None:
        return "patch_application", MutationTarget(application_id=int(app_id))
    return None, None


def _resolve_application_id_by_company(company_name: str, context: dict) -> int | None:
    """Resolve a single non-archived application_id from context by company name match."""
    applications = context.get("applications")
    if not isinstance(applications, list):
        return None
    normalized_target = _normalize(company_name)
    matches = [
        app for app in applications
        if isinstance(app, dict)
        and _normalize(app.get("company", "")) == normalized_target
        and not app.get("archived_at")
    ]
    if len(matches) == 1:
        return matches[0].get("id")
    # If multiple rows share the same company (different roles), can't resolve by company alone
    return None


def _resolve_archived_application_id_by_company(company_name: str, context: dict) -> int | None:
    """Resolve a single archived application_id from context by company name match."""
    applications = context.get("applications")
    if not isinstance(applications, list):
        return None
    normalized_target = _normalize(company_name)
    matches = [
        app for app in applications
        if isinstance(app, dict)
        and _normalize(app.get("company", "")) == normalized_target
        and app.get("archived_at")
    ]
    if len(matches) == 1:
        return matches[0].get("id")
    return None


def try_parse(transcript: str, context: dict) -> MutationPayload | None:
    normalized = _normalize(transcript)

    # Save draft
    save_triggers = {"save it", "save", "save this"}
    if normalized in save_triggers:
        draft_id = context.get("draft_id")
        if draft_id is None:
            return None
        return MutationPayload(
            operation="save_draft",
            target=MutationTarget(draft_id=str(draft_id)),
            changes=ApplicationChanges(),
        )

    # Discard draft
    discard_triggers = {"discard it", "discard", "discard this", "cancel it", "cancel draft"}
    if normalized in discard_triggers:
        draft_id = context.get("draft_id")
        if draft_id is None:
            return None
        return MutationPayload(
            operation="discard_draft",
            target=MutationTarget(draft_id=str(draft_id)),
            changes=ApplicationChanges(),
        )

    # Priority high
    priority_high_triggers = {"priority high", "high priority", "set priority high", "set high priority"}
    if normalized in priority_high_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(priority="HIGH"),
        )

    # Priority medium
    priority_medium_triggers = {"priority medium", "medium priority", "set priority medium", "set medium priority"}
    if normalized in priority_medium_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(priority="MEDIUM"),
        )

    # Priority low
    priority_low_triggers = {"priority low", "low priority", "set priority low", "set low priority"}
    if normalized in priority_low_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(priority="LOW"),
        )

    # Location remote
    remote_triggers = {"remote", "it is remote", "location remote", "set remote", "working remotely"}
    if normalized in remote_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(location_mode="remote"),
        )

    # Location hybrid
    hybrid_triggers = {"hybrid", "it is hybrid", "location hybrid", "set hybrid"}
    if normalized in hybrid_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(location_mode="hybrid"),
        )

    # Location onsite
    onsite_triggers = {"onsite", "on site", "it is onsite", "location onsite", "set onsite", "on-site"}
    if normalized in onsite_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(location_mode="on-site"),
        )

    # Mark applied
    applied_triggers = {"mark applied", "applied", "i applied", "i have applied", "already applied", "status applied"}
    if normalized in applied_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(status="applied"),
        )

    # Mark rejected
    rejected_triggers = {"mark rejected", "rejected", "got rejected", "they rejected", "status rejected"}
    if normalized in rejected_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return None
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(status="rejected"),
        )

    # Archive application: "archive {company}", "remove {company}", "hide {company}"
    archive_prefixes = ["archive ", "remove ", "hide "]
    for prefix in archive_prefixes:
        if normalized.startswith(prefix):
            company_name = normalized[len(prefix):].strip()
            if company_name:
                application_id = _resolve_application_id_by_company(company_name, context)
                if application_id is None:
                    return None
                return MutationPayload(
                    operation="archive_application",
                    target=MutationTarget(application_id=application_id),
                    changes=ApplicationChanges(),
                )

    # Restore application: "restore {company}", "unarchive {company}", "bring back {company}"
    restore_prefixes = ["restore ", "unarchive ", "bring back "]
    for prefix in restore_prefixes:
        if normalized.startswith(prefix):
            company_name = normalized[len(prefix):].strip()
            if company_name:
                application_id = _resolve_archived_application_id_by_company(company_name, context)
                if application_id is None:
                    return None
                return MutationPayload(
                    operation="restore_application",
                    target=MutationTarget(application_id=application_id),
                    changes=ApplicationChanges(),
                )

    return None
