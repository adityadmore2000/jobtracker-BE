import re

from .constants import ALLOWED_PRIORITIES, ALLOWED_LOCATIONS, STATUS_OPTIONS
from .mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget


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
            changes=ApplicationChanges(location_mode="onsite"),
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
            changes=ApplicationChanges(status="Applied"),
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
            changes=ApplicationChanges(status="Rejected"),
        )

    return None
