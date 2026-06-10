import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .company_matching import has_meaningful_company_difference, normalize_company_name
from .company_resolution import (
    ensure_canonical_company,
    get_application_matches_for_company,
    get_canonical_company_by_normalized_name,
    get_company_alias_by_normalized_name,
    get_or_create_company,
    resolve_company_name,
)
from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    ALLOWED_ROLES,
    STATUS_OPTIONS,
)
from .database import get_db
from .livekit_config import LiveKitConfigurationError, get_livekit_settings
from .migrations import run_startup_migrations_if_enabled
from .models import ApplicationEvent, ApplicationNote, AsrCompanyCorrectionEvent, BrowserContext, CanonicalCompany, Company, CompanyAlias, JobApplication
from .mutation_dispatcher import dispatch
from .mutation_schemas import MutationPayload, MutationTarget, ApplicationChanges
from .public_schemas import PublicApplicationDTO, PublicTranscriptResponse
from .role_resolution import normalize_role_name
from .transcript_response_adapter import to_public_application, to_public_transcript_response
from .schemas import (
    ApplicationCompanyConfirmationRequest,
    ApplicationCreateCandidateRequest,
    ApplicationCreateCandidateResponse,
    AsrHotwordsResponse,
    BrowserContextCreate,
    BrowserContextResponse,
    DraftPatchRequest,
    JobApplicationCreate,
    JobApplicationRead,
    JobApplicationUpdate,
    LiveKitTokenRequest,
    LiveKitTokenResponse,
    SemanticTranscriptResponse,
    TranscriptParseRequest,
)
from .semantic_interpreter import OllamaSemanticInterpreter, get_semantic_interpreter
from .semantic_validation import interpret_transcript_command, normalize_role_title


@asynccontextmanager
async def lifespan(_app: FastAPI):
    run_startup_migrations_if_enabled()
    yield


app = FastAPI(title="Job Tracker API", lifespan=lifespan)
HOTWORD_LIMIT = 100
DEFAULT_LIVEKIT_ROOM_NAME = "job-tracker-local"
LIVEKIT_BROWSER_TOKEN_TTL = timedelta(minutes=15)
STATIC_HOTWORDS = [
    *ALLOWED_ROLES,
    *ALLOWED_EMPLOYMENT_TYPES,
    *ALLOWED_CURRENT_STAGES,
    *ALLOWED_LOCATIONS,
    *ALLOWED_PRIORITIES,
    *STATUS_OPTIONS,
    "referral",
    "next action",
]


def get_frontend_origins() -> list[str]:
    configured_origin = os.getenv("FRONTEND_ORIGIN", "")
    configured_origins = [origin.strip() for origin in configured_origin.split(",") if origin.strip()]
    local_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    return list(dict.fromkeys([*local_origins, *configured_origins]))


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(),
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_job_application(db: Session, payload: JobApplicationCreate) -> JobApplication:
    company_obj = get_or_create_company(db, payload.company)
    application = JobApplication(
        company_id=company_obj.id,
        role=payload.role,
        normalized_role=normalize_role_name(payload.role),
        employment_types_json=list(payload.employment_types_json),
        job_link=payload.job_link,
        location=payload.location,
        status=payload.status,
        current_stages_json=list(payload.current_stages_json),
        priority=payload.priority,
        engaged_days=payload.engaged_days,
        next_action=payload.next_action,
        comments=payload.comments,
    )
    db.add(application)
    db.flush()
    return application


def find_duplicate_role(db: Session, company_name: str, incoming_role: str) -> bool:
    """Return True if there is already a non-archived saved application at this company with this role."""
    normalized_incoming = normalize_role_title(incoming_role).casefold()
    for application in get_application_matches_for_company(db, company_name):
        if normalize_role_title(application.role).casefold() == normalized_incoming:
            return True
    return False


def maybe_create_alias(
    db: Session,
    canonical_company: CanonicalCompany,
    original_company_name: str | None,
) -> bool:
    if not original_company_name or not has_meaningful_company_difference(original_company_name, canonical_company.canonical_name):
        return False

    normalized_original = normalize_company_name(original_company_name)
    existing_alias = get_company_alias_by_normalized_name(db, normalized_original)
    if existing_alias is not None:
        return False

    if get_canonical_company_by_normalized_name(db, normalized_original) is not None:
        return False

    db.add(CompanyAlias(canonical_company_id=canonical_company.id, alias_text=original_company_name.strip()))
    db.flush()
    return True


def maybe_create_correction_event(
    db: Session,
    payload: ApplicationCompanyConfirmationRequest,
    canonical_company: CanonicalCompany,
    application: JobApplication,
    alias_created: bool,
) -> None:
    if payload.raw_transcript is None and payload.original_extracted_company_name is None and payload.audio_reference is None:
        return

    db.add(
        AsrCompanyCorrectionEvent(
            raw_transcript=payload.raw_transcript or "",
            original_extracted_company_name=payload.original_extracted_company_name or "",
            confirmed_company_name=canonical_company.canonical_name,
            canonical_company_id=canonical_company.id,
            application_id=application.id,
            alias_created=alias_created,
            audio_reference=payload.audio_reference,
        )
    )


def build_hotword_list(db: Session, limit: int = HOTWORD_LIMIT) -> list[str]:
    hotwords: list[str] = []
    normalized_seen: set[str] = set()

    def add_values(values: list[str]) -> None:
        for value in values:
            cleaned = value.strip()
            normalized = normalize_company_name(cleaned)
            if not cleaned or not normalized or normalized in normalized_seen:
                continue
            hotwords.append(cleaned)
            normalized_seen.add(normalized)
            if len(hotwords) >= limit:
                return

    canonical_values = [row.canonical_name for row in db.query(CanonicalCompany).order_by(CanonicalCompany.canonical_name.asc()).all()]

    add_values(canonical_values)
    if len(hotwords) < limit:
        add_values(STATIC_HOTWORDS)

    return hotwords[:limit]


def create_livekit_browser_token(room_name: str, participant_identity: str) -> tuple[str, datetime, str]:
    settings = get_livekit_settings()
    expires_at = datetime.now(timezone.utc) + LIVEKIT_BROWSER_TOKEN_TTL
    access_token = (
        AccessToken(settings.api_key, settings.api_secret)
        .with_identity(participant_identity)
        .with_ttl(LIVEKIT_BROWSER_TOKEN_TTL)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                can_publish_sources=["microphone"],
            )
        )
        .to_jwt()
    )
    return access_token, expires_at, settings.url


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/semantic-interpreter/health")
async def semantic_interpreter_health(interpreter: OllamaSemanticInterpreter = Depends(get_semantic_interpreter)) -> dict[str, str]:
    return interpreter.health_check()


@app.patch("/drafts/{draft_id}", response_model=PublicApplicationDTO)
async def patch_draft(
    draft_id: int,
    payload: DraftPatchRequest,
    db: Session = Depends(get_db),
) -> PublicApplicationDTO:
    app_row = db.get(JobApplication, draft_id)
    if app_row is None or not app_row.is_draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")

    changes = ApplicationChanges(
        company=payload.company,
        role=payload.role,
        status=payload.status,
        priority=payload.priority,
        location_mode=payload.location,
        job_link=payload.job_link,
        employment_types=payload.employment_types,
        current_stages=payload.current_stages,
    )
    mutation = MutationPayload(
        operation="patch_draft",
        target=MutationTarget(draft_id=str(draft_id)),
        changes=changes,
    )
    result = dispatch(mutation, db)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    if payload.engaged_days is not None:
        app_row.engaged_days = payload.engaged_days
    if payload.next_action is not None:
        app_row.next_action = payload.next_action
    if payload.comments is not None:
        app_row.comments = payload.comments
    db.commit()
    db.refresh(app_row)

    return to_public_application(app_row)


@app.post("/drafts/{draft_id}/save", response_model=PublicApplicationDTO)
async def save_draft(draft_id: int, db: Session = Depends(get_db)) -> PublicApplicationDTO:
    app_row = db.get(JobApplication, draft_id)
    if app_row is None or not app_row.is_draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")

    mutation = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=str(draft_id)),
        changes=ApplicationChanges(),
    )
    result = dispatch(mutation, db)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    db.refresh(app_row)
    return to_public_application(app_row)


@app.post("/drafts/{draft_id}/discard", status_code=status.HTTP_204_NO_CONTENT)
async def discard_draft(draft_id: int, db: Session = Depends(get_db)) -> Response:
    app_row = db.get(JobApplication, draft_id)
    if app_row is None or not app_row.is_draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Draft {draft_id} not found")

    mutation = MutationPayload(
        operation="discard_draft",
        target=MutationTarget(draft_id=str(draft_id)),
        changes=ApplicationChanges(),
    )
    result = dispatch(mutation, db)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/applications/archived", response_model=list[PublicApplicationDTO])
async def list_archived_applications(db: Session = Depends(get_db)) -> list[PublicApplicationDTO]:
    rows = (
        db.query(JobApplication)
        .filter(JobApplication.is_draft == False)  # noqa: E712
        .filter(JobApplication.archived_at != None)  # noqa: E711
        .order_by(JobApplication.archived_at.desc())
        .all()
    )
    return [to_public_application(row) for row in rows]


@app.get("/applications", response_model=list[PublicApplicationDTO])
async def list_applications(db: Session = Depends(get_db)) -> list[PublicApplicationDTO]:
    rows = (
        db.query(JobApplication)
        .filter(JobApplication.is_draft == False)  # noqa: E712
        .filter(JobApplication.archived_at == None)  # noqa: E711
        .order_by(JobApplication.updated_at.desc())
        .all()
    )
    return [to_public_application(row) for row in rows]


@app.get("/asr/hotwords", response_model=AsrHotwordsResponse)
async def get_asr_hotwords(db: Session = Depends(get_db)) -> AsrHotwordsResponse:
    return AsrHotwordsResponse(hotwords=build_hotword_list(db), limit=HOTWORD_LIMIT)


@app.post("/livekit/token", response_model=LiveKitTokenResponse)
async def create_livekit_token(payload: LiveKitTokenRequest) -> LiveKitTokenResponse:
    participant_identity = f"browser-{uuid4()}"
    try:
        access_token, expires_at, url = create_livekit_browser_token(payload.room_name, participant_identity)
    except LiveKitConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return LiveKitTokenResponse(
        url=url,
        room_name=payload.room_name,
        participant_identity=participant_identity,
        access_token=access_token,
        expires_at=expires_at,
    )


@app.post("/browser-context", response_model=BrowserContextResponse, status_code=status.HTTP_201_CREATED)
async def create_browser_context(payload: BrowserContextCreate, db: Session = Depends(get_db)) -> dict[str, BrowserContext]:
    context = BrowserContext(**payload.model_dump())
    db.add(context)
    db.commit()
    db.refresh(context)
    return {"context": context}


@app.get("/browser-context/latest", response_model=BrowserContextResponse)
async def get_latest_browser_context(db: Session = Depends(get_db)) -> dict[str, BrowserContext | None]:
    context = db.query(BrowserContext).order_by(BrowserContext.captured_at.desc(), BrowserContext.id.desc()).first()
    return {"context": context}


@app.post("/transcript/parse", response_model=PublicTranscriptResponse)
async def parse_transcript_command(
    payload: TranscriptParseRequest,
    db: Session = Depends(get_db),
    interpreter: OllamaSemanticInterpreter = Depends(get_semantic_interpreter),
) -> PublicTranscriptResponse:
    internal = interpret_transcript_command(db, payload, interpreter)
    return to_public_transcript_response(internal)


@app.post("/applications", response_model=JobApplicationRead, status_code=status.HTTP_201_CREATED)
async def create_application(payload: JobApplicationCreate, db: Session = Depends(get_db)) -> JobApplication:
    application = create_job_application(db, payload)
    db.commit()
    db.refresh(application)
    return application


@app.post("/applications/create-candidate", response_model=ApplicationCreateCandidateResponse)
async def create_application_candidate(
    payload: ApplicationCreateCandidateRequest,
    db: Session = Depends(get_db),
) -> ApplicationCreateCandidateResponse:
    resolved_company_name, _canonical_company = resolve_company_name(db, payload.company)
    if resolved_company_name is None:
        return {"status": "confirmation_required", "requires_confirmation": True, "candidate": payload}

    application_payload = JobApplicationCreate(
        **(payload.model_dump(exclude={"raw_transcript", "original_extracted_company_name", "audio_reference"})
           | {"company": resolved_company_name})
    )

    if find_duplicate_role(db, resolved_company_name, application_payload.role):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application for {resolved_company_name} — {application_payload.role} already exists.",
        )

    application = create_job_application(db, application_payload)
    db.commit()
    db.refresh(application)
    return {"status": "created", "requires_confirmation": False, "application": application}


@app.post("/applications/confirm-company", response_model=JobApplicationRead, status_code=status.HTTP_201_CREATED)
async def confirm_company_and_create_application(
    payload: ApplicationCompanyConfirmationRequest,
    db: Session = Depends(get_db),
) -> JobApplication:
    resolved_company_name, existing_canonical_company = resolve_company_name(db, payload.confirmed_company_name)
    canonical_company = existing_canonical_company
    final_company_name = resolved_company_name or payload.confirmed_company_name.strip()
    if canonical_company is None:
        canonical_company = ensure_canonical_company(db, final_company_name)
        final_company_name = canonical_company.canonical_name

    application_payload = JobApplicationCreate(
        **(payload.model_dump(exclude={"confirmed_company_name", "raw_transcript", "original_extracted_company_name", "audio_reference"})
           | {"company": final_company_name})
    )
    application = create_job_application(db, application_payload)

    alias_created = maybe_create_alias(db, canonical_company, payload.original_extracted_company_name)
    maybe_create_correction_event(db, payload, canonical_company, application, alias_created)

    db.commit()
    db.refresh(application)
    return application


@app.get("/applications/{application_id}", response_model=JobApplicationRead)
async def get_application(application_id: int, db: Session = Depends(get_db)) -> JobApplication:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    return application


@app.patch("/applications/{application_id}", response_model=JobApplicationRead)
async def update_application(
    application_id: int,
    payload: JobApplicationUpdate,
    db: Session = Depends(get_db),
) -> JobApplication:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if application.is_draft:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update a draft application directly. Use the transcript interface.",
        )

    update_data = payload.model_dump(exclude_unset=True)

    new_company_id = application.company_id
    new_normalized_role = application.normalized_role

    # Company change must go through get_or_create_company
    if "company" in update_data:
        new_company_name = update_data.pop("company")
        company_obj = get_or_create_company(db, new_company_name)
        new_company_id = company_obj.id

    if "role" in update_data:
        new_normalized_role = normalize_role_name(update_data["role"])

    # Enforce uniqueness if company or role is changing
    if "role" in update_data or new_company_id != application.company_id:
        collision = (
            db.query(JobApplication)
            .filter(
                JobApplication.company_id == new_company_id,
                JobApplication.normalized_role == new_normalized_role,
                JobApplication.id != application.id,
            )
            .first()
        )
        if collision is not None:
            cname = collision.company_rel.name if collision.company_rel else str(new_company_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An application for {cname} — {collision.role} already exists.",
            )

    if new_company_id != application.company_id:
        application.company_id = new_company_id

    for field, value in update_data.items():
        setattr(application, field, value)

    # Keep normalized_role in sync when role changes
    if "role" in update_data:
        application.normalized_role = new_normalized_role

    db.commit()
    db.refresh(application)
    return application


@app.delete("/applications/{application_id}")
async def delete_application(application_id: int, db: Session = Depends(get_db)) -> dict:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    role_label = application.role or "application"
    return {
        "requires_confirmation": True,
        "confirmation_kind": "archive",
        "message": f"Archive {application.company} — {role_label}?",
        "application_id": application_id,
    }


@app.post("/applications/{application_id}/archive")
async def archive_application(application_id: int, db: Session = Depends(get_db)) -> dict:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    payload = MutationPayload(
        operation="archive_application",
        target=MutationTarget(application_id=application_id),
        changes=ApplicationChanges(),
    )
    result = dispatch(payload, db)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)
    return {"success": True, "message": result.message, "application": result.application}


@app.post("/applications/{application_id}/restore")
async def restore_application(application_id: int, db: Session = Depends(get_db)) -> dict:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    payload = MutationPayload(
        operation="restore_application",
        target=MutationTarget(application_id=application_id),
        changes=ApplicationChanges(),
    )
    result = dispatch(payload, db)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)
    return {"success": True, "message": result.message, "application": result.application}


@app.get("/applications/{application_id}/notes")
async def get_application_notes(application_id: int, db: Session = Depends(get_db)) -> dict:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if application.is_draft:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot get notes for a draft application")
    notes = (
        db.query(ApplicationNote)
        .filter(ApplicationNote.application_id == application_id)
        .order_by(ApplicationNote.created_at.asc())
        .all()
    )
    return {
        "application_id": application_id,
        "notes": [
            {"id": n.id, "text": n.text, "created_at": n.created_at.isoformat()}
            for n in notes
        ],
    }


@app.get("/applications/{application_id}/timeline")
async def get_application_timeline(application_id: int, db: Session = Depends(get_db)) -> dict:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if application.is_draft:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot get timeline for a draft application")
    events = (
        db.query(ApplicationEvent)
        .filter(ApplicationEvent.application_id == application_id)
        .order_by(ApplicationEvent.created_at.asc())
        .all()
    )
    return {
        "application_id": application_id,
        "timeline": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "payload": e.payload,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
    }
