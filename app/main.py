import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .company_matching import has_meaningful_company_difference, normalize_company_name
from .company_resolution import (
    ensure_canonical_company,
    get_canonical_company_by_normalized_name,
    get_company_alias_by_normalized_name,
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
from .migrations import run_startup_migrations_if_enabled
from .models import AsrCompanyCorrectionEvent, BrowserContext, CanonicalCompany, CompanyAlias, JobApplication
from .schemas import (
    ApplicationCompanyConfirmationRequest,
    ApplicationCreateCandidateRequest,
    ApplicationCreateCandidateResponse,
    BrowserContextCreate,
    BrowserContextResponse,
    AsrHotwordsResponse,
    JobApplicationCreate,
    JobApplicationRead,
    JobApplicationUpdate,
    SemanticTranscriptResponse,
    TranscriptParseRequest,
)
from .semantic_interpreter import OllamaSemanticInterpreter, get_semantic_interpreter
from .semantic_validation import interpret_transcript_command


@asynccontextmanager
async def lifespan(_app: FastAPI):
    run_startup_migrations_if_enabled()
    yield


app = FastAPI(title="Job Tracker API", lifespan=lifespan)
HOTWORD_LIMIT = 100
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
    application = JobApplication(**payload.model_dump())
    db.add(application)
    db.flush()
    return application


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/semantic-interpreter/health")
async def semantic_interpreter_health(interpreter: OllamaSemanticInterpreter = Depends(get_semantic_interpreter)) -> dict[str, str]:
    return interpreter.health_check()


@app.get("/applications", response_model=list[JobApplicationRead])
async def list_applications(db: Session = Depends(get_db)) -> list[JobApplication]:
    return db.query(JobApplication).order_by(JobApplication.updated_at.desc()).all()


@app.get("/asr/hotwords", response_model=AsrHotwordsResponse)
async def get_asr_hotwords(db: Session = Depends(get_db)) -> AsrHotwordsResponse:
    return AsrHotwordsResponse(hotwords=build_hotword_list(db), limit=HOTWORD_LIMIT)


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


@app.post("/transcript/parse", response_model=SemanticTranscriptResponse)
async def parse_transcript_command(
    payload: TranscriptParseRequest,
    db: Session = Depends(get_db),
    interpreter: OllamaSemanticInterpreter = Depends(get_semantic_interpreter),
) -> SemanticTranscriptResponse:
    return interpret_transcript_command(db, payload, interpreter)


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

    application_payload = JobApplicationCreate(**(payload.model_dump(exclude={"raw_transcript", "original_extracted_company_name", "audio_reference"}) | {"company": resolved_company_name}))
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
        **(payload.model_dump(exclude={"confirmed_company_name", "raw_transcript", "original_extracted_company_name", "audio_reference"}) | {"company": final_company_name})
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

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(application, field, value)

    db.commit()
    db.refresh(application)
    return application


@app.delete("/applications/{application_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_application(application_id: int, db: Session = Depends(get_db)) -> Response:
    application = db.get(JobApplication, application_id)
    if application is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")

    try:
        db.delete(application)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Application could not be deleted because related records are still enforcing integrity constraints.",
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
