import os

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import JobApplication
from .schemas import JobApplicationCreate, JobApplicationRead, JobApplicationUpdate

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ApplicationOps Tracker API")


def get_frontend_origins() -> list[str]:
    configured_origin = os.getenv("FRONTEND_ORIGIN", "")
    configured_origins = [origin.strip() for origin in configured_origin.split(",") if origin.strip()]
    local_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    return list(dict.fromkeys([*local_origins, *configured_origins]))


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/applications", response_model=list[JobApplicationRead])
async def list_applications(db: Session = Depends(get_db)) -> list[JobApplication]:
    return db.query(JobApplication).order_by(JobApplication.updated_at.desc()).all()


@app.post("/applications", response_model=JobApplicationRead, status_code=status.HTTP_201_CREATED)
async def create_application(payload: JobApplicationCreate, db: Session = Depends(get_db)) -> JobApplication:
    application = JobApplication(**payload.model_dump())
    db.add(application)
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

    db.delete(application)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
