from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobApplication(Base):
    __tablename__ = "job_applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    company: Mapped[str] = mapped_column(String, nullable=False, index=True)
    roles_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    employment_types_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    job_link: Mapped[str] = mapped_column(String, nullable=False, default="")
    location: Mapped[str] = mapped_column(String, nullable=False, default="")
    status: Mapped[str] = mapped_column(String, nullable=False, default="")
    current_stages_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    priority: Mapped[str] = mapped_column(String, nullable=False, default="")
    engaged_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_action: Mapped[str] = mapped_column(String, nullable=False, default="")
    comments: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class CanonicalCompany(Base):
    __tablename__ = "canonical_companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    canonical_name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    aliases: Mapped[list["CompanyAlias"]] = relationship(back_populates="canonical_company", cascade="all, delete-orphan")
    correction_events: Mapped[list["AsrCompanyCorrectionEvent"]] = relationship(
        back_populates="canonical_company",
        cascade="all, delete-orphan",
    )


class CompanyAlias(Base):
    __tablename__ = "company_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    canonical_company_id: Mapped[int] = mapped_column(ForeignKey("canonical_companies.id"), nullable=False, index=True)
    alias_text: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    canonical_company: Mapped["CanonicalCompany"] = relationship(back_populates="aliases")


class AsrCompanyCorrectionEvent(Base):
    __tablename__ = "asr_company_correction_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    raw_transcript: Mapped[str] = mapped_column(String, nullable=False, default="")
    original_extracted_company_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    confirmed_company_name: Mapped[str] = mapped_column(String, nullable=False)
    canonical_company_id: Mapped[int] = mapped_column(ForeignKey("canonical_companies.id"), nullable=False, index=True)
    application_id: Mapped[int | None] = mapped_column(ForeignKey("job_applications.id"), nullable=True, index=True)
    alias_created: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    audio_reference: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    canonical_company: Mapped["CanonicalCompany"] = relationship(back_populates="correction_events")


class BrowserContext(Base):
    __tablename__ = "browser_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    url: Mapped[str] = mapped_column(String, nullable=False)
    page_title: Mapped[str] = mapped_column(String, nullable=False, default="")
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
