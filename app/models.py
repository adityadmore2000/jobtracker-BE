from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    website: Mapped[str | None] = mapped_column(Text, nullable=True)
    career_page: Mapped[str | None] = mapped_column(Text, nullable=True)
    linkedin_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    applications: Mapped[list["JobApplication"]] = relationship(back_populates="company_rel")


class JobApplication(Base):
    __tablename__ = "job_applications"
    __table_args__ = (
        UniqueConstraint("company_id", "normalized_role", name="uq_job_applications_company_role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("companies.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False, default="")
    normalized_role: Mapped[str] = mapped_column(Text, nullable=False, default="")
    employment_types_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    job_link: Mapped[str] = mapped_column(String, nullable=False, default="")
    location: Mapped[str] = mapped_column(String, nullable=False, default="")
    status: Mapped[str] = mapped_column(String, nullable=False, default="")
    current_stages_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    priority: Mapped[str] = mapped_column(String, nullable=False, default="")
    engaged_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_action: Mapped[str] = mapped_column(String, nullable=False, default="")
    comments: Mapped[str] = mapped_column(String, nullable=False, default="")
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    draft_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    company_rel: Mapped["Company"] = relationship(back_populates="applications")

    correction_events: Mapped[list["AsrCompanyCorrectionEvent"]] = relationship(
        back_populates="application",
        passive_deletes=True,
    )
    notes_rel: Mapped[list["ApplicationNote"]] = relationship(
        "ApplicationNote",
        order_by="ApplicationNote.created_at",
        cascade="all, delete-orphan",
    )
    events: Mapped[list["ApplicationEvent"]] = relationship(
        "ApplicationEvent",
        order_by="ApplicationEvent.created_at",
        cascade="all, delete-orphan",
    )

    @property
    def company(self) -> str:
        return self.company_rel.name if self.company_rel else ""


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
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_applications.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    alias_created: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    audio_reference: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    canonical_company: Mapped["CanonicalCompany"] = relationship(back_populates="correction_events")
    application: Mapped["JobApplication | None"] = relationship(back_populates="correction_events")


class BrowserContext(Base):
    __tablename__ = "browser_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    url: Mapped[str] = mapped_column(String, nullable=False)
    page_title: Mapped[str] = mapped_column(String, nullable=False, default="")
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ApplicationNote(Base):
    __tablename__ = "application_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("job_applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class ApplicationEvent(Base):
    __tablename__ = "application_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("job_applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
