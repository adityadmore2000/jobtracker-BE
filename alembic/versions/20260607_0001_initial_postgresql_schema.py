"""Initial PostgreSQL schema.

Revision ID: 20260607_0001
Revises:
Create Date: 2026-06-07 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260607_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "browser_context",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("page_title", sa.String(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_browser_context_id"), "browser_context", ["id"], unique=False)

    op.create_table(
        "canonical_companies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("canonical_name"),
    )
    op.create_index(op.f("ix_canonical_companies_canonical_name"), "canonical_companies", ["canonical_name"], unique=True)
    op.create_index(op.f("ix_canonical_companies_id"), "canonical_companies", ["id"], unique=False)

    op.create_table(
        "job_applications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company", sa.String(), nullable=False),
        sa.Column("roles_json", sa.JSON(), nullable=False),
        sa.Column("employment_types_json", sa.JSON(), nullable=False),
        sa.Column("job_link", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("current_stages_json", sa.JSON(), nullable=False),
        sa.Column("priority", sa.String(), nullable=False),
        sa.Column("engaged_days", sa.Integer(), nullable=False),
        sa.Column("next_action", sa.String(), nullable=False),
        sa.Column("comments", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_applications_company"), "job_applications", ["company"], unique=False)
    op.create_index(op.f("ix_job_applications_id"), "job_applications", ["id"], unique=False)

    op.create_table(
        "company_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_company_id", sa.Integer(), nullable=False),
        sa.Column("alias_text", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["canonical_company_id"], ["canonical_companies.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("alias_text"),
    )
    op.create_index(op.f("ix_company_aliases_alias_text"), "company_aliases", ["alias_text"], unique=True)
    op.create_index(op.f("ix_company_aliases_canonical_company_id"), "company_aliases", ["canonical_company_id"], unique=False)
    op.create_index(op.f("ix_company_aliases_id"), "company_aliases", ["id"], unique=False)

    op.create_table(
        "asr_company_correction_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("raw_transcript", sa.String(), nullable=False),
        sa.Column("original_extracted_company_name", sa.String(), nullable=False),
        sa.Column("confirmed_company_name", sa.String(), nullable=False),
        sa.Column("canonical_company_id", sa.Integer(), nullable=False),
        sa.Column("application_id", sa.Integer(), nullable=True),
        sa.Column("alias_created", sa.Boolean(), nullable=False),
        sa.Column("audio_reference", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["application_id"], ["job_applications.id"]),
        sa.ForeignKeyConstraint(["canonical_company_id"], ["canonical_companies.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_asr_company_correction_events_application_id"),
        "asr_company_correction_events",
        ["application_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_asr_company_correction_events_canonical_company_id"),
        "asr_company_correction_events",
        ["canonical_company_id"],
        unique=False,
    )
    op.create_index(op.f("ix_asr_company_correction_events_id"), "asr_company_correction_events", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_asr_company_correction_events_id"), table_name="asr_company_correction_events")
    op.drop_index(op.f("ix_asr_company_correction_events_canonical_company_id"), table_name="asr_company_correction_events")
    op.drop_index(op.f("ix_asr_company_correction_events_application_id"), table_name="asr_company_correction_events")
    op.drop_table("asr_company_correction_events")
    op.drop_index(op.f("ix_company_aliases_id"), table_name="company_aliases")
    op.drop_index(op.f("ix_company_aliases_canonical_company_id"), table_name="company_aliases")
    op.drop_index(op.f("ix_company_aliases_alias_text"), table_name="company_aliases")
    op.drop_table("company_aliases")
    op.drop_index(op.f("ix_job_applications_id"), table_name="job_applications")
    op.drop_index(op.f("ix_job_applications_company"), table_name="job_applications")
    op.drop_table("job_applications")
    op.drop_index(op.f("ix_canonical_companies_id"), table_name="canonical_companies")
    op.drop_index(op.f("ix_canonical_companies_canonical_name"), table_name="canonical_companies")
    op.drop_table("canonical_companies")
    op.drop_index(op.f("ix_browser_context_id"), table_name="browser_context")
    op.drop_table("browser_context")
