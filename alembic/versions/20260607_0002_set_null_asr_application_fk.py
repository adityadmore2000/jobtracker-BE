"""Set ASR correction event application FK to ON DELETE SET NULL.

Revision ID: 20260607_0002
Revises: 20260607_0001
Create Date: 2026-06-07 00:30:00
"""

from __future__ import annotations

from alembic import op


revision = "20260607_0002"
down_revision = "20260607_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "asr_company_correction_events_application_id_fkey",
        "asr_company_correction_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "asr_company_correction_events_application_id_fkey",
        "asr_company_correction_events",
        "job_applications",
        ["application_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "asr_company_correction_events_application_id_fkey",
        "asr_company_correction_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "asr_company_correction_events_application_id_fkey",
        "asr_company_correction_events",
        "job_applications",
        ["application_id"],
        ["id"],
    )
