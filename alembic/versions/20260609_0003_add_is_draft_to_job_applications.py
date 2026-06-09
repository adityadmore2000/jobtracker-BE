"""Add is_draft flag and draft_created_at to job_applications.

Revision ID: 20260609_0003
Revises: 20260607_0002
Create Date: 2026-06-09 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260609_0003"
down_revision = "20260607_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "job_applications",
        sa.Column("is_draft", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "job_applications",
        sa.Column("draft_created_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("job_applications", "draft_created_at")
    op.drop_column("job_applications", "is_draft")
