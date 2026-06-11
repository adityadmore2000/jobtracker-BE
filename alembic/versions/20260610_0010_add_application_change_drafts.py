"""add application_change_drafts table

Revision ID: 20260610_0010
Revises: 20260610_0009
Create Date: 2026-06-10
"""

from alembic import op
import sqlalchemy as sa

revision = "20260610_0010"
down_revision = "20260610_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "application_change_drafts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.Text(), nullable=False, server_default="update"),
        sa.Column(
            "target_application_id",
            sa.Integer(),
            sa.ForeignKey("job_applications.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("changes_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_application_change_drafts_target_application_id",
        "application_change_drafts",
        ["target_application_id"],
    )
    op.create_unique_constraint(
        "uq_application_change_drafts_target",
        "application_change_drafts",
        ["target_application_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_application_change_drafts_target",
        "application_change_drafts",
        type_="unique",
    )
    op.drop_index(
        "ix_application_change_drafts_target_application_id",
        table_name="application_change_drafts",
    )
    op.drop_table("application_change_drafts")
