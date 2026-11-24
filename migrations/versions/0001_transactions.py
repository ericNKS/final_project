"""Create payment attempts and immutable prediction snapshots on PostgreSQL 18."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Initial schema; no existing research artifacts or datasets are modified."""
    op.execute("""DO $$ BEGIN
        IF current_setting('server_version_num')::int < 180000 THEN
            RAISE EXCEPTION 'Requires PostgreSQL 18 or higher';
        END IF;
    END $$""")
    op.create_table("transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("uuidv7()"), primary_key=True),
        sa.Column("scenario", sa.String(3), nullable=False),
        sa.Column("external_id", sa.String(128), nullable=False),
        sa.Column("payer_id", sa.String(128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("amount", sa.Numeric(20, 2), nullable=False),
        sa.Column("attributes", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True)),
        sa.Column("confirmed_at", sa.DateTime(timezone=True)),
        sa.Column("analyzed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("model_version", sa.String(80)),
        sa.Column("model_family", sa.String(40)),
        sa.Column("score", sa.Float()),
        sa.Column("threshold", sa.Float()),
        sa.Column("prediction", sa.Integer()),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.String(24), nullable=False),
        sa.UniqueConstraint("scenario", "external_id", name="transactions_external_unique"),
        sa.CheckConstraint("scenario IN ('10k', '2m')", name="transactions_scenario_check"),
        sa.CheckConstraint("status IN ('pending', 'completed', 'cancelled')", name="transactions_status_check"),
        sa.CheckConstraint("amount >= 0", name="transactions_amount_check"),
        sa.CheckConstraint("(status = 'completed') = (executed_at IS NOT NULL)", name="transactions_execution_check"),
        sa.CheckConstraint("(status = 'pending') = (confirmed_at IS NULL)", name="transactions_confirmation_check"),
        sa.CheckConstraint("executed_at IS NULL OR executed_at >= occurred_at", name="transactions_time_check"),
        sa.CheckConstraint("source IN ('api', 'simulation_import')", name="transactions_source_check"),
        sa.CheckConstraint("score IS NULL OR (score >= 0 AND score <= 1)", name="transactions_score_check"),
        sa.CheckConstraint("threshold IS NULL OR (threshold >= 0 AND threshold <= 1)", name="transactions_threshold_check"),
        sa.CheckConstraint("prediction IS NULL OR prediction IN (0, 1)", name="transactions_prediction_check"),
    )
    op.create_index("idx_transactions_history", "transactions", ["scenario", "payer_id", "executed_at"],
                    postgresql_where=sa.text("status = 'completed'"))


def downgrade() -> None:
    """Rollback is safe only while empty; never implicitly erase recorded decisions."""
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM transactions)")):
        raise RuntimeError("Banco contém transações: faça backup e preserve os dados antes de reverter")
    op.drop_index("idx_transactions_history", table_name="transactions")
    op.drop_table("transactions")
