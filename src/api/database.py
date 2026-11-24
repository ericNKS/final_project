"""PostgreSQL persistence; monetary values remain exact until model conversion."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (CheckConstraint, DateTime, Float, Index, Integer, Numeric,
                         String, UniqueConstraint, create_engine, text)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Database metadata owned by Alembic migrations."""


class Transaction(Base):
    """Immutable decision snapshot plus a single terminal confirmation."""
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("scenario", "external_id", name="transactions_external_unique"),
        CheckConstraint("scenario IN ('10k', '2m')", name="transactions_scenario_check"),
        CheckConstraint("status IN ('pending', 'completed', 'cancelled')", name="transactions_status_check"),
        CheckConstraint("amount >= 0", name="transactions_amount_check"),
        CheckConstraint("(status = 'completed') = (executed_at IS NOT NULL)", name="transactions_execution_check"),
        CheckConstraint("(status = 'pending') = (confirmed_at IS NULL)", name="transactions_confirmation_check"),
        CheckConstraint("executed_at IS NULL OR executed_at >= occurred_at", name="transactions_time_check"),
        CheckConstraint("source IN ('api', 'simulation_import')", name="transactions_source_check"),
        CheckConstraint("score IS NULL OR (score >= 0 AND score <= 1)", name="transactions_score_check"),
        CheckConstraint("threshold IS NULL OR (threshold >= 0 AND threshold <= 1)", name="transactions_threshold_check"),
        CheckConstraint("prediction IS NULL OR prediction IN (0, 1)", name="transactions_prediction_check"),
        Index("idx_transactions_history", "scenario", "payer_id", "executed_at",
              postgresql_where=text("status = 'completed'")),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()"))
    scenario: Mapped[str] = mapped_column(String(3))
    external_id: Mapped[str] = mapped_column(String(128))
    payer_id: Mapped[str] = mapped_column(String(128))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 2))
    attributes: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(10), default="pending")
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("clock_timestamp()"))
    model_version: Mapped[str | None] = mapped_column(String(80))
    model_family: Mapped[str | None] = mapped_column(String(40))
    score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    prediction: Mapped[int | None] = mapped_column(Integer)
    features: Mapped[dict] = mapped_column(JSONB, default=dict)
    source: Mapped[str] = mapped_column(String(24), default="api")


def make_engine(url: str) -> Engine:
    """READ COMMITTED plus payer locks provides visibility after a lock is acquired."""
    if not url.startswith("postgresql"):
        raise ValueError("Esta API exige PostgreSQL 18 ou superior")
    return create_engine(url, pool_pre_ping=True, isolation_level="READ COMMITTED")
