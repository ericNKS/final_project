"""Transactional payment lifecycle and point-in-time history ownership."""

import hashlib
from decimal import Decimal
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.api.database import Transaction
from src.api.registry import Registry
from src.api.schemas import Confirmation, TransactionRequest, TransactionResponse
from src.data.temporal import feature_row, timestamp


def lock_payer(session: Session, scenario: str, payer_id: str) -> None:
    """Serialize analysis, confirmation and imports for one scenario/payer."""
    key = int.from_bytes(hashlib.sha256(f"{scenario}:{payer_id}".encode()).digest()[:8], "big", signed=True)
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def request_payload(request: TransactionRequest) -> dict:
    """Canonical money/time representations make semantic retries idempotent."""
    payload = request.model_dump(mode="json")
    payload["occurred_at"] = timestamp(request.occurred_at).isoformat()
    payload["amount"] = str(request.amount.quantize(Decimal("0.01")))
    if request.scenario == "2m":
        for name in ("saldo_anterior_pagador", "saldo_anterior_recebedor"):
            payload["attributes"][name] = str(getattr(request.attributes, name).quantize(Decimal("0.01")))
    return payload


def response(transaction: Transaction) -> TransactionResponse:
    """Return persisted decision values, never recompute a historical prediction."""
    return TransactionResponse(**{name: getattr(transaction, name) for name in TransactionResponse.model_fields})


def payer_history(session: Session, scenario: str, payer_id: str,
                  occurred_at, known_at) -> tuple[int, float | None, float | None]:
    """Use only completed, strictly earlier payments known at analysis time."""
    row = session.execute(text("""
        SELECT count(*) AS count, avg(amount) AS mean,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY amount::double precision) AS median
        FROM transactions
        WHERE scenario = :scenario AND payer_id = :payer
          AND status = 'completed' AND executed_at < :occurred
          AND confirmed_at <= :known
    """), {"scenario": scenario, "payer": payer_id, "occurred": occurred_at, "known": known_at}).one()
    values = row._mapping
    return int(values["count"]), float(values["mean"]) if values["mean"] is not None else None, float(values["median"]) if values["median"] is not None else None


def analyze(session: Session, registry: Registry, request: TransactionRequest) -> Transaction:
    """Atomically snapshot history and persist a pending attempt; never execute a Pix."""
    lock_payer(session, request.scenario, request.payer_id)
    payload = request_payload(request)
    existing = session.scalar(select(Transaction).where(
        Transaction.scenario == request.scenario, Transaction.external_id == request.external_id))
    if existing is not None:
        if (existing.source != "api" or existing.payer_id != request.payer_id or
                existing.amount != request.amount or timestamp(existing.occurred_at) != timestamp(request.occurred_at) or
                existing.attributes != payload["attributes"]):
            raise HTTPException(409, "Identificador já registrado com conteúdo diferente")
        return existing
    known_at = session.scalar(text("SELECT clock_timestamp()"))
    occurred_at = timestamp(request.occurred_at).to_pydatetime()
    history = payer_history(session, request.scenario, request.payer_id, occurred_at, known_at)
    attributes = request.attributes.model_dump()
    features = feature_row(request.scenario, timestamp(occurred_at), float(request.amount), attributes, history)
    # JSON snapshots and model input use exactly the same numeric values.
    for key, value in features.items():
        if isinstance(value, Decimal):
            features[key] = float(value)
    try:
        score, threshold, prediction, version, family = registry.predict(request.scenario, features)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    transaction = Transaction(scenario=request.scenario, external_id=request.external_id,
        payer_id=request.payer_id, occurred_at=occurred_at, amount=request.amount,
        attributes=payload["attributes"], status="pending", analyzed_at=known_at,
        model_version=version, model_family=family, score=score, threshold=threshold,
        prediction=prediction, features=features, source="api")
    session.add(transaction)
    session.flush()
    return transaction


def confirm(session: Session, identifier: UUID, confirmation: Confirmation) -> Transaction:
    """A terminal state is immutable; replaying the same confirmation is safe."""
    transaction = session.get(Transaction, identifier)
    if transaction is None:
        raise HTTPException(404, "Transação não encontrada")
    lock_payer(session, transaction.scenario, transaction.payer_id)
    session.refresh(transaction)
    if transaction.status != "pending":
        if transaction.status != confirmation.status or transaction.executed_at != confirmation.executed_at:
            raise HTTPException(409, "Confirmação final não pode ser alterada")
        return transaction
    if confirmation.executed_at is not None and confirmation.executed_at < transaction.occurred_at:
        raise HTTPException(422, "Execução anterior ao horário da transação")
    # Fetch before changing state: ORM autoflush must not observe half a transition.
    confirmed_at = session.scalar(text("SELECT clock_timestamp()"))
    transaction.status = confirmation.status
    transaction.executed_at = confirmation.executed_at
    transaction.confirmed_at = confirmed_at
    session.flush()
    return transaction
