"""Optional, idempotent simulation bootstrap with real import knowledge times."""

import hashlib
import hmac
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.api.database import Transaction, make_engine
from src.api.schemas import FinancialTransaction, RegistrationTransaction
from src.api.service import lock_payer, request_payload
from src.data.temporal import Scenario, normalize, timestamp
from src.workflow import load_source


def pseudonym(scenario: str, payer_id: str, key: str) -> str:
    """Keyed identifier: raw names/documents never enter the API database."""
    return hmac.new(key.encode(), f"{scenario}:{payer_id}".encode(), hashlib.sha256).hexdigest()


def import_history(scenario: Scenario, cutoff: str, source: Path | None = None) -> dict:
    """Import completed records before cutoff; commit in restartable 500-row batches."""
    key = os.environ.get("PAYER_HMAC_KEY", "")
    if len(key) < 32 or key.startswith("troque_"):
        raise ValueError("Defina PAYER_HMAC_KEY privada e estável, com pelo menos 32 caracteres")
    if timestamp(cutoff) > datetime.now(timezone.utc):
        raise ValueError("Corte futuro não é permitido para importar pagamentos concluídos")
    engine = make_engine(os.environ["DATABASE_URL"])
    frame, quality = normalize(load_source(scenario, source), scenario)
    frame = frame.loc[frame.completed & (frame.occurred_at < timestamp(cutoff))].copy()
    frame["payer_id"] = frame.payer_id.map(lambda value: pseudonym(scenario, value, key))
    frame = frame.sort_values(["payer_id", "occurred_at", "source_id"], kind="stable")
    imported, existing_count = 0, 0
    try:
        for start in range(0, len(frame), 500):
            added, skipped = 0, 0
            chunk = frame.iloc[start:start + 500]
            with Session(engine) as session, session.begin():
                for payer in sorted(chunk.payer_id.unique()):
                    lock_payer(session, scenario, payer)
                for row in chunk.to_dict("records"):
                    attributes = ({name: row[name] for name in ("tipo_transacao", "saldo_anterior_pagador", "saldo_anterior_recebedor")}
                        if scenario == "2m" else {name: row[name] for name in (
                            "tipo_chave", "pagador_banco", "recebedor_banco", "pagador_is_cnpj", "recebedor_is_cnpj", "pagador_cpf_zerado")})
                    schema = FinancialTransaction if scenario == "2m" else RegistrationTransaction
                    request = schema(scenario=scenario, external_id="history:" + hashlib.sha256(row["source_id"].encode()).hexdigest(),
                        payer_id=row["payer_id"], occurred_at=row["occurred_at"].to_pydatetime(),
                        amount=str(row["amount"]), attributes=attributes)
                    payload = request_payload(request)
                    existing = session.scalar(select(Transaction).where(Transaction.scenario == scenario,
                        Transaction.external_id == request.external_id))
                    if existing is not None:
                        if (existing.source != "simulation_import" or existing.payer_id != request.payer_id or
                            existing.amount != request.amount or timestamp(existing.occurred_at) != timestamp(request.occurred_at) or
                            existing.attributes != payload["attributes"]):
                            raise ValueError("Histórico já importado com conteúdo diferente; lote revertido")
                        skipped += 1
                        continue
                    session.add(Transaction(scenario=scenario, external_id=request.external_id,
                        payer_id=request.payer_id, occurred_at=request.occurred_at, amount=request.amount,
                        attributes=payload["attributes"], status="completed", executed_at=request.occurred_at,
                        confirmed_at=session.scalar(text("SELECT clock_timestamp()")),
                        features={}, source="simulation_import"))
                    session.flush()
                    added += 1
            imported += added
            existing_count += skipped
    finally:
        engine.dispose()
    return {"imported": imported, "already_present": existing_count,
            "invalid_dates_excluded": quality["invalid_dates"], "cutoff_exclusive": timestamp(cutoff).isoformat(),
            "source": "simulation_import", "knowledge_time": "actual_import_time"}
