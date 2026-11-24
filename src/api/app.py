"""HTTP application served through python main.py serve."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.database import Transaction, make_engine
from src.api.registry import Registry
from src.api.schemas import (BatchRequest, Confirmation, ErrorResponse, TransactionPage,
                              TransactionRequest, TransactionResponse)
from src.api.service import analyze, confirm, lock_payer, response
from src.workflow import GPU_RESULTS as RESULTS

logger = logging.getLogger(__name__)
ERRORS = {code: {"model": ErrorResponse} for code in (404, 409, 503)}


def create_app(database_url: str | None = None, model_root: Path = RESULTS) -> FastAPI:
    """Construct without database side effects; schema is applied explicitly by Alembic."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        url = database_url or os.environ.get("DATABASE_URL")
        application.state.engine = make_engine(url) if url else None
        application.state.registry = Registry(model_root)
        yield
        if application.state.engine is not None:
            application.state.engine.dispose()

    application = FastAPI(title="Pix — classificação temporal", version="1.0.0", lifespan=lifespan,
        description="API local de pesquisa. Classifica antes da execução; confirmações alimentam o histórico. "
                    "Escores não são probabilidades calibradas. Sem execução de pagamentos ou autenticação pública.")

    def session_dependency():
        engine = application.state.engine
        if engine is None:
            raise HTTPException(503, "Configure DATABASE_URL e aplique as migrações")
        try:
            with Session(engine) as session, session.begin():
                yield session
        except IntegrityError as exc:
            raise HTTPException(409, "Conflito de identificador ou integridade da transação") from exc
        except SQLAlchemyError as exc:
            logger.error("Falha de persistência: %s", type(exc).__name__)
            raise HTTPException(503, "Banco de dados indisponível ou migração pendente") from exc

    Database = Annotated[Session, Depends(session_dependency)]

    @application.get("/health", tags=["Operação"], responses=ERRORS, summary="Verificar banco e modelos ativos")
    def health(session: Database) -> dict:
        version = int(session.scalar(text("SHOW server_version_num")))
        if version < 180000:
            raise HTTPException(503, "PostgreSQL 18 ou superior é necessário")
        session.execute(select(Transaction.id).limit(1))
        registry = application.state.registry
        if len(registry.models) != 2:
            raise HTTPException(503, "Treine e ative os modelos dos dois cenários; consulte /v1/models")
        return {"status": "ok", "database": "postgresql", "models": list(registry.models)}

    @application.get("/v1/models", tags=["Modelos"], summary="Consultar contratos, seleção e versões ativas")
    def models() -> dict:
        return application.state.registry.describe()

    @application.post("/v1/transactions", tags=["Transações"], response_model=TransactionResponse,
                      responses=ERRORS, summary="Registrar e analisar uma tentativa antes da execução")
    def create_transaction(request: TransactionRequest, session: Database) -> TransactionResponse:
        return response(analyze(session, application.state.registry, request))

    @application.post("/v1/transactions/batch", tags=["Transações"], response_model=list[TransactionResponse],
                      responses=ERRORS, summary="Analisar até 100 tentativas em uma operação atômica")
    def create_batch(request: BatchRequest, session: Database) -> list[TransactionResponse]:
        # Global lock ordering prevents opposite payer orders from deadlocking batches.
        for scenario, payer in sorted({(item.scenario, item.payer_id) for item in request.transactions}):
            lock_payer(session, scenario, payer)
        return [response(analyze(session, application.state.registry, item)) for item in request.transactions]

    @application.post("/v1/transactions/{identifier}/confirmation", tags=["Transações"],
                      response_model=TransactionResponse, responses=ERRORS,
                      summary="Confirmar execução ou cancelamento, sem reclassificar")
    def confirmation(identifier: UUID, request: Confirmation, session: Database) -> TransactionResponse:
        return response(confirm(session, identifier, request))

    @application.get("/v1/transactions/{identifier}", tags=["Transações"], response_model=TransactionResponse,
                     responses=ERRORS, summary="Consultar a fotografia original de uma decisão")
    def get_transaction(identifier: UUID, session: Database) -> TransactionResponse:
        transaction = session.get(Transaction, identifier)
        if transaction is None:
            raise HTTPException(404, "Transação não encontrada")
        return response(transaction)

    @application.get("/v1/transactions", tags=["Transações"], response_model=TransactionPage,
                     responses=ERRORS, summary="Listar transações por cursor e cenário")
    def list_transactions(session: Database, scenario: Literal["10k", "2m"] | None = None,
                          after: UUID | None = None, limit: int = Query(default=50, ge=1, le=100)) -> TransactionPage:
        query = select(Transaction).order_by(Transaction.id).limit(limit + 1)
        if scenario is not None:
            query = query.where(Transaction.scenario == scenario)
        if after is not None:
            query = query.where(Transaction.id > after)
        rows = session.scalars(query).all()
        return TransactionPage(items=[response(row) for row in rows[:limit]],
                               next_cursor=rows[limit - 1].id if len(rows) > limit else None)

    return application


app = create_app(model_root=Path(os.environ.get("MODEL_ROOT", str(RESULTS))))
