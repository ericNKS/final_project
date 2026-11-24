"""Strict, scenario-specific HTTP contracts; no post-execution model inputs."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Money = Annotated[Decimal, Field(max_digits=20, decimal_places=2, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


class StrictModel(BaseModel):
    """Reject extra attributes rather than silently ignoring unavailable features."""
    model_config = ConfigDict(extra="forbid")


class FinancialAttributes(StrictModel):
    tipo_transacao: str = Field(min_length=1, max_length=80)
    saldo_anterior_pagador: Money = Field(description="Saldo observado antes da decisão, obtido de fonte autorizada. Não enviar saldo posterior ou estimado.")
    saldo_anterior_recebedor: Money = Field(description="Exige acesso autorizado ao saldo do recebedor antes da decisão. Sem esse acesso, o modelo 2m não é aplicável. A API não comprova a origem ou o instante de observação.")


class RegistrationAttributes(StrictModel):
    tipo_chave: str = Field(min_length=1, max_length=80)
    pagador_banco: str = Field(min_length=1, max_length=120)
    recebedor_banco: str = Field(min_length=1, max_length=120)
    pagador_is_cnpj: Literal[0, 1]
    recebedor_is_cnpj: Literal[0, 1]
    pagador_cpf_zerado: Literal[0, 1]


class TransactionBase(StrictModel):
    external_id: Identifier = Field(examples=["pix-demo-001"])
    payer_id: Identifier = Field(description="Identificador opaco/pseudonimizado; não enviar CPF ou nome")
    occurred_at: AwareDatetime = Field(examples=["2026-11-22T12:00:00-03:00"])
    amount: Money = Field(ge=0, examples=["150.00"])


class FinancialTransaction(TransactionBase):
    scenario: Literal["2m"]
    attributes: FinancialAttributes
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "scenario": "2m", "external_id": "pix-demo-001", "payer_id": "payer-opaque-001",
        "occurred_at": "2026-11-22T12:00:00-03:00", "amount": "150.00", "attributes": {
            "tipo_transacao": "chave_pix", "saldo_anterior_pagador": "1000.00", "saldo_anterior_recebedor": "500.00"}}]})


class RegistrationTransaction(TransactionBase):
    scenario: Literal["10k"]
    attributes: RegistrationAttributes
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "scenario": "10k", "external_id": "registration-demo-001", "payer_id": "payer-opaque-001",
        "occurred_at": "2026-11-22T12:00:00-03:00", "amount": "150.00", "attributes": {
            "tipo_chave": "CPF", "pagador_banco": "A", "recebedor_banco": "B",
            "pagador_is_cnpj": 0, "recebedor_is_cnpj": 1, "pagador_cpf_zerado": 0}}]})


TransactionRequest = Annotated[FinancialTransaction | RegistrationTransaction, Field(discriminator="scenario")]


class BatchRequest(StrictModel):
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=100)


class Confirmation(StrictModel):
    status: Literal["completed", "cancelled"]
    executed_at: AwareDatetime | None = None
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"status": "completed", "executed_at": "2026-11-22T12:00:00-03:00"}, {"status": "cancelled"}]})

    @model_validator(mode="after")
    def validate_execution(self) -> "Confirmation":
        """Only completed payments have an actual execution time."""
        if (self.status == "completed") != (self.executed_at is not None):
            raise ValueError("completed exige executed_at; cancelled não aceita executed_at")
        if self.executed_at is not None and self.executed_at > datetime.now(timezone.utc):
            raise ValueError("Execução não pode estar no futuro")
        return self


class TransactionResponse(StrictModel):
    id: UUID
    scenario: str
    external_id: str
    payer_id: str
    occurred_at: datetime
    amount: Decimal
    status: str
    executed_at: datetime | None
    confirmed_at: datetime | None
    analyzed_at: datetime | None
    model_version: str | None
    model_family: str | None
    score: float | None
    threshold: float | None
    prediction: int | None
    features: dict
    source: str


class TransactionPage(StrictModel):
    items: list[TransactionResponse]
    next_cursor: UUID | None


class ErrorResponse(StrictModel):
    detail: str
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"detail": "Identificador já registrado com conteúdo diferente"}]})
