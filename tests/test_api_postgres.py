"""Real PostgreSQL tests; isolated temporary schema, real fitted model artifacts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.api.app import create_app
from src.api.history_import import import_history, pseudonym
from src.api.database import make_engine
from src.api.service import payer_history
from src.models.experiments import Candidate, FAMILIES
from src.workflow import digest, evaluate, train, verify
from sqlalchemy.orm import Session

pytestmark = pytest.mark.postgres


def source_fixture(scenario: str, n: int = 240) -> pd.DataFrame:
    """Sufficient positives in each progressive window; no persistence mocks."""
    dates = pd.date_range("2025-01-01", periods=n, freq="h").astype(str)
    if scenario == "2m":
        return pd.DataFrame({"datetime_brasilia": dates, "id_pagador": [f"p-{i % 3}" for i in range(n)],
            "valor_brl": (np.arange(n) % 9 + 1).astype(float), "tipo_transacao": ["pix"] * n,
            "saldo_anterior_pagador": np.full(n, 1000.), "saldo_anterior_recebedor": np.full(n, 500.),
            "fraude": [int(i % 7 == 0) for i in range(n)]})
    return pd.DataFrame({"EndToEndId": [f"id-{i}" for i in range(n)], "DataHora": dates,
        "Valor": (np.arange(n) % 9 + 1).astype(float), "TipoChave": ["CPF"] * n,
        "Pagador_CPF_CNPJ": [f"p-{i % 3}" for i in range(n)], "Recebedor_CPF_CNPJ": ["receiver"] * n,
        "Pagador_Banco": ["A"] * n, "Recebedor_Banco": ["B"] * n,
        "Status": ["Concluída"] * n, "Anomalia": [int(i % 7 == 0) for i in range(n)]})


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Defina TEST_DATABASE_URL para PostgreSQL 18 dedicado aos testes")
    schema = "test_pix_" + uuid4().hex
    root_engine = create_engine(url)
    with root_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped_url = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    root = tmp_path_factory.mktemp("temporal_models")
    try:
        with patch.dict(os.environ, {"DATABASE_URL": scoped_url}):
            command.upgrade(Config("alembic.ini"), "head")
            command.upgrade(Config("alembic.ini"), "head")  # repeat is idempotent
        with patch("src.workflow.candidates", return_value=[Candidate(f, "original") for f in FAMILIES]):
            for scenario in ("10k", "2m"):
                source = root / f"{scenario}.parquet"
                source_fixture(scenario).to_parquet(source)
                train(scenario, source=source, backend="cpu", results=root)
        yield scoped_url, root
    finally:
        with root_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        root_engine.dispose()


@pytest.fixture
def client(environment):
    url, root = environment
    with TestClient(create_app(url, root)) as instance:
        yield instance


def request(payer: str | None = None, moment="2025-01-02T12:00:00Z") -> dict:
    return {"scenario": "2m", "external_id": uuid4().hex,
        "payer_id": payer or uuid4().hex, "occurred_at": moment, "amount": "50.00",
        "attributes": {"tipo_transacao": "pix", "saldo_anterior_pagador": "500.00",
                       "saldo_anterior_recebedor": "100.00"}}


def post(client, body):
    result = client.post("/v1/transactions", json=body)
    assert result.status_code == 200, result.text
    return result.json()


def complete(client, transaction, executed_at=None):
    result = client.post(f"/v1/transactions/{transaction['id']}/confirmation", json={
        "status": "completed", "executed_at": executed_at or transaction["occurred_at"]})
    assert result.status_code == 200, result.text
    return result.json()


def test_lifecycle_history_and_late_confirmation_snapshot(client):
    payer = uuid4().hex
    prior = post(client, request(payer, "2025-01-01T12:00:00Z"))
    before = post(client, request(payer))
    assert before["features"]["qtd_transacoes_pagador"] == 0  # pending is not a payment
    complete(client, prior)
    after = post(client, request(payer))
    assert after["features"]["qtd_transacoes_pagador"] == 1
    assert after["features"]["media_valor_pagador"] == 50
    assert client.get(f"/v1/transactions/{before['id']}").json() == before
    tied = complete(client, after)
    later_attempt = post(client, request(payer))
    assert later_attempt["features"]["qtd_transacoes_pagador"] == 1  # tied event excluded
    assert complete(client, tied) == tied  # idempotent
    conflict = client.post(f"/v1/transactions/{prior['id']}/confirmation", json={"status": "cancelled"})
    assert conflict.status_code == 409


def test_confirmation_knowledge_time_is_preserved(environment, client):
    payer = uuid4().hex
    prior = post(client, request(payer, "2025-01-01T12:00:00Z"))
    before = post(client, request(payer))
    complete(client, prior)
    url, _ = environment
    engine = make_engine(url)
    try:
        with Session(engine) as session:
            old = payer_history(session, "2m", payer, before["occurred_at"], before["analyzed_at"])
            current = payer_history(session, "2m", payer, before["occurred_at"], datetime.now(timezone.utc))
        assert old[0] == 0 and current[0] == 1
    finally:
        engine.dispose()


def test_idempotency_validation_batch_atomicity_and_cancellation(client):
    body = request()
    first = post(client, body)
    body["amount"] = "50"  # same decimal amount
    assert post(client, body) == first
    body["amount"] = "51"
    assert client.post("/v1/transactions", json=body).status_code == 409
    invalid = request()
    invalid["attributes"]["saldo_posterior_pagador"] = "0"
    assert client.post("/v1/transactions", json=invalid).status_code == 422
    invalid = request()
    invalid["occurred_at"] = "invalid-date"
    assert client.post("/v1/transactions", json=invalid).status_code == 422
    failed_batch = request()
    result = client.post("/v1/transactions/batch", json={"transactions": [failed_batch, body]})
    assert result.status_code == 409
    assert all(row["external_id"] != failed_batch["external_id"] for row in client.get("/v1/transactions?limit=100").json()["items"])
    result = client.post(f"/v1/transactions/{first['id']}/confirmation", json={"status": "cancelled"})
    assert result.status_code == 200
    next_attempt = post(client, request(first["payer_id"], "2025-01-03T12:00:00Z"))
    assert next_attempt["features"]["qtd_transacoes_pagador"] == 0


def test_concurrent_duplicates_and_restart_persistence(environment, client):
    body = request()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: post(client, body), range(4)))
    assert len({row["id"] for row in results}) == 1
    assert client.get("/health").status_code == 200
    page = client.get("/v1/transactions?limit=1").json()
    assert len(page["items"]) == 1
    assert page["next_cursor"] is not None
    url, root = environment
    with TestClient(create_app(url, root)) as restarted:
        assert restarted.get(f"/v1/transactions/{results[0]['id']}").json() == results[0]


def test_optional_import_cutoff_idempotency_and_scenario_isolation(environment, client):
    url, root = environment
    key = "test_key_" + "x" * 40
    with patch.dict(os.environ, {"DATABASE_URL": url, "PAYER_HMAC_KEY": key}):
        imported = import_history("2m", "2025-01-02", root / "2m.parquet")
        repeated = import_history("2m", "2025-01-02", root / "2m.parquet")
    assert imported["imported"] == 24
    assert repeated["already_present"] == 24 and repeated["imported"] == 0
    payer = pseudonym("2m", "p-0", key)
    transaction = post(client, request(payer, "2025-01-03T12:00:00Z"))
    assert transaction["features"]["qtd_transacoes_pagador"] == 8
    assert pseudonym("10k", "p-0", key) != payer


def test_missing_models_fail_explicitly_and_schema_is_documented(environment, tmp_path):
    url, _ = environment
    with TestClient(create_app(url, tmp_path)) as empty:
        assert empty.post("/v1/transactions", json=request()).status_code == 503
        assert empty.get("/health").status_code == 503
        spec = empty.get("/openapi.json").json()
        assert "/v1/transactions/{identifier}/confirmation" in spec["paths"]
        assert "FinancialAttributes" in spec["components"]["schemas"]


def test_test_labels_cannot_change_artifacts_or_champion(environment):
    _, root = environment
    pointer = json.loads((root / "2m" / "active.json").read_text())
    run = root / "2m" / pointer["run_id"]
    manifest_hash = digest(run / "manifest.json")
    manifest = json.loads((run / "manifest.json").read_text())
    artifacts = {family: digest(run / config["artifact"]) for family, config in manifest["models"].items()}
    test = pd.read_parquet(run / "test.parquet")
    original = test.copy()
    test["target"] = 1 - test.target
    test.to_parquet(run / "test.parquet", index=False)
    evaluate(run)
    assert digest(run / "manifest.json") == manifest_hash
    assert all(digest(run / manifest["models"][family]["artifact"]) == value for family, value in artifacts.items())
    original.to_parquet(run / "test.parquet", index=False)
    evaluate(run)
    assert verify(run)["status"] == "verified"


def test_registration_route_has_its_own_contract(client):
    body = {"scenario": "10k", "external_id": uuid4().hex, "payer_id": uuid4().hex,
        "occurred_at": "2025-01-02T12:00:00Z", "amount": "10.00", "attributes": {
            "tipo_chave": "CPF", "pagador_banco": "A", "recebedor_banco": "B",
            "pagador_is_cnpj": 0, "recebedor_is_cnpj": 1, "pagador_cpf_zerado": 0}}
    result = post(client, body)
    assert result["scenario"] == "10k" and "saldo_anterior_pagador" not in result["features"]
    body["attributes"]["saldo_anterior_pagador"] = "100"
    assert client.post("/v1/transactions", json=body).status_code == 422


def test_rollback_refuses_to_drop_nonempty_database(environment):
    url, _ = environment
    with patch.dict(os.environ, {"DATABASE_URL": url}):
        with pytest.raises(RuntimeError, match="contém transações"):
            command.downgrade(Config("alembic.ini"), "base")


def test_concurrent_confirmations_are_single_terminal_transition(client):
    first = post(client, request())
    url = f"/v1/transactions/{first['id']}/confirmation"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.post(url, json={"status": "cancelled"}), range(2)))
    assert [item.status_code for item in results] == [200, 200]
    assert results[0].json() == results[1].json()


@pytest.mark.parametrize("prediction", [0, 1])
def test_prediction_does_not_confirm_execution(client, monkeypatch, prediction):
    monkeypatch.setattr(client.app.state.registry, "predict", lambda scenario, features: (
        .9 if prediction else .1, .5, prediction, "test-decision", "xgboost"))
    transaction = post(client, request())
    assert transaction["prediction"] == prediction
    assert transaction["status"] == "pending"
    assert transaction["executed_at"] is None and transaction["confirmed_at"] is None


def test_migration_roundtrip_when_empty(environment):
    url, _ = environment
    # A separate schema proves reversible initial migration without touching recorded data.
    schema = "test_empty_" + uuid4().hex
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    clean_url = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    try:
        with patch.dict(os.environ, {"DATABASE_URL": clean_url}):
            command.upgrade(Config("alembic.ini"), "head")
            command.downgrade(Config("alembic.ini"), "base")
            command.upgrade(Config("alembic.ini"), "head")
    finally:
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()
