"""OpenAPI stays generated from the same validated contracts used by the service."""

import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from src.api.app import create_app
from src.api.schemas import FinancialTransaction


def test_committed_openapi_matches_live_contract(tmp_path):
    application = create_app(model_root=tmp_path)
    expected = json.loads((Path(__file__).resolve().parents[1] / "docs/openapi.json").read_text())
    assert application.openapi() == expected


def test_money_and_timezone_are_validated():
    body = {"scenario": "2m", "external_id": "example", "payer_id": "opaque",
            "occurred_at": "2025-01-01T12:00:00-03:00", "amount": "1.00", "attributes": {
                "tipo_transacao": "pix", "saldo_anterior_pagador": "10", "saldo_anterior_recebedor": "10"}}
    for name, invalid in (("amount", "NaN"), ("amount", "1.001"), ("occurred_at", "2025-01-01 12:00")):
        with pytest.raises(ValidationError):
            FinancialTransaction(**{**body, name: invalid})
