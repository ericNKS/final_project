"""Behavioral proofs for point-in-time features and train/test isolation."""

import numpy as np
import pandas as pd
import pytest

from src.data.temporal import (feature_frame, feature_row, normalize,
                               parse_dates, temporal_folds, temporal_split, timestamp)
from src.models.experiments import Candidate, make_pipeline
from src.workflow import calibrate, select


def financial_raw(n=120):
    """Repeated payers and recurring positives across every temporal window."""
    return pd.DataFrame({
        "datetime_brasilia": pd.date_range("2025-01-01", periods=n, freq="h").astype(str),
        "id_pagador": [f"payer-{i % 3}" for i in range(n)],
        "valor_brl": np.arange(n, dtype=float) + 1,
        "tipo_transacao": ["pix" if i % 2 else "transfer" for i in range(n)],
        "saldo_anterior_pagador": np.full(n, 1000.),
        "saldo_anterior_recebedor": np.full(n, 500.),
        "saldo_posterior_pagador": np.zeros(n),
        "fraude": [int(i % 7 == 0) for i in range(n)],
    })


def test_future_and_shuffle_do_not_change_past_features():
    raw = financial_raw()
    old, _ = normalize(raw.iloc[:60], "2m")
    full, _ = normalize(raw.sample(frac=1, random_state=1), "2m")
    # Source fallback ids are ordering-only; no equal timestamps in this fixture.
    pd.testing.assert_frame_equal(feature_frame(old, "2m"), feature_frame(full, "2m").iloc[:60])
    assert "saldo_posterior_pagador" not in feature_frame(full, "2m")


def test_tied_events_exclude_each_other_and_match_serving():
    raw = financial_raw(20)
    raw.loc[:3, "id_pagador"] = "payer"
    raw.loc[1:2, "datetime_brasilia"] = raw.loc[1, "datetime_brasilia"]
    frame, _ = normalize(raw, "2m")
    features = feature_frame(frame, "2m")
    assert features.loc[1, "qtd_transacoes_pagador"] == 1
    assert features.loc[2, "qtd_transacoes_pagador"] == 1
    assert features.loc[1, "media_valor_pagador"] == features.loc[2, "media_valor_pagador"] == 1
    attributes = frame.loc[2, ["tipo_transacao", "saldo_anterior_pagador", "saldo_anterior_recebedor"]].to_dict()
    actual = feature_row("2m", frame.loc[2, "occurred_at"], frame.loc[2, "amount"], attributes, (1, 1., 1.))
    for key, value in actual.items():
        assert features.loc[2, key] == value
    assert pd.isna(features.loc[0, "media_valor_pagador"])


def test_partitions_and_folds_never_share_instants():
    frame, _ = normalize(financial_raw(), "2m")
    split = temporal_split(frame)
    assert frame.loc[split.development, "occurred_at"].max() < frame.loc[split.calibration, "occurred_at"].min()
    assert frame.loc[split.calibration, "occurred_at"].max() < frame.loc[split.test, "occurred_at"].min()
    for fit, val in temporal_folds(frame.loc[split.development]):
        assert frame.loc[fit, "occurred_at"].max() < frame.loc[val, "occurred_at"].min()


def test_invalid_date_quality_and_missing_payer():
    raw = financial_raw()
    raw.loc[0, "datetime_brasilia"] = "2025-02-30"
    frame, quality = normalize(raw, "2m")
    assert len(frame) == 119 and quality["invalid_dates"] == 1
    assert quality["invalid_dates_by_class"] == {"1": 1}
    raw.loc[1, "id_pagador"] = None
    with pytest.raises(ValueError, match="Pagador ausente"):
        normalize(raw, "2m")


def test_selection_has_no_test_metric_dependency():
    rows = [{"family": "xgboost", "candidate": "b", "status": "ok", "validation_ap": .8, "test_f1": 1},
            {"family": "random_forest", "candidate": "a", "status": "ok", "validation_ap": .9, "test_f1": 0}]
    assert select(rows)["candidate"] == "a"
    rows[0]["test_f1"], rows[1]["test_f1"] = 0, 1
    assert select(rows)["candidate"] == "a"
    assert calibrate(pd.Series([0, 1, 1, 0]), np.array([.1, .4, .8, .3])) == .4
    with pytest.raises(ValueError, match="duas classes"):
        calibrate(pd.Series([0, 0]), np.array([.1, .9]))


@pytest.mark.parametrize("strategy", ["original", "class_weight", "under", "smotenc", "smotenc_tomek"])
def test_each_strategy_has_valid_probabilities_and_discrete_categories(strategy):
    frame, _ = normalize(financial_raw(), "2m")
    X, y = feature_frame(frame, "2m"), frame.target
    pipeline = make_pipeline(Candidate("logistic_regression", strategy), "2m", y)
    pipeline.fit(X, y)
    unseen = X.iloc[:2].copy()
    unseen["tipo_transacao"] = "unseen"
    probabilities = pipeline.predict_proba(unseen)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    if "sampler" in pipeline.named_steps and strategy.startswith("smotenc"):
        prepared = pipeline.named_steps["prepare"].transform(X)
        sampled, _ = pipeline.named_steps["sampler"].fit_resample(prepared, y)
        assert set(sampled[:, -1]) <= {0, 1}


def test_brasilia_naive_date_is_normalized():
    assert timestamp("2025-01-01 12:00").isoformat() == "2025-01-01T15:00:00+00:00"


def test_mixed_offsets_invalid_dates_and_first_tied_history():
    dates = parse_dates(pd.Series(["2025-01-01 12:00", "2025-01-01T15:00:00Z",
                                  "2025-01-01T12:00:00-03:00", "2025-02-30"]))
    assert dates.iloc[:3].nunique() == 1 and pd.isna(dates.iloc[3])
    raw = financial_raw(20)
    raw.loc[:1, "id_pagador"] = "payer"
    raw.loc[1, "datetime_brasilia"] = raw.loc[0, "datetime_brasilia"]
    frame, _ = normalize(raw, "2m")
    features = feature_frame(frame, "2m")
    assert features.loc[:1, "qtd_transacoes_pagador"].eq(0).all()
    assert features.loc[:1, "mediana_valor_pagador"].isna().all()


def test_registration_contract_never_fabricates_financial_columns():
    raw = pd.DataFrame({"DataHora": ["2025-01-01 12:00"], "Valor": [5.],
        "TipoChave": ["CPF"], "Pagador_CPF_CNPJ": ["000.000.000-00"],
        "Recebedor_CPF_CNPJ": ["12.345.678/0001-00"], "Pagador_Banco": ["A"],
        "Recebedor_Banco": ["A"], "Status": ["Pendente"], "Anomalia": [1]})
    frame, _ = normalize(raw, "10k")
    features = feature_frame(frame, "10k")
    assert not any("saldo" in name for name in features)
    assert "Status" not in features
    row = frame.iloc[0]
    attrs = row[["tipo_chave", "pagador_banco", "recebedor_banco", "pagador_is_cnpj",
                 "recebedor_is_cnpj", "pagador_cpf_zerado"]].to_dict()
    assert features.iloc[0].to_dict() == feature_row("10k", row.occurred_at, row.amount, attrs)
