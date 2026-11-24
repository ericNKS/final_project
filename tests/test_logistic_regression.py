"""Classifier facades must use the same scenario contract and train-only transforms."""
import numpy as np
import pandas as pd
import pytest
from src.models.logistic_regression import getPipeline
from src.data.temporal import feature_frame, normalize
from test_temporal import financial_raw


def test_scaler_is_fitted_only_on_training_and_unknown_categories_are_supported():
    frame, _ = normalize(financial_raw(180), "2m")
    X, y = feature_frame(frame, "2m"), frame.target
    pipeline = getPipeline(y.iloc[:120]).fit(X.iloc[:120], y.iloc[:120])
    transformer = pipeline.named_steps["prepare"].named_transformers_["numeric"]
    scaler = transformer.named_steps["scaler"]
    before = scaler.mean_.copy()
    unseen = X.iloc[120:].copy()
    unseen["tipo_transacao"] = "unknown"
    unseen["amount"] = 1e9
    probabilities = pipeline.predict_proba(unseen)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(before, scaler.mean_)
    assert scaler.mean_[0] == X.amount.iloc[:120].mean()
