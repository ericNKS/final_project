"""Real CUDA acceptance: both scenarios, every strategy, exact history and persisted models."""
import io
import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest

if os.environ.get("RUN_GPU_TESTS") != "1":
    pytest.skip("Use RUN_GPU_TESTS=1 in .venv-gpu to exercise actual CUDA", allow_module_level=True)

import cupy as cp
from imblearn.under_sampling import TomekLinks
from sklearn.neighbors import NearestNeighbors
from src.models.experiments import Candidate, candidates, make_pipeline
from src.models.gpu import GPUTomek, GPUResampler, nearest_indices, verify_backend
from src.data.temporal import feature_frame, normalize, history_features
from src.data.gpu_features import history_features as gpu_history
from src.workflow import metrics, calibrate
from test_temporal import financial_raw
from test_api_postgres import source_fixture


@pytest.fixture(scope="module", autouse=True)
def cuda_runtime():
    information = verify_backend()
    assert information["total_vram_bytes"] > 0


@pytest.mark.parametrize("scenario", ["10k", "2m"])
@pytest.mark.parametrize("candidate", candidates(), ids=lambda item: item.name)
def test_gpu_candidates_and_roundtrip(scenario, candidate):
    raw = source_fixture(scenario, 180)
    frame, _ = normalize(raw, scenario)
    X, y = feature_frame(frame, scenario, "gpu"), frame.target
    model = make_pipeline(candidate, scenario, y, "gpu").fit(X, y)
    classifier = model.named_steps["classifier"]
    if candidate.family == "xgboost":
        device = json.loads(classifier.get_booster().save_config())["learner"]["generic_param"]["device"]
        assert device.startswith("cuda")
    else:
        assert type(classifier).__module__.startswith("cuml.")
    assert "CUDA" in model.execution_["prepare"]
    unknown = X.iloc[:3].copy()
    for name in ("tipo_transacao", "tipo_chave", "pagador_banco", "recebedor_banco"):
        if name in unknown:
            unknown[name] = "unseen"
    probability = model.predict_proba(unknown)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
    if candidate.strategy == "original":
        artifact = io.BytesIO()
        joblib.dump(model, artifact)
        artifact.seek(0)
        np.testing.assert_allclose(joblib.load(artifact).predict_proba(unknown), probability, atol=1e-6)
    del model


def test_gpu_history_matches_cpu_with_ties_duplicates_and_future():
    raw = financial_raw(200)
    raw.loc[:12, "id_pagador"] = "same"
    raw.loc[1:4, "datetime_brasilia"] = raw.loc[0, "datetime_brasilia"]
    raw.loc[5:12, "valor_brl"] = 7.
    frame, _ = normalize(raw.sample(frac=1, random_state=3), "2m")
    pd.testing.assert_frame_equal(gpu_history(frame), history_features(frame), check_dtype=False, atol=1e-10)
    pd.testing.assert_frame_equal(feature_frame(frame, "2m", "gpu"), feature_frame(frame, "2m", "cpu"),
                                  check_dtype=False, atol=1e-10)
    older = frame.iloc[:60]
    pd.testing.assert_frame_equal(gpu_history(older), gpu_history(frame).iloc[:60], check_dtype=False)
    first = gpu_history(frame).iloc[0]
    assert first.qtd_transacoes_pagador == 0 and pd.isna(first.mediana_valor_pagador)


def test_large_single_payer_prefix_median():
    n = 10001
    frame = pd.DataFrame({"payer_id": ["same"] * n, "source_id": np.arange(n).astype(str),
        "occurred_at": pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC"),
        "amount": np.random.default_rng(42).integers(0, 1000, n).astype(float)})
    actual = gpu_history(frame)
    for index in (0, 1, 2, 100, 10000):
        assert actual.loc[index, "qtd_transacoes_pagador"] == index
        if index:
            assert actual.loc[index, "mediana_valor_pagador"] == np.median(frame.amount.iloc[:index])
            assert actual.loc[index, "media_valor_pagador"] == frame.amount.iloc[:index].mean()


def test_cuda_metrics_match_sklearn_for_ties_and_absent_classes():
    probability = np.array([0., .2, .2, .5, .5, 1.])
    for y in (pd.Series([0, 1, 0, 1, 1, 0]), pd.Series(np.zeros(6, dtype=int)), pd.Series(np.ones(6, dtype=int))):
        for threshold in (0, .2, .5, 1, 1.01):
            cpu, gpu = metrics(y, probability, threshold), metrics(y, probability, threshold, "gpu")
            for name in cpu:
                if cpu[name] is None:
                    assert gpu[name] is None
                else:
                    assert gpu[name] == pytest.approx(cpu[name], abs=1e-12)
        if y.nunique() == 2:
            assert calibrate(y, probability, "gpu") == calibrate(y, probability)


def test_exact_cuda_neighbours_and_tomek_match_reference():
    values = np.random.default_rng(1).normal(size=(100, 6)).astype(np.float32)
    labels = np.array([0] * 80 + [1] * 20)
    actual = cp.asnumpy(nearest_indices(cp.asarray(values), 5))
    expected = NearestNeighbors(n_neighbors=6).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    np.testing.assert_array_equal(actual, expected)
    sampler = GPUTomek()
    sampler.fit_resample(cp.asarray(values), cp.asarray(labels))
    reference = TomekLinks()
    reference.fit_resample(values, labels)
    np.testing.assert_array_equal(cp.asnumpy(sampler.sample_indices_), reference.sample_indices_)
    duplicates = cp.zeros((10, 2), dtype=cp.float32)
    neighbours = nearest_indices(duplicates, 1)[:, 0]
    assert bool(cp.all(neighbours != cp.arange(10)))


def test_synthesis_keeps_discrete_categories_and_training_medians():
    values = cp.asarray(np.random.default_rng(2).normal(size=(100, 3)), dtype=cp.float32)
    values[:, 2] = cp.arange(100) % 3
    labels = cp.asarray([0] * 80 + [1] * 20, dtype=cp.int32)
    sampler = GPUResampler("smotenc", 2)
    balanced, target = sampler.fit_resample(values, labels)
    assert set(cp.asnumpy(cp.unique(balanced[:, 2]))) <= {0, 1, 2}
    np.testing.assert_array_equal(cp.asnumpy(cp.bincount(target)), [80, 80])
