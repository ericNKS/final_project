"""Confere métricas salvas contra previsões e verifica artefatos do experimento."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from evaluate_models import FAMILIES, SLUGS, METRICS


def main():
    root = Path(__file__).resolve().parent / "resultados"
    cv = pd.read_csv(root / "tabelas" / "metricas_validacao_cruzada.csv")
    test = pd.read_csv(root / "tabelas" / "metricas_teste.csv")
    predictions = pd.read_csv(root / "tabelas" / "predicoes_teste.csv")
    metadata = json.loads((root / "metadados" / "metadados.json").read_text(encoding="utf-8"))
    assert len(cv) == 24 and cv["nome"].nunique() == 24
    assert len(test) == 3 and set(test["modelo"]) == set(FAMILIES)
    assert len(predictions) == metadata["teste"]
    assert predictions["indice_original"].is_unique
    assert np.isfinite(cv[["cv_" + m for m in METRICS]].to_numpy()).all()
    y = predictions["anomalia_real"]
    scorers = dict(accuracy=accuracy_score, precision=precision_score,
                   recall=recall_score, f1=f1_score, roc_auc=roc_auc_score,
                   average_precision=average_precision_score)
    for family in FAMILIES:
        candidates = cv[cv["modelo"] == family]
        assert len(candidates) == 8
        selected = candidates.sort_values(["cv_f1", "cv_average_precision", "cv_roc_auc"],
                                          ascending=False, kind="stable").iloc[0]
        row = test[test["modelo"] == family].iloc[0]
        assert selected["nome"] == row["nome"] == metadata["selecionados"][family]
        prob = predictions["prob_" + SLUGS[family]]
        pred = predictions["pred_" + SLUGS[family]]
        assert prob.between(0, 1).all() and pred.isin([0, 1]).all()
        for metric, scorer in scorers.items():
            actual = scorer(y, prob if metric in ("roc_auc", "average_precision") else pred)
            np.testing.assert_allclose(actual, row[metric], atol=1e-12)
        np.testing.assert_array_equal(confusion_matrix(y, pred, labels=[0, 1]).ravel(),
                                      row[["tn", "fp", "fn", "tp"]].to_numpy())
        pipeline = joblib.load(root / "modelos" / (SLUGS[family] + ".joblib"))
        np.testing.assert_array_equal(pipeline.classes_, [0, 1])
        assert pipeline.named_steps["classifier"].n_features_in_ > 0
    images = [p for folder in ("dados", "random_forest", "xgboost", "logistic_regression", "comparacao")
              for p in (root / "graficos" / folder).glob("*.png")]
    assert len(images) == 20, len(images)
    for path in images:
        with Image.open(path) as image:
            image.verify()
    print("OK: 24 configurações, 3 modelos, métricas recalculadas, seleção e 20 imagens verificadas.")


if __name__ == "__main__":
    main()

