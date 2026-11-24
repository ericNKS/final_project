import json
import joblib
import kagglehub
import pandas as pd
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

def loadPix10kData():
    """Carrega o dataset de 10.000 transações do Kaggle."""
    cached = Path.home() / ".cache/kagglehub/datasets/juniorbueno/pix-banking-transaction/versions/1"
    filepath = str(cached) if (cached / "comprovantes_pix_10000_anomalias.csv").exists() else kagglehub.dataset_download("juniorbueno/pix-banking-transaction")
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        return pd.read_csv(f"{path}/comprovantes_pix_10000_anomalias.csv", sep=";")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise


def loadData():
    """Alias para loadPix10kData para compatibilidade com códigos legados."""
    return loadPix10kData()


def loadFraud2mData():
    """Carrega o dataset completo de 2 milhões de transações do Hugging Face."""
    logger.info("Carregando dataset de 2M transações do Hugging Face...")
    url = "hf://datasets/andremessina/pix-fraud-br/data/train-00000-of-00001.parquet"
    try:
        return pd.read_parquet(url)
    except Exception as e:
        logger.error(f"Error loading 2M dataset from {url}: {e}")
        raise


def getTrainTestData():
    """Removed unsafe cross-domain train/test route."""
    raise ValueError("Use main.py train --scenario 10k ou 2m: os alvos e contratos são distintos")


def getTrainTuningTestData(tuning_size: float = 0.1, random_state: int = 42):
    """Compatibility route: temporal development/calibration/test, all from 2M."""
    if tuning_size != 0.1:
        raise ValueError("O protocolo temporal fixa 70/10/20; use o fluxo canônico")
    from src.data.temporal import normalize, feature_frame, temporal_split
    frame, _ = normalize(loadFraud2mData(), "2m")
    X, y = feature_frame(frame, "2m"), frame.target
    split = temporal_split(frame)
    return (X.loc[split.development], y.loc[split.development],
            X.loc[split.calibration], y.loc[split.calibration],
            X.loc[split.test], y.loc[split.test])


def get2mDataSplit(test_size: float = 0.2, random_state: int = 42):
    """Compatibility chronological 80/20; the canonical training reserves calibration."""
    if test_size != .2:
        raise ValueError("Use a divisão temporal canônica")
    from src.data.temporal import normalize, feature_frame, temporal_split
    frame, _ = normalize(loadFraud2mData(), "2m")
    X, y = feature_frame(frame, "2m"), frame.target
    split = temporal_split(frame)
    development = split.development.append(split.calibration)
    return X.loc[development], X.loc[split.test], y.loc[development], y.loc[split.test]


def getTuningSample(XTrain, yTrain, sample_size: int = 80000, random_state: int = 42):
    """Legacy sampling cannot preserve tied timestamps without metadata."""
    raise ValueError("Use main.py train --tuning-rows: a amostra exige metadados temporais")


def saveModelArtifact(model, metadata: dict, filepath: str = "resultados/modelos/melhor_modelo_pix.joblib"):
    """
    Serializa o pipeline do modelo com joblib e salva arquivo de metadados JSON.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Escreve em arquivo novo e só então substitui o anterior (também no WSL/NTFS).
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        joblib.dump(model, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    meta_path = path.with_suffix(".json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".json", delete=False) as temp:
        meta_temp = Path(temp.name)
        temp.write(json.dumps(metadata, indent=2, ensure_ascii=False))
    try:
        os.replace(meta_temp, meta_path)
    finally:
        meta_temp.unlink(missing_ok=True)
    print(f"\nModelo salvo com sucesso em: {path}")
    print(f"Metadados salvos em: {meta_path}")


def printTrainResult(modelNames, *results):
    modelResults = pd.DataFrame({
        "Modelo": modelNames,
        "Accuracy": [
            result["test_accuracy"].mean() * 100
            for result in results
        ],
        "Precision": [
            result["test_precision"].mean() * 100
            for result in results
        ],
        "Recall": [
            result["test_recall"].mean() * 100
            for result in results
        ],
        "F1": [
            result["test_f1"].mean() * 100
            for result in results
        ],
        "ROC-AUC": [
            result["test_roc_auc"].mean() * 100
            for result in results
        ]
    })

    print(modelResults)


def printTestResult(modelNames, *results):
    """Exibe tabela comparativa com métricas de avaliação no conjunto de teste."""
    modelResults = pd.DataFrame({
        "Modelo": modelNames,
        "Accuracy": [
            result["accuracy"] * 100
            for result in results
        ],
        "Precision": [
            result["precision"] * 100
            for result in results
        ],
        "Recall": [
            result["recall"] * 100
            for result in results
        ],
        "F1": [
            result["f1"] * 100
            for result in results
        ],
        "ROC-AUC": [
            result["roc_auc"] * 100
            for result in results
        ],
        "PR-AUC": [
            result["pr_auc"] * 100
            for result in results
        ]
    })

    print(modelResults)
