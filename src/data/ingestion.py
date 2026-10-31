import json
import joblib
import kagglehub
import pandas as pd
import logging
import os
import tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def loadPix10kData():
    """Carrega o dataset de 10.000 transações do Kaggle."""
    filepath = kagglehub.dataset_download("juniorbueno/pix-banking-transaction")
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
    """
    Carrega o dataset de 2M (treino) e o dataset de 10k (teste),
    aplica o tratamento e retorna (XTrain, yTrain, XTest, yTest)
    com schemas compatíveis.
    """
    from src.data.cleaning import prepare2mDataset, prepare10kDataset

    logger.info("Iniciando ingestão e preparação dos dados de Treino (2M) e Teste (10k)...")
    df_2m = loadFraud2mData()
    XTrain, yTrain = prepare2mDataset(df_2m)

    df_10k = loadPix10kData()
    XTest, yTest = prepare10kDataset(df_10k, reference_columns=list(XTrain.columns))

    logger.info(f"Dados prontos: XTrain={XTrain.shape}, yTrain={yTrain.shape}, XTest={XTest.shape}, yTest={yTest.shape}")
    return XTrain, yTrain, XTest, yTest


def getTrainTuningTestData(tuning_size: float = 0.2, random_state: int = 42):
    """
    Retorna os dados separados em:
    - XTrain, yTrain: parcela de treino do 2M (ex: 80% = 1.6M)
    - XTuning, yTuning: coletânea estratificada de 20% do 2M para testes de hiperparâmetros
    - XTest, yTest: dataset independente de 10k Pix para verificação final
    """
    X2m, y2m, XTest, yTest = getTrainTestData()

    logger.info(f"Separando partição de {tuning_size*100:.0f}% dos dados de 2M para tunagem de hiperparâmetros...")
    XTrain, XTuning, yTrain, yTuning = train_test_split(
        X2m, y2m,
        test_size=tuning_size,
        stratify=y2m,
        random_state=random_state
    )

    logger.info(f"Partições geradas: Treino={XTrain.shape}, Tunagem={XTuning.shape}, Teste={XTest.shape}")
    return XTrain, yTrain, XTuning, yTuning, XTest, yTest


def get2mDataSplit(test_size: float = 0.2, random_state: int = 42):
    """
    Carrega e prepara exclusivamente o dataset de 2M transações,
    dividindo em Treino (ex: 80% = 1.6M) e Teste (ex: 20% = 400k) estratificado.
    """
    from src.data.cleaning import prepare2mDataset
    logger.info("Carregando e preparando dataset completo de 2M de transações...")
    df_2m = loadFraud2mData()
    X, y = prepare2mDataset(df_2m)

    logger.info(f"Dividindo dataset de 2M em Treino ({(1-test_size)*100:.0f}%) e Teste ({test_size*100:.0f}%)...")
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def getTuningSample(XTrain, yTrain, sample_size: int = 100000, random_state: int = 42):
    """
    Extrai uma amostra representativa estratificada para testes rápidos de hiperparâmetros.
    """
    if len(yTrain) <= sample_size:
        return XTrain, yTrain
    frac = sample_size / len(yTrain)
    XSample, _, ySample, _ = train_test_split(
        XTrain, yTrain,
        train_size=frac,
        stratify=yTrain,
        random_state=random_state
    )
    return XSample, ySample


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
