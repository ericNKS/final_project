import kagglehub
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def loadData():
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