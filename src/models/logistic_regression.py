"""Regressão logística com padronização e reamostragem dentro dos folds."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from src.models.xgboost import _getPipeline, scoring


def getClassifier(adjusted=False, sampler=None):
    """Configurações manuais; não representam busca de hiperparâmetros."""
    return LogisticRegression(
        C=0.1 if adjusted else 1.0,
        class_weight="balanced" if adjusted and sampler is None else None,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )


def getPipeline(sampler=None, classifier=None):
    """Ajusta o scaler no treino, antes da reamostragem opcional."""
    if classifier is None:
        classifier = getClassifier()
    return _getPipeline(
        sampler=sampler,
        classifier=classifier,
        numeric_transformer=StandardScaler(),
    )


def crossValidate(df, XTrain, yTrain, sampler=None, classifier=None):
    """Avalia cinco folds; df mantém a interface dos outros modelos."""
    return cross_validate(
        getPipeline(sampler=sampler, classifier=classifier),
        XTrain,
        yTrain,
        scoring=scoring,
        cv=5,
        error_score="raise",
    )
