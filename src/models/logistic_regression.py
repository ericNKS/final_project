"""Regressão logística com padronização e reamostragem dentro dos folds."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV, cross_validate
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


def crossValidate(df, XTrain, yTrain, sampler=None, classifier=None, tuned=False):
    """Avalia cinco folds; df mantém a interface dos outros modelos."""
    pipeline = getPipeline(sampler=sampler, classifier=classifier)
    if tuned:
        pipeline = TunedThresholdClassifierCV(pipeline, scoring="f1")
        
    return cross_validate(
        pipeline,
        XTrain,
        yTrain,
        scoring=scoring,
        cv=5,
        error_score="raise",
    )
