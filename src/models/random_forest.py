"""Random Forest usando o mesmo pre-processamento e metricas do XGBoost."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TunedThresholdClassifierCV, cross_validate

from src.models.xgboost import _getPipeline, scoring


def getPipeline(sampler=None, classifier=None):
    if classifier is None:
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    return _getPipeline(sampler=sampler, classifier=classifier)


def crossValidate(df, XTrain, yTrain, sampler=None, classifier=None, tuned=False):
    """Avalia o modelo em cinco folds, com reamostragem apenas no treino."""
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
