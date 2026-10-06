"""Random Forest usando o mesmo pre-processamento e metricas do XGBoost."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from src.models.xgboost import _getPipeline, scoring


def crossValidate(df, XTrain, yTrain, sampler=None, classifier=None):
    """Avalia o modelo em cinco folds, com reamostragem apenas no treino."""
    if classifier is None:
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

    pipeline = _getPipeline(sampler=sampler, classifier=classifier)
    return cross_validate(
        pipeline,
        XTrain,
        yTrain,
        scoring=scoring,
        cv=5,
        error_score="raise",
    )
