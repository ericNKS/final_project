"""One experiment factory: preprocessing and sampling are fitted only on training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data.temporal import CATEGORIES, NUMERICS, Scenario

if TYPE_CHECKING:
    from src.models.gpu import GPUPipeline

FAMILIES = ("logistic_regression", "random_forest", "xgboost")
STRATEGIES = ("original", "class_weight", "under", "smotenc", "smotenc_tomek")


@dataclass(frozen=True, order=True)
class Candidate:
    family: str
    strategy: str
    adjusted: bool = False

    @property
    def name(self) -> str:
        return f"{self.family}__{self.strategy}__{'adjusted' if self.adjusted else 'default'}"


def candidates() -> list[Candidate]:
    """Fixed, declared search space; adjusted means candidate parameters, not a claim of tuning."""
    return [Candidate(family, strategy, adjusted)
            for family in FAMILIES for strategy in STRATEGIES for adjusted in (False, True)]


def make_pipeline(candidate: Candidate, scenario: Scenario, y: pd.Series,
                  backend: str = "cpu") -> Pipeline | GPUPipeline:
    """Keep categories discrete through synthesis, then one-hot encode before Tomek/model."""
    if set(y.unique()) != {0, 1}:
        raise ValueError("Treino precisa das duas classes")
    if backend == "gpu":
        from src.models.gpu import GPUPipeline
        return GPUPipeline(candidate, scenario)
    if backend != "cpu":
        raise ValueError("Backend deve ser gpu ou cpu")
    categorical, numeric = CATEGORIES[scenario], NUMERICS[scenario]
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if candidate.family == "logistic_regression" or candidate.strategy.startswith("smotenc"):
        numeric_steps.append(("scaler", StandardScaler()))
    prepare = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
    ])
    cat_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    steps = [("prepare", prepare)]
    if candidate.strategy == "under":
        steps.append(("sampler", RandomUnderSampler(random_state=42)))
    elif candidate.strategy.startswith("smotenc"):
        if int(y.value_counts().min()) < 6:
            raise ValueError("SMOTENC exige ao menos seis exemplos por classe no treino")
        steps.append(("sampler", SMOTENC(categorical_features=cat_indices, random_state=42)))
    steps.append(("encode", ColumnTransformer([
        ("numeric", "passthrough", list(range(len(numeric)))),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_indices),
    ])))
    if candidate.strategy == "smotenc_tomek":
        steps.append(("tomek", TomekLinks()))
    weighted = candidate.strategy == "class_weight"
    if candidate.family == "logistic_regression":
        classifier = LogisticRegression(C=.1 if candidate.adjusted else 1.,
                                        class_weight="balanced" if weighted else None,
                                        max_iter=2000, random_state=42)
    elif candidate.family == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100,
                                            max_depth=12 if candidate.adjusted else None,
                                            min_samples_leaf=2 if candidate.adjusted else 1,
                                            class_weight="balanced" if weighted else None,
                                            n_jobs=-1, random_state=42)
    elif candidate.family == "xgboost":
        classifier = XGBClassifier(n_estimators=100, max_depth=6 if candidate.adjusted else 3,
                                   learning_rate=.1, tree_method="hist",
                                   device="cuda" if backend == "gpu" else "cpu",
                                   scale_pos_weight=float((y == 0).sum() / (y == 1).sum()) if weighted else 1.,
                                   random_state=42, n_jobs=-1)
    else:
        raise ValueError("Família desconhecida")
    steps.append(("classifier", classifier))
    return Pipeline(steps)


class ThresholdModel:
    """Persist a frozen estimator and its calibration-period F1 threshold."""

    def __init__(self, pipeline: Pipeline, threshold: float):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
