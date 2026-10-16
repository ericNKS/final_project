"""Adaptador sklearn para estimadores cuML nativos, sem fallback de treino."""

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class GPUClassifier(ClassifierMixin, BaseEstimator):
    """Preserva a configuração do experimento e usa cuML para fit/predict."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
        from cuml.linear_model import LogisticRegression

        params = self.estimator.get_params()
        name = type(self.estimator).__name__
        if name == "RandomForestClassifier":
            keys = ("n_estimators", "max_depth", "min_samples_leaf",
                    "min_samples_split", "max_features", "bootstrap",
                    "class_weight", "random_state")
            self.model_ = RandomForestClassifier(
                **{key: params[key] for key in keys},
                n_streams=1, output_type="numpy",
            )
        elif name == "LogisticRegression":
            self.model_ = LogisticRegression(
                C=params["C"], class_weight=params["class_weight"],
                max_iter=params["max_iter"], tol=params["tol"],
                fit_intercept=params["fit_intercept"], output_type="numpy",
            )
        else:
            raise TypeError(f"Estimador GPU não suportado: {name}")

        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("O pipeline de fraude GPU espera classes 0 e 1")
        self.n_features_in_ = X.shape[1]
        logger.info("%s.fit: cuML nativo na GPU | shape=%s", name, X.shape)
        self.model_.fit(cp.asarray(X, dtype=cp.float32), cp.asarray(y, dtype=cp.int32))
        cp.cuda.get_current_stream().synchronize()
        logger.info("%s.fit: concluído na GPU", name)
        return self

    def predict_proba(self, X):
        import cupy as cp
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict_proba(cp.asarray(X, dtype=cp.float32)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
