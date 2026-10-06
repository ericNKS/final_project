"""Verificação de integração com dados sintéticos, sem acesso ao Kaggle."""

import unittest
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.exceptions import ConvergenceWarning

from src.models.logistic_regression import crossValidate, getClassifier, getPipeline
from src.models.xgboost import _getPipeline, scoring


class LogisticRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.y = pd.Series([0] * 280 + [1] * 20)
        cls.X = pd.DataFrame({
            "TipoChave": rng.choice(["CPF", "Email"], 300),
            "Status": rng.choice(["Concluído", "Pendente"], 300),
            "Pagador_Banco": rng.choice(["A", "B"], 300),
            "Recebedor_Banco": rng.choice(["A", "B"], 300),
            "Valor": rng.lognormal(7, 2, 300),
            "Hora": rng.integers(0, 24, 300),
            "Dia_Da_Semana": rng.integers(0, 7, 300),
            "Dia_Do_Mes": rng.integers(1, 29, 300),
            "Pagador_Is_CNPJ": rng.integers(0, 2, 300),
            "Recebedor_Is_CNPJ": rng.integers(0, 2, 300),
            "Pagador_CPF_Zerado": rng.integers(0, 2, 300),
            "Mesmo_Banco": rng.integers(0, 2, 300),
        })

    def test_all_sampling_strategies_and_settings(self):
        for sampler in (None, SMOTE(random_state=42),
                        RandomUnderSampler(sampling_strategy=.1, random_state=42),
                        SMOTETomek(random_state=42)):
            for adjusted in (False, True):
                with self.subTest(sampler=sampler, adjusted=adjusted):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        result = crossValidate(None, self.X, self.y, sampler=sampler,
                            classifier=getClassifier(adjusted=adjusted, sampler=sampler))
                    for metric in scoring:
                        values = result["test_" + metric]
                        self.assertEqual(len(values), 5)
                        self.assertTrue(np.isfinite(values).all())

    def test_scaler_uses_training_data_and_accepts_unseen_categories(self):
        pipeline = getPipeline().fit(self.X, self.y)
        prep = pipeline.named_steps["preprocessor"]
        scaler = prep.named_transformers_["numeric"]
        numeric_columns = prep.transformers_[1][2]
        np.testing.assert_allclose(scaler.mean_, self.X[numeric_columns].mean())
        before = scaler.mean_.copy()
        unseen = self.X.iloc[:3].copy()
        unseen["TipoChave"] = "Nova categoria"
        unseen["Valor"] = 1e9
        probabilities = pipeline.predict_proba(unseen)
        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_tree_pipeline_keeps_unscaled_numeric_features(self):
        prep = _getPipeline(None, None).named_steps["preprocessor"]
        self.assertEqual(prep.transformers[1][1], "passthrough")


if __name__ == "__main__":
    unittest.main()
