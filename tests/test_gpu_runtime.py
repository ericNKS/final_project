"""Integração CUDA: execute no WSL com python -m unittest discover -s tests -p test_gpu_runtime.py."""
import ast
import io
from pathlib import Path
import unittest

from cuml.accel import install
install(log_level="info")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler

from src.models.xgboost import _getPipeline


class GPURuntimeTest(unittest.TestCase):
    def test_all_experiment_configurations(self):
        # Reutiliza as 24 configurações reais sem carregar o dataset completo.
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            )
        start = next(i for i, node in enumerate(tree.body) if assigns(node, "experiments"))
        end = next(i for i, node in enumerate(tree.body) if assigns(node, "results_list"))
        scope = dict(globals(), training_backend="gpu", xgb_device="cuda")
        exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), "main.py", "exec"), scope)

        template = _getPipeline(None, None)
        numeric = template.named_steps["preprocessor"].transformers[1][2]
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(600, len(numeric))), columns=numeric)
        X["tipo_transacao"] = np.where(np.arange(600) % 2, "pix", "transferencia")
        y = (X[numeric[0]] > 1.65).astype(int).to_numpy()
        X.loc[0, numeric[1]] = np.nan

        self.assertEqual(len(scope["experiments"]), 24)
        for exp in scope["experiments"]:
            with self.subTest(model=exp["modelo"], sampler=exp["balanceamento"], tuning=exp["tunagem"]):
                pipe = clone(_getPipeline(exp["sampler"], exp["clf"], exp["numeric_trans"]))
                pipe.fit(X.iloc[:450], y[:450])
                classifier = pipe.named_steps["classifier"]
                if isinstance(classifier, xgb.XGBClassifier):
                    import json
                    config = json.loads(classifier.get_booster().save_config())
                    self.assertTrue(config["learner"]["generic_param"]["device"].startswith("cuda"))
                else:
                    self.assertTrue(type(classifier.model_).__module__.startswith("cuml."))
                calibrated = TunedThresholdClassifierCV(pipe, scoring="f1", cv="prefit", refit=False)
                calibrated.fit(X.iloc[450:], y[450:])
                probs = calibrated.predict_proba(X.iloc[450:])
                self.assertTrue(np.isfinite(probs).all())
                np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
                artifact = io.BytesIO()
                joblib.dump(calibrated, artifact)
                artifact.seek(0)
                restored = joblib.load(artifact)
                np.testing.assert_array_equal(restored.predict(X.iloc[450:]), calibrated.predict(X.iloc[450:]))


if __name__ == "__main__":
    unittest.main()
