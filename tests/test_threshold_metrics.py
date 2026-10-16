import unittest
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from src.visualization.model_diagnostics import threshold_metrics


class ThresholdMetricsTest(unittest.TestCase):
    def test_ties_extremes_and_absent_classes(self):
        probabilities = np.array([0, .2, .2, .5, 1, 1])
        thresholds = np.array([0, .2, .200001, .5, 1, 1.01])
        for y in [np.array([0, 1, 0, 1, 0, 1]), np.zeros(6), np.ones(6)]:
            actual = threshold_metrics(y, probabilities, thresholds)
            for label, metric in [("F1", f1_score), ("Precisão", precision_score), ("Recall", recall_score)]:
                expected = [metric(y, probabilities >= t, zero_division=0) for t in thresholds]
                np.testing.assert_allclose(actual[label], expected)
