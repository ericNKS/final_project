"""Recria o gráfico de métricas a partir do teste salvo, sem treinar novamente."""
import pandas as pd
from evaluate_models import TABLES, plot_test_metrics

if __name__ == "__main__":
    plot_test_metrics(pd.read_csv(TABLES / "metricas_teste.csv"))

