"""Recria contagens e trade-offs dos modelos finais usando previsões salvas."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from src.visualization.model_diagnostics import plot_model_diagnostics


def main():
    root = Path(__file__).resolve().parent
    output = root / "src/notebooks/graficos"
    for dataset in ("2m", "10k"):
        if dataset == "2m":
            folder = root / "resultados/tabelas"
            predictions = pd.read_parquet(folder / "predicoes_teste_2m.parquet")
            metrics = pd.read_csv(folder / "metricas_teste_2m.csv")
            limits = dict(zip(metrics["modelo"], metrics["limiar"]))
            destination = output
            number = "15"
        else:
            folder = root / "resultados/10k"
            predictions = pd.read_csv(folder / "predicoes_teste.csv")
            metrics = pd.read_csv(folder / "metricas_teste.csv")
            metadata = json.loads((folder / "metadados.json").read_text(encoding="utf-8"))
            selected = metadata["selecionados_por_familia"].values()
            metrics = metrics[metrics["Variante"].isin(selected)]
            limits = dict(zip(metrics["Variante"], metrics["Limiar"]))
            destination = output / "10k"
            number = "10"
        plot_model_diagnostics(
            predictions["classe_real"],
            {name: predictions[name + "_prob"] for name in limits},
            {name: predictions[name + "_pred"] for name in limits},
            limits, destination, f"Dataset {dataset.upper()}",
            f"07_tradeoff_limiar_metricas_teste_{dataset}.png",
            f"{number}_acertos_falsos_positivos_teste_{dataset}.png",
        )
        print(f"Gráficos de {dataset} atualizados: {destination}")


if __name__ == "__main__":
    main()
