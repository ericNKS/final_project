"""Comparações descritivas entre bases com alvos e prevalências distintos."""
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from src.utils.artifacts import savefig

logger = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[1] / "notebooks/graficos"


def generate_comparison_plots(metrics_2m, probs_2m, y_test_2m, financial_2m,
                              data_10k, tuning_2m):
    OUT.mkdir(parents=True, exist_ok=True)

    def save(name):
        plt.tight_layout()
        path = OUT / name
        savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Salvo: %s", path)

    families = ["XGBoost", "Random Forest", "Regressão Logística"]
    rows = []
    keys = {"f1": "F1", "precision": "Precision", "recall": "Recall",
            "pr_auc": "PR_AUC", "roc_auc": "ROC_AUC"}
    for row in metrics_2m:
        family = next(f for f in families if row["modelo"].startswith(f))
        rows.append({"Dataset": "2M", "Modelo": family,
                     **{key: row[key] for key in keys}})
    for row in data_10k["metrics"]:
        rows.append({"Dataset": "10k", "Modelo": row["Modelo"],
                     **{key: row[column] for key, column in keys.items()}})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, key in zip(axes.flat, keys):
        sns.barplot(data=table, x="Modelo", y=key, hue="Dataset", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title(keys[key])
        ax.tick_params(axis="x", rotation=15)
    axes.flat[-1].axis("off")
    fig.suptitle("Comparação descritiva: 10k (anomalias sintéticas) e 2M (fraude Pix)\n"
                 "Bases distintas: diferenças não isolam o efeito do tamanho do dataset")
    save("10_comparacao_10k_vs_2m_metricas.png")

    positives = [int(np.sum(data_10k["y_test"])), int(np.sum(y_test_2m))]
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=["10k", "2M"], y=[100 / n for n in positives])
    ax.bar_label(ax.containers[0], fmt="%.4f p.p.")
    ax.set_ylabel("Variação do recall por um positivo detectado (p.p.)")
    ax.set_title(f"Resolução do recall no teste: {positives[0]} vs {positives[1]:,} positivos")
    save("11_comparacao_10k_vs_2m_resolucao_degrau.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for column, (label, truth, predictions) in enumerate([
        ("10k", data_10k["y_test"], {
            row["Modelo"]: data_10k["predictions"][row["Variante"]][0]
            for row in data_10k["metrics"]}),
        ("2M", y_test_2m, probs_2m),
    ]):
        for model, prob in predictions.items():
            fpr, tpr, _ = roc_curve(truth, prob)
            precision, recall, _ = precision_recall_curve(truth, prob)
            axes[0, column].plot(fpr, tpr, label=model)
            axes[1, column].plot(recall, precision, label=model)
        axes[0, column].plot([0, 1], [0, 1], "k--")
        axes[1, column].axhline(np.mean(truth), color="gray", linestyle="--", label="Prevalência")
        axes[0, column].set(title=f"ROC — {label}", xlabel="Taxa de falsos positivos", ylabel="Recall")
        axes[1, column].set(title=f"Precisão-recall — {label}", xlabel="Recall", ylabel="Precisão")
        for ax in axes[:, column]:
            ax.legend(fontsize=8)
    save("12_comparacao_10k_vs_2m_curvas_pr_roc.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, frame, title in zip(axes, [data_10k["validation"], tuning_2m],
                               ["10k: validação cruzada", "2M: amostra de tunagem"]):
        sns.barplot(data=frame, x="Balanceamento", y="F1_Tunado", hue="Modelo", ax=ax)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Balanceamento: comparação descritiva com protocolos e hiperparâmetros distintos")
    save("13_comparacao_10k_vs_2m_balanceamento.png")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    small = data_10k["financial"].set_index("Variante")
    large = pd.DataFrame(financial_2m).set_index("modelo")[[
        "fraudes_interceptadas_brl", "fraudes_nao_detectadas_brl", "falsos_positivos_brl"]]
    large.columns = ["Fraudes detectadas", "Fraudes não detectadas", "Falsos alarmes"]
    for ax, frame, title in zip(axes, [small, large], [
        "10k: valores associados a anomalias sintéticas", "2M: valores associados a fraude Pix"]):
        frame.plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Valor (R$); escalas independentes")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.suptitle("Escala financeira por dataset — anomalia sintética não equivale a prejuízo")
    save("14_comparacao_10k_vs_2m_escala_financeira.png")
    table.to_csv(OUT.parents[2] / "resultados/tabelas/comparacao_10k_2m.csv", index=False)
    return table
