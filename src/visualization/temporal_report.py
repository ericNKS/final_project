"""Readable offline figures for frozen temporal results; no model selection."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src.utils.artifacts import atomic_image_save

MODEL_NAMES = {"logistic_regression": "Regressão logística",
               "random_forest": "Random Forest", "xgboost": "XGBoost"}
MODEL_COLORS = {"logistic_regression": "#0072B2",
                "random_forest": "#D55E00", "xgboost": "#009E73"}
STRATEGY_NAMES = {"original": "Sem balanceamento", "class_weight": "Pesos por classe",
                  "under": "Subamostragem", "smotenc": "SMOTENC",
                  "smotenc_tomek": "SMOTENC + Tomek"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 11,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.labelcolor": "#334155", "text.color": "#0f172a",
         "xtick.color": "#475569", "ytick.color": "#475569",
         "axes.edgecolor": "#cbd5e1", "figure.facecolor": "white",
         "axes.facecolor": "white", "savefig.facecolor": "white"}


def _save(fig, destination: Path) -> None:
    """Publish complete images and release figures even when writing fails."""
    try:
        atomic_image_save(destination, lambda path: fig.savefig(path, dpi=180, bbox_inches="tight"))
    finally:
        plt.close(fig)


def _unit_axis(axis, horizontal: bool = False) -> None:
    """Use a shared zero-to-one scale so comparisons do not exaggerate changes."""
    if horizontal:
        axis.set_xlim(0, 1.15)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.grid(axis="x", color="#e2e8f0", linewidth=.8)
    else:
        axis.set_ylim(0, 1.05)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color="#e2e8f0", linewidth=.8)
    axis.set_axisbelow(True)


def render_report_plots(manifest: dict, predictions: pd.DataFrame,
                        validation: pd.DataFrame, calibration: pd.DataFrame,
                        output: Path) -> None:
    """Render the four report figures using only saved predictions and metrics."""
    families = list(manifest["models"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5.6),
                                 squeeze=False, layout="constrained")
        fig.suptitle("Matrizes de confusão", fontsize=16, fontweight="bold")
        fig.get_layout_engine().set(w_pad=.3, h_pad=.25, wspace=.12)
        for axis, family in zip(axes.flat, families):
            counts = confusion_matrix(predictions.target, predictions[f"{family}_prediction"], labels=[0, 1])
            totals = counts.sum(axis=1, keepdims=True)
            rates = np.divide(counts, totals, out=np.zeros((2, 2)), where=totals != 0)
            axis.imshow([[0, 0], [0, 1]], cmap=ListedColormap(["#f1f5f9", "#0072B2"]),
                        vmin=0, vmax=1)
            for row, column in np.ndindex(2, 2):
                share = f"{rates[row, column]:.1%}" if totals[row, 0] else "Classe ausente"
                color = "white" if (row, column) == (1, 1) else "#0f172a"
                axis.text(column, row - .07, f"{counts[row, column]:,}".replace(",", "."),
                          ha="center", va="center", fontsize=18, color=color)
                axis.text(column, row + .14, share, ha="center", va="center",
                          fontsize=11, color=color, alpha=.85)
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negativo", "Positivo"],
                     yticklabels=["Negativo", "Positivo"])
            axis.set_title(MODEL_NAMES.get(family, family), pad=20)
            axis.tick_params(length=0, pad=10)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.grid(False)
        fig.supylabel("Classe real", fontsize=11)
        fig.supxlabel("Classe prevista · percentuais por linha", fontsize=11)
        _save(fig, output / "confusion.png")

        if predictions.target.nunique() == 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
            fig.suptitle("Discriminação no teste reservado", fontsize=18, fontweight="bold")
            for index, family in enumerate(families):
                scores = predictions[f"{family}_score"]
                color = MODEL_COLORS.get(family, f"C{index}")
                line = ["-", "--", "-."][index % 3]
                name = MODEL_NAMES.get(family, family)
                precision, recall, _ = precision_recall_curve(predictions.target, scores)
                ap = average_precision_score(predictions.target, scores)
                axes[0].step(recall, precision, where="post", color=color, linestyle=line,
                             linewidth=2, label=f"{name} · AP {ap:.3f}")
                fpr, tpr, _ = roc_curve(predictions.target, scores)
                auc = roc_auc_score(predictions.target, scores)
                axes[1].plot(fpr, tpr, color=color, linestyle=line, linewidth=2,
                             label=f"{name} · AUC {auc:.3f}")
            prevalence = predictions.target.mean()
            axes[0].axhline(prevalence, color="#64748b", linestyle=":",
                               label=f"Referência · prevalência {prevalence:.2%}")
            axes[1].plot([0, 1], [0, 1], color="#64748b", linestyle=":", label="Referência aleatória")
            for axis, title, xlabel, ylabel in zip(axes,
                    ["Precisão × recall", "Curva ROC"],
                    ["Recall · positivos encontrados", "Taxa de falsos positivos"],
                    ["Precisão · acerto entre os sinalizados", "Recall · taxa de verdadeiros positivos"]):
                _unit_axis(axis)
                axis.set(xlim=(0, 1), title=title, xlabel=xlabel, ylabel=ylabel)
                axis.xaxis.set_major_formatter(PercentFormatter(1))
                axis.legend(loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False, fontsize=9)
            _save(fig, output / "curves.png")

        valid = validation.loc[validation.status.eq("ok")]
        comparison = valid.groupby(["family", "strategy"]).validation_ap.mean().unstack()
        fig, axes = plt.subplots(len(families), 1, figsize=(11, 4 * len(families) + 1),
                                 squeeze=False, layout="constrained")
        fig.get_layout_engine().set(w_pad=.2, h_pad=.2, hspace=.10)
        fig.suptitle("Balanceamento na validação temporal", fontsize=18, fontweight="bold")
        for index, (axis, family) in enumerate(zip(axes.flat, families)):
            scores = comparison.loc[family].dropna()
            bars = axis.barh([STRATEGY_NAMES.get(key, key) for key in scores.index], scores,
                             color=MODEL_COLORS.get(family, f"C{index}"), height=.48)
            axis.bar_label(bars, labels=[f"{score:.1%}" for score in scores], padding=9, fontsize=12)
            axis.invert_yaxis()
            _unit_axis(axis, horizontal=True)
            axis.set_title(MODEL_NAMES.get(family, family), pad=18, loc="left", fontsize=14)
            axis.set_xlabel("AP média · maior é melhor", labelpad=16)
            axis.tick_params(axis="both", length=0, pad=10, labelsize=12)
            axis.margins(y=.14)
        fig.supxlabel("Média das configurações manuais viáveis · Estratégias inviáveis são omitidas", fontsize=10)
        _save(fig, output / "balancing.png")

        fig, axis = plt.subplots(figsize=(11, 6), layout="constrained")
        positions = np.arange(len(families))
        for offset, kind, name, color in [(-.19, "default", "Limiar padrão · 0,5", "#94a3b8"),
                                          (.19, "tuned", "Limiar ajustado por F1", "#0072B2")]:
            rows = calibration.loc[calibration.threshold_type.eq(kind)].set_index("family").loc[families]
            bars = axis.bar(positions + offset, rows.f1, width=.34, color=color, label=name)
            axis.bar_label(bars, labels=[f"{value:.1%}" for value in rows.f1], padding=5)
        _unit_axis(axis)
        axis.set(xticks=positions, xticklabels=[MODEL_NAMES.get(family, family) for family in families],
                 ylabel="F1 na calibração · maior é melhor", title="Efeito do ajuste de limiar")
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
        fig.supxlabel("Período usado para ajustar o limiar · Não é uma estimativa independente de desempenho", fontsize=10)
        _save(fig, output / "thresholds.png")
