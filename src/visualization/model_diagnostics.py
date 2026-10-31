"""Contagens de classificação e curvas descritivas de limiar no teste."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.utils.artifacts import savefig


def threshold_metrics(y_true, probabilities, thresholds):
    """Métricas para regra probabilidade >= limiar, incluindo empates e extremos."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities)
    order = np.argsort(p)
    cumulative = np.r_[0, np.cumsum(y[order])]
    cut = np.searchsorted(p[order], thresholds, side="left")
    tp = cumulative[-1] - cumulative[cut]
    predicted = len(y) - cut
    fp = predicted - tp
    fn = cumulative[-1] - tp
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted != 0)
    recall = np.divide(tp, cumulative[-1], out=np.zeros_like(tp, dtype=float), where=cumulative[-1] != 0)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    return {"F1": f1, "Precisão": precision, "Recall": recall}


def plot_model_diagnostics(y_true, probabilities, predictions, limits, output_dir,
                           dataset, tradeoff_filename, counts_filename):
    """Um painel por modelo; limiares vêm do treino/validação, nunca do teste."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(probabilities)
    rows = []
    for name in names:
        tn, fp, fn, tp = confusion_matrix(y_true, predictions[name], labels=[0, 1]).ravel()
        rows.append({"Modelo": name, "Acertos totais": int(tn + tp),
                     "Positivos corretos": int(tp), "Negativos corretos": int(tn),
                     "Falsos positivos": int(fp), "Falsos negativos": int(fn),
                     "Limiar": float(limits[name])})
    counts = pd.DataFrame(rows)
    labels = [textwrap.fill(n, 29) for n in names]
    fig, axes = plt.subplots(2, 3, figsize=(18, max(9, len(names) * 1.6)), layout="constrained")
    panels = [
        ("Acertos totais", "Acertos totais (VP + VN)", "#2563eb"),
        ("Positivos corretos", "Verdadeiros positivos (VP)", "#16a34a"),
        ("Falsos positivos", "Falsos positivos (FP)", "#dc2626"),
        ("Negativos corretos", "Verdadeiros negativos (VN)", "#0891b2"),
        ("Falsos negativos", "Falsos negativos (FN)", "#ea580c"),
    ]
    for ax, (metric, title, color) in zip(axes.flat, panels):
        bars = ax.barh(labels, counts[metric], color=color, height=.55)
        ax.bar_label(bars, labels=[f"{v:,}".replace(",", ".") for v in counts[metric]], padding=5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1, counts[metric].max()) * 1.25)
        ax.set_title(title)
        ax.set_xlabel("Quantidade de transações")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    legend = axes[1, 2]
    legend.axis("off")
    legend.text(.02, .9, "Como interpretar", fontsize=14, weight="bold", va="top")
    legend.text(.02, .78,
                "VP: fraudes/anomalias detectadas.\n\n"
                "VN: transações legítimas classificadas\ncorretamente.\n\n"
                "FP: transações legítimas sinalizadas\ncomo fraude/anomalia.\n\n"
                "FN: fraudes/anomalias não detectadas.\n\n"
                "Acertos totais = VP + VN\n"
                "Total do teste = VP + VN + FP + FN",
                fontsize=11, va="top", linespacing=1.25)
    fig.suptitle(f"{dataset}: acertos e erros de classificação no teste (n={len(y_true):,})\n"
                 "Escalas independentes; cada modelo usa seu limiar calibrado", fontsize=13)
    savefig(output_dir / counts_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    counts.to_csv(output_dir / counts_filename.replace(".png", ".csv"), index=False)

    cols = min(3, len(names))
    fig, axes = plt.subplots(int(np.ceil(len(names) / cols)), cols,
                             figsize=(6 * cols, 5 * int(np.ceil(len(names) / cols))),
                             squeeze=False, layout="constrained")
    for ax, name in zip(axes.flat, names):
        threshold = float(limits[name])
        grid = np.unique(np.r_[np.linspace(0, 1, 201), threshold])
        metrics = threshold_metrics(y_true, probabilities[name], grid)
        for (metric, values), color in zip(metrics.items(), ["#2563eb", "#16a34a", "#dc2626"]):
            ax.plot(grid, values, label=metric, color=color, linewidth=2)
        chosen = threshold_metrics(y_true, probabilities[name], [threshold])["F1"][0]
        ax.axvline(threshold, color="#7c3aed", linestyle="--", label=f"Limiar calibrado: {threshold:.4f}")
        ax.scatter([threshold], [chosen], color="#7c3aed", zorder=5, label=f"F1 nesse limiar: {chosen:.1%}")
        ax.set(title=textwrap.fill(name, 43), xlabel="Limiar de decisão", ylabel="Métrica",
               xlim=(0, 1), ylim=(0, 1.04))
        ax.grid(alpha=.2)
        ax.legend(loc="lower left", fontsize=9)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"{dataset}: trade-off de limiar por modelo\n"
                 "Curvas descritivas no teste; o limiar destacado foi escolhido no treino/validação", fontsize=13)
    savefig(output_dir / tradeoff_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return counts
