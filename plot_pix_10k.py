"""Executa pix.py e exporta nove gráficos equivalentes aos do pipeline de 2M."""
from pathlib import Path
import json
import runpy
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.artifacts import savefig, atomic_image_save
from sklearn.metrics import (f1_score,
                            precision_score, recall_score, confusion_matrix,
                            ConfusionMatrixDisplay, roc_curve, precision_recall_curve)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados/graficos/10k"
TABLES = ROOT / "resultados/10k"


def save(number, name):
    plt.tight_layout()
    savefig(OUT / f"{number:02d}_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {number:02d}_{name}.png", flush=True)


def generate_10k_plots(use_saved=False):
    # Usa a cadeia de confiança do sistema sem desativar a validação TLS.
    import truststore
    truststore.inject_into_ssl()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if use_saved:
        metadata = json.loads((TABLES / "metadados.json").read_text(encoding="utf-8"))
        saved = pd.read_csv(TABLES / "predicoes_teste.csv")
        test = pd.read_csv(TABLES / "metricas_teste.csv")
        ns = {"comparisonResults": pd.read_csv(TABLES / "metricas_validacao_cruzada.csv"),
              "testResults": test, "selectedVariants": metadata["selecionados_por_familia"],
              "bestVariant": metadata["modelo_final_notebook"],
              "testPredictions": {label: (saved[label + "_prob"].to_numpy(), saved[label + "_pred"].to_numpy())
                                  for label in test["Variante"]},
              "XTrain": pd.DataFrame(index=range(metadata["treino"])),
              "yTrain": pd.Series(index=range(metadata["treino"]), dtype=float),
              "XTest": pd.DataFrame({"Valor": saved["valor_brl"]}), "yTest": saved["classe_real"]}
    else:
        show = plt.show
        plt.show = lambda: plt.close()
        try:
            ns = runpy.run_path(str(ROOT / "src/notebooks/pix.py"))
        finally:
            plt.show = show
    X, y = ns["XTrain"], ns["yTrain"]
    Xt, yt = ns["XTest"], ns["yTest"]
    df, mt = ns["comparisonResults"], ns["testResults"]
    selected = ns["selectedVariants"]
    metrics = mt[mt["Variante"].isin(selected.values())].to_dict("records")
    predictions = ns["testPredictions"]
    families = list(selected)
    palette = {"XGBoost": "#1f77b4", "Random Forest": "#2ca02c", "Regressão Logística": "#ff7f0e"}
    threshold_palette = {"F1_Padrao": "#94a3b8", "F1_Tunado": "#10b981"}
    colors = {row["Variante"]: palette[row["Modelo"]] for row in metrics}
    financial = []
    for row in metrics:
        name = row["Variante"]
        prob, pred = predictions[name]
        anomaly = yt.to_numpy() == 1
        value = Xt["Valor"].to_numpy()
        financial.append({"Variante": name,
                          "Anomalias sinalizadas": value[anomaly & (pred == 1)].sum(),
                          "Anomalias não detectadas": value[anomaly & (pred == 0)].sum(),
                          "Falsos alarmes": value[~anomaly & (pred == 1)].sum()})
    money = pd.DataFrame(financial)
    df.to_csv(TABLES / "metricas_validacao_cruzada.csv", index=False)
    mt.to_csv(TABLES / "metricas_teste.csv", index=False)
    money.to_csv(TABLES / "impacto_financeiro.csv", index=False)
    pd.DataFrame({"classe_real": yt, "valor_brl": Xt["Valor"],
                  **{name + "_prob": pair[0] for name, pair in predictions.items()},
                  **{name + "_pred": pair[1] for name, pair in predictions.items()}}).to_csv(
                      TABLES / "predicoes_teste.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key in zip(axes, ("PR_AUC", "F1_Tunado", "ROC_AUC")):
        sns.boxplot(data=df, x="Modelo", y=key, hue="Modelo", palette=palette, legend=False, ax=ax)
        sns.stripplot(data=df, x="Modelo", y=key, color="black", ax=ax)
        ax.set_title(key + " — validação cruzada")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Dataset 10k: comparação de três modelos (4 balanceamentos × 5 folds)")
    save(1, "comparacao_por_modelo")

    plt.figure(figsize=(18, 7))
    melted = df.melt(id_vars=["Variante", "Fold"], value_vars=["F1_Padrao", "F1_Tunado"],
                     var_name="Limiar", value_name="F1")
    sns.barplot(data=melted, x="Variante", y="F1", hue="Limiar", palette=threshold_palette, errorbar="sd")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.title("Dataset 10k: impacto da calibração do limiar (média ± desvio padrão)")
    save(2, "impacto_por_tunagem")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, family in zip(axes, families):
        subset = df[df["Modelo"] == family].melt(id_vars=["Balanceamento", "Fold"],
                   value_vars=["F1_Padrao", "F1_Tunado"], var_name="Limiar", value_name="F1")
        sns.barplot(data=subset, x="Balanceamento", y="F1", hue="Limiar",
                    palette=threshold_palette, errorbar="sd", ax=ax)
        ax.set_title(family)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Dataset 10k: F1 por balanceamento e calibração (5 folds)")
    save(3, "tunagem_por_balanceamento")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, row in zip(axes, metrics):
        name = row["Variante"]
        ConfusionMatrixDisplay(confusion_matrix(yt, predictions[name][1]),
                               display_labels=["Normal", "Anomalia"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nF1: {row['F1']:.1%} | limiar: {row['Limiar']:.3f}")
    fig.suptitle("Dataset 10k: matrizes de confusão no teste de 2.000 transações")
    save(4, "matrizes_confusao_teste_10k")

    plt.figure(figsize=(9, 7))
    for row in metrics:
        name = row["Variante"]
        fpr, tpr, _ = roc_curve(yt, predictions[name][0])
        plt.plot(fpr, tpr, color=colors[name], label=f"{name}: {row['ROC_AUC']:.2%}")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Recall")
    plt.title("Dataset 10k: curvas ROC no teste de 2.000 transações")
    plt.legend()
    save(5, "curvas_roc_teste_10k")

    plt.figure(figsize=(9, 7))
    for row in metrics:
        name = row["Variante"]
        p, r, _ = precision_recall_curve(yt, predictions[name][0])
        plt.plot(r, p, color=colors[name], label=f"{name}: PR-AUC {row['PR_AUC']:.2%}")
        plt.scatter(row["Recall"], row["Precision"], color=colors[name], edgecolors="black")
    plt.axhline(yt.mean(), linestyle="--", color="red", label=f"Aleatório: {yt.mean():.1%}")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Dataset 10k: curvas precisão-recall no teste de 2.000 transações")
    plt.legend()
    save(6, "curvas_precision_recall_teste_10k")

    # Escolha exclusivamente pelo treino; o teste não escolhe modelo nem limiar.
    best = ns["bestVariant"]
    row = next(item for item in metrics if item["Variante"] == best)
    prob = predictions[best][0]
    from src.visualization.model_diagnostics import plot_model_diagnostics
    plot_model_diagnostics(
        yt, {item["Variante"]: predictions[item["Variante"]][0] for item in metrics},
        {item["Variante"]: predictions[item["Variante"]][1] for item in metrics},
        {item["Variante"]: item["Limiar"] for item in metrics},
        OUT, "Dataset 10k", "07_tradeoff_limiar_metricas_teste_10k.png",
        "10_acertos_falsos_positivos_teste_10k.png",
    )

    plt.figure(figsize=(10, 6))
    for cls, label, color in ((0, "Normal", "#16a34a"), (1, "Anomalia", "#dc2626")):
        sns.kdeplot(prob[yt.to_numpy() == cls], fill=True, clip=(0, 1), cut=0,
                    color=color, label=f"{label} (n={(yt == cls).sum()})")
    plt.axvline(row["Limiar"], linestyle="--", color="purple", label="Limiar do treino")
    plt.xlabel("Probabilidade de anomalia")
    plt.title(f"Dataset 10k: distribuição das probabilidades — {best}")
    plt.legend()
    save(8, "distribuicao_probabilidades_teste_10k")

    ax = money.set_index("Variante").plot.bar(figsize=(12, 6), color=["#16a34a", "#dc2626", "#f59e0b"])
    ax.set_ylabel("Valor das transações (R$)")
    ax.set_title("Dataset 10k: volume financeiro associado às classificações no teste\n"
                 "Anomalias sintéticas; os valores não representam prejuízo ou dinheiro protegido")
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)
    save(9, "impacto_financeiro_teste_10k")
    (TABLES / "metadados.json").write_text(json.dumps({"dataset": 10000, "treino": len(y),
        "teste": len(yt), "anomalias_teste": int(yt.sum()), "modelo_final_notebook": best,
        "selecionados_por_familia": selected,
        "nota": "Variantes possuem hiperparâmetros diferentes; comparação não isola apenas balanceamento.",
        "metricas": mt.to_dict("records")}, indent=2, ensure_ascii=False), encoding="utf-8")
    from PIL import Image, ImageOps, ImageDraw
    overview_paths = sorted(OUT.glob("*.png"))
    overview = Image.new("RGB", (1800, 450 * int(np.ceil(len(overview_paths) / 3))), "white")
    draw = ImageDraw.Draw(overview)
    for index, path in enumerate(overview_paths):
        x, top = (index % 3) * 600, (index // 3) * 450
        with Image.open(path) as source:
            thumbnail = ImageOps.contain(source, (590, 420))
            overview.paste(thumbnail, (x, top + 25))
        draw.text((x + 5, top + 5), path.name, fill="black")
    atomic_image_save(TABLES / "visao_geral.png", overview.save)
    print(mt.to_string(index=False))
    return {"metrics": metrics, "predictions": predictions, "y_test": yt,
            "financial": money, "validation": df, "selected": selected}


if __name__ == "__main__":
    generate_10k_plots(use_saved="--saved" in sys.argv)
