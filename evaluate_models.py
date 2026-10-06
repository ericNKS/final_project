"""Executa os experimentos Pix e salva graficos exclusivos e comparativos."""
from pathlib import Path
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from src.data.ingestion import loadData
from src.data.cleaning import (
    extractDateFeatures, extractPaymentTypeFeatures,
    extractSameBankTransactionFeature, removeUnnecessaryColumns,
)
from src.models.xgboost import _getPipeline, scoring
from src.models.logistic_regression import getPipeline as getLogisticPipeline, getClassifier as getLogisticClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados"
PLOTS = OUT / "graficos"
TABLES = OUT / "tabelas"
MODELS = OUT / "modelos"
REPORTS = OUT / "relatorios"
META = OUT / "metadados"
RF_OUT = PLOTS / "random_forest"
COMPARE_OUT = PLOTS / "comparacao"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
LABELS = ["Acurácia", "Precisão", "Recall", "F1", "ROC-AUC", "AP"]
JOBS = min(4, os.cpu_count() or 1)
FAMILIES = ("Random Forest", "XGBoost", "Regressão Logística")
SLUGS = {"Random Forest": "random_forest", "XGBoost": "xgboost", "Regressão Logística": "logistic_regression"}
COLORS = {"Random Forest": "#1976d2", "XGBoost": "#e76f51", "Regressão Logística": "#2a9d8f"}
LR_OUT = PLOTS / "logistic_regression"

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gráfico salvo: {path.relative_to(ROOT)}", flush=True)

def heatmap(frame, path, title):
    values = frame[[f"cv_{m}" for m in METRICS]].to_numpy()
    fig, ax = plt.subplots(figsize=(11, max(5, len(frame) * .47)))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(METRICS)), LABELS)
    ax.set_yticks(range(len(frame)), frame["nome"])
    ax.set_title(title, pad=16)
    for row in range(len(frame)):
        for col in range(len(METRICS)):
            value = values[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > .65 else "#222")
    fig.colorbar(im, ax=ax, label="Média nos 5 folds")
    save_plot(fig, path)

def metrics_table(frame, prefix=""):
    lines = ["| Modelo | " + " | ".join(LABELS) + " |",
             "| --- | " + " | ".join(["---"] * len(METRICS)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + str(row["nome"]) + " | " +
                     " | ".join(f"{row[prefix + m]:.4f}" for m in METRICS) + " |")
    return "\n".join(lines)

def curves(y_test, probabilities, families, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for family in families:
        RocCurveDisplay.from_predictions(y_test, probabilities[family], name=family,
                                         ax=axes[0], curve_kwargs={"color": COLORS[family]})
        PrecisionRecallDisplay.from_predictions(y_test, probabilities[family], name=family,
                                                ax=axes[1], color=COLORS[family])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="ROC no teste", xlabel="Taxa de falsos positivos", ylabel="Recall")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Prevalência")
    axes[1].set(title="Precisão–recall no teste", xlabel="Recall", ylabel="Precisão")
    axes[1].legend()
    save_plot(fig, path)

def plot_test_metrics(test_frame):
    fig, ax = plt.subplots(figsize=(14, 6))
    xx = np.arange(len(METRICS))
    width = .8 / len(FAMILIES)
    for i, family in enumerate(FAMILIES):
        row = test_frame[test_frame["modelo"] == family].iloc[0]
        bars = ax.bar(xx + (i - (len(FAMILIES) - 1) / 2) * width,
                      [row[m] for m in METRICS], width, label=family, color=COLORS[family])
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xticks(xx, LABELS)
    ax.set(title="Modelos selecionados: métricas no mesmo teste", ylabel="Pontuação", ylim=(0, 1.13))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=3, frameon=False)
    save_plot(fig, COMPARE_OUT / "02_metricas_teste.png")


def main():
    for folder in (RF_OUT, LR_OUT, COMPARE_OUT, PLOTS / "xgboost", PLOTS / "dados", TABLES, MODELS, REPORTS, META):
        folder.mkdir(parents=True, exist_ok=True)
    print("Carregando a base Pix do Kaggle...", flush=True)
    df = loadData()
    for prepare in (extractDateFeatures, extractPaymentTypeFeatures,
                    extractSameBankTransactionFeature, removeUnnecessaryColumns):
        df = prepare(df)
    assert set(df["Anomalia"].unique()) == {0, 1}, "Alvo precisa conter classes 0 e 1."
    X, y = df.drop(columns="Anomalia"), df["Anomalia"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42)
    folds = list(StratifiedKFold(n_splits=5, shuffle=False).split(X_train, y_train))
    print(f"Base: {len(df)}; treino: {len(X_train)}; teste: {len(X_test)}; "
          f"classes: {y.value_counts().to_dict()}", flush=True)
    metadata = {
        "dataset": "juniorbueno/pix-banking-transaction",
        "arquivo": "comprovantes_pix_10000_anomalias.csv",
        "registros": len(df), "treino": len(X_train), "teste": len(X_test),
        "classes_total": {str(k): int(v) for k, v in y.value_counts().items()},
        "classes_treino": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "classes_teste": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "random_state": 42, "folds": 5, "embaralhamento_folds": False,
        "selecao": "Maior F1 medio de validacao; desempate por AP e ROC-AUC",
        "limiar": .5, "jobs": JOBS,
    }
    (META / "metadados.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = y.value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Legítimas (0)", "Anômalas (1)"], counts, color=["#2a9d8f", "#e76f51"])
    ax.bar_label(bars, padding=4)
    ax.set(title="Distribuição dos rótulos na base Pix", ylabel="Transações")
    save_plot(fig, PLOTS / "dados" / "01_distribuicao_classes.png")
    samplers = {
        "Sem reamostragem": lambda: None,
        "SMOTE": lambda: SMOTE(random_state=42),
        "Under": lambda: RandomUnderSampler(sampling_strategy=.1, random_state=42),
        "SMOTETomek": lambda: SMOTETomek(random_state=42),
    }
    adjusted_xgb = {
        "Sem reamostragem": dict(max_depth=8, learning_rate=.1, n_estimators=40,
            random_state=0, min_child_weight=1, subsample=1, colsample_bytree=1,
            colsample_bynode=.9, num_parallel_tree=1, scale_pos_weight=3, tree_method="hist"),
        "SMOTE": dict(max_depth=12, learning_rate=.1, n_estimators=100,
            random_state=0, min_child_weight=1, subsample=1, colsample_bytree=1,
            colsample_bynode=.9, num_parallel_tree=1, scale_pos_weight=1, tree_method="hist"),
        "Under": dict(max_depth=3, learning_rate=.1, n_estimators=200,
            random_state=0, colsample_bytree=.8, scale_pos_weight=1, tree_method="hist"),
        "SMOTETomek": dict(max_depth=14, learning_rate=.1, n_estimators=40,
            random_state=5, min_child_weight=0, subsample=1, colsample_bytree=.5,
            colsample_bynode=1, num_parallel_tree=2, scale_pos_weight=2.8, tree_method="hist"),
    }
    candidates, rows = {}, []
    for family in FAMILIES:
        for strategy, sampler_factory in samplers.items():
            for adjusted in (False, True):
                setting = "Ajustada" if adjusted else "Padrão"
                name = f"{family} | {strategy} | {setting}"
                if family == "Random Forest":
                    params = dict(random_state=42, n_jobs=JOBS)
                    if adjusted:
                        params.update(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                      class_weight="balanced" if strategy == "Sem reamostragem" else None)
                    classifier = RandomForestClassifier(**params)
                elif family == "XGBoost":
                    params = dict(adjusted_xgb[strategy]) if adjusted else dict(random_state=42)
                    classifier = XGBClassifier(**params, n_jobs=JOBS)
                sampler = sampler_factory()
                if family == "Regressão Logística":
                    classifier = getLogisticClassifier(adjusted=adjusted, sampler=sampler)
                    pipeline = getLogisticPipeline(sampler=sampler, classifier=classifier)
                else:
                    pipeline = _getPipeline(sampler=sampler, classifier=classifier)
                candidates[name] = pipeline
                print(f"Avaliando: {name}", flush=True)
                start = time.perf_counter()
                result = cross_validate(pipeline, X_train, y_train, cv=folds,
                    scoring=dict(scoring, average_precision="average_precision"),
                    error_score="raise", n_jobs=1)
                row = dict(nome=name, modelo=family, estrategia=strategy, configuracao=setting,
                           tempo_cv_s=time.perf_counter() - start)
                for metric in METRICS:
                    values = result["test_" + metric]
                    assert np.isfinite(values).all(), (name, metric)
                    row["cv_" + metric], row["std_" + metric] = float(values.mean()), float(values.std())
                row["tempo_fit_medio_s"] = float(result["fit_time"].mean())
                rows.append(row)
                pd.DataFrame(rows).to_csv(TABLES / "metricas_validacao_cruzada.csv", index=False)
                print(f"F1 CV={row['cv_f1']:.4f}; AP CV={row['cv_average_precision']:.4f}; "
                      f"tempo={row['tempo_cv_s']:.1f}s", flush=True)
    cv_frame = pd.DataFrame(rows)
    rf_frame = cv_frame[cv_frame["modelo"] == "Random Forest"].copy()
    rf_plot = rf_frame.copy()
    rf_plot["nome"] = rf_plot["estrategia"] + " | " + rf_plot["configuracao"]
    heatmap(rf_plot, RF_OUT / "02_configuracoes_validacao.png",
            "Random Forest: comparação das configurações no treino")
    heatmap(cv_frame, COMPARE_OUT / "01_validacao_todas_configuracoes.png",
            "Comparação dos três algoritmos: mesmas partições de validação")
    lr_frame = cv_frame[cv_frame["modelo"] == "Regressão Logística"].copy()
    lr_plot = lr_frame.copy()
    lr_plot["nome"] = lr_plot["estrategia"] + " | " + lr_plot["configuracao"]
    heatmap(lr_plot, LR_OUT / "01_configuracoes_validacao.png",
            "Regressão Logística: comparação das configurações no treino")
    test_rows, fitted, probabilities, predictions, chosen = [], {}, {}, {}, {}
    for family in FAMILIES:
        selected = cv_frame[cv_frame["modelo"] == family].sort_values(
            ["cv_f1", "cv_average_precision", "cv_roc_auc"], ascending=False, kind="stable").iloc[0]
        name = selected["nome"]
        chosen[family] = name
        print(f"Treinando configuração selecionada: {name}", flush=True)
        pipeline = candidates[name]
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        prob, pred = pipeline.predict_proba(X_test)[:, 1], pipeline.predict(X_test)
        values = dict(accuracy=accuracy_score(y_test, pred),
            precision=precision_score(y_test, pred, zero_division=0),
            recall=recall_score(y_test, pred, zero_division=0),
            f1=f1_score(y_test, pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, prob),
            average_precision=average_precision_score(y_test, prob))
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        test_rows.append(dict(modelo=family, nome=name, **values, tn=int(tn), fp=int(fp),
                              fn=int(fn), tp=int(tp), tempo_fit_s=fit_time))
        fitted[family], probabilities[family], predictions[family] = pipeline, prob, pred
        joblib.dump(pipeline, MODELS / (SLUGS[family] + ".joblib"))
        print(f"Teste {family}: {values}; TN={tn}; FP={fp}; FN={fn}; TP={tp}", flush=True)
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(TABLES / "metricas_teste.csv", index=False)
    prediction_columns = dict(indice_original=X_test.index, anomalia_real=y_test.to_numpy())
    for family in FAMILIES:
        prediction_columns["prob_" + SLUGS[family]] = probabilities[family]
        prediction_columns["pred_" + SLUGS[family]] = predictions[family]
    pd.DataFrame(prediction_columns).to_csv(TABLES / "predicoes_teste.csv", index=False)
    metadata["selecionados"] = chosen
    (META / "metadados.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions["Random Forest"],
        display_labels=["Legítima", "Anômala"], cmap="Blues", colorbar=False, ax=ax)
    ax.set(title="Random Forest: matriz de confusão no teste", xlabel="Classe prevista", ylabel="Classe real")
    save_plot(fig, RF_OUT / "03_matriz_confusao.png")
    curves(y_test, probabilities, ["Random Forest"], RF_OUT / "04_curvas_roc_precisao_recall.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions["Regressão Logística"],
        display_labels=["Legítima", "Anômala"], cmap="Greens", colorbar=False, ax=ax)
    ax.set(title="Regressão Logística: matriz de confusão no teste", xlabel="Classe prevista", ylabel="Classe real")
    save_plot(fig, LR_OUT / "02_matriz_confusao.png")
    curves(y_test, probabilities, ["Regressão Logística"], LR_OUT / "03_curvas_roc_precisao_recall.png")
    prep = fitted["Random Forest"].named_steps["preprocessor"]
    categorical = list(prep.transformers_[0][2])
    grouped = {}
    for feature, importance in zip(prep.get_feature_names_out(),
            fitted["Random Forest"].named_steps["classifier"].feature_importances_):
        key = (next(key for key in categorical if feature.startswith("categorical__" + key + "_"))
               if feature.startswith("categorical__") else feature.split("__", 1)[1])
        grouped[key] = grouped.get(key, 0) + float(importance)
    feature_frame = pd.Series(grouped, name="importancia").sort_values()
    feature_frame.rename_axis("atributo").to_csv(TABLES / "importancia_atributos_random_forest.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feature_frame.index, feature_frame.values, color=COLORS["Random Forest"])
    ax.set(title="Random Forest: importância dos atributos", xlabel="Importância por redução de impureza")
    save_plot(fig, RF_OUT / "05_importancia_atributos.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [(0, "#2a9d8f"), (1, "#e76f51")]:
        ax.hist(probabilities["Random Forest"][y_test.to_numpy() == label],
                bins=np.linspace(0, 1, 26), alpha=.65, color=color,
                label="Legítimas" if label == 0 else "Anômalas")
    ax.axvline(.5, linestyle="--", color="black", label="Limiar 0,5")
    ax.set_yscale("symlog", linthresh=1)
    ax.set(title="Random Forest: probabilidades no teste", xlabel="Probabilidade prevista de anomalia",
           ylabel="Transações (escala symlog)")
    ax.legend()
    save_plot(fig, RF_OUT / "06_distribuicao_probabilidades.png")
    plot_test_metrics(test_frame)
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(6 * len(FAMILIES), 5))
    for ax, family in zip(axes, FAMILIES):
        ConfusionMatrixDisplay.from_predictions(y_test, predictions[family],
            display_labels=["Legítima", "Anômala"], cmap="Blues", colorbar=False, ax=ax)
        ax.set(title=family, xlabel="Classe prevista", ylabel="Classe real")
    fig.suptitle("Matrizes de confusão no mesmo conjunto de teste")
    save_plot(fig, COMPARE_OUT / "03_matrizes_confusao.png")
    curves(y_test, probabilities, FAMILIES, COMPARE_OUT / "04_curvas_comparativas.png")
    for family in ("XGBoost", "Regressão Logística"):
        folder = PLOTS / SLUGS[family]
        frame = cv_frame[cv_frame["modelo"] == family].copy()
        frame["nome"] = frame["estrategia"] + " | " + frame["configuracao"]
        if family == "XGBoost":
            heatmap(frame, folder / "01_configuracoes_validacao.png", family + ": validação no treino")
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, predictions[family],
                display_labels=["Legítima", "Anômala"], cmap="Blues", colorbar=False, ax=ax)
            ax.set(title=family + ": matriz de confusão", xlabel="Classe prevista", ylabel="Classe real")
            save_plot(fig, folder / "02_matriz_confusao.png")
            curves(y_test, probabilities, [family], folder / "03_curvas_roc_precisao_recall.png")
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, color in [(0, "#2a9d8f"), (1, "#e76f51")]:
            ax.hist(probabilities[family][y_test.to_numpy() == label], bins=np.linspace(0, 1, 26),
                    alpha=.65, color=color, label="Legítimas" if label == 0 else "Anômalas")
        ax.axvline(.5, linestyle="--", color="black", label="Limiar 0,5")
        ax.set_yscale("symlog", linthresh=1)
        ax.set(title=family + ": probabilidades no teste", xlabel="Probabilidade de anomalia", ylabel="Transações (symlog)")
        ax.legend()
        save_plot(fig, folder / "04_distribuicao_probabilidades.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ("fp", "fn"), ("Falsos positivos", "Falsos negativos")):
        bars = ax.bar(test_frame["modelo"], test_frame[metric], color=[COLORS[f] for f in FAMILIES])
        ax.bar_label(bars, padding=4)
        ax.set(title=title, ylabel="Transações", ylim=(0, max(1, test_frame[metric].max()) * 1.25))
    save_plot(fig, COMPARE_OUT / "06_erros_classificacao.png")
    report = (
        "# Resultados: Random Forest, XGBoost e Regressão Logística\n\n"
        f"Base {metadata['dataset']}: {len(df)} registros; {len(X_train)} no treino "
        f"e {len(X_test)} no teste. Rótulos: {metadata['classes_total']}. "
        "Anomalia é o rótulo positivo e não constitui confirmação de fraude.\n\n"
        "## Procedimento\n\n"
        "Divisão estratificada 80/20, semente 42. Avaliação de 24 configurações "
        "(3 algoritmos × 4 estratégias × 2 conjuntos de parâmetros) nos mesmos 5 folds do treino. "
        "Encoder, padronização numérica da regressão logística e reamostragem ajustados somente no treino de cada fold. "
        "Seleção pelo maior F1 médio de validação, com desempate por AP e ROC-AUC. "
        "Parâmetros ajustados manualmente, reproduzindo os valores de main.py. "
        "O XGBoost padrão recebe semente 42. Os modelos de árvores usam até 4 processadores. "
        "Limiar padrão 0,5, sem otimização. Teste reservado usado somente após a seleção.\n\n"
        "## Random Forest: validação no treino\n\n" + metrics_table(rf_frame, "cv_") +
        "\n\n## XGBoost: validação no treino\n\n" + metrics_table(cv_frame[cv_frame["modelo"] == "XGBoost"], "cv_") +
        "\n\n## Regressão Logística: validação no treino\n\n" + metrics_table(lr_frame, "cv_") +
        "\n\n## Configurações selecionadas\n\n" +
        "\n".join(f"- {family}: {name}" for family, name in chosen.items()) +
        "\n\n## Comparação no teste reservado\n\n" + metrics_table(test_frame) +
        "\n\n" + "\n".join(
            f"- {row['modelo']}: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}."
            for row in test_rows) +
        "\n\nAP é average precision, separada de ROC-AUC; não é área trapezoidal "
        "da curva precisão–recall. Médias, desvios e tempos de validação estão no CSV. "
        "Métricas de teste vêm de uma única divisão e não são intervalos de confiança.\n\n"
        "## Leitura da comparação\n\n"
        + f"Nesta divisão, {test_frame.sort_values('f1', ascending=False).iloc[0]['modelo']} "
        "obteve o maior F1 no teste. Essa ordenação descreve este teste; "
        "a configuração de cada algoritmo foi selecionada previamente pela validação. "
        "O conjunto de teste contém apenas " + str(int(y_test.sum())) + " anomalias; "
        "cada acerto ou erro positivo altera o recall em " + f"{100 / int(y_test.sum()):.1f}" + " pontos percentuais. "
        "Não foi realizado teste de significância entre algoritmos.\n\n"
        "## Arquivos\n\n"
        "- graficos/dados/: distribuição das classes.\n"
        "- graficos/random_forest/: validação, matriz, curvas, importância e probabilidades.\n"
        "- graficos/xgboost/ e logistic_regression/: validação, matriz, curvas e probabilidades.\n"
        "- graficos/comparacao/: validação, métricas de teste, matrizes, curvas e erros.\n"
        "- tabelas/metricas_validacao_cruzada.csv: médias, desvios e tempos das 24 configurações.\n"
        "- tabelas/metricas_teste.csv: métricas e contagens das matrizes de confusão.\n"
        "- tabelas/predicoes_teste.csv: probabilidades e previsões com índices originais.\n"
        "- metadados/metadados.json: distribuição dos rótulos, divisão e configurações selecionadas.\n"
        "- modelos/: random_forest.joblib, xgboost.joblib e logistic_regression.joblib: pipelines ajustados no treino.\n\n"
        "## Limites de interpretação\n\n"
        "Os resultados descrevem esta base e divisão, sem comprovar generalização para "
        "transações em produção. O uso de Status pressupõe disponibilidade no momento da previsão. "
        "Não foi auditada a origem ou independência temporal dos registros. "
        "SMOTE após one-hot pode gerar atributos categóricos fracionários. "
        "Importância soma categorias por atributo e usa redução de impureza; "
        "não representa causalidade e pode favorecer atributos com mais possibilidades de divisão.\n\n"
        "Para reproduzir, execute na raiz: .\\.venv\\Scripts\\python.exe evaluate_models.py\n"
    )
    (REPORTS / "RELATORIO.md").write_text(report, encoding="utf-8")
    print("\nCONCLUÍDO\n" + test_frame.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()


