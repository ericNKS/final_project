# %%
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score,
    roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.data.ingestion import (
    loadFraud2mData, getTuningSample, printTestResult, saveModelArtifact
)
from src.data.cleaning import prepare2mDataset
from src.models.xgboost import _getPipeline

os.makedirs("src/notebooks/graficos", exist_ok=True)
os.makedirs("resultados/modelos", exist_ok=True)

# %%
"""
================================================================================
ETAPA 1: CARREGAMENTO E DIVISÃO DO DATASET DE 2 MILHÕES
================================================================================
Utilizando exclusivamente o dataset de 2M transações reais de Pix:
- Treino: 80% (1.600.000 transações) para ajuste dos modelos finais
- Teste Independente: 20% (400.000 transações) para verificação do desempenho real
- Amostra de Tunagem: 80.000 transações estratificadas para testes rápidos de hiperparâmetros
"""
print("="*80)
print("ETAPA 1: CARREGAMENTO E DIVISÃO DO DATASET DE 2M PIX")
print("="*80)

t0 = time.time()
df_raw = loadFraud2mData()
print(f"Dataset de 2M carregado em {time.time()-t0:.2f}s | Linhas: {len(df_raw):,}")

X, y = prepare2mDataset(df_raw)
print(f"Dataset preparado | X shape: {X.shape} | Fraudes totais: {y.sum():,} ({y.mean()*100:.2f}%)")

XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"XTrain (80% treino): {XTrain.shape} | Fraudes: {yTrain.sum():,} ({yTrain.mean()*100:.2f}%)")
print(f"XTest  (20% teste):  {XTest.shape}  | Fraudes: {yTest.sum():,}  ({yTest.mean()*100:.2f}%)")

# Extração da amostra representativa estratificada para testes de hiperparâmetros
XSample, ySample = getTuningSample(XTrain, yTrain, sample_size=80000, random_state=42)
XSubTrain, XSubVal, ySubTrain, ySubVal = train_test_split(
    XSample, ySample, test_size=0.25, stratify=ySample, random_state=42
)
print(f"Amostra de Tunagem: {XSample.shape} (SubTreino: {XSubTrain.shape}, SubVal: {XSubVal.shape})")

# %%
"""
================================================================================
ETAPA 2: EXPERIMENTOS DE TUNAGEM DE HIPERPARÂMETROS EM TODOS OS MODELOS
================================================================================
Testando sistematicamente:
- Modelos: XGBoost, Random Forest, Regressão Logística
- Balanceamentos: Baseline, Under-sampling (RandomUnderSampler), SMOTE
- Tunagens de Hiperparâmetros: Variações baseadas em pix.py e df2m.py
- Tunagem de Limiar de Decisão: Limiar Padrão (0.5) vs Limiar Otimizado (F1)
"""
print("\n" + "="*80)
print("ETAPA 2: EXPERIMENTOS DE TUNAGEM DE HIPERPARÂMETROS (AMOSTRA DE 80K)")
print("="*80)

experiments = []

# --- 1. XGBoost ---
# Baseline
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Baseline", "tunagem": "Config A (depth=6, n=40, spw=2.5)",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=6, n_estimators=40, scale_pos_weight=2.5, learning_rate=0.1, tree_method="hist", random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Baseline", "tunagem": "Config B (depth=8, n=50, spw=3.5)",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=8, n_estimators=50, scale_pos_weight=3.5, learning_rate=0.1, tree_method="hist", random_state=42, n_jobs=-1)
})
# Under-sampling
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Under-sampling", "tunagem": "Config A (depth=3, n=100, col=0.8)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=3, n_estimators=100, colsample_bytree=0.8, scale_pos_weight=1.0, learning_rate=0.1, tree_method="hist", random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Under-sampling", "tunagem": "Config B (depth=4, n=150, col=0.8)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=4, n_estimators=150, colsample_bytree=0.8, scale_pos_weight=1.0, learning_rate=0.1, tree_method="hist", random_state=42, n_jobs=-1)
})
# SMOTE
experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTE", "tunagem": "Config A (depth=6, n=60, child=2)",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=6, n_estimators=60, min_child_weight=2, subsample=0.8, tree_method="hist", random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTE", "tunagem": "Config B (depth=8, n=80, child=4)",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=8, n_estimators=80, min_child_weight=4, subsample=0.8, tree_method="hist", random_state=42, n_jobs=-1)
})

# --- 2. Random Forest ---
# Baseline
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Baseline", "tunagem": "Config A (n=50, depth=10, balanced)",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Baseline", "tunagem": "Config B (n=60, depth=14, balanced)",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=60, max_depth=14, class_weight="balanced", random_state=42, n_jobs=-1)
})
# Under-sampling
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Under-sampling", "tunagem": "Config A (n=50, depth=10, leaf=2)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Under-sampling", "tunagem": "Config B (n=80, depth=12, leaf=1)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=1, random_state=42, n_jobs=-1)
})
# SMOTE
experiments.append({
    "modelo": "Random Forest", "balanceamento": "SMOTE", "tunagem": "Config A (n=40, depth=10, leaf=2)",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "SMOTE", "tunagem": "Config B (n=60, depth=12, leaf=1)",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=1, random_state=42, n_jobs=-1)
})

# --- 3. Regressão Logística ---
# Baseline
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Baseline", "tunagem": "Config A (C=0.1, balanced)",
    "sampler": None, "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Baseline", "tunagem": "Config B (C=1.0, balanced)",
    "sampler": None, "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
})
# Under-sampling
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Under-sampling", "tunagem": "Config A (C=0.1, standard)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Under-sampling", "tunagem": "Config B (C=1.0, standard)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
})
# SMOTE
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "SMOTE", "tunagem": "Config A (C=0.1, standard)",
    "sampler": SMOTE(random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "SMOTE", "tunagem": "Config B (C=1.0, standard)",
    "sampler": SMOTE(random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
})

results_list = []

for idx, exp in enumerate(experiments):
    m_name = exp["modelo"]
    bal = exp["balanceamento"]
    tun = exp["tunagem"]
    print(f"[{idx+1:02d}/{len(experiments):02d}] Testando {m_name} | {bal} | {tun}...")

    pipe = _getPipeline(
        sampler=exp["sampler"],
        classifier=exp["clf"],
        numeric_transformer=exp["numeric_trans"]
    )
    t_start = time.time()
    pipe.fit(XSubTrain, ySubTrain)
    fit_duration = time.time() - t_start

    # Avaliação com Limiar Padrão (0.5)
    y_prob = pipe.predict_proba(XSubVal)[:, 1]
    y_pred_def = (y_prob >= 0.5).astype(int)

    f1_def = f1_score(ySubVal, y_pred_def, zero_division=0)
    prec_def = precision_score(ySubVal, y_pred_def, zero_division=0)
    rec_def = recall_score(ySubVal, y_pred_def, zero_division=0)

    # Avaliação com Limiar Otimizado (TunedThresholdClassifierCV)
    tuned_cv = TunedThresholdClassifierCV(pipe, scoring="f1", cv="prefit", refit=False)
    tuned_cv.fit(XSubVal, ySubVal)
    thresh_opt = tuned_cv.best_threshold_

    y_pred_tun = tuned_cv.predict(XSubVal)
    f1_tun = f1_score(ySubVal, y_pred_tun, zero_division=0)
    prec_tun = precision_score(ySubVal, y_pred_tun, zero_division=0)
    rec_tun = recall_score(ySubVal, y_pred_tun, zero_division=0)

    pr_auc = average_precision_score(ySubVal, y_prob)
    roc_auc = roc_auc_score(ySubVal, y_prob)

    results_list.append({
        "Modelo": m_name,
        "Balanceamento": bal,
        "Tunagem": tun,
        "Fit_Tempo_s": fit_duration,
        "F1_Padrao": f1_def,
        "Precision_Padrao": prec_def,
        "Recall_Padrao": rec_def,
        "Limiar_Tunado": thresh_opt,
        "F1_Tunado": f1_tun,
        "Precision_Tunado": prec_tun,
        "Recall_Tunado": rec_tun,
        "Ganho_F1": (f1_tun - f1_def) * 100,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    })
    print(f"      -> F1 Padrão: {f1_def*100:.1f}% | Limiar Otimizado: {thresh_opt:.4f} -> F1 Tunado: {f1_tun*100:.1f}% (Ganho: +{(f1_tun-f1_def)*100:.1f}%) | PR-AUC: {pr_auc*100:.1f}%")

df_tuning = pd.DataFrame(results_list)

# %%
"""
================================================================================
ETAPA 3: GERAÇÃO DE GRÁFICOS AGRUPADOS (MODELO, TUNAGEM, BALANCEAMENTO)
================================================================================
"""
print("\n" + "="*80)
print("ETAPA 3: GERANDO E SALVANDO GRÁFICOS AGRUPADOS DE TUNAGEM")
print("="*80)

models_order = ["XGBoost", "Random Forest", "Regressão Logística"]
palette = {"XGBoost": "#1f77b4", "Random Forest": "#2ca02c", "Regressão Logística": "#ff7f0e"}

# --- Gráfico 1: Agrupamento por Modelos ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=df_tuning, x="Modelo", y="PR_AUC", order=models_order, hue="Modelo", legend=False, palette=palette, ax=axes[0])
sns.stripplot(data=df_tuning, x="Modelo", y="PR_AUC", order=models_order, color="black", alpha=0.6, jitter=0.2, ax=axes[0])
axes[0].set_title("PR-AUC por Modelo\n(Distribuição nas Tunagens)", fontweight="bold")
axes[0].set_ylabel("PR-AUC")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

sns.boxplot(data=df_tuning, x="Modelo", y="F1_Tunado", order=models_order, hue="Modelo", legend=False, palette=palette, ax=axes[1])
sns.stripplot(data=df_tuning, x="Modelo", y="F1_Tunado", order=models_order, color="black", alpha=0.6, jitter=0.2, ax=axes[1])
axes[1].set_title("F1-Score Tunado por Modelo\n(Limiar Calibrado)", fontweight="bold")
axes[1].set_ylabel("F1-Score")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

sns.boxplot(data=df_tuning, x="Modelo", y="ROC_AUC", order=models_order, hue="Modelo", legend=False, palette=palette, ax=axes[2])
sns.stripplot(data=df_tuning, x="Modelo", y="ROC_AUC", order=models_order, color="black", alpha=0.6, jitter=0.2, ax=axes[2])
axes[2].set_title("ROC-AUC por Modelo", fontweight="bold")
axes[2].set_ylabel("ROC-AUC")
axes[2].grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("AGRUPAMENTO POR MODELOS: Comparação Geral de Desempenho", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/01_comparacao_por_modelo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/01_comparacao_por_modelo.png")

# --- Gráfico 2: Agrupamento por Tunagem (Impacto da Calibração do Limiar) ---
fig, ax = plt.subplots(figsize=(16, 7))
df_melted = df_tuning.melt(
    id_vars=["Modelo", "Balanceamento", "Tunagem"],
    value_vars=["F1_Padrao", "F1_Tunado"],
    var_name="Tipo_Limiar", value_name="F1_Score"
)
df_melted["Tipo_Limiar"] = df_melted["Tipo_Limiar"].map({"F1_Padrao": "Limiar Padrão (0.50)", "F1_Tunado": "Limiar Tunado (Otimizado)"})
df_melted["Experimento_Label"] = df_melted["Modelo"] + " - " + df_melted["Balanceamento"] + " (" + df_melted["Tunagem"].apply(lambda s: s[:8]) + ")"

sns.barplot(
    data=df_melted, x="Experimento_Label", y="F1_Score", hue="Tipo_Limiar",
    palette={"Limiar Padrão (0.50)": "#94a3b8", "Limiar Tunado (Otimizado)": "#10b981"},
    ax=ax
)
ax.set_title("AGRUPAMENTO POR TUNAGEM: Impacto da Calibração do Limiar no F1-Score", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("F1-Score", fontweight="bold")
ax.set_xlabel("Configuração de Hiperparâmetros", fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(title="Estratégia de Limiar", loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig("src/notebooks/graficos/02_impacto_por_tunagem.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/02_impacto_por_tunagem.png")

# --- Gráfico 3: Agrupamento por Tunagem e Balanceamento ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for i, m_name in enumerate(models_order):
    sub_df = df_tuning[df_tuning["Modelo"] == m_name]
    sns.barplot(
        data=sub_df, x="Balanceamento", y="F1_Tunado", hue="Tunagem",
        palette="viridis", ax=axes[i]
    )
    axes[i].set_title(f"{m_name}\nF1 Tunado por Balanceamento", fontweight="bold", fontsize=12)
    axes[i].set_ylabel("F1-Score" if i == 0 else "")
    axes[i].set_xlabel("Técnica de Balanceamento", fontweight="bold")
    axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    axes[i].legend(title="Configuração", fontsize=8, title_fontsize=9, loc="upper right")

plt.suptitle("AGRUPAMENTO POR TUNAGEM E BALANCEAMENTO: Cruzamento entre Reamostragem e Hiperparâmetros", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/03_tunagem_por_balanceamento.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/03_tunagem_por_balanceamento.png")

# %%
"""
================================================================================
ETAPA 4: TREINAMENTO DOS MELHORES MODELOS EM XTRAIN (1.6M)
================================================================================
"""
print("\n" + "="*80)
print("ETAPA 4: TREINAMENTO DOS MELHORES MODELOS EM 1.6M E PREDIÇÃO NO TESTE DE 400K")
print("="*80)

best_configs = {}
for m_name in models_order:
    sub = df_tuning[df_tuning["Modelo"] == m_name]
    best_row_idx = (sub["PR_AUC"] * 0.5 + sub["F1_Tunado"] * 0.5).idxmax()
    best_configs[m_name] = experiments[best_row_idx]
    print(f"\nMelhor configuração selecionada para {m_name}:")
    print(f"  Balanceamento: {best_configs[m_name]['balanceamento']}")
    print(f"  Tunagem:       {best_configs[m_name]['tunagem']}")

final_model_names = []
final_test_results = []
final_pipelines = []
final_y_probs = {}
final_y_preds = {}

for m_name in models_order:
    cfg = best_configs[m_name]
    display_name = f"{m_name} ({cfg['balanceamento']})"
    print(f"\nTreinando modelo final: {display_name} no conjunto de 1.6M...")

    pipe_final = _getPipeline(
        sampler=cfg["sampler"],
        classifier=cfg["clf"],
        numeric_transformer=cfg["numeric_trans"]
    )
    t_start = time.time()
    pipe_final.fit(XTrain, yTrain)
    print(f"  -> Treinamento em 1.6M concluído em {time.time()-t_start:.2f}s")

    print("  -> Calibrando limiar ótimo de decisão na partição de validação...")
    tuned_final = TunedThresholdClassifierCV(pipe_final, scoring="f1", cv="prefit", refit=False)
    tuned_final.fit(XSubVal, ySubVal)
    print(f"  -> Limiar ótimo calibrado: {tuned_final.best_threshold_:.4f}")

    t_start = time.time()
    y_pred = tuned_final.predict(XTest)
    y_prob = tuned_final.predict_proba(XTest)[:, 1]
    print(f"  -> Inferência em 400k concluída em {time.time()-t_start:.2f}s")

    acc = accuracy_score(yTest, y_pred)
    prec = precision_score(yTest, y_pred, zero_division=0)
    rec = recall_score(yTest, y_pred, zero_division=0)
    f1 = f1_score(yTest, y_pred, zero_division=0)
    roc = roc_auc_score(yTest, y_prob)
    pr = average_precision_score(yTest, y_prob)
    cm = confusion_matrix(yTest, y_pred)

    final_model_names.append(display_name)
    final_test_results.append({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "limiar": tuned_final.best_threshold_,
        "confusion_matrix": cm
    })
    final_pipelines.append(tuned_final)
    final_y_probs[display_name] = y_prob
    final_y_preds[display_name] = y_pred

# %%
"""
================================================================================
ETAPA 5: TESTE DE DESEMPENHO APROFUNDADO NO CONJUNTO DE TESTE INDEPENDENTE (400K)
E GERAÇÃO DE NOVOS GRÁFICOS DE DESEMPENHO
================================================================================
Avaliação completa no conjunto que NÃO participou do treinamento:
1. Tabela Estatística de Métricas no Teste
2. Avaliação de Impacto Financeiro (Volume de Fraudes Salvo em R$)
3. Novos Gráficos:
   - Matrizes de Confusão no Teste (04)
   - Curvas ROC Comparativas no Teste (05)
   - Curvas Precision-Recall Comparativas no Teste (06)
   - Curva de Trade-off Limiar vs Métricas no Teste (07)
   - Distribuição de Densidade das Probabilidades Preditas no Teste (08)
   - Impacto Financeiro / Fraudes Interceptadas em R$ no Teste (09)
"""
print("\n" + "="*80)
print("ETAPA 5: TESTE DE DESEMPENHO APROFUNDADO NO CONJUNTO DE TESTE (400.000 TRANSAÇÕES)")
print("="*80)

# 1. Tabela de Métricas
printTestResult(final_model_names, *final_test_results)

# 2. Avaliação de Impacto Financeiro
print("\n" + "="*80)
print("AVALIAÇÃO DE IMPACTO FINANCEIRO NO CONJUNTO DE TESTE (400.000 TRANSAÇÕES)")
print("="*80)

frauds_mask = (yTest == 1)
total_fraud_val = float(XTest.loc[frauds_mask, "valor_brl"].sum())
total_fraud_count = int(frauds_mask.sum())
print(f"Volume Total de Fraudes no Teste: R$ {total_fraud_val:,.2f} ({total_fraud_count:,} ocorrências)")

financial_metrics_list = []
for name in final_model_names:
    y_p = final_y_preds[name]
    val_intercepted = float(XTest.loc[frauds_mask & (y_p == 1), "valor_brl"].sum())
    count_intercepted = int((frauds_mask & (y_p == 1)).sum())
    val_lost = float(XTest.loc[frauds_mask & (y_p == 0), "valor_brl"].sum())
    count_lost = int((frauds_mask & (y_p == 0)).sum())
    val_fp = float(XTest.loc[(~frauds_mask) & (y_p == 1), "valor_brl"].sum())
    count_fp = int(((~frauds_mask) & (y_p == 1)).sum())
    pct_saved = (val_intercepted / total_fraud_val) * 100 if total_fraud_val > 0 else 0

    financial_metrics_list.append({
        "modelo": name,
        "fraudes_interceptadas_brl": val_intercepted,
        "fraudes_interceptadas_qtd": count_intercepted,
        "fraudes_nao_detectadas_brl": val_lost,
        "fraudes_nao_detectadas_qtd": count_lost,
        "falsos_positivos_brl": val_fp,
        "falsos_positivos_qtd": count_fp,
        "pct_financeiro_salvo": pct_saved
    })

    print(f"\n[{name}]")
    print(f"  -> Fraudes Bloqueadas com Sucesso: R$ {val_intercepted:,.2f} ({pct_saved:.2f}% do volume total | {count_intercepted}/{total_fraud_count} fraudes)")
    print(f"  -> Fraudes Não Detectadas (Perda Residual): R$ {val_lost:,.2f} ({100-pct_saved:.2f}% | {count_lost} fraudes)")
    print(f"  -> Volume Financeiro de Falsos Alarmes: R$ {val_fp:,.2f} ({count_fp:,} transações legítimas)")

# --- Gráfico 4: Matrizes de Confusão no Teste de 2M ---
fig, axes = plt.subplots(1, len(final_model_names), figsize=(5.5 * len(final_model_names), 4.5))
if len(final_model_names) == 1:
    axes = [axes]

for idx, (name, res) in enumerate(zip(final_model_names, final_test_results)):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=res["confusion_matrix"],
        display_labels=["Legítima", "Fraude"]
    )
    cm_display.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"{name}\nPR-AUC: {res['pr_auc']*100:.1f}% | F1: {res['f1']*100:.1f}%\nLimiar: {res['limiar']:.4f}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Classe Prevista", fontweight="bold")
    axes[idx].set_ylabel("Classe Real" if idx == 0 else "", fontweight="bold")

plt.suptitle("TESTE NO DATASET INDEPENDENTE (400k): Matrizes de Confusão", fontsize=14, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/04_matrizes_confusao_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSalvo: src/notebooks/graficos/04_matrizes_confusao_teste_2m.png")

# --- Gráfico 5: Curvas ROC no Teste Independente ---
plt.figure(figsize=(9, 7))
colors_map = [palette.get(m.split()[0], "#333333") for m in final_model_names]

for name, color in zip(final_model_names, colors_map):
    fpr, tpr, _ = roc_curve(yTest, final_y_probs[name])
    roc_val = roc_auc_score(yTest, final_y_probs[name])
    plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_val*100:.2f}%)", color=color, linewidth=2.5)

plt.plot([0, 1], [0, 1], "k--", label="Classificador Aleatório (AUC = 50.0%)", alpha=0.6)
plt.title("TESTE NO DATASET INDEPENDENTE (400k): Curvas ROC", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Taxa de Falsos Positivos (FPR)", fontweight="bold")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)", fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower right", fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/05_curvas_roc_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/05_curvas_roc_teste_2m.png")

# --- Gráfico 6: Curvas Precision-Recall no Teste Independente ---
plt.figure(figsize=(9, 7))
baseline_pr = float(yTest.mean())

for name, color in zip(final_model_names, colors_map):
    prec_vals, rec_vals, _ = precision_recall_curve(yTest, final_y_probs[name])
    pr_val = average_precision_score(yTest, final_y_probs[name])
    plt.plot(rec_vals, prec_vals, label=f"{name} (PR-AUC = {pr_val*100:.2f}%)", color=color, linewidth=2.5)

    # Ponto de operação com limiar calibrado
    idx_model = final_model_names.index(name)
    p_op = final_test_results[idx_model]["precision"]
    r_op = final_test_results[idx_model]["recall"]
    plt.scatter(r_op, p_op, color=color, s=90, zorder=5, edgecolors="black", label=f"Ponto de Operação ({name})")

plt.axhline(y=baseline_pr, color="r", linestyle="--", alpha=0.7, label=f"Baseline Aleatório (PR = {baseline_pr*100:.2f}%)")
plt.title("TESTE NO DATASET INDEPENDENTE (400k): Curvas Precision-Recall", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Recall (Taxa de Detecção de Fraude)", fontweight="bold")
plt.ylabel("Precision (Taxa de Acerto nas Suspeitas)", fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/06_curvas_precision_recall_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/06_curvas_precision_recall_teste_2m.png")

# --- Gráfico 7: Trade-off Limiar vs Métricas no Teste (Melhor Modelo: XGBoost) ---
best_name = final_model_names[0]
best_probs = final_y_probs[best_name]
best_limiar = final_test_results[0]["limiar"]

thresholds_range = np.linspace(0.01, 0.99, 150)
f1_curve, prec_curve, rec_curve = [], [], []

for t_val in thresholds_range:
    y_temp = (best_probs >= t_val).astype(int)
    f1_curve.append(f1_score(yTest, y_temp, zero_division=0))
    prec_curve.append(precision_score(yTest, y_temp, zero_division=0))
    rec_curve.append(recall_score(yTest, y_temp, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(thresholds_range, f1_curve, label="F1-Score", color="#2563eb", linewidth=2.5)
plt.plot(thresholds_range, prec_curve, label="Precision", color="#16a34a", linewidth=2.0)
plt.plot(thresholds_range, rec_curve, label="Recall", color="#dc2626", linewidth=2.0)
plt.axvline(x=best_limiar, color="purple", linestyle="--", linewidth=2, label=f"Limiar Otimizado ({best_limiar:.4f})")
plt.scatter(best_limiar, final_test_results[0]["f1"], color="purple", s=100, zorder=5, label=f"F1 Máximo = {final_test_results[0]['f1']*100:.1f}%")

plt.title(f"TESTE NO DATASET INDEPENDENTE (400k): Trade-off Limiar vs Métricas ({best_name})", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Limiar de Decisão de Fraude (Probability Threshold)", fontweight="bold")
plt.ylabel("Pontuação da Métrica", fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="center left", frameon=True)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/07_tradeoff_limiar_metricas_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/07_tradeoff_limiar_metricas_teste_2m.png")

# --- Gráfico 8: Distribuição de Densidade das Probabilidades Preditas no Teste ---
plt.figure(figsize=(10, 6))
sns.kdeplot(best_probs[yTest == 0], label="Transações Legítimas (Classe 0)", color="#16a34a", fill=True, alpha=0.4, linewidth=2)
sns.kdeplot(best_probs[yTest == 1], label="Transações Fraudulentas (Classe 1)", color="#dc2626", fill=True, alpha=0.4, linewidth=2)
plt.axvline(x=best_limiar, color="purple", linestyle="--", linewidth=2, label=f"Limiar de Corte ({best_limiar:.4f})")

plt.title(f"TESTE NO DATASET INDEPENDENTE (400k): Separação de Densidades das Probabilidades ({best_name})", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Probabilidade Predita de Fraude", fontweight="bold")
plt.ylabel("Densidade de Frequência", fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper center", frameon=True)
plt.tight_layout()
plt.savefig("src/notebooks/graficos/08_distribuicao_probabilidades_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/08_distribuicao_probabilidades_teste_2m.png")

# --- Gráfico 9: Impacto Financeiro no Conjunto de Teste ---
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
indices = np.arange(len(final_model_names))

val_salvo = [f["fraudes_interceptadas_brl"] / 1e9 for f in financial_metrics_list]
val_perdido = [f["fraudes_nao_detectadas_brl"] / 1e9 for f in financial_metrics_list]

b1 = ax.bar(indices - bar_width/2, val_salvo, bar_width, label="Fraudes Bloqueadas com Sucesso (R$ Bilhões)", color="#16a34a")
b2 = ax.bar(indices + bar_width/2, val_perdido, bar_width, label="Perda Residual Não Detectada (R$ Bilhões)", color="#dc2626")

ax.set_title("TESTE NO DATASET INDEPENDENTE (400k): Volume Financeiro de Fraude Interceptado", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Volume Financeiro (R$ em Bilhões)", fontweight="bold")
ax.set_xticks(indices)
ax.set_xticklabels(final_model_names, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(frameon=True)

for bar, f_data in zip(b1, financial_metrics_list):
    ax.annotate(f"R$ {bar.get_height():.2f}B\n({f_data['pct_financeiro_salvo']:.1f}%)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)

for bar in b2:
    ax.annotate(f"R$ {bar.get_height():.2f}B",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1),
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig("src/notebooks/graficos/09_impacto_financeiro_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: src/notebooks/graficos/09_impacto_financeiro_teste_2m.png")

# %%
"""
================================================================================
ETAPA 6: SELEÇÃO E PERSISTÊNCIA DO MELHOR MODELO FINAL
================================================================================
"""
best_idx = int(np.argmax([r["pr_auc"] for r in final_test_results]))
best_model_name = final_model_names[best_idx]
best_pipeline = final_pipelines[best_idx]
best_metrics = final_test_results[best_idx]
best_financial = financial_metrics_list[best_idx]

print("\n" + "="*80)
print(f"MELHOR MODELO SELECIONADO: {best_model_name}")
print(f"PR-AUC: {best_metrics['pr_auc']*100:.2f}% | F1: {best_metrics['f1']*100:.2f}% | ROC-AUC: {best_metrics['roc_auc']*100:.2f}% | Recall: {best_metrics['recall']*100:.2f}% | Precision: {best_metrics['precision']*100:.2f}%")
print(f"Impacto Financeiro: R$ {best_financial['fraudes_interceptadas_brl']:,.2f} interceptados ({best_financial['pct_financeiro_salvo']:.2f}% do total de fraudes)")
print("="*80)

model_metadata = {
    "modelo": best_model_name,
    "limiar_otimizado": float(best_metrics["limiar"]),
    "metricas_teste_400k": {
        "accuracy": float(best_metrics["accuracy"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1": float(best_metrics["f1"]),
        "roc_auc": float(best_metrics["roc_auc"]),
        "pr_auc": float(best_metrics["pr_auc"])
    },
    "impacto_financeiro_teste_400k": {
        "volume_total_fraudes_brl": total_fraud_val,
        "volume_fraudes_interceptadas_brl": best_financial["fraudes_interceptadas_brl"],
        "volume_fraudes_nao_detectadas_brl": best_financial["fraudes_nao_detectadas_brl"],
        "volume_falsos_positivos_brl": best_financial["falsos_positivos_brl"],
        "pct_financeiro_salvo": best_financial["pct_financeiro_salvo"]
    },
    "particoes": {
        "treino_80_2m": int(len(yTrain)),
        "teste_20_2m": int(len(yTest)),
        "amostra_tunagem": int(len(ySample))
    },
    "features_utilizadas": list(X.columns),
    "data_treinamento": pd.Timestamp.now().isoformat()
}

saveModelArtifact(best_pipeline, model_metadata, filepath="resultados/modelos/melhor_modelo_pix.joblib")
