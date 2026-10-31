# %%
import os
import argparse
parser = argparse.ArgumentParser(description="Treinamento e gráficos Pix")
parser.add_argument("--reuse-10k-results", action="store_true", help="Recriar gráficos de 10k usando previsões salvas")
args = parser.parse_args()
training_backend = os.environ.get("TRAINING_BACKEND", "gpu")
if training_backend not in {"gpu", "cpu"}:
    raise ValueError("TRAINING_BACKEND deve ser gpu ou cpu")
if training_backend == "gpu":
    # Instalar antes de sklearn/imblearn também acelera os vizinhos de SMOTETomek.
    try:
        from cuml.accel import install
    except ImportError as exc:
        raise RuntimeError(
            "O modo GPU requer cuML no WSL2. No PowerShell, execute ./run_gpu.ps1. "
            "Para CPU explicitamente, defina TRAINING_BACKEND=cpu."
        ) from exc
    install(log_level="info")

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
from imblearn.combine import SMOTETomek
import xgboost as xgb

from src.data.ingestion import (
    loadFraud2mData, getTuningSample, printTestResult, saveModelArtifact
)
from src.data.cleaning import prepare2mDataset
from src.models.xgboost import _getPipeline
from src.utils.logger import setup_logging, logger
from src.utils.artifacts import savefig

os.makedirs("logs", exist_ok=True)
os.makedirs("src/notebooks/graficos", exist_ok=True)
os.makedirs("resultados/modelos", exist_ok=True)

setup_logging()

# O modo CPU precisa ser solicitado explicitamente com TRAINING_BACKEND=cpu.
xgb_device = "cuda" if training_backend == "gpu" else "cpu"
logger.info("Backend de treinamento: %s", training_backend)
logger.info("Dispositivo solicitado para XGBoost: %s", xgb_device)
if xgb_device.startswith("cuda"):
    # Confirma CUDA antes de carregar o dataset; evita fallback silencioso para CPU.
    gpu_probe = xgb.XGBClassifier(
        tree_method="hist", device=xgb_device, n_estimators=1, max_depth=1
    )
    gpu_probe.fit(np.array([[0.], [1.], [2.], [3.]]), np.array([0, 0, 1, 1]))
    active_device = json.loads(gpu_probe.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not active_device.startswith("cuda"):
        raise RuntimeError("CUDA indisponível no XGBoost. Verifique a instalação e o driver NVIDIA.")
    logger.info("CUDA confirmado pelo XGBoost: %s", active_device)
    del gpu_probe

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

logger.info("=" * 80)
logger.info("ETAPA 1: CARREGAMENTO E DIVISÃO DO DATASET DE 2M PIX")
logger.info("=" * 80)

t0 = time.time()
df_raw = loadFraud2mData()
logger.info(
    f"Dataset de 2M carregado em {time.time()-t0:.2f}s "
    f"| Linhas: {len(df_raw):,}"
)

X, y = prepare2mDataset(df_raw)

logger.info(
    f"Dataset preparado | X shape: {X.shape} "
    f"| Fraudes totais: {y.sum():,} ({y.mean()*100:.2f}%)"
)

# 1. Separação inicial: 80% desenvolvimento e 20% teste
XTrain, XTest, yTrain, yTest = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 2. Separação dos dados de desenvolvimento:
# 70% do total para treinamento
# 10% do total para validação independente
XTrainFinal, XVal, yTrainFinal, yVal = train_test_split(
    XTrain,
    yTrain,
    test_size=0.125,
    stratify=yTrain,
    random_state=42
)

logger.info(
    f"Treinamento (70%): {XTrainFinal.shape} "
    f"| Fraudes: {yTrainFinal.sum():,} "
    f"({yTrainFinal.mean()*100:.2f}%)"
)

logger.info(
    f"Validação (10%): {XVal.shape} "
    f"| Fraudes: {yVal.sum():,} "
    f"({yVal.mean()*100:.2f}%)"
)

logger.info(
    f"Teste (20%): {XTest.shape} "
    f"| Fraudes: {yTest.sum():,} "
    f"({yTest.mean()*100:.2f}%)"
)

# 3. Extração de 80 mil registros exclusivamente do treinamento
XSample, ySample = getTuningSample(
    XTrainFinal,
    yTrainFinal,
    sample_size=80000,
    random_state=42
)

# 4. Divisão da amostra para comparar configurações
XSubTrain, XSubVal, ySubTrain, ySubVal = train_test_split(
    XSample,
    ySample,
    test_size=0.25,
    stratify=ySample,
    random_state=42
)

logger.info(
    f"Amostra de Tunagem: {XSample.shape} "
    f"(SubTreino: {XSubTrain.shape}, "
    f"SubVal: {XSubVal.shape})"
)






# %%
"""
================================================================================
ETAPA 2: EXPERIMENTOS DE TUNAGEM DE HIPERPARÂMETROS EM TODOS OS MODELOS
================================================================================
Testando sistematicamente:
- Modelos: XGBoost, Random Forest, Regressão Logística
- Balanceamentos: Sem Reamostragem (Base Original), Under-sampling, SMOTE e SMOTETomek
- Configurações: Sem Tunagem (Padrão) vs Com Tunagem (Hiperparâmetros Ajustados)
- Tunagem de Limiar de Decisão: Limiar Padrão (0.50) vs Limiar Otimizado (F1)
"""
logger.info("=" * 80)
logger.info("ETAPA 2: EXPERIMENTOS DE TUNAGEM DE HIPERPARÂMETROS (AMOSTRA DE 80K)")
logger.info("=" * 80)

experiments = []

# --- 1. XGBoost ---
# 1.1 Sem Reamostragem
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Sem Reamostragem", "tunagem": "Sem Tunagem",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(random_state=42, tree_method="hist", n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Sem Reamostragem", "tunagem": "Com Tunagem",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=40, min_child_weight=1, subsample=1, colsample_bytree=1, colsample_bynode=0.9, num_parallel_tree=1, scale_pos_weight=3, tree_method="hist", random_state=42, n_jobs=-1)
})
# 1.2 Under-sampling
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Under-sampling", "tunagem": "Sem Tunagem",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(random_state=42, tree_method="hist", n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Under-sampling", "tunagem": "Com Tunagem",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=1, learning_rate=0.1, n_estimators=500, colsample_bytree=0.8, scale_pos_weight=1, tree_method="hist", random_state=42, n_jobs=-1)
})
# 1.3 SMOTE
experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTE", "tunagem": "Sem Tunagem",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(random_state=42, tree_method="hist", n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTE", "tunagem": "Com Tunagem",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=20, learning_rate=0.2, n_estimators=1000, random_state=0, min_child_weight=1, subsample=1, colsample_bytree=0.8, colsample_bynode=0.7, num_parallel_tree=1, scale_pos_weight=1, tree_method="hist", n_jobs=-1 )
})
# 1.4 SMOTETomek
experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTETomek", "tunagem": "Sem Tunagem",
    "sampler": SMOTETomek(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(random_state=42, tree_method="hist", n_jobs=-1)
})
experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTETomek", "tunagem": "Com Tunagem",
    "sampler": SMOTETomek(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(max_depth=14, learning_rate=0.1, n_estimators=40, colsample_bytree=0.5, scale_pos_weight=2.8, random_state=42, tree_method="hist", n_jobs=-1)
})

# --- 2. Random Forest ---
# 2.1 Sem Reamostragem
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Sem Reamostragem", "tunagem": "Sem Tunagem",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Sem Reamostragem", "tunagem": "Com Tunagem",
    "sampler": None, "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
})
# 2.2 Under-sampling
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Under-sampling", "tunagem": "Sem Tunagem",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "Under-sampling", "tunagem": "Com Tunagem",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
})
# 2.3 SMOTE
experiments.append({
    "modelo": "Random Forest", "balanceamento": "SMOTE", "tunagem": "Sem Tunagem",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "SMOTE", "tunagem": "Com Tunagem",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
})
# 2.4 SMOTETomek
experiments.append({
    "modelo": "Random Forest", "balanceamento": "SMOTETomek", "tunagem": "Sem Tunagem",
    "sampler": SMOTETomek(random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(random_state=42, n_jobs=-1)
})
experiments.append({
    "modelo": "Random Forest", "balanceamento": "SMOTETomek", "tunagem": "Com Tunagem",
    "sampler": SMOTETomek(random_state=42), "numeric_trans": "passthrough",
    "clf": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=1, random_state=42, n_jobs=-1)
})

# --- 3. Regressão Logística ---
# 3.1 Sem Reamostragem
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Sem Reamostragem", "tunagem": "Sem Tunagem",
    "sampler": None, "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Sem Reamostragem", "tunagem": "Com Tunagem",
    "sampler": None, "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
})
# 3.2 Under-sampling
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Under-sampling", "tunagem": "Sem Tunagem",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "Under-sampling", "tunagem": "Com Tunagem",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, max_iter=1000, random_state=42)
})
# 3.3 SMOTE
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "SMOTE", "tunagem": "Sem Tunagem",
    "sampler": SMOTE(random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "SMOTE", "tunagem": "Com Tunagem",
    "sampler": SMOTE(random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, max_iter=1000, random_state=42)
})
# 3.4 SMOTETomek
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "SMOTETomek", "tunagem": "Sem Tunagem",
    "sampler": SMOTETomek(random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
})
experiments.append({
    "modelo": "Regressão Logística", "balanceamento": "SMOTETomek", "tunagem": "Com Tunagem",
    "sampler": SMOTETomek(random_state=42), "numeric_trans": StandardScaler(),
    "clf": LogisticRegression(C=0.1, max_iter=1000, random_state=42)
})

for exp in experiments:
    if isinstance(exp["clf"], xgb.XGBClassifier):
        exp["clf"].set_params(device=xgb_device)
    elif training_backend == "gpu":
        from src.models.gpu import GPUClassifier
        exp["clf"] = GPUClassifier(exp["clf"])

results_list = []

for idx, exp in enumerate(experiments):
    m_name = exp["modelo"]
    bal = exp["balanceamento"]
    tun = exp["tunagem"]
    logger.info(f"[{idx+1:02d}/{len(experiments):02d}] Testando {m_name} | {bal} | {tun}...")

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
    logger.info(f"      -> F1 Padrão: {f1_def*100:.1f}% | Limiar Otimizado: {thresh_opt:.4f} -> F1 Tunado: {f1_tun*100:.1f}% (Ganho: +{(f1_tun-f1_def)*100:.1f}%) | PR-AUC: {pr_auc*100:.1f}%")

df_tuning = pd.DataFrame(results_list)

# %%
"""
================================================================================
ETAPA 3: GERAÇÃO DE GRÁFICOS AGRUPADOS (MODELO, TUNAGEM, BALANCEAMENTO)
================================================================================
"""
logger.info("=" * 80)
logger.info("ETAPA 3: GERANDO E SALVANDO GRÁFICOS AGRUPADOS DE TUNAGEM")
logger.info("=" * 80)

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
savefig("src/notebooks/graficos/01_comparacao_por_modelo.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/01_comparacao_por_modelo.png")

# --- Gráfico 2: Agrupamento por Tunagem (Impacto da Calibração do Limiar) ---
fig, ax = plt.subplots(figsize=(20, 8))
df_melted = df_tuning.melt(
    id_vars=["Modelo", "Balanceamento", "Tunagem"],
    value_vars=["F1_Padrao", "F1_Tunado"],
    var_name="Tipo_Limiar", value_name="F1_Score"
)
df_melted["Tipo_Limiar"] = df_melted["Tipo_Limiar"].map({"F1_Padrao": "Limiar Padrão (0.50)", "F1_Tunado": "Limiar Tunado (Otimizado)"})
df_melted["Experimento_Label"] = df_melted["Modelo"] + " - " + df_melted["Balanceamento"] + " (" + df_melted["Tunagem"] + ")"

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
savefig("src/notebooks/graficos/02_impacto_por_tunagem.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/02_impacto_por_tunagem.png")

# --- Gráfico 3: Agrupamento por Tunagem e Balanceamento ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
balance_order = ["Sem Reamostragem", "Under-sampling", "SMOTE", "SMOTETomek"]
for i, m_name in enumerate(models_order):
    sub_df = df_tuning[df_tuning["Modelo"] == m_name]
    sns.barplot(
        data=sub_df, x="Balanceamento", y="F1_Tunado", hue="Tunagem",
        order=balance_order,
        palette={"Sem Tunagem": "#94a3b8", "Com Tunagem": "#10b981"}, ax=axes[i]
    )
    axes[i].set_title(f"{m_name}\nF1 Tunado por Balanceamento", fontweight="bold", fontsize=12)
    axes[i].set_ylabel("F1-Score" if i == 0 else "")
    axes[i].set_xlabel("Técnica de Balanceamento", fontweight="bold")
    axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    axes[i].legend(title="Configuração", fontsize=8, title_fontsize=9, loc="upper right")

plt.suptitle("AGRUPAMENTO POR TUNAGEM E BALANCEAMENTO: Cruzamento entre Reamostragem e Hiperparâmetros", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
savefig("src/notebooks/graficos/03_tunagem_por_balanceamento.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/03_tunagem_por_balanceamento.png")

# %%
"""
================================================================================
ETAPA 4: TREINAMENTO DOS MELHORES MODELOS EM XTRAIN (1.6M)
================================================================================
"""
logger.info("=" * 80)
logger.info("ETAPA 4: TREINAMENTO DOS MELHORES MODELOS EM 1.6M E PREDIÇÃO NO TESTE DE 400K")
logger.info("=" * 80)

best_configs = {}
for m_name in models_order:
    sub = df_tuning[df_tuning["Modelo"] == m_name]
    best_row_idx = (sub["PR_AUC"] * 0.5 + sub["F1_Tunado"] * 0.5).idxmax()
    best_configs[m_name] = experiments[best_row_idx]
    logger.info(f"Melhor configuração selecionada para {m_name}:")
    logger.info(f"  Balanceamento: {best_configs[m_name]['balanceamento']}")
    logger.info(f"  Tunagem:       {best_configs[m_name]['tunagem']}")

final_model_names = []
final_test_results = []
final_pipelines = []
final_y_probs = {}
final_y_preds = {}

for m_name in models_order:
    cfg = best_configs[m_name]
    display_name = f"{m_name} ({cfg['balanceamento']})"
    logger.info(f"Treinando modelo final: {display_name} no conjunto de 1.6M...")

    pipe_final = _getPipeline(
        sampler=cfg["sampler"],
        classifier=cfg["clf"],
        numeric_transformer=cfg["numeric_trans"]
    )
    t_start = time.time()
    pipe_final.fit(XTrainFinal, yTrainFinal)
    logger.info(f"  -> Treinamento em 1.6M concluído em {time.time()-t_start:.2f}s")

    logger.info("  -> Calibrando limiar ótimo de decisão na partição de validação...")
    tuned_final = TunedThresholdClassifierCV(pipe_final, scoring="f1", cv="prefit", refit=False)
    tuned_final.fit(XVal, yVal)
    logger.info(f"  -> Limiar ótimo calibrado: {tuned_final.best_threshold_:.4f}")

    t_start = time.time()
    y_pred = tuned_final.predict(XTest)
    y_prob = tuned_final.predict_proba(XTest)[:, 1]
    logger.info(f"  -> Inferência em 400k concluída em {time.time()-t_start:.2f}s")

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
logger.info("=" * 80)
logger.info("ETAPA 5: TESTE DE DESEMPENHO APROFUNDADO NO CONJUNTO DE TESTE (400.000 TRANSAÇÕES)")
logger.info("=" * 80)

# 1. Tabela de Métricas
printTestResult(final_model_names, *final_test_results)

# 2. Avaliação de Impacto Financeiro
logger.info("=" * 80)
logger.info("AVALIAÇÃO DE IMPACTO FINANCEIRO NO CONJUNTO DE TESTE (400.000 TRANSAÇÕES)")
logger.info("=" * 80)

frauds_mask = (yTest == 1)
total_fraud_val = float(XTest.loc[frauds_mask, "valor_brl"].sum())
total_fraud_count = int(frauds_mask.sum())
logger.info(f"Volume Total de Fraudes no Teste: R$ {total_fraud_val:,.2f} ({total_fraud_count:,} ocorrências)")

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

    logger.info(f"[{name}]")
    logger.info(f"  -> Fraudes Bloqueadas com Sucesso: R$ {val_intercepted:,.2f} ({pct_saved:.2f}% do volume total | {count_intercepted}/{total_fraud_count} fraudes)")
    logger.info(f"  -> Fraudes Não Detectadas (Perda Residual): R$ {val_lost:,.2f} ({100-pct_saved:.2f}% | {count_lost} fraudes)")
    logger.info(f"  -> Volume Financeiro de Falsos Alarmes: R$ {val_fp:,.2f} ({count_fp:,} transações legítimas)")

df_test_2m = pd.DataFrame([{
    "modelo": name,
    "accuracy": res["accuracy"],
    "precision": res["precision"],
    "recall": res["recall"],
    "f1": res["f1"],
    "roc_auc": res["roc_auc"],
    "pr_auc": res["pr_auc"],
    "limiar": res["limiar"],
    "fraudes_interceptadas_brl": fin["fraudes_interceptadas_brl"],
    "fraudes_nao_detectadas_brl": fin["fraudes_nao_detectadas_brl"],
    "falsos_positivos_brl": fin["falsos_positivos_brl"],
    "pct_financeiro_salvo": fin["pct_financeiro_salvo"]
} for name, res, fin in zip(final_model_names, final_test_results, financial_metrics_list)])
os.makedirs("resultados/tabelas", exist_ok=True)
df_test_2m.to_csv("resultados/tabelas/metricas_teste_2m.csv", index=False)
logger.info("Tabela de métricas de teste de 2M salva em: resultados/tabelas/metricas_teste_2m.csv")

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
savefig("src/notebooks/graficos/04_matrizes_confusao_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/04_matrizes_confusao_teste_2m.png")

# --- Gráfico 5: Curvas ROC no Teste Independente ---
plt.figure(figsize=(9, 7))
colors_map = [next((v for k, v in palette.items() if k in m), "#333333") for m in final_model_names]

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
savefig("src/notebooks/graficos/05_curvas_roc_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/05_curvas_roc_teste_2m.png")

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
savefig("src/notebooks/graficos/06_curvas_precision_recall_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/06_curvas_precision_recall_teste_2m.png")

# Preserva probabilidades para novos gráficos sem repetir o treinamento.
pd.DataFrame({
    "classe_real": np.asarray(yTest),
    **{name + "_prob": probs for name, probs in final_y_probs.items()},
    **{name + "_pred": preds for name, preds in final_y_preds.items()},
}).to_parquet("resultados/tabelas/predicoes_teste_2m.parquet", index=False)

# --- Gráfico 7: trade-off dos três modelos finais e contagens de acertos ---
from src.visualization.model_diagnostics import plot_model_diagnostics
plot_model_diagnostics(
    yTest, final_y_probs, final_y_preds,
    {name: res["limiar"] for name, res in zip(final_model_names, final_test_results)},
    "src/notebooks/graficos", "Dataset 2M",
    "07_tradeoff_limiar_metricas_teste_2m.png",
    "15_acertos_falsos_positivos_teste_2m.png",
)
logger.info("Salvos: trade-off de todos os modelos e contagens de acertos/falsos positivos (2M)")
best_name = final_model_names[0]
best_probs = final_y_probs[best_name]
best_limiar = final_test_results[0]["limiar"]

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
savefig("src/notebooks/graficos/08_distribuicao_probabilidades_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/08_distribuicao_probabilidades_teste_2m.png")

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
savefig("src/notebooks/graficos/09_impacto_financeiro_teste_2m.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Salvo: src/notebooks/graficos/09_impacto_financeiro_teste_2m.png")

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

logger.info("=" * 80)
logger.info(f"MELHOR MODELO SELECIONADO: {best_model_name}")
logger.info(f"PR-AUC: {best_metrics['pr_auc']*100:.2f}% | F1: {best_metrics['f1']*100:.2f}% | ROC-AUC: {best_metrics['roc_auc']*100:.2f}% | Recall: {best_metrics['recall']*100:.2f}% | Precision: {best_metrics['precision']*100:.2f}%")
logger.info(f"Impacto Financeiro: R$ {best_financial['fraudes_interceptadas_brl']:,.2f} interceptados ({best_financial['pct_financeiro_salvo']:.2f}% do total de fraudes)")
logger.info("=" * 80)

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

# %%
"""
================================================================================
ETAPA 7: PROCESSAMENTO E GERAÇÃO DOS GRÁFICOS DO DATASET DE 10K PIX
================================================================================
Gera todos os 9 gráficos individuais do dataset de 10k (anomalias cadastrais sintéticas)
em 'src/notebooks/graficos/10k/' e o painel consolidado em 'resultados/10k/visao_geral.png'.
"""
logger.info("=" * 80)
logger.info("ETAPA 7: PROCESSAMENTO E GERAÇÃO DOS GRÁFICOS DO DATASET DE 10K PIX")
logger.info("=" * 80)

from plot_pix_10k import generate_10k_plots
data_10k_res = generate_10k_plots(use_saved=args.reuse_10k_results)
logger.info("Gráficos do dataset de 10k gerados com sucesso em: src/notebooks/graficos/10k/")

# %%
"""
================================================================================
ETAPA 8: COMPARAÇÃO ESTRUTURAL ENTRE DATASET PEQUENO (10K) E GRANDE (2M)
================================================================================
Gera os gráficos comparativos diretos lado a lado (10 a 14) em 'src/notebooks/graficos/':
- 10_comparacao_10k_vs_2m_metricas.png: F1, Precision, Recall, PR-AUC, ROC-AUC lado a lado
- 11_comparacao_10k_vs_2m_resolucao_degrau.png: Resolução estatística (5% por erro no 10k vs 0.0325% no 2M)
- 12_comparacao_10k_vs_2m_curvas_pr_roc.png: Painel 2x2 com Curvas ROC e PR e limiares de prevalência
- 13_comparacao_10k_vs_2m_balanceamento.png: Papel do balanceamento (vital em 10k vs custo/ruído em 2M)
- 14_comparacao_10k_vs_2m_escala_financeira.png: Escala de negócio (R$ milhares vs R$ bilhões)
"""
logger.info("=" * 80)
logger.info("ETAPA 8: COMPARAÇÃO ESTRUTURAL ENTRE DATASET PEQUENO (10K) E GRANDE (2M)")
logger.info("=" * 80)

from src.visualization.comparison_10k_2m import generate_comparison_plots
comp_table = generate_comparison_plots(
    metrics_2m=[{
        "modelo": name,
        "accuracy": res["accuracy"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "roc_auc": res["roc_auc"],
        "pr_auc": res["pr_auc"],
        "limiar": res["limiar"],
        "nome_completo": name
    } for name, res in zip(final_model_names, final_test_results)],
    probs_2m=final_y_probs,
    y_test_2m=yTest,
    financial_2m=financial_metrics_list,
    data_10k=data_10k_res,
    tuning_2m=df_tuning,
)
logger.info("Painéis comparativos gerados com sucesso:")
logger.info("  -> 10_comparacao_10k_vs_2m_metricas.png")
logger.info("  -> 11_comparacao_10k_vs_2m_resolucao_degrau.png")
logger.info("  -> 12_comparacao_10k_vs_2m_curvas_pr_roc.png")
logger.info("  -> 13_comparacao_10k_vs_2m_balanceamento.png")
logger.info("  -> 14_comparacao_10k_vs_2m_escala_financeira.png")
logger.info("\nTabela Resumo Comparativo 10k vs 2M:\n" + comp_table.to_string(index=False))
logger.info("EXECUÇÃO COMPLETA: treinamentos, métricas e gráficos de 2M e 10k concluídos.")
