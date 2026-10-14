# %%
import sys
import importlib
from pathlib import Path

ROOT_DIR = Path.cwd().resolve()

while not (ROOT_DIR / "src" / "data" / "ingestion.py").is_file():
    if ROOT_DIR == ROOT_DIR.parent:
        raise RuntimeError("Não foi encontrado src/data/ingestion.py")
    ROOT_DIR = ROOT_DIR.parent

sys.path.insert(0, str(ROOT_DIR))
importlib.invalidate_caches()

print("Raiz:", ROOT_DIR)
print("Ingestion:", ROOT_DIR / "src/data/ingestion.py")

# %%
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
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
    "clf": xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.1,
            n_estimators=40,
            min_child_weight=1,
            subsample=1,
            colsample_bytree=1,
            colsample_bynode=0.9,
            num_parallel_tree=1,
            scale_pos_weight=3,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
})

# Under-sampling
experiments.append({
    "modelo": "XGBoost", "balanceamento": "Under-sampling", "tunagem": "Config A (depth=3, n=100, col=0.8)",
    "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(
            max_depth=1,
            learning_rate=0.1,
            n_estimators=500,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
})

experiments.append({
    "modelo": "XGBoost", "balanceamento": "SMOTE", "tunagem": "Config B (depth=8, n=80, child=4)",
    "sampler": SMOTE(random_state=42), "numeric_trans": "passthrough",
    "clf": xgb.XGBClassifier(
                max_depth=20,
                learning_rate=0.2,
                n_estimators=1000,
                random_state=0,
                min_child_weight=1,
                subsample=1,
                colsample_bytree=0.8,
                colsample_bynode=0.7,
                num_parallel_tree=1,
                scale_pos_weight=1,
                tree_method="hist",
                n_jobs=-1
            )
})


results_list = []

for idx, exp in enumerate(experiments):
    m_name = exp["modelo"]
    bal = exp["balanceamento"]
    tun = exp["tunagem"]

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
    
    print({
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




# %%
results_list

df_tuning = pd.DataFrame(results_list)

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

plt.show()

# %%



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