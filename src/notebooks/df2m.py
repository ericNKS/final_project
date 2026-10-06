# %%
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import TunedThresholdClassifierCV, TunedThresholdClassifierCV, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline

#%%
dataDf = pd.read_parquet("hf://datasets/andremessina/pix-fraud-br/data/train-00000-of-00001.parquet")

# %%
dataDf['DataHora'] = pd.to_datetime(dataDf['datetime_brasilia'], errors='coerce')
dataDf['Hora'] = dataDf['DataHora'].dt.hour.fillna(-1).astype(int)
dataDf['Dia_Da_Semana'] = dataDf['DataHora'].dt.dayofweek.fillna(-1).astype(int)  # 0 = Monday, 6 = Sunday
dataDf['Dia_Do_Mes'] = dataDf['DataHora'].dt.day.fillna(-1).astype(int)

columnsToDrop = [
    'datetime_brasilia', 
    'hora_dia', 
    'dia_semana', 
    'id_recebedor', 
    'id_pagador',
    'datetime_brasilia',
]

# %%

frequencia_chaves = dataDf[dataDf['fraude'] == 1]['tipo_transacao'].value_counts()

ax = frequencia_chaves.plot(
    kind='bar',
    color='#1f77b4',
    figsize=(10, 6)
)

plt.title(
    'Distribuição dos Tipos de Chave Pix Utilizadas',
    fontsize=14,
    pad=15,
    fontweight='bold'
)

plt.xlabel('Tipo de Chave', fontweight='bold')
plt.ylabel('Volume de Transações', fontweight='bold')
plt.xticks(rotation=45, ha='right')

# Rótulos numéricos em cima das barras
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()
plt.show()


# %%
categoricalFeatures = [
    'tipo_transacao',
]

numericFeatures = [
    'valor_brl',
    'saldo_anterior_pagador',
    'saldo_posterior_pagador',
    'saldo_anterior_recebedor',
    'saldo_posterior_recebedor',
    'Hora',
    'Dia_Da_Semana',
    'Dia_Do_Mes',
    'dia_util',
    'horario_noturno',
    'acima_limite_noturno',
    'razao_saldo_residual',
    'proporcao_valor_recebedor',
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ),
            categoricalFeatures
        ),
        (
            'numeric',
            'passthrough',
            numericFeatures
        )
    ]
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

# %%
dataDf = dataDf.drop(columns=columnsToDrop)
XDataDf = dataDf.drop(["fraude"], axis=1)
YDataDf = dataDf["fraude"]

XTrain, XTest, yTrain, yTest = train_test_split(
    XDataDf, YDataDf, test_size=0.2,
    stratify=YDataDf, random_state=42
)

# %%
classifier = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=40,
    random_state=0,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    colsample_bynode=0.9,
    num_parallel_tree=1,
    scale_pos_weight=3,
    tree_method='hist',
    n_jobs=16,
)

rawPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

rawTuned = TunedThresholdClassifierCV(rawPipeline, scoring="f1", cv=5)

results = cross_validate(
    rawTuned,
    XTrain,
    yTrain,
    scoring=scoring,
    cv=5,
    n_jobs=1
)

results

# %%
underClassifier = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=200,
    random_state=0,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    tree_method='hist',
    n_jobs=-1,
)

underPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
    ("classifier", underClassifier)
])

underTuned = TunedThresholdClassifierCV(underPipeline, scoring="f1", cv=5)

cross_validate(
    underTuned,
    XTrain,
    yTrain,
    scoring=scoring,
    cv=5
)


# %%
mixClassifier = xgb.XGBClassifier(
    max_depth=14,
    learning_rate=0.1,
    n_estimators=40,
    random_state=5,
    min_child_weight=0,
    subsample=1,
    colsample_bytree=0.5,
    colsample_bynode=1,
    num_parallel_tree=2,
    scale_pos_weight=2.8,
    n_jobs=-1,
    tree_method='hist',
)

mixPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", SMOTETomek(random_state=42)),
    ("classifier", mixClassifier)
])

mixTuned = TunedThresholdClassifierCV(mixPipeline, scoring="f1")

cross_validate(
    mixTuned,
    XTrain,
    yTrain,
    scoring=scoring,
    cv=5
)