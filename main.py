from sklearn.ensemble import RandomForestClassifier
from src.models.random_forest import crossValidate as crossValidateRandomForest
from src.models.logistic_regression import (
    crossValidate as crossValidateLogisticRegression,
    getClassifier as getLogisticClassifier,
)
# %%
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

from src.models.xgboost import getTrainTestSplit, crossValidate
from src.data.ingestion import loadData, printTrainResult
from src.data.cleaning import extractDateFeatures, extractPaymentTypeFeatures, extractSameBankTransactionFeature, removeUnnecessaryColumns
import matplotlib.pyplot as plt
import xgboost as xgb

# %%
pixDf = loadData()

pixDf = extractDateFeatures(pixDf)
pixDf = extractPaymentTypeFeatures(pixDf)
pixDf = extractSameBankTransactionFeature(pixDf)
pixDf = removeUnnecessaryColumns(pixDf)

class_counts = pixDf['Anomalia'].value_counts()
legitTransactionColor = '#2ca02c'
fraudTransactionColor = '#d62728'
ax = class_counts.plot(
    kind='bar', 
    color=[legitTransactionColor, fraudTransactionColor], # Verde para normal, Vermelho para fraude
    figsize=(8, 5)
)

plt.title('Distribuição do Target: O Abismo do Desbalanceamento', fontsize=14, pad=15)
plt.xlabel('Classe (0 = Legítima, 1 = Fraude)', fontweight='bold')
plt.ylabel('Volume de Transações', fontweight='bold')
plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3, fontweight='bold')

plt.tight_layout()
plt.show()


"""
    Verifica se a transação foi feita para o mesmo banco do remetente
"""
contagem = pixDf[pixDf['Anomalia'] == 1]['Mesmo_Banco'].value_counts()

contagem.index = ['Bancos diferentes', 'Mesmo banco']

contagem.plot(
    kind='bar',
    xlabel='Tipo de transação',
    ylabel='Quantidade',
    title='Transações entre bancos',
    rot=0
)

plt.show()


"""
    Mostrando os tipos de chaves usadas e suas respectivas quantidades
"""

frequencia_chaves = pixDf[pixDf['Anomalia'] == 1]['TipoChave'].value_counts()

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


XTrain, XTest, yTrain, yTest = getTrainTestSplit(
    df=pixDf,
    testSize=0.2,
    randomState=42
)


"""
    Cross Validation with different sampling strategies
    but without hyperparameter tuning, to see the impact of sampling on the model's performance.
"""

xgbClassifier = xgb.XGBClassifier()
xgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier=xgbClassifier,
    sampler=None,
)

overXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier=xgbClassifier,
    sampler=SMOTE(random_state=42),
)

underXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier=xgbClassifier,
    sampler=RandomUnderSampler(sampling_strategy=0.1, random_state=42),
)

mixXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier=xgbClassifier,
    sampler=SMOTETomek(random_state=42)
)

modelNames = [
    "XGBoost",
    "XGBoost + SMOTE",
    "XGBoost + Under",
    "XGBoost + SMOTETomek"
]

results = [
    xgbResults,
    overXgbResults,
    underXgbResults,
    mixXgbResults
]

print("Cross Validation with different sampling strategies but without hyperparameter tuning, to see the impact of sampling on the model's performance.")
printTrainResult(modelNames, *results)


"""
    Cross Validation with different sampling strategies
    and hyperparameter tuning, to see the impact of both on the model's performance.
"""


tunedXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier= xgb.XGBClassifier(
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
    ),
    sampler=None,
)

tunedOverXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier= xgb.XGBClassifier(
        max_depth=12,
        learning_rate=0.1,
        n_estimators=100,
        random_state=0,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        colsample_bynode=0.9,
        num_parallel_tree=1,
        scale_pos_weight=1,
        tree_method='hist',
    ),
    sampler=SMOTE(random_state=42),
)

tunedUnderXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier= xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=200,
        random_state=0,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        tree_method='hist',
    ),
    sampler=RandomUnderSampler(sampling_strategy=0.1, random_state=42),
)

tunedUixXgbResults = crossValidate(
    df=pixDf,
    XTrain=XTrain,
    yTrain=yTrain,
    classifier= xgb.XGBClassifier(
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
        tree_method='hist',
    ),
    sampler=SMOTETomek(random_state=42)
)


tunedModelNames = [
    "XGBoost",
    "XGBoost + SMOTE",
    "XGBoost + Under",
    "XGBoost + SMOTETomek"
]

tunedResults = [
    tunedXgbResults,
    tunedOverXgbResults,
    tunedUnderXgbResults,
    tunedUixXgbResults
]

print("Cross Validation with different sampling strategies and hyperparameter tuning, to see the impact of both on the model's performance.")
printTrainResult(tunedModelNames, *tunedResults)

# %%

# Random Forest: mesmas particoes, metricas e estrategias do XGBoost.
# Os parametros ajustados sao configuracoes iniciais, sem busca automatica.
forestStrategies = [
    ("Random Forest", None),
    ("Random Forest + SMOTE", SMOTE(random_state=42)),
    ("Random Forest + Under", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
    ("Random Forest + SMOTETomek", SMOTETomek(random_state=42)),
]

forestResults = []
adjustedForestResults = []

for forestName, forestSampler in forestStrategies:
    forestResults.append(crossValidateRandomForest(
        df=pixDf,
        XTrain=XTrain,
        yTrain=yTrain,
        sampler=forestSampler,
    ))
    adjustedForestResults.append(crossValidateRandomForest(
        df=pixDf,
        XTrain=XTrain,
        yTrain=yTrain,
        sampler=forestSampler,
        classifier=RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced" if forestSampler is None else None,
            random_state=42,
            n_jobs=-1,
        ),
    ))

forestNames = [name for name, _ in forestStrategies]
print("Random Forest: configuracao padrao")
printTrainResult(forestNames, *forestResults)
print("Random Forest: configuracao ajustada")
printTrainResult(forestNames, *adjustedForestResults)

# Inclui Random Forest nos graficos comparativos existentes.
modelNames.extend(forestNames)
results.extend(forestResults)
tunedModelNames.extend(forestNames)
tunedResults.extend(adjustedForestResults)

# Regressão logística: mesmas partições, com padronização dentro dos folds.
logisticStrategies = [
    ("Regressão Logística", None),
    ("Regressão Logística + SMOTE", SMOTE(random_state=42)),
    ("Regressão Logística + Under", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
    ("Regressão Logística + SMOTETomek", SMOTETomek(random_state=42)),
]
logisticResults, adjustedLogisticResults = [], []
for logisticName, logisticSampler in logisticStrategies:
    for adjusted, destination in ((False, logisticResults), (True, adjustedLogisticResults)):
        destination.append(crossValidateLogisticRegression(
            df=pixDf, XTrain=XTrain, yTrain=yTrain, sampler=logisticSampler,
            classifier=getLogisticClassifier(adjusted=adjusted, sampler=logisticSampler),
        ))
logisticNames = [name for name, _ in logisticStrategies]
print("Regressão Logística: configuração padrão")
printTrainResult(logisticNames, *logisticResults)
print("Regressão Logística: configuração ajustada manualmente")
printTrainResult(logisticNames, *adjustedLogisticResults)
modelNames.extend(logisticNames)
results.extend(logisticResults)
tunedModelNames.extend(logisticNames)
tunedResults.extend(adjustedLogisticResults)

metrics = {
    "Accuracy": "test_accuracy",
    "Precision": "test_precision",
    "Recall": "test_recall",
    "F1": "test_f1",
    "ROC-AUC": "test_roc_auc"
}

# Calcula a média de cada métrica
comparison = {}

for metricName, metricKey in metrics.items():
    comparison[metricName] = {
        "Sem Tunagem": [
            result[metricKey].mean() * 100
            for result in results
        ],
        "Com Tunagem": [
            result[metricKey].mean() * 100
            for result in tunedResults
        ]
    }

# Cria um gráfico para cada métrica
for metricName in metrics:

    x = np.arange(len(modelNames))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 7))

    semTunagem = comparison[metricName]["Sem Tunagem"]
    comTunagem = comparison[metricName]["Com Tunagem"]

    bars1 = ax.bar(
        x - width / 2,
        semTunagem,
        width,
        label="Sem Tunagem"
    )

    bars2 = ax.bar(
        x + width / 2,
        comTunagem,
        width,
        label="Com Tunagem"
    )

    ax.set_title(
        f"Comparação de {metricName}: Com e Sem Tunagem",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(f"{metricName} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(modelNames, rotation=20, ha="right")

    ax.set_ylim(
        max(0, min(semTunagem + comTunagem) - 5),
        100
    )

    ax.legend()

    # Valores sobre as barras
    ax.bar_label(
        bars1,
        fmt="%.2f",
        padding=3
    )

    ax.bar_label(
        bars2,
        fmt="%.2f",
        padding=3
    )

    plt.tight_layout()
    plt.show()
