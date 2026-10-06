# %%
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate, TunedThresholdClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, average_precision_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# %%
datasetDir = kagglehub.dataset_download("juniorbueno/pix-banking-transaction")

# %%
dataDf = pd.read_csv(f"{datasetDir}/comprovantes_pix_10000_anomalias.csv", sep=";")

# %%
dataDf.info()

# %%
dataDf.head()

# %%
dataDf.count()

# %%
dataDf.describe()

# %%
"""
    Como todos os dados são na moeda brasileira não faz necessario essa coluna
"""

(dataDf['Moeda'].str.strip() == 'BRL').sum()

# %%
"""
    Como sempre existe mais transações legitimas do que fraudulentas, é pressupostos que o dataset é
    desbalanceado. Porém é necessário uma confirmação
"""
class_counts = dataDf['Anomalia'].value_counts()
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

# %%
"""
    Transformando a coluna DataHora em datetime, tratando valores nulos transformando em -1
    e criando as colunas de Hora, dia da semana e dia do mês
    que são dados importantes para conseguir identificar o padrão de compra
"""

dataDf['DataHora'] = pd.to_datetime(dataDf['DataHora'], errors='coerce')
dataDf['Hora'] = dataDf['DataHora'].dt.hour.fillna(-1).astype(int)
dataDf['Dia_Da_Semana'] = dataDf['DataHora'].dt.dayofweek.fillna(-1).astype(int)  # 0 = Monday, 6 = Sunday
dataDf['Dia_Do_Mes'] = dataDf['DataHora'].dt.day.fillna(-1).astype(int)

# %%
dataDf['Pagador_Is_CNPJ'] = dataDf['Pagador_CPF_CNPJ'].str.contains('/').astype(int)
dataDf['Recebedor_Is_CNPJ'] = dataDf['Recebedor_CPF_CNPJ'].str.contains('/').astype(int)
dataDf['Pagador_CPF_Zerado'] = (dataDf['Pagador_CPF_CNPJ'] == '000.000.000-00').astype(int)

# %%
"""
    Verifica se a transação foi feita para o mesmo banco do remetente
"""
dataDf['Mesmo_Banco'] = (
    dataDf['Pagador_Banco'] == dataDf['Recebedor_Banco']
).astype(int)

contagem = dataDf[dataDf['Anomalia'] == 1]['Mesmo_Banco'].value_counts()

contagem.index = ['Bancos diferentes', 'Mesmo banco']

contagem.plot(
    kind='bar',
    xlabel='Tipo de transação',
    ylabel='Quantidade',
    title='Transações entre bancos',
    rot=0
)

plt.show()

# %%
"""
    Mostrando os tipos de chaves usadas e suas respectivas quantidades
"""

frequencia_chaves = dataDf[dataDf['Anomalia'] == 1]['TipoChave'].value_counts()

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

"""
    Limpar dados sensíveis e que possa causar ruído no aprendizado
"""
columnsToDrop = [
    'EndToEndId', 
    'Pagador_Nome', 
    'Pagador_CPF_CNPJ', 
    'Recebedor_Nome', 
    'Recebedor_CPF_CNPJ',
    'ChavePix_Utilizada',
    'DataHora',
    'Moeda',
    'Descricao'
]

dataDf = dataDf.drop(columns=columnsToDrop)

# %%
"""
    Separação dos dados de treinamento e de teste
    Garantindo que o target (YdataDf) vai ter a proporção mantida
"""
XDataDf = dataDf.drop(["Anomalia"], axis=1)
YDataDf = dataDf["Anomalia"]

XTrain, XTest, yTrain, yTest = train_test_split(
    XDataDf, YDataDf, test_size=0.2,
    stratify=YDataDf, random_state=42
)

# %%
XTrain.shape, XTest.shape

# %%
categoricalFeatures = [
    'TipoChave',
    'Status',
    'Pagador_Banco',
    'Recebedor_Banco'
]

numericFeatures = [
    'Valor',
    'Hora',
    'Dia_Da_Semana',
    'Dia_Do_Mes',
    'Pagador_Is_CNPJ',
    'Recebedor_Is_CNPJ',
    'Pagador_CPF_Zerado',
    'Mesmo_Banco'
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
}

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
)

rawPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

results = cross_validate(
    rawPipeline,
    XTrain,
    yTrain,
    scoring=scoring,
    cv=5
)

# %%

overClassifier = xgb.XGBClassifier(
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
)

overPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", SMOTE(random_state=42)),
    ("classifier", overClassifier)
])

overTuned = TunedThresholdClassifierCV(overPipeline, scoring="f1", cv=5)

overResults = cross_validate(
    overTuned,
    XTrain,
    yTrain,
    scoring=scoring,
    cv=5
)

# %%
underClassifier = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=200,
    random_state=0,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    tree_method='hist',
)

underPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
    ("classifier", underClassifier)
])

underTuned = TunedThresholdClassifierCV(underPipeline, scoring="f1", cv=5)

underResults = cross_validate(
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
    tree_method='hist',
)

mixPipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", SMOTETomek(random_state=42)),
    ("classifier", mixClassifier)
])

mixTuned = TunedThresholdClassifierCV(mixPipeline, scoring="f1", cv=5)

mixResults = cross_validate(
    mixTuned,
    XTrain,
    yTrain,
    scoring=scoring,
    cv=5
)

# %%
overTuned.fit(XTrain, yTrain)

# Probabilidades da classe fraude no conjunto de teste
y_scores = overTuned.predict_proba(XTest)[:, 1]

pr_auc = average_precision_score(yTest, y_scores)
print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
print(f"Baseline (aleatório): {yTest.mean():.4f}")