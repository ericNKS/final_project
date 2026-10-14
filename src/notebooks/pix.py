# %%
import kagglehub
import truststore
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate, TunedThresholdClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# %%
truststore.inject_into_ssl()
# Reutiliza o dataset já baixado; consulta o Kaggle apenas se o CSV não existir.
cachedDataset = Path.home() / '.cache/kagglehub/datasets/juniorbueno/pix-banking-transaction/versions/1'
datasetDir = str(cachedDataset) if (cachedDataset / 'comprovantes_pix_10000_anomalias.csv').exists() else kagglehub.dataset_download("juniorbueno/pix-banking-transaction")

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

contagem = contagem.reindex([0, 1], fill_value=0)
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

# As três famílias são avaliadas juntas ao final do notebook.

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


# %%
# Regressão Logística usa escala padronizada antes da reamostragem.
# Todos os transformadores e samplers são ajustados dentro de cada fold.
modelPipelines = {
    'XGBoost | Baseline': rawPipeline,
    'XGBoost | SMOTE': overPipeline,
    'XGBoost | Under-sampling': underPipeline,
    'XGBoost | SMOTE + Tomek': mixPipeline,
}
samplers = {
    'Baseline': None,
    'SMOTE': SMOTE(random_state=42),
    'Under-sampling': RandomUnderSampler(sampling_strategy=0.1, random_state=42),
    'SMOTE + Tomek': SMOTETomek(random_state=42),
}
for family in ('Random Forest', 'Regressão Logística'):
    for balancing, sampler in samplers.items():
        weight = 'balanced' if sampler is None else None
        estimator = (RandomForestClassifier(n_estimators=100, max_depth=12,
                     class_weight=weight, random_state=42, n_jobs=-1)
                     if family == 'Random Forest' else
                     LogisticRegression(C=1.0, max_iter=3000, class_weight=weight,
                                        random_state=42))
        steps = [('preprocessor', clone(preprocessor))]
        if family == 'Regressão Logística':
            steps.append(('scaler', StandardScaler()))
        if sampler is not None:
            steps.append(('sampler', clone(sampler)))
        steps.append(('classifier', estimator))
        modelPipelines[f'{family} | {balancing}'] = ImbPipeline(steps)

# %%
def default_f1(estimator, X, y):
    return f1_score(y, estimator.predict_proba(X)[:, 1] >= 0.5, zero_division=0)


comparisonScoring = {**scoring, 'PR_AUC': 'average_precision', 'ROC_AUC': 'roc_auc',
                     'F1_Tunado': 'f1', 'F1_Padrao': default_f1}
comparisonRows, testRows = [], []
fittedModels, testPredictions = {}, {}
for label, pipeline in modelPipelines.items():
    family, balancing = label.split(' | ')
    print(f'Avaliando {label}', flush=True)
    tuned = TunedThresholdClassifierCV(clone(pipeline), scoring='f1', cv=5)
    cvResults = cross_validate(tuned, XTrain, yTrain, scoring=comparisonScoring,
                               cv=5, error_score='raise')
    for fold in range(5):
        comparisonRows.append({'Modelo': family, 'Balanceamento': balancing,
                               'Variante': label, 'Fold': fold + 1,
                               **{key: float(cvResults['test_' + key][fold])
                                  for key in comparisonScoring}})
    tuned.fit(XTrain, yTrain)
    fittedModels[label] = tuned
    prob, pred = tuned.predict_proba(XTest)[:, 1], tuned.predict(XTest)
    testPredictions[label] = (prob, pred)
    testRows.append({'Modelo': family, 'Balanceamento': balancing, 'Variante': label,
                     'Limiar': float(tuned.best_threshold_),
                     'PR_AUC': average_precision_score(yTest, prob),
                     'ROC_AUC': roc_auc_score(yTest, prob),
                     'F1': f1_score(yTest, pred, zero_division=0),
                     'Precision': precision_score(yTest, pred, zero_division=0),
                     'Recall': recall_score(yTest, pred, zero_division=0)})

comparisonResults = pd.DataFrame(comparisonRows)
testResults = pd.DataFrame(testRows)
# Escolha por F1 médio na validação externa, sem usar o teste.
cvMeans = comparisonResults.groupby('Variante')['F1_Tunado'].mean()
selectedVariants = {family: cvMeans[[label for label in modelPipelines
                    if label.startswith(family + ' | ')]].idxmax()
                    for family in ('XGBoost', 'Random Forest', 'Regressão Logística')}
bestVariant = cvMeans.idxmax()
outputDir = Path(__file__).resolve().parents[2] / 'resultados/10k'
outputDir.mkdir(parents=True, exist_ok=True)
comparisonResults.to_csv(outputDir / 'metricas_validacao_cruzada.csv', index=False)
testResults.to_csv(outputDir / 'metricas_teste.csv', index=False)
print('\nResultados no teste de 2.000 transações:')
print(testResults.to_string(index=False))
print(f'Escolha pela validação cruzada: {bestVariant}')

# Compatibilidade com as células originais do XGBoost.
overTuned = fittedModels['XGBoost | SMOTE']
y_scores = testPredictions['XGBoost | SMOTE'][0]
pr_auc = average_precision_score(yTest, y_scores)
print(f'PR-AUC XGBoost SMOTE: {pr_auc:.4f}')
