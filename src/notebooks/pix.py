# %%
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score, auc
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
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
dataDf['Mesmo_Banco'] = (dataDf['Pagador_Banco'] == dataDf['Recebedor_Banco']).astype(int)

# %%
"""
    Aqui vamos converter as colunas do tipo string que não pode ser interpretado pelo modelo matemático
    em colunas categoricas binárias, ou seja, se a quela linha faz parte daquela categoria vai receber 1
    se não vai receber 0
"""
dataDf = pd.get_dummies(dataDf, columns=['TipoChave'], dtype=int)
dataDf = pd.get_dummies(dataDf, columns=['Status'], dtype=int)
dataDf = pd.get_dummies(dataDf, columns=['Pagador_Banco'], dtype=int)
dataDf = pd.get_dummies(dataDf, columns=['Recebedor_Banco'], dtype=int)

# %%
"""
    Mostrando os tipos de chaves usadas e suas respectivas quantidades
"""

colunas_chave = [col for col in dataDf.columns if 'TipoChave' in col]

frequencia_chaves = dataDf[colunas_chave].sum().sort_values(ascending=False)

# Limpar o nome das colunas para o gráfico ficar legível (removendo o "TipoChave_")
frequencia_chaves.index = frequencia_chaves.index.str.replace('TipoChave_', '')

ax = frequencia_chaves.plot(
    kind='bar', 
    color='#1f77b4', 
    figsize=(10, 6)
)

plt.title('Distribuição dos Tipos de Chave Pix Utilizadas', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Tipo de Chave', fontweight='bold')
plt.ylabel('Volume de Transações', fontweight='bold')
plt.xticks(rotation=45, ha='right')

# Adiciona os rótulos numéricos em cima de cada barra
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()
plt.show()

# %%
frequencia_clientes = dataDf['Pagador_CPF_CNPJ'].value_counts()
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
"""
Criando treinamentos com diferentes tipos de balanceamento
"""

# Oversampling
overXtrain, overYtrain = SMOTE(random_state=42).fit_resample(XTrain, yTrain)
# Undersampling
underXtrain, underYtrain = RandomUnderSampler(random_state=42).fit_resample(XTrain, yTrain)
# Combinado
mixXtrain, mixYtrain = SMOTETomek(random_state=42).fit_resample(XTrain, yTrain)

# %%
