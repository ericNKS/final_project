# %%
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

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

# %%

# 2. Transforme as categorias do Tipo de Chave em variáveis matemáticas binárias
# Isso criará colunas separadas como 'TipoChave_E-mail', 'TipoChave_CPF', onde 1 = Sim, 0 = Não
dataDf = pd.get_dummies(dataDf, columns=['TipoChave'], dtype=int)
# %%
columnsToDrop = [
    'EndToEndId', 
    'Pagador_Nome', 
    'Pagador_CPF_CNPJ', 
    'Recebedor_Nome', 
    'Recebedor_CPF_CNPJ',
    'ChavePix_Utilizada',
    'DataHora'
]

dataDf = dataDf.drop(columns=columnsToDrop)

# %%
dataDf.info()