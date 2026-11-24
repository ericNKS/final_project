# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import kagglehub

# %%
datasetDir = kagglehub.dataset_download("juniorbueno/pix-banking-transaction")

# %%
trainDf = pd.read_csv(f"{datasetDir}/comprovantes_pix_10000_anomalias.csv", sep=";")

# %%
trainDf.info()

# %%
trainDf.head()

# %%
trainDf.describe()
# %%
print(trainDf.columns)

# %%

# Remoção de atributos não relacionados diretamente
# ao comportamento transacional
trainDf = trainDf.drop(columns=[
    'Unnamed: 0',
    'first',
    'last',
    'gender',
    'street',
    'city',
    'state',
    'zip',
    'job',
    'dob',
])

# %%
trainDf["is_fraud"].value_counts().plot(kind="bar")

plt.xlabel("Fraude")
plt.ylabel("Quantidade")
plt.title("Quantidade de transações fraudulentas e não fraudulentas")
plt.xticks(
    ticks=[0, 1],
    labels=["Não fraude", "Fraude"],
    rotation=0
)
plt.show()

print(trainDf["is_fraud"].value_counts())

print(trainDf["is_fraud"].value_counts(normalize=True) * 100)

# %%
# Verificando valores nulos
msno.matrix(trainDf)

# %%
# Transformando a coluna trans_date_trans_time de str para datetime
# e extraindo informações de hora, dia da semana e mês

trainDf["trans_date_trans_time"] = pd.to_datetime(
    trainDf["trans_date_trans_time"]
)

trainDf["hour"] = trainDf["trans_date_trans_time"].dt.hour
trainDf["day_of_week"] = trainDf["trans_date_trans_time"].dt.dayofweek
trainDf["month"] = trainDf["trans_date_trans_time"].dt.month

# %%
# Calculando a distância entre a localização do cliente
# e a localização do estabelecimento em quilômetros

EARTH_RADIUS_KM = 6371.0

lat1 = np.radians(trainDf["lat"])
lon1 = np.radians(trainDf["long"])

lat2 = np.radians(trainDf["merch_lat"])
lon2 = np.radians(trainDf["merch_long"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1)
    * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

c = 2 * np.arctan2(
    np.sqrt(a),
    np.sqrt(1 - a)
)

trainDf["distance_to_merchant"] = EARTH_RADIUS_KM * c

