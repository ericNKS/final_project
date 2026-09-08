# %%
from src.data.ingestion import loadData
from src.data.cleaning import calculate_haversine_distance, clean_transactions, extract_time_features

# %%
rawDataDf = loadData("./data/archive/fraudTrain.csv")

# %%
dataCleanedDf = clean_transactions(rawDataDf)
dataCleanedDf = extract_time_features(dataCleanedDf)
dataCleanedDf = calculate_haversine_distance(dataCleanedDf)
