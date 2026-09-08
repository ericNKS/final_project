import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Iniciando limpeza. Shape original: {df.shape}")
    df_clean = df.copy()
    
    # Remoção de atributos não relacionados diretamente
    # ao comportamento transacional
    cols_to_drop = [
        'Unnamed: 0', 'first', 'last', 'gender', 'street', 
        'city', 'state', 'zip', 'job', 'dob'
    ]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    linhas_antes = len(df_clean)
    df_clean = df_clean.dropna().drop_duplicates()
    linhas_perdidas = linhas_antes - len(df_clean)
    
    if linhas_perdidas > 0:
        logger.warning(f"Limpeza removeu {linhas_perdidas} linhas (NAs ou duplicadas).")
        
    return df_clean

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extraindo features temporais...")
    df_feat = df.copy()
    
    df_feat["trans_date_trans_time"] = pd.to_datetime(df_feat["trans_date_trans_time"])
    df_feat["hour"] = df_feat["trans_date_trans_time"].dt.hour
    df_feat["day_of_week"] = df_feat["trans_date_trans_time"].dt.day_of_week
    df_feat["month"] = df_feat["trans_date_trans_time"].dt.month
    
    return df_feat

def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculando distância espacial entre cliente e estabelecimento...")
    df_feat = df.copy()
    
    EARTH_RADIUS_KM = 6371.0
    lat1, lon1 = np.radians(df_feat["lat"]), np.radians(df_feat["long"])
    lat2, lon2 = np.radians(df_feat["merch_lat"]), np.radians(df_feat["merch_long"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df_feat["distance_to_merchant"] = EARTH_RADIUS_KM * c
    return df_feat