import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extractDateFeatures(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extraindo features de data...")
    df_feat = df.copy()
    
    df_feat['DataHora'] = pd.to_datetime(df_feat['DataHora'], errors='coerce')
    df_feat['Hora'] = df_feat['DataHora'].dt.hour.fillna(-1).astype(int)
    df_feat['Dia_Da_Semana'] = df_feat['DataHora'].dt.dayofweek.fillna(-1).astype(int)  # 0 = Monday, 6 = Sunday
    df_feat['Dia_Do_Mes'] = df_feat['DataHora'].dt.day.fillna(-1).astype(int)
    
    return df_feat

def extractPaymentTypeFeatures(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extraindo features de tipo de pagamento...")
    df_feat = df.copy()
    
    df_feat['Pagador_Is_CNPJ'] = df_feat['Pagador_CPF_CNPJ'].str.contains('/').astype(int)
    df_feat['Recebedor_Is_CNPJ'] = df_feat['Recebedor_CPF_CNPJ'].str.contains('/').astype(int)
    df_feat['Pagador_CPF_Zerado'] = (df_feat['Pagador_CPF_CNPJ'] == '000.000.000-00').astype(int)
    
    return df_feat

def extractSameBankTransactionFeature(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extraindo feature de transações entre o mesmo banco...")
    df_feat = df.copy()
    
    df_feat['Mesmo_Banco'] = (df_feat['Pagador_Banco'] == df_feat['Recebedor_Banco']).astype(int)
    
    return df_feat

def removeUnnecessaryColumns(
    df: pd.DataFrame,
    columns_to_remove: list = [
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
) -> pd.DataFrame:
    logger.info("Removendo colunas desnecessárias...")
    
    
    df_cleaned = df.copy()
    
    df_cleaned.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    return df_cleaned