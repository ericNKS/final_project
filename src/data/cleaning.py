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


def prepare2mDataset(df: pd.DataFrame):
    """
    Realiza o tratamento e engenharia de atributos do dataset completo de 2M,
    conforme estruturado em df2m.py.
    Retorna (X, y).
    """
    logger.info("Preparando dataset de 2M transações...")
    dataDf = df.copy()

    dataHora = pd.to_datetime(dataDf['datetime_brasilia'], errors='coerce')
    dataDf['Hora'] = dataHora.dt.hour.fillna(-1).astype(int)
    dataDf['Dia_Da_Semana'] = dataHora.dt.dayofweek.fillna(-1).astype(int)

    pagadorGroup = dataDf.groupby("id_pagador", sort=False)
    dataDf["qtd_transacoes_pagador"] = pagadorGroup.cumcount()
    somaPagador = pagadorGroup["valor_brl"].cumsum()
    qtdAnteriorPagador = dataDf["qtd_transacoes_pagador"]

    dataDf["media_valor_pagador"] = np.where(
        qtdAnteriorPagador > 0,
        (somaPagador - dataDf["valor_brl"]) / qtdAnteriorPagador,
        np.nan
    )

    dataDf["mediana_valor_pagador"] = (
        dataDf
        .groupby("id_pagador", sort=False)["valor_brl"]
        .expanding()
        .median()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
        .reindex(dataDf.index)
    )

    columnsToDrop = [
        'hora_dia', 
        'dia_semana', 
        'id_recebedor', 
        'id_pagador',
        'datetime_brasilia',
    ]

    dataDf = dataDf.drop(columns=columnsToDrop, errors='ignore')

    X = dataDf.drop(columns=["fraude"])
    y = dataDf["fraude"]

    return X, y


def prepare10kDataset(df: pd.DataFrame, reference_columns: list = None):
    """
    Trata e adapta o dataset de 10k (pix.py / comprovantes_pix_10000_anomalias.csv)
    para que seja totalmente compatível com o schema do dataset de 2M treinado.
    Retorna (X, y).
    """
    logger.info("Preparando e compatibilizando dataset de 10k transações com o schema de 2M...")
    dataDf = df.copy()

    # Mapeamento do valor e tipo de transação
    dataDf['valor_brl'] = dataDf['Valor'].astype(float)
    dataDf['tipo_transacao'] = 'chave_pix'

    # Tratamento de data/hora resiliente a datas anômalas
    dataHora = pd.to_datetime(dataDf['DataHora'], errors='coerce')
    dataDf['Hora'] = dataHora.dt.hour.fillna(-1).astype(int)
    dataDf['Dia_Da_Semana'] = dataHora.dt.dayofweek.fillna(-1).astype(int)
    dataDf['dia_util'] = (dataDf['Dia_Da_Semana'] < 5) & (dataDf['Dia_Da_Semana'] >= 0)
    dataDf['horario_noturno'] = (dataDf['Hora'] >= 20) | (dataDf['Hora'] < 6)
    dataDf['acima_limite_noturno'] = dataDf['horario_noturno'] & (dataDf['valor_brl'] > 1000)

    # Métricas agregadas do pagador
    payerCol = 'Pagador_CPF_CNPJ' if 'Pagador_CPF_CNPJ' in dataDf.columns else 'Pagador_Nome'
    pagadorGroup = dataDf.groupby(payerCol, sort=False)
    dataDf["qtd_transacoes_pagador"] = pagadorGroup.cumcount()
    somaPagador = pagadorGroup["valor_brl"].cumsum()
    qtdAnteriorPagador = dataDf["qtd_transacoes_pagador"]

    dataDf["media_valor_pagador"] = np.where(
        qtdAnteriorPagador > 0,
        (somaPagador - dataDf["valor_brl"]) / qtdAnteriorPagador,
        np.nan
    )

    dataDf["mediana_valor_pagador"] = (
        dataDf
        .groupby(payerCol, sort=False)["valor_brl"]
        .expanding()
        .median()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
        .reindex(dataDf.index)
    )

    # Inclusão de colunas de saldo ausentes como NaN para alinhamento dimensional
    balanceCols = [
        'saldo_anterior_pagador',
        'saldo_posterior_pagador',
        'saldo_anterior_recebedor',
        'saldo_posterior_recebedor',
        'razao_saldo_residual',
        'proporcao_valor_recebedor'
    ]
    for col in balanceCols:
        if col not in dataDf.columns:
            dataDf[col] = np.nan

    # Target
    y = dataDf['Anomalia'] if 'Anomalia' in dataDf.columns else dataDf.get('fraude')

    # Garantir a mesma ordem exata de features do dataset de 2M
    defaultFeatures = [
        'tipo_transacao',
        'valor_brl',
        'saldo_anterior_pagador',
        'saldo_posterior_pagador',
        'saldo_anterior_recebedor',
        'saldo_posterior_recebedor',
        'Hora',
        'Dia_Da_Semana',
        'dia_util',
        'horario_noturno',
        'acima_limite_noturno',
        'razao_saldo_residual',
        'proporcao_valor_recebedor',
        'qtd_transacoes_pagador',
        'media_valor_pagador',
        'mediana_valor_pagador'
    ]

    targetFeatures = reference_columns if reference_columns is not None else defaultFeatures
    X = dataDf[targetFeatures].copy()

    return X, y