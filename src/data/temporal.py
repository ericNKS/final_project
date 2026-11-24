"""Scenario contracts and point-in-time features shared by training and serving."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Scenario = Literal["10k", "2m"]
CONTRACT_VERSION = 1
CATEGORIES = {
    "2m": ["tipo_transacao"],
    "10k": ["tipo_chave", "pagador_banco", "recebedor_banco"],
}
NUMERICS = {
    "2m": ["amount", "saldo_anterior_pagador", "saldo_anterior_recebedor",
           "hora", "dia_semana", "dia_util", "horario_noturno",
           "acima_limite_noturno", "qtd_transacoes_pagador",
           "media_valor_pagador", "mediana_valor_pagador"],
    "10k": ["amount", "hora", "dia_semana", "dia_mes", "pagador_is_cnpj",
            "recebedor_is_cnpj", "pagador_cpf_zerado", "mesmo_banco"],
}


def timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Interpret naive dataset dates in Brasília and store instants in UTC."""
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Data ausente ou inválida")
    if result.tzinfo is None:
        result = result.tz_localize("America/Sao_Paulo", ambiguous="raise", nonexistent="raise")
    return result.tz_convert("UTC")


def parse_dates(values: pd.Series) -> pd.Series:
    """Vectorized mixed-format parsing; preserve offsets and localize naive dates in Brasília."""
    result = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    aware = values.astype("string").str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False)
    naive = result.loc[~aware].dt.tz_localize(None).dt.tz_localize(
        "America/Sao_Paulo", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    result.loc[~aware] = naive
    return result


def feature_row(scenario: Scenario, occurred_at: pd.Timestamp, amount: float,
                attributes: dict, history: tuple[int, float | None, float | None] = (0, None, None)) -> dict:
    """Build only attributes known before execution; history excludes current/tied events."""
    local = timestamp(occurred_at).tz_convert("America/Sao_Paulo")
    row = {"amount": amount, "hora": local.hour, "dia_semana": local.dayofweek}
    if scenario == "2m":
        row.update({name: attributes[name] for name in (
            "tipo_transacao", "saldo_anterior_pagador", "saldo_anterior_recebedor")})
        night = int(local.hour >= 20 or local.hour < 6)
        row.update(dia_util=int(local.dayofweek < 5), horario_noturno=night,
                   acima_limite_noturno=int(night and amount > 1000),
                   qtd_transacoes_pagador=history[0], media_valor_pagador=history[1],
                   mediana_valor_pagador=history[2])
    elif scenario == "10k":
        row.update({name: attributes[name] for name in (
            "tipo_chave", "pagador_banco", "recebedor_banco", "pagador_is_cnpj",
            "recebedor_is_cnpj", "pagador_cpf_zerado")})
        row.update(dia_mes=local.day,
                   mesmo_banco=int(attributes["pagador_banco"] == attributes["recebedor_banco"]))
    else:
        raise ValueError("Cenário desconhecido")
    return row


def normalize(raw: pd.DataFrame, scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    """Adapt each source without fabricating the other scenario's missing balances."""
    raw = raw.reset_index(drop=True).copy()
    if scenario == "2m":
        frame = raw.rename(columns={"datetime_brasilia": "occurred_at", "valor_brl": "amount",
                                    "id_pagador": "payer_id", "fraude": "target"})
        attrs = ["tipo_transacao", "saldo_anterior_pagador", "saldo_anterior_recebedor"]
        frame["completed"] = True  # Source represents executed transactions; see methodology.
    elif scenario == "10k":
        frame = raw.rename(columns={"DataHora": "occurred_at", "Valor": "amount",
                                    "Pagador_CPF_CNPJ": "payer_id", "Anomalia": "target",
                                    "TipoChave": "tipo_chave", "Pagador_Banco": "pagador_banco",
                                    "Recebedor_Banco": "recebedor_banco"})
        frame["pagador_is_cnpj"] = frame.payer_id.astype("string").str.contains("/", na=False).astype(int)
        frame["recebedor_is_cnpj"] = raw.Recebedor_CPF_CNPJ.astype("string").str.contains("/", na=False).astype(int)
        frame["pagador_cpf_zerado"] = frame.payer_id.eq("000.000.000-00").astype(int)
        attrs = CATEGORIES[scenario] + ["pagador_is_cnpj", "recebedor_is_cnpj", "pagador_cpf_zerado"]
        frame["completed"] = raw["Status"].astype("string").str.casefold().isin(["concluída", "concluida", "concluído", "concluido", "completed"])
    else:
        raise ValueError("Cenário desconhecido")
    frame["source_id"] = raw.get("EndToEndId", pd.Series(raw.index.astype(str))).astype(str)
    frame["occurred_at"] = parse_dates(frame.occurred_at)
    invalid = frame.occurred_at.isna()
    quality = {"total": len(frame), "invalid_dates": int(invalid.sum()),
               "invalid_dates_by_class": {str(k): int(v) for k, v in frame.loc[invalid, "target"].value_counts().items()},
               "invalid_date_source_ids": frame.loc[invalid, "source_id"].tolist()}
    frame = frame.loc[~invalid, ["source_id", "payer_id", "occurred_at", "amount", "target", "completed", *attrs]].copy()
    if frame.payer_id.isna().any() or frame.payer_id.astype(str).str.strip().eq("").any():
        raise ValueError("Pagador ausente: histórico não pode agrupar identidades desconhecidas")
    frame["payer_id"] = frame.payer_id.astype(str)
    frame["amount"] = pd.to_numeric(frame.amount, errors="raise")
    numeric = ["amount"] + (["saldo_anterior_pagador", "saldo_anterior_recebedor"] if scenario == "2m" else [])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Valores monetários ausentes ou não finitos")
    if (frame.amount < 0).any() or not frame.target.isin([0, 1]).all():
        raise ValueError("Valor negativo ou classe diferente de 0/1")
    if frame[CATEGORIES[scenario]].isna().any().any():
        raise ValueError("Categoria obrigatória ausente")
    return frame.sort_values(["occurred_at", "source_id"], kind="stable").reset_index(drop=True), quality


def history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """O(n log n) expanding statistics; tied timestamps all see the same past."""
    ordered = frame.sort_values(["payer_id", "occurred_at", "source_id"], kind="stable").copy()
    group = ordered.groupby("payer_id", sort=False)
    ordered["qtd_transacoes_pagador"] = group.cumcount()
    ordered["media_valor_pagador"] = (group.amount.cumsum() - ordered.amount) / ordered.qtd_transacoes_pagador.replace(0, np.nan)
    ordered["mediana_valor_pagador"] = (
        group.amount.expanding().median().groupby(level=0).shift(1).reset_index(level=0, drop=True)
    )
    names = ["qtd_transacoes_pagador", "media_valor_pagador", "mediana_valor_pagador"]
    # nth(0) preserves a missing first-history value; groupby.first would skip it.
    first = ordered.drop_duplicates(["payer_id", "occurred_at"])[["payer_id", "occurred_at", *names]]
    result = frame[["payer_id", "occurred_at"]].merge(first, on=["payer_id", "occurred_at"], how="left", sort=False)
    result.index = frame.index
    return result[names]


def feature_frame(frame: pd.DataFrame, scenario: Scenario, backend: str = "cpu") -> pd.DataFrame:
    """Vectorized equivalent of feature_row for large offline datasets."""
    if backend == "gpu":
        from src.data.gpu_features import feature_frame as cuda_features
        return cuda_features(frame, scenario)
    local = frame.occurred_at.dt.tz_convert("America/Sao_Paulo")
    result = frame[["amount", *CATEGORIES[scenario]]].copy()
    result["hora"], result["dia_semana"] = local.dt.hour, local.dt.dayofweek
    if scenario == "2m":
        for col in ("saldo_anterior_pagador", "saldo_anterior_recebedor"):
            result[col] = frame[col]
        result["dia_util"] = (local.dt.dayofweek < 5).astype(int)
        result["horario_noturno"] = ((local.dt.hour >= 20) | (local.dt.hour < 6)).astype(int)
        result["acima_limite_noturno"] = (result.horario_noturno.eq(1) & frame.amount.gt(1000)).astype(int)
        result = result.join(history_features(frame))
    else:
        result["dia_mes"] = local.dt.day
        for col in ("pagador_is_cnpj", "recebedor_is_cnpj", "pagador_cpf_zerado"):
            result[col] = frame[col]
        result["mesmo_banco"] = frame.pagador_banco.eq(frame.recebedor_banco).astype(int)
    return result[CATEGORIES[scenario] + NUMERICS[scenario]]


@dataclass(frozen=True)
class TemporalSplit:
    development: pd.Index
    calibration: pd.Index
    test: pd.Index


def temporal_split(frame: pd.DataFrame) -> TemporalSplit:
    """70/10/20 by rows, shifting boundaries forward to keep timestamp groups whole."""
    dates = frame.occurred_at
    if not dates.is_monotonic_increasing or dates.isna().any():
        raise ValueError("A divisão exige datas válidas e ordenadas")
    n = len(frame)
    if n < 10:
        raise ValueError("Dados insuficientes para três períodos")
    a = int(dates.searchsorted(dates.iloc[int(n * .7)], side="left"))
    b = int(dates.searchsorted(dates.iloc[int(n * .8)], side="left"))
    if not 0 < a < b < n:
        raise ValueError("Instantes distintos insuficientes para divisão temporal")
    return TemporalSplit(frame.index[:a], frame.index[a:b], frame.index[b:])


def temporal_folds(frame: pd.DataFrame, folds: int = 3) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding windows based on unique instants; never shuffle or stratify."""
    dates = frame.occurred_at
    unique = dates.drop_duplicates().to_numpy()
    if len(unique) < folds + 1:
        raise ValueError("Instantes insuficientes para validação temporal")
    blocks = np.array_split(unique, folds + 1)
    return [(frame.index[dates < blocks[i][0]], frame.index[dates.isin(blocks[i])])
            for i in range(1, folds + 1)]
