"""CUDA feature engineering with exact segmented prefix medians and no future events."""

import numpy as np
import pandas as pd

from src.data.temporal import CATEGORIES, NUMERICS, Scenario
from src.utils.cuda import initialize

_PREFIX_MEDIAN = r'''
extern "C" __global__ void prefix_history(
    const long long* starts, const long long* times, const double* values,
    const long long* ranks, const double* sorted_values, int* tree,
    long long* counts, double* means, double* medians, long long groups, double missing) {
  long long g = blockDim.x * (long long)blockIdx.x + threadIdx.x;
  if (g >= groups) return;
  long long start = starts[g], end = starts[g + 1], length = end - start;
  int* frequencies = tree + start + g;
  long long count = 0;
  double sum = 0.0;
  for (long long i = start; i < end;) {
    long long j = i + 1;
    while (j < end && times[j] == times[i]) ++j;
    double median = missing;
    if (count > 0) {
      long long targets[2] = {(count + 1) / 2, count / 2 + 1};
      double middle[2];
      for (int k = 0; k < 2; ++k) {
        long long position = 0, remaining = targets[k], bit = 1;
        while ((bit << 1) <= length) bit <<= 1;
        for (; bit > 0; bit >>= 1) {
          long long next = position + bit;
          if (next <= length && frequencies[next] < remaining) {
            remaining -= frequencies[next]; position = next;
          }
        }
        middle[k] = sorted_values[start + position];
      }
      median = (middle[0] + middle[1]) / 2.0;
    }
    // All tied events see the state before this entire timestamp group.
    for (long long row = i; row < j; ++row) {
      counts[row] = count;
      means[row] = count ? sum / count : missing;
      medians[row] = median;
    }
    for (long long row = i; row < j; ++row) {
      for (long long rank = ranks[row]; rank <= length; rank += rank & -rank)
        ++frequencies[rank];
      sum += values[row]; ++count;
    }
    i = j;
  }
}
'''


def history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fenwick order statistics: O(n log n), linear storage, no growing pairwise matrix."""
    import cupy as cp
    import cudf
    initialize()
    if frame.empty:
        return pd.DataFrame(columns=["qtd_transacoes_pagador", "media_valor_pagador", "mediana_valor_pagador"], index=frame.index)
    source = frame[["payer_id", "source_id", "amount"]].copy()
    source["occurred_at"] = frame.occurred_at.dt.tz_convert("UTC").dt.tz_localize(None)
    source["_row"] = np.arange(len(frame))
    ordered = cudf.from_pandas(source).sort_values(["payer_id", "occurred_at", "source_id"])
    payer_codes, _ = ordered.payer_id.factorize(sort=False)
    codes = cp.asarray(payer_codes, dtype=cp.int64)
    starts = cp.r_[cp.flatnonzero(cp.r_[True, codes[1:] != codes[:-1]]), len(frame)].astype(cp.int64)
    groups = len(starts) - 1
    # Use contiguous segment numbers rather than dictionary code ordering.
    group = cp.cumsum(cp.r_[True, codes[1:] != codes[:-1]]) - 1
    values = ordered.amount.astype("float64").to_cupy()
    value_order = cp.lexsort(cp.stack([values, group]))
    ranks = cp.empty(len(frame), dtype=cp.int64)
    ranks[value_order] = cp.arange(len(frame)) - starts[group[value_order]] + 1
    sorted_values = values[value_order]
    counts = cp.empty(len(frame), dtype=cp.int64)
    means, medians = cp.empty(len(frame), dtype=cp.float64), cp.empty(len(frame), dtype=cp.float64)
    frequencies = cp.zeros(len(frame) + groups, dtype=cp.int32)
    times = ordered.occurred_at.astype("int64").to_cupy()
    cp.RawKernel(_PREFIX_MEDIAN, "prefix_history")(((groups + 127) // 128,), (128,),
        (starts, times, values, ranks, sorted_values, frequencies, counts, means, medians, np.int64(groups), np.float64(np.nan)))
    restore = cp.argsort(ordered["_row"].to_cupy())
    return pd.DataFrame({"qtd_transacoes_pagador": cp.asnumpy(counts[restore]),
                         "media_valor_pagador": cp.asnumpy(means[restore]),
                         "mediana_valor_pagador": cp.asnumpy(medians[restore])}, index=frame.index)


def feature_frame(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """CUDA arithmetic/calendar extraction; host frames are only workflow/I/O boundaries."""
    import cupy as cp
    import cudf
    initialize()
    local = cudf.from_pandas(frame.occurred_at.dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None))
    result = frame[["amount", *CATEGORIES[scenario]]].copy()
    result["hora"] = local.dt.hour.to_numpy()
    result["dia_semana"] = local.dt.weekday.to_numpy()
    if scenario == "2m":
        for name in ("saldo_anterior_pagador", "saldo_anterior_recebedor"):
            result[name] = frame[name]
        hour = local.dt.hour.to_cupy()
        night = (hour >= 20) | (hour < 6)
        result["dia_util"] = cp.asnumpy((local.dt.weekday.to_cupy() < 5).astype(cp.int64))
        result["horario_noturno"] = cp.asnumpy(night.astype(cp.int64))
        result["acima_limite_noturno"] = cp.asnumpy((night & (cp.asarray(frame.amount.to_numpy()) > 1000)).astype(cp.int64))
        result = result.join(history_features(frame))
    else:
        result["dia_mes"] = local.dt.day.to_numpy()
        for name in ("pagador_is_cnpj", "recebedor_is_cnpj", "pagador_cpf_zerado"):
            result[name] = frame[name]
        banks = cudf.from_pandas(frame[["pagador_banco", "recebedor_banco"]])
        result["mesmo_banco"] = (banks.pagador_banco == banks.recebedor_banco).astype("int64").to_numpy()
    return result[CATEGORIES[scenario] + NUMERICS[scenario]]
