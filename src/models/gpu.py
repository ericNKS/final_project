"""Native CUDA pipeline: transforms, categorical synthesis, neighbours and estimators."""

import json
import numpy as np
import pandas as pd

from src.data.temporal import CATEGORIES, NUMERICS
from src.utils.cuda import GPUExecutionError, initialize, memory_failure


def verify_backend() -> dict:
    """Probe actual CUDA training before reading datasets; no sklearn accelerator fallback."""
    information = initialize()
    import cupy as cp
    import xgboost as xgb
    from cuml.ensemble import RandomForestClassifier
    from cuml.linear_model import LogisticRegression
    X = cp.asarray([[0.], [1.], [2.], [3.]], dtype=cp.float32)
    y = cp.asarray([0, 0, 1, 1], dtype=cp.int32)
    probe = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1, max_depth=1)
    probe.fit(X, y)
    device = json.loads(probe.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise GPUExecutionError("XGBoost ignorou CUDA; treinamento interrompido, sem fallback para CPU")
    RandomForestClassifier(n_estimators=1, max_depth=1, n_bins=4, output_type="cupy").fit(X, y)
    LogisticRegression(max_iter=50, output_type="cupy").fit(X, y)
    cp.cuda.get_current_stream().synchronize()
    return information


class GPUPreprocessor:
    """Train-only GPU medians/scaling and categorical dictionaries shared with inference."""

    def __init__(self, scenario: str, scale: bool):
        self.scenario, self.scale = scenario, scale

    def fit_transform(self, X: pd.DataFrame):
        import cupy as cp
        import cudf
        initialize()
        numeric = cp.asarray(X[NUMERICS[self.scenario]].to_numpy(dtype=np.float64, na_value=np.nan))
        self.median_ = cp.nanmedian(numeric, axis=0)
        self.median_ = cp.nan_to_num(self.median_, nan=0.)
        numeric = cp.where(cp.isnan(numeric), self.median_, numeric)
        self.mean_ = cp.mean(numeric, axis=0)
        std = cp.std(numeric, axis=0)
        self.scale_ = cp.where(std > 0, std, 1.)
        self.categories_ = []
        categorical = cudf.from_pandas(X[CATEGORIES[self.scenario]])
        for name in CATEGORIES[self.scenario]:
            _, categories = categorical[name].factorize(sort=True)
            self.categories_.append(categories.to_arrow().to_pylist())
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        import cupy as cp
        import cudf
        initialize()
        numeric = cp.asarray(X[NUMERICS[self.scenario]].to_numpy(dtype=np.float64, na_value=np.nan))
        numeric = cp.where(cp.isnan(numeric), self.median_, numeric)
        if self.scale:
            numeric = (numeric - self.mean_) / self.scale_
        categorical = cudf.from_pandas(X[CATEGORIES[self.scenario]])
        codes = [cp.asarray(cudf.Index(categories).get_indexer(categorical[name]), dtype=cp.float32)
                 for name, categories in zip(CATEGORIES[self.scenario], self.categories_)]
        return cp.column_stack([numeric.astype(cp.float32), *codes])

    def encode(self, X):
        """One-hot only after categorical resampling; unknown categories map to all zeros."""
        import cupy as cp
        n = len(NUMERICS[self.scenario])
        return cp.concatenate([X[:, :n]] + [
            (X[:, n + i, None] == cp.arange(len(categories))[None, :]).astype(cp.float32)
            for i, categories in enumerate(self.categories_)], axis=1)


class GPUResampler:
    """Binary under-sampling or SMOTENC semantics, entirely using CUDA arrays."""

    def __init__(self, strategy: str, numeric_count: int):
        self.strategy, self.numeric_count = strategy, numeric_count

    def fit_resample(self, X, y):
        import cupy as cp
        rng = cp.random.RandomState(42)
        counts = cp.bincount(y, minlength=2)
        minority = int(cp.argmin(counts))
        majority = 1 - minority
        self.sampling_strategy_ = {}
        if self.strategy == "under":
            keep = [rng.choice(cp.flatnonzero(y == label), int(counts.min()), replace=False) for label in (0, 1)]
            indices = cp.concatenate(keep)
            self.sample_indices_ = indices
            self.sampling_strategy_ = {label: int(counts.min()) for label in (0, 1)}
            return X[indices], y[indices]
        added = int(counts[majority] - counts[minority])
        self.sampling_strategy_ = {minority: added}
        if int(counts[minority]) < 6:
            raise ValueError("SMOTENC exige ao menos seis exemplos por classe no treino")
        if added == 0:
            return X, y
        minority_values = X[y == minority]
        # SMOTENC distance: categorical mismatch contributes the median numeric std squared.
        numeric = minority_values[:, :self.numeric_count]
        weight = cp.median(cp.std(numeric, axis=0)) / cp.sqrt(2.)
        categories = [cp.unique(minority_values[:, column]) for column in range(self.numeric_count, X.shape[1])]
        encoded = cp.concatenate([numeric] + [
            (minority_values[:, self.numeric_count + i, None] == values[None, :]).astype(cp.float32) * weight
            for i, values in enumerate(categories)], axis=1)
        neighbours = nearest_indices(encoded, k=5)
        generated = cp.empty((added, X.shape[1]), dtype=cp.float32)
        # Synthesis workspaces stay bounded even when expanding millions of rows.
        for start in range(0, added, 32768):
            end = min(start + 32768, added)
            rows = rng.randint(0, len(minority_values), size=end - start)
            choices = rng.randint(0, 5, size=end - start)
            other = neighbours[rows, choices]
            steps = rng.uniform(size=(end - start, 1)).astype(cp.float32)
            generated[start:end, :self.numeric_count] = numeric[rows] + steps * (numeric[other] - numeric[rows])
            for i, values in enumerate(categories):
                column = self.numeric_count + i
                neighbourhood = minority_values[neighbours[rows], column]
                votes = (neighbourhood[:, :, None] == values[None, None, :]).sum(axis=1)
                # Uniform random tie breaking among the most frequent neighbour categories.
                ties = votes == votes.max(axis=1, keepdims=True)
                random_ties = cp.where(ties, rng.uniform(size=ties.shape), -1.)
                generated[start:end, column] = values[cp.argmax(random_ties, axis=1)]
        return cp.concatenate([X, generated]), cp.concatenate([y, cp.full(added, minority, dtype=y.dtype)])


class GPUTomek:
    """Remove majority-class mutual nearest-neighbour links after one-hot encoding."""

    def fit_resample(self, X, y):
        import cupy as cp
        neighbour = nearest_indices(X, k=1)[:, 0]
        rows = cp.arange(len(y))
        # imblearn 'auto' cleans every class except the minority; ties designate class 0 as minority.
        target = 1 if int(cp.count_nonzero(y == 0)) <= int(cp.count_nonzero(y == 1)) else 0
        links = (neighbour[neighbour] == rows) & (y[neighbour] != y) & (y == target)
        self.sample_indices_ = cp.flatnonzero(~links)
        return X[self.sample_indices_], y[self.sample_indices_]


def nearest_indices(X, k: int):
    """Exact CUDA neighbours, excluding self explicitly even when duplicate rows tie."""
    import cupy as cp
    from cuml.neighbors import NearestNeighbors
    estimator = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", output_type="cupy")
    estimator.fit(X)
    result = cp.empty((len(X), k), dtype=cp.int64)
    for start in range(0, len(X), 1024):
        end = min(start + 1024, len(X))
        indices = estimator.kneighbors(X[start:end], return_distance=False)
        self_mask = indices == cp.arange(start, end)[:, None]
        # Move the self index to the end; preserve native distance order for all other neighbours.
        order = cp.argsort(self_mask.astype(cp.int8), axis=1, kind="stable")
        result[start:end] = cp.take_along_axis(indices, order, axis=1)[:, :k]
    return result


class GPUPipeline:
    """One resident CUDA fit path; only public prediction results cross back to host."""

    def __init__(self, candidate, scenario: str):
        self.candidate, self.scenario = candidate, scenario
        self.named_steps = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import cupy as cp
        initialize()
        try:
            return self._fit(X, y)
        except Exception as exc:
            memory_failure(exc, self.candidate.name)
            raise

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
        from cuml.linear_model import LogisticRegression
        import xgboost as xgb
        candidate = self.candidate
        prepare = GPUPreprocessor(self.scenario, candidate.family == "logistic_regression" or candidate.strategy.startswith("smotenc"))
        values, labels = prepare.fit_transform(X), cp.asarray(y.to_numpy(), dtype=cp.int32)
        self.named_steps = {"prepare": prepare}
        if candidate.strategy in {"under", "smotenc", "smotenc_tomek"}:
            sampler = GPUResampler(candidate.strategy, len(NUMERICS[self.scenario]))
            values, labels = sampler.fit_resample(values, labels)
            self.named_steps["sampler"] = sampler
        values = prepare.encode(values)
        if candidate.strategy == "smotenc_tomek":
            tomek = GPUTomek()
            values, labels = tomek.fit_resample(values, labels)
            self.named_steps["tomek"] = tomek
        weighted = candidate.strategy == "class_weight"
        if candidate.family == "logistic_regression":
            classifier = LogisticRegression(C=.1 if candidate.adjusted else 1., max_iter=2000,
                class_weight="balanced" if weighted else None, output_type="cupy")
        elif candidate.family == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, max_depth=12 if candidate.adjusted else None,
                min_samples_leaf=2 if candidate.adjusted else 1, random_state=42,
                n_streams=1, output_type="cupy", class_weight="balanced" if weighted else None)
        else:
            classifier = xgb.XGBClassifier(n_estimators=100, max_depth=6 if candidate.adjusted else 3,
                learning_rate=.1, tree_method="hist", device="cuda", random_state=42,
                scale_pos_weight=float(cp.count_nonzero(labels == 0) / cp.count_nonzero(labels == 1)) if weighted else 1.)
        classifier.fit(values, labels)
        cp.cuda.get_current_stream().synchronize()
        if candidate.family == "xgboost":
            device = json.loads(classifier.get_booster().save_config())["learner"]["generic_param"]["device"]
            if not device.startswith("cuda"):
                raise GPUExecutionError("XGBoost não executou na GPU; não haverá fallback")
        self.named_steps["classifier"] = classifier
        self.classes_ = np.array([0, 1])
        parameters = {}
        for key, value in classifier.get_params().items():
            if key == "handle":
                continue
            if isinstance(value, float) and not np.isfinite(value):
                value = str(value)
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = str(value)
            parameters[key] = value
        self.execution_ = {"prepare": "cupy/cudf CUDA", "encode": "cupy CUDA",
            "resampling": "cupy/cuml CUDA" if "sampler" in self.named_steps else "not_requested",
            "tomek": "cuml/cupy CUDA" if "tomek" in self.named_steps else "not_requested",
            "classifier": "xgboost CUDA" if candidate.family == "xgboost" else "cuml CUDA",
            "classifier_parameters": parameters}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import cupy as cp
        initialize()
        output = np.empty((len(X), 2), dtype=float)
        prepare = self.named_steps["prepare"]
        for start in range(0, len(X), 32768):
            end = min(start + 32768, len(X))
            values = prepare.encode(prepare.transform(X.iloc[start:end]))
            probability = self.named_steps["classifier"].predict_proba(values)
            output[start:end] = cp.asnumpy(probability)
        return output

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)
