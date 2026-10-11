from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate, TunedThresholdClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, make_scorer, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline


scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

def getTrainTestSplit(df: DataFrame, testSize: float = 0.2, randomState: int = 42):
    targetCol = "fraude" if "fraude" in df.columns else "Anomalia"
    XDataDf = df.drop([targetCol], axis=1)
    YDataDf = df[targetCol]

    return train_test_split(
        XDataDf, YDataDf, test_size=testSize,
        stratify=YDataDf, random_state=randomState
    )


def _getPipeline(
    sampler,
    classifier,
    numeric_transformer="passthrough",
):
    categoricalFeatures = [
        'tipo_transacao',
    ]

    numericFeatures = [
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

    if numeric_transformer == "passthrough":
        num_transformer = SimpleImputer(strategy="median")
    else:
        num_transformer = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("transformer", numeric_transformer)
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categorical',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                categoricalFeatures
            ),
            (
                'numeric',
                num_transformer,
                numericFeatures
            )
        ]
    )
    
    pipeArray = [("preprocessor", preprocessor)]
    if sampler is not None:
        pipeArray.append(("sampler", sampler))
        
    if classifier is not None:
        pipeArray.append(("classifier", classifier))
    return ImbPipeline(pipeArray)
    
    
def crossValidate(
    df: DataFrame,
    XTrain,
    yTrain,
    sampler=None,
    classifier=xgb.XGBClassifier(),
    tuned=False
):
    rawPipeline = _getPipeline(classifier=classifier, sampler=sampler)
    if tuned:
        rawPipeline = TunedThresholdClassifierCV(rawPipeline, scoring="f1")
    
    return cross_validate(
        rawPipeline,
        XTrain,
        yTrain,
        scoring=scoring,
        cv=5
    )


def evaluateModel(pipeline, XTrain, yTrain, XTest, yTest, XTuning=None, yTuning=None):
    """
    Ajusta o pipeline nos dados de treino (XTrain, yTrain).
    Se XTuning e yTuning forem fornecidos, calibra o limiar ótimo de decisão
    via TunedThresholdClassifierCV(cv='prefit', refit=False) usando a partição de tunagem (20%).
    Calcula métricas completas nos dados de teste (XTest, yTest).
    """
    if XTuning is not None and yTuning is not None and not isinstance(pipeline, TunedThresholdClassifierCV):
        pipeline.fit(XTrain, yTrain)
        print("   -> Calibrando limiar ótimo de decisão na partição de tunagem (20%)...")
        tuned_model = TunedThresholdClassifierCV(pipeline, scoring="f1", cv="prefit", refit=False)
        tuned_model.fit(XTuning, yTuning)
        active_model = tuned_model
        print(f"   -> Limiar ótimo calibrado: {active_model.best_threshold_:.4f}")
    else:
        pipeline.fit(XTrain, yTrain)
        active_model = pipeline

    yPred = active_model.predict(XTest)

    if hasattr(active_model, "predict_proba"):
        yProb = active_model.predict_proba(XTest)[:, 1]
    else:
        yProb = yPred

    try:
        rocAuc = roc_auc_score(yTest, yProb)
    except Exception:
        rocAuc = float("nan")

    try:
        prAuc = average_precision_score(yTest, yProb)
    except Exception:
        prAuc = float("nan")

    cm = confusion_matrix(yTest, yPred)

    return {
        "accuracy": accuracy_score(yTest, yPred),
        "precision": precision_score(yTest, yPred, zero_division=0),
        "recall": recall_score(yTest, yPred, zero_division=0),
        "f1": f1_score(yTest, yPred, zero_division=0),
        "roc_auc": rocAuc,
        "pr_auc": prAuc,
        "confusion_matrix": cm,
        "y_pred": yPred,
        "y_prob": yProb,
        "fitted_model": active_model
    }


# Configurações especializadas de hiperparâmetros por estratégia de balanceamento
BALANCED_CONFIGS = {
    "baseline": {
        "sampler": None,
        "classifier_params": {
            "max_depth": 8,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "scale_pos_weight": 3.0,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }
    },
    "under": {
        "sampler": RandomUnderSampler(sampling_strategy=0.1, random_state=42),
        "classifier_params": {
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 150,
            "scale_pos_weight": 1.0,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }
    },
    "smote": {
        "sampler": SMOTE(random_state=42),
        "classifier_params": {
            "max_depth": 8,
            "learning_rate": 0.1,
            "n_estimators": 80,
            "scale_pos_weight": 1.0,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }
    }
}


def getTunedPipeline(strategy: str = "under", custom_params: dict = None, cv: int = 3, scoring: str = "f1"):
    """
    Retorna o pipeline base com a estratégia de amostragem e hiperparâmetros
    especializados configurados.
    """
    config = BALANCED_CONFIGS.get(strategy, BALANCED_CONFIGS["under"])
    sampler = config["sampler"]
    params = config["classifier_params"].copy()
    if custom_params:
        params.update(custom_params)

    classifier = xgb.XGBClassifier(**params)
    return _getPipeline(sampler=sampler, classifier=classifier)


def tuneHyperparametersOnSample(XSample, ySample, strategy: str = "under", param_candidates: list = None, val_size: float = 0.25):
    """
    Executa busca/comparação de hiperparâmetros na coletânea de 20% para a estratégia informada.
    Utiliza uma divisão interna de treino/validação na coletânea para calibrar o limiar e pontuar F1/PR-AUC.
    Retorna o melhor conjunto de parâmetros e o pipeline correspondente.
    """
    print(f"\n[Tunagem] Testando configurações de hiperparâmetros para estratégia '{strategy}' na coletânea de 20%...")
    config = BALANCED_CONFIGS.get(strategy, BALANCED_CONFIGS["under"])
    sampler = config["sampler"]
    base_params = config["classifier_params"].copy()

    if param_candidates is None:
        if strategy == "under":
            param_candidates = [
                {"max_depth": 3, "n_estimators": 100, "colsample_bytree": 0.8},
                {"max_depth": 4, "n_estimators": 150, "colsample_bytree": 0.8},
                {"max_depth": 6, "n_estimators": 100, "colsample_bytree": 0.9},
            ]
        elif strategy == "baseline":
            param_candidates = [
                {"scale_pos_weight": 2.0, "max_depth": 6},
                {"scale_pos_weight": 3.0, "max_depth": 8},
                {"scale_pos_weight": 5.0, "max_depth": 8},
            ]
        elif strategy == "smote":
            param_candidates = [
                {"max_depth": 6, "min_child_weight": 2, "subsample": 0.8},
                {"max_depth": 8, "min_child_weight": 3, "subsample": 0.8},
                {"max_depth": 8, "min_child_weight": 5, "subsample": 0.9},
            ]

    # Divisão interna estratificada na amostra de 20%
    XSubTrain, XSubVal, ySubTrain, ySubVal = train_test_split(
        XSample, ySample, test_size=val_size, stratify=ySample, random_state=42
    )

    best_score = -1.0
    best_params = base_params
    best_pipeline = None

    for i, candidate in enumerate(param_candidates):
        current_params = base_params.copy()
        current_params.update(candidate)
        clf = xgb.XGBClassifier(**current_params)
        raw_pipe = _getPipeline(sampler=sampler, classifier=clf)

        raw_pipe.fit(XSubTrain, ySubTrain)
        y_prob = raw_pipe.predict_proba(XSubVal)[:, 1]
        val_pr_auc = average_precision_score(ySubVal, y_prob)

        # Calibra limiar ótimo no split de validação da amostra
        tuned_sub = TunedThresholdClassifierCV(raw_pipe, scoring="f1", cv="prefit", refit=False)
        tuned_sub.fit(XSubVal, ySubVal)
        val_f1 = f1_score(ySubVal, tuned_sub.predict(XSubVal), zero_division=0)
        threshold = tuned_sub.best_threshold_

        print(f"  Candidato {i+1}/{len(param_candidates)}: {candidate} -> F1 Val: {val_f1*100:.2f}%, PR-AUC Val: {val_pr_auc*100:.2f}%, Limiar: {threshold:.4f}")

        # Otimiza balanceando PR-AUC e F1
        combined_score = 0.5 * val_f1 + 0.5 * val_pr_auc
        if combined_score > best_score:
            best_score = combined_score
            best_params = current_params
            best_pipeline = raw_pipe

    print(f"Melhor configuração para '{strategy}': {best_params}")
    return best_params, best_pipeline
