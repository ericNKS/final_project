from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate, TunedThresholdClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
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
}

def getTrainTestSplit(df: DataFrame, testSize: float = 0.2, randomState: int = 42):
    XDataDf = df.drop(["Anomalia"], axis=1)
    YDataDf = df["Anomalia"]

    return train_test_split(
        XDataDf, YDataDf, test_size=0.2,
        stratify=YDataDf, random_state=42
    )


def _getPipeline(
    sampler,
    classifier,
    numeric_transformer="passthrough",
):
    categoricalFeatures = [
        'TipoChave',
        'Status',
        'Pagador_Banco',
        'Recebedor_Banco'
    ]

    numericFeatures = [
        'Valor',
        'Hora',
        'Dia_Da_Semana',
        'Dia_Do_Mes',
        'Pagador_Is_CNPJ',
        'Recebedor_Is_CNPJ',
        'Pagador_CPF_Zerado',
        'Mesmo_Banco'
    ]

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
                numeric_transformer,
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
    classifier=xgb.XGBClassifier()
):
    rawPipeline = _getPipeline(classifier=classifier, sampler=sampler)

    return cross_validate(
        rawPipeline,
        XTrain,
        yTrain,
        scoring=scoring,
        cv=5
    )
