"""xgboost: scenario-aware facade over the shared experiment factory."""

import pandas as pd
from src.data.temporal import Scenario
from src.models.experiments import Candidate, make_pipeline


def getPipeline(y: pd.Series, scenario: Scenario = "2m", strategy: str = "original",
                adjusted: bool = False, backend: str = "cpu"):
    """Training labels are required so class weights are derived only from this fit set."""
    return make_pipeline(Candidate("xgboost", strategy, adjusted), scenario, y, backend)
