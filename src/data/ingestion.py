import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def loadData(filepath: str | Path):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise