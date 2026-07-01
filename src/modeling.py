import pandas as pd

from src.ml_model import train_engagement_models


def train_models(df: pd.DataFrame) -> dict:
    """Train engagement classification models and return evaluation artifacts."""
    return train_engagement_models(df)
