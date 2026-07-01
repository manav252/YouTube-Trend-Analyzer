import pandas as pd

from src.data_processing import add_features


def build_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engagement and publishing-time features for analysis/modeling."""
    return add_features(df)
