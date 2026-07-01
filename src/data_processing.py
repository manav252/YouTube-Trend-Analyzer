from pathlib import Path
import json

import numpy as np
import pandas as pd


# These paths are built from the project folder, so the app works on any machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VIDEO_PATH = DATA_DIR / "INvideos_small.csv"
DEFAULT_CATEGORY_PATH = DATA_DIR / "IN_category_id.json"


def load_category_mapping(category_path: Path = DEFAULT_CATEGORY_PATH) -> dict[int, str]:
    """Read YouTube category metadata and return {category_id: category_name}."""
    with category_path.open("r", encoding="utf-8") as file:
        category_data = json.load(file)

    return {
        int(item["id"]): item["snippet"]["title"]
        for item in category_data.get("items", [])
    }


def load_raw_data(video_path: Path = DEFAULT_VIDEO_PATH) -> pd.DataFrame:
    """Load the raw trending videos CSV."""
    return pd.read_csv(video_path)


def clean_data(df: pd.DataFrame, category_mapping: dict[int, str]) -> pd.DataFrame:
    """Clean missing values, dates, duplicates, and category names."""
    cleaned_df = df.copy()

    # Remove exact duplicate rows if the dataset contains repeated records.
    cleaned_df = cleaned_df.drop_duplicates()

    # Convert publish_time into a real datetime column for time-based analysis.
    cleaned_df["publish_time"] = pd.to_datetime(
        cleaned_df["publish_time"], errors="coerce", utc=True
    )

    # Keep text columns usable in filters/tables even when the original data is blank.
    text_columns = ["title", "channel_title", "tags", "thumbnail_link", "description"]
    for column in text_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].fillna("")

    # Fill numeric gaps with 0, then keep values non-negative for ratio calculations.
    numeric_columns = ["views", "likes", "dislikes", "comment_count", "category_id"]
    for column in numeric_columns:
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce").fillna(0)
        cleaned_df[column] = cleaned_df[column].clip(lower=0)

    cleaned_df["category_id"] = cleaned_df["category_id"].astype(int)
    cleaned_df["category_name"] = cleaned_df["category_id"].map(category_mapping)
    cleaned_df["category_name"] = cleaned_df["category_name"].fillna("Unknown")

    # The original trending_date format is YY.DD.MM, for example 17.14.11.
    cleaned_df["trending_date"] = pd.to_datetime(
        cleaned_df["trending_date"], format="%y.%d.%m", errors="coerce"
    )
    cleaned_df["year"] = cleaned_df["trending_date"].dt.year.astype("Int64").astype(str)

    return cleaned_df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create portfolio-friendly features for analysis and modeling."""
    featured_df = df.copy()

    featured_df["title_length"] = featured_df["title"].str.len()
    featured_df["publish_hour"] = featured_df["publish_time"].dt.hour.fillna(0).astype(int)
    featured_df["publish_day"] = featured_df["publish_time"].dt.day_name().fillna("Unknown")

    # Avoid division by zero by replacing 0 views with NaN, then filling ratios with 0.
    safe_views = featured_df["views"].replace(0, np.nan)
    featured_df["engagement_rate"] = (
        (featured_df["likes"] + featured_df["comment_count"]) / safe_views
    ).fillna(0)
    featured_df["like_ratio"] = (featured_df["likes"] / safe_views).fillna(0)
    featured_df["comment_ratio"] = (featured_df["comment_count"] / safe_views).fillna(0)

    median_engagement = featured_df["engagement_rate"].median()
    featured_df["high_engagement"] = (
        featured_df["engagement_rate"] > median_engagement
    ).astype(int)

    return featured_df


def load_clean_featured_data() -> pd.DataFrame:
    """Load, clean, and feature-engineer the dataset in one app-friendly call."""
    category_mapping = load_category_mapping()
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df, category_mapping)
    return add_features(cleaned_df)
