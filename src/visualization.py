import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def category_trending_count_chart(df: pd.DataFrame) -> Figure:
    """Build a category-wise trending-count bar chart."""
    category_count = (
        df["category_name"]
        .value_counts()
        .reset_index(name="trending_count")
        .rename(columns={"index": "category_name"})
    )
    return px.bar(
        category_count,
        x="trending_count",
        y="category_name",
        orientation="h",
        labels={"trending_count": "Trending videos", "category_name": "Category"},
    )


def views_distribution_chart(df: pd.DataFrame) -> Figure:
    """Build a log-scale view distribution chart."""
    return px.histogram(
        df,
        x="views",
        nbins=60,
        log_y=True,
        labels={"views": "Views", "count": "Video count"},
    )
