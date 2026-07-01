import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import load_clean_featured_data
from src.ml_model import train_engagement_models


MODEL_CACHE_VERSION = "engagement-model-v2"


st.set_page_config(
    page_title="YouTube Trending Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    """Load cleaned and feature-engineered YouTube trending data."""
    return load_clean_featured_data()


@st.cache_data
def get_model_results(df: pd.DataFrame, cache_version: str) -> dict:
    """Train engagement prediction models once and reuse the results."""
    return train_engagement_models(df)


def format_number(value: float) -> str:
    """Show large dashboard numbers in a readable way."""
    if pd.isna(value):
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def format_metric_columns(metric_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Format model metrics without crashing on older cached result shapes."""
    formatted_df = metric_df.copy()
    for column in columns:
        if column in formatted_df.columns:
            formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.3f}")
    return formatted_df


df = get_data()

st.title("YouTube Trending Video Analytics & Engagement Classification")
st.caption(
    "An interactive Data Science portfolio project for analyzing Indian YouTube "
    "trending videos and classifying high-engagement content."
)
st.info(
    "Model note: this dataset contains videos that already became trending. "
    "The ML section classifies high vs lower engagement within trending videos; "
    "it does not predict whether a brand-new video will become trending."
)

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header("Filters")

selected_categories = st.sidebar.multiselect(
    "Category",
    options=sorted(df["category_name"].dropna().unique()),
    default=sorted(df["category_name"].dropna().unique()),
)

selected_years = st.sidebar.multiselect(
    "Trending year",
    options=sorted(df["year"].dropna().unique()),
    default=sorted(df["year"].dropna().unique()),
)

min_views, max_views = int(df["views"].min()), int(df["views"].max())
selected_view_range = st.sidebar.slider(
    "Views range",
    min_value=min_views,
    max_value=max_views,
    value=(min_views, max_views),
)

filtered_df = df[
    (df["category_name"].isin(selected_categories))
    & (df["year"].isin(selected_years))
    & (df["views"].between(selected_view_range[0], selected_view_range[1]))
].copy()

if filtered_df.empty:
    st.warning("No videos match the selected filters. Adjust the sidebar filters.")
    st.stop()

# ------------------ KPIS ------------------
total_videos = len(filtered_df)
average_views = filtered_df["views"].mean()
average_engagement = filtered_df["engagement_rate"].mean() * 100
unique_channels = filtered_df["channel_title"].nunique()

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Trending Videos", format_number(total_videos))
kpi_2.metric("Average Views", format_number(average_views))
kpi_3.metric("Avg Engagement Rate", f"{average_engagement:.2f}%")
kpi_4.metric("Unique Channels", format_number(unique_channels))

overview_tab, behavior_tab, engagement_insights_tab, ml_tab, ml_insights_tab, data_tab = st.tabs(
    [
        "Overview",
        "Engagement Patterns",
        "Engagement Insights",
        "ML Classification",
        "ML Model Insights",
        "Data Preview",
    ]
)

with overview_tab:
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Category-wise Trending Count")
        category_count = (
            filtered_df["category_name"]
            .value_counts()
            .reset_index(name="trending_count")
            .rename(columns={"index": "category_name"})
        )
        fig_category = px.bar(
            category_count,
            x="trending_count",
            y="category_name",
            orientation="h",
            color="trending_count",
            color_continuous_scale="Blues",
            labels={
                "trending_count": "Trending videos",
                "category_name": "Category",
            },
        )
        fig_category.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_category, use_container_width=True)

    with right_col:
        st.subheader("Views Distribution")
        fig_views = px.histogram(
            filtered_df,
            x="views",
            nbins=60,
            log_y=True,
            labels={"views": "Views", "count": "Video count"},
            title="Log-scaled distribution highlights the long-tail of viral videos",
        )
        st.plotly_chart(fig_views, use_container_width=True)

    st.subheader("Top Channels by Trending Count")
    top_channels = (
        filtered_df.groupby("channel_title", as_index=False)
        .agg(trending_count=("video_id", "count"), avg_views=("views", "mean"))
        .sort_values("trending_count", ascending=False)
        .head(15)
    )
    fig_channels = px.bar(
        top_channels,
        x="trending_count",
        y="channel_title",
        orientation="h",
        color="avg_views",
        color_continuous_scale="Viridis",
        labels={
            "trending_count": "Trending videos",
            "channel_title": "Channel",
            "avg_views": "Average views",
        },
    )
    fig_channels.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_channels, use_container_width=True)

with behavior_tab:
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Engagement by Category")
        category_engagement = (
            filtered_df.groupby("category_name", as_index=False)
            .agg(
                avg_engagement_rate=("engagement_rate", "mean"),
                avg_like_ratio=("like_ratio", "mean"),
                avg_comment_ratio=("comment_ratio", "mean"),
            )
            .sort_values("avg_engagement_rate", ascending=False)
        )
        fig_engagement = px.bar(
            category_engagement,
            x="avg_engagement_rate",
            y="category_name",
            orientation="h",
            color="avg_engagement_rate",
            color_continuous_scale="Teal",
            labels={
                "avg_engagement_rate": "Average engagement rate",
                "category_name": "Category",
            },
        )
        fig_engagement.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_engagement, use_container_width=True)

    with right_col:
        st.subheader("Publish Hour vs Views and Engagement")
        hourly_performance = (
            filtered_df.groupby("publish_hour", as_index=False)
            .agg(avg_views=("views", "mean"), avg_engagement=("engagement_rate", "mean"))
        )
        fig_hour = px.scatter(
            hourly_performance,
            x="publish_hour",
            y="avg_views",
            size="avg_engagement",
            color="avg_engagement",
            color_continuous_scale="Plasma",
            labels={
                "publish_hour": "Publish hour",
                "avg_views": "Average views",
                "avg_engagement": "Average engagement rate",
            },
        )
        fig_hour.update_xaxes(dtick=1)
        st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Correlation Heatmap")
    correlation_columns = [
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "title_length",
        "publish_hour",
        "engagement_rate",
        "like_ratio",
        "comment_ratio",
    ]
    corr = filtered_df[correlation_columns].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with engagement_insights_tab:
    st.subheader("Engagement Insights")
    st.write(
        "This section compares engagement quality across categories and publishing "
        "times, using ratios instead of only raw views."
    )

    category_ratio_summary = (
        filtered_df.groupby("category_name", as_index=False)
        .agg(
            engagement_rate=("engagement_rate", "mean"),
            like_ratio=("like_ratio", "mean"),
            comment_ratio=("comment_ratio", "mean"),
            trending_videos=("video_id", "count"),
        )
        .sort_values("engagement_rate", ascending=False)
    )

    insight_col_1, insight_col_2 = st.columns(2)

    with insight_col_1:
        st.subheader("Engagement Rate by Category")
        fig_engagement_rate_category = px.bar(
            category_ratio_summary,
            x="engagement_rate",
            y="category_name",
            orientation="h",
            color="engagement_rate",
            color_continuous_scale="Teal",
            hover_data=["trending_videos"],
            labels={
                "engagement_rate": "Avg engagement rate",
                "category_name": "Category",
                "trending_videos": "Trending videos",
            },
        )
        fig_engagement_rate_category.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig_engagement_rate_category, use_container_width=True)

    with insight_col_2:
        st.subheader("Like Ratio by Category")
        fig_like_ratio_category = px.bar(
            category_ratio_summary.sort_values("like_ratio", ascending=False),
            x="like_ratio",
            y="category_name",
            orientation="h",
            color="like_ratio",
            color_continuous_scale="Blues",
            hover_data=["trending_videos"],
            labels={
                "like_ratio": "Avg like ratio",
                "category_name": "Category",
                "trending_videos": "Trending videos",
            },
        )
        fig_like_ratio_category.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_like_ratio_category, use_container_width=True)

    insight_col_3, insight_col_4 = st.columns(2)

    with insight_col_3:
        st.subheader("Comment Ratio by Category")
        fig_comment_ratio_category = px.bar(
            category_ratio_summary.sort_values("comment_ratio", ascending=False),
            x="comment_ratio",
            y="category_name",
            orientation="h",
            color="comment_ratio",
            color_continuous_scale="Purples",
            hover_data=["trending_videos"],
            labels={
                "comment_ratio": "Avg comment ratio",
                "category_name": "Category",
                "trending_videos": "Trending videos",
            },
        )
        fig_comment_ratio_category.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig_comment_ratio_category, use_container_width=True)

    with insight_col_4:
        st.subheader("Publish Hour vs Engagement Rate")
        hourly_engagement = (
            filtered_df.groupby("publish_hour", as_index=False)
            .agg(
                engagement_rate=("engagement_rate", "mean"),
                trending_videos=("video_id", "count"),
            )
            .sort_values("publish_hour")
        )
        fig_hour_engagement = px.line(
            hourly_engagement,
            x="publish_hour",
            y="engagement_rate",
            markers=True,
            labels={
                "publish_hour": "Publish hour",
                "engagement_rate": "Avg engagement rate",
            },
            hover_data=["trending_videos"],
        )
        fig_hour_engagement.update_xaxes(dtick=1)
        st.plotly_chart(fig_hour_engagement, use_container_width=True)

    st.subheader("Publish Day vs Engagement Rate")
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Unknown",
    ]
    daily_engagement = (
        filtered_df.groupby("publish_day", as_index=False)
        .agg(
            engagement_rate=("engagement_rate", "mean"),
            trending_videos=("video_id", "count"),
        )
        .sort_values("engagement_rate", ascending=False)
    )
    fig_day_engagement = px.bar(
        daily_engagement,
        x="publish_day",
        y="engagement_rate",
        color="engagement_rate",
        color_continuous_scale="Viridis",
        category_orders={"publish_day": day_order},
        labels={
            "publish_day": "Publish day",
            "engagement_rate": "Avg engagement rate",
        },
        hover_data=["trending_videos"],
    )
    st.plotly_chart(fig_day_engagement, use_container_width=True)

with ml_tab:
    st.subheader("High Engagement Classification")
    st.write(
        "The target variable is `high_engagement`, where a video is labeled 1 when "
        "its engagement rate is above the dataset median."
    )
    st.warning(
        "This is not a pre-publish trend predictor. Features such as views, likes, "
        "and comments are available only after a video receives audience activity, "
        "so the model is best understood as an engagement analysis/classification model."
    )

    model_results = get_model_results(df, MODEL_CACHE_VERSION)

    metric_table = model_results["metrics"].copy()
    metric_table = format_metric_columns(
        metric_table, ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    )

    st.dataframe(metric_table, use_container_width=True, hide_index=True)
    st.caption(
        f"Training rows: {model_results['train_rows']:,} | "
        f"Testing rows: {model_results['test_rows']:,}"
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Random Forest Confusion Matrix")
        confusion_df = pd.DataFrame(
            model_results["confusion_matrix"],
            index=["Actual Low", "Actual High"],
            columns=["Predicted Low", "Predicted High"],
        )
        fig_confusion = px.imshow(
            confusion_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
        )
        st.plotly_chart(fig_confusion, use_container_width=True)

    with right_col:
        st.subheader("Random Forest Feature Importance")
        fig_importance = px.bar(
            model_results["feature_importance"],
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Greens",
        )
        fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_importance, use_container_width=True)

with ml_insights_tab:
    st.subheader("ML Model Insights")
    st.write(
        "This section compares baseline and ensemble models for predicting whether "
        "a video has above-median engagement."
    )
    st.info(
        "Because the dataset includes only trending videos, ROC-AUC and the other "
        "metrics measure high-engagement classification inside the trending dataset, "
        "not real-world trending prediction for new uploads."
    )

    model_results = get_model_results(df, MODEL_CACHE_VERSION)
    model_comparison = model_results["metrics"].copy()
    model_comparison = model_comparison.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
            "roc_auc": "ROC-AUC",
        }
    )
    model_comparison = format_metric_columns(
        model_comparison, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    st.subheader("Logistic Regression vs Random Forest")
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    st.caption(
        f"Training rows: {model_results['train_rows']:,} | "
        f"Testing rows: {model_results['test_rows']:,}"
    )

    ml_col_1, ml_col_2 = st.columns(2)

    with ml_col_1:
        st.subheader("Confusion Matrix")
        confusion_df = pd.DataFrame(
            model_results["confusion_matrix"],
            index=["Actual Low", "Actual High"],
            columns=["Predicted Low", "Predicted High"],
        )
        fig_ml_confusion = px.imshow(
            confusion_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            labels={"color": "Videos"},
        )
        st.plotly_chart(fig_ml_confusion, use_container_width=True)

    with ml_col_2:
        st.subheader("ROC Curve")
        fig_roc = px.line(
            model_results["roc_curve"],
            x="false_positive_rate",
            y="true_positive_rate",
            color="model",
            labels={
                "false_positive_rate": "False positive rate",
                "true_positive_rate": "True positive rate",
                "model": "Model",
            },
        )
        fig_roc.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash", "color": "gray"},
        )
        fig_roc.update_layout(legend_title_text="Model")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Random Forest Feature Importance")
    fig_ml_importance = px.bar(
        model_results["feature_importance"],
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Greens",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig_ml_importance.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_ml_importance, use_container_width=True)

with data_tab:
    st.subheader("Top Trending Videos")
    top_videos = filtered_df.sort_values("views", ascending=False).head(20)
    st.dataframe(
        top_videos[
            [
                "title",
                "channel_title",
                "category_name",
                "views",
                "likes",
                "comment_count",
                "engagement_rate",
                "publish_day",
                "publish_hour",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show cleaned dataset"):
        st.dataframe(filtered_df, use_container_width=True)
