import streamlit as st
import pandas as pd
import plotly.express as px
import json

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="YouTube Dashboard", layout="wide")

st.title("📊 YouTube Trending Dashboard (India)")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    # Read CSV (must be in same folder)
    df = pd.read_csv("INvideos_small.csv")

    # Preprocessing
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    df["publishing_hour"] = df["publish_time"].dt.hour
    df["publishing_day"] = df["publish_time"].dt.day_name()
    df["year"] = df["trending_date"].apply(lambda x: '20' + x[:2])

    df["description"] = df["description"].fillna("")

    # Load category mapping
    with open("/Users/manavdoshi/IN_category_id.json") as f:
        data = json.load(f)

    cat_dict = {int(item["id"]): item["snippet"]["title"] for item in data["items"]}
    df["category_name"] = df["category_id"].map(cat_dict)

    return df  # ✅ correctly inside function

# Load data
df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔍 Filters")

category = st.sidebar.multiselect(
    "Category",
    df["category_name"].dropna().unique(),
    default=df["category_name"].dropna().unique()
)

year = st.sidebar.multiselect(
    "Year",
    df["year"].unique(),
    default=df["year"].unique()
)

filtered_df = df[
    (df["category_name"].isin(category)) &
    (df["year"].isin(year))
]

# ------------------ KPIs ------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Videos", len(filtered_df))
col2.metric("Avg Views", int(filtered_df["views"].mean()))
col3.metric("Avg Likes", int(filtered_df["likes"].mean()))

# ------------------ CHARTS ------------------

# 📈 Videos per Year
st.subheader("📈 Videos per Year")
year_df = filtered_df["year"].value_counts().reset_index()
year_df.columns = ["year", "count"]

fig1 = px.bar(year_df, x="year", y="count")
st.plotly_chart(fig1, use_container_width=True)

# 📊 Category Distribution
st.subheader("📊 Category Distribution")
cat_df = filtered_df["category_name"].value_counts().reset_index()
cat_df.columns = ["category", "count"]

fig2 = px.pie(cat_df, names="category", values="count")
st.plotly_chart(fig2, use_container_width=True)

# ⏰ Publishing Hour
st.subheader("⏰ Publishing Hour")
hour_df = filtered_df["publishing_hour"].value_counts().sort_index().reset_index()
hour_df.columns = ["hour", "count"]

fig3 = px.bar(hour_df, x="hour", y="count")
st.plotly_chart(fig3, use_container_width=True)

# 📅 Publishing Day
st.subheader("📅 Publishing Day")
day_df = filtered_df["publishing_day"].value_counts().reset_index()
day_df.columns = ["day", "count"]

fig4 = px.bar(day_df, x="day", y="count")
st.plotly_chart(fig4, use_container_width=True)

# 🔥 Top Channels
st.subheader("🔥 Top Channels")
top_channels = (
    filtered_df.groupby("channel_title")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)

fig5 = px.bar(top_channels, x="count", y="channel_title", orientation="h")
st.plotly_chart(fig5, use_container_width=True)

# 🔥 Top Videos
st.subheader("🔥 Top 10 Videos")
top_videos = filtered_df.sort_values("views", ascending=False).head(10)

st.dataframe(top_videos[["title", "channel_title", "views", "likes"]])

# 📄 Raw Data
with st.expander("📄 Show Raw Data"):
    st.dataframe(filtered_df)
