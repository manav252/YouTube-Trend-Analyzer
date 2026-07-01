# YouTube Trending Video Analytics & Engagement Prediction

## Problem Statement

YouTube trending videos generate large amounts of engagement data, but raw metrics alone do not explain why some videos perform better than others. This project analyzes Indian YouTube trending videos to identify patterns across categories, publishing time, views, likes, comments, and engagement rates. It also includes a simple machine learning section that predicts whether a video is likely to receive high engagement.

The goal is to turn a raw Kaggle-style dataset into a clean, interactive Data Science portfolio project suitable for Master's applications.

## Dataset Source

The project uses the India trending videos dataset from the public YouTube Trending Video Statistics dataset. The included files are:

- `data/INvideos_small.csv`: video-level trending data
- `data/IN_category_id.json`: category ID to category name mapping

## Tech Stack

- Python
- Pandas and NumPy
- Streamlit
- Plotly
- scikit-learn

## Project Workflow

1. Load YouTube trending video data and category metadata.
2. Clean missing values, duplicate rows, dates, and numeric columns.
3. Map `category_id` values to readable category names.
4. Engineer engagement and time-based features.
5. Build interactive Streamlit visualizations.
6. Train Logistic Regression and Random Forest models.
7. Compare model metrics and show feature importance.

## Data Cleaning

The project includes cleaning steps in `src/data_processing.py`:

- Handles missing text values such as descriptions.
- Converts `publish_time` into datetime format.
- Converts `trending_date` into a usable date.
- Maps `category_id` to category names using the JSON metadata.
- Removes duplicate rows.
- Converts numeric fields such as views, likes, dislikes, and comments.

## Feature Engineering

The app creates the following features:

- `title_length`: number of characters in the video title
- `publish_hour`: hour of day when the video was published
- `publish_day`: weekday when the video was published
- `engagement_rate`: `(likes + comment_count) / views`
- `like_ratio`: `likes / views`
- `comment_ratio`: `comment_count / views`
- `high_engagement`: target variable where engagement rate is above the median

## Key Insights

- Entertainment, news, and music categories tend to dominate trending counts.
- Views follow a long-tail pattern, where a small number of videos collect very large audiences.
- Likes and comments are strongly related to views, but engagement ratios help compare videos more fairly across different audience sizes.
- Publishing hour can be used as an additional behavioral feature, but engagement is usually driven more by category and audience response.
- Random Forest provides interpretable feature importance for understanding which metrics contribute most to high engagement.

## ML Approach

The machine learning section predicts `high_engagement`.

Models used:

- Logistic Regression
- Random Forest Classifier

Metrics shown in the app:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Random Forest feature importance

## Repository Structure

```text
.
├── app.py
├── data/
│   ├── IN_category_id.json
│   └── INvideos_small.csv
├── notebooks/
│   └── youtube_trend_analyser.py
├── reports/
├── screenshots/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   └── ml_model.py
├── requirements.txt
└── README.md
```

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/manav252/YouTube-Trend-Analyzer.git
cd YouTube-Trend-Analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Screenshots

Add screenshots of the final Streamlit dashboard in the `screenshots/` folder.

Suggested screenshots:

- Overview dashboard
- Engagement patterns tab
- ML prediction tab
- Data preview tab

## Future Scope

- Add more countries and compare trending behavior across regions.
- Add NLP features from video titles and tags.
- Use time-series analysis for trending patterns over time.
- Deploy the app on Streamlit Community Cloud.
- Add SHAP or permutation importance for more detailed model explainability.
- Build a recommendation-style model for predicting high-performing video categories.

## Portfolio Value

This project demonstrates practical Data Science skills across data cleaning, feature engineering, exploratory analysis, visualization, machine learning, and dashboard development. It is designed to be easy to run, easy to inspect, and suitable for showcasing in a graduate Data Science portfolio.
