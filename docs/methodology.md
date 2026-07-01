# Methodology

## Data Cleaning

- Removed duplicate rows.
- Converted `publish_time` to datetime.
- Parsed `trending_date`.
- Filled missing text values such as descriptions.
- Converted numeric fields such as views, likes, dislikes, and comments.
- Mapped YouTube category IDs to category names.

## Feature Engineering

- `title_length`
- `publish_hour`
- `publish_day`
- `engagement_rate`
- `like_ratio`
- `comment_ratio`
- `high_engagement`

## Modeling

The model classifies high-engagement videos within the trending dataset. It does not predict whether a new upload will trend because the dataset contains only videos that already became trending.

Models:

- Logistic Regression
- Random Forest Classifier

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
