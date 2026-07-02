# YouTube Trending Video Analytics Insights

## Executive Summary

This project analyzes Indian YouTube trending videos to understand category performance, engagement quality, publishing behavior, and channel-level patterns. The dashboard also includes a machine learning section that classifies high-engagement videos within the already-trending dataset.

## Important Modeling Limitation

The dataset contains videos that already became trending. Because of that, this project does not claim to predict whether a brand-new video will trend. The model predicts whether an already-trending video has above-median engagement based on available video performance features.

## Data Quality Notes

- `publish_time` is converted into datetime format.
- `category_id` is mapped to readable category names using the category metadata JSON.
- Missing text values are handled before analysis.
- Duplicate records are removed when present.
- Engagement ratios are protected against division-by-zero issues.

## Key Insights

- Trending counts are concentrated in a small number of high-volume categories.
- Views follow a long-tail distribution, so log-scale visualization is more informative than a raw linear scale.
- Engagement rate, like ratio, and comment ratio help compare videos more fairly than raw likes or comments alone.
- Publish hour and publish day show useful behavioral patterns, but category and audience response are stronger engagement signals.
- Channel-level aggregation helps identify creators or publishers that repeatedly appear in trending data.

## Model Insights

- Logistic Regression and Random Forest are trained on engineered engagement and publishing features.
- Metrics include accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and feature importance.
- Strong results are expected because post-publication metrics such as views, likes, and comments are highly predictive of engagement classification.
- For a true pre-upload prediction model, the dataset would need non-trending videos and early-stage features such as title text, tags, channel history, and first-hour engagement.

## Business / Product Impact

- Content teams can compare categories and publishing windows to understand where engagement is strongest.
- Marketing teams can identify high-performing content types and channels.
- Analysts can use the Power BI-ready cleaned CSV and DAX plan as a starting point for business intelligence reporting.

## Future Improvements

- Add non-trending video examples for true trending prediction.
- Add NLP features from titles and tags.
- Add channel-level historical features.
- Add cross-country comparisons.
- Deploy the dashboard publicly for easier portfolio review.
