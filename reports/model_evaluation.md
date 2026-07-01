# Model Evaluation

## Objective

Classify whether an already-trending YouTube video has high engagement, where `high_engagement = 1` when engagement rate is above the dataset median.

## Models

- Logistic Regression
- Random Forest Classifier

## Metrics

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.986 | 1.000 | 0.971 | 0.985 | 1.000 |
| Random Forest | 0.993 | 0.997 | 0.988 | 0.993 | 1.000 |

## Interpretation

Both models perform strongly because post-publication engagement metrics such as views, likes, and comments are highly informative. This makes the model useful as an engagement classifier, not as a pre-upload trending predictor.

## Limitation

The dataset only includes trending videos. A true trending prediction task would need examples of both trending and non-trending videos.
