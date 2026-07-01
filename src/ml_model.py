import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "views",
    "likes",
    "comment_count",
    "dislikes",
    "title_length",
    "publish_hour",
    "like_ratio",
    "comment_ratio",
]


def _score_model(model_name: str, y_true, y_pred, y_probability) -> dict:
    """Return a simple metric dictionary for Streamlit display."""
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
    }


def train_engagement_models(df: pd.DataFrame) -> dict:
    """Train Logistic Regression and Random Forest to predict high engagement."""
    model_df = df.dropna(subset=FEATURE_COLUMNS + ["high_engagement"]).copy()

    X = model_df[FEATURE_COLUMNS]
    y = model_df["high_engagement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    random_forest_model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)
    forest_predictions = random_forest_model.predict(X_test)
    logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]
    forest_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        [
            _score_model(
                "Logistic Regression",
                y_test,
                logistic_predictions,
                logistic_probabilities,
            ),
            _score_model(
                "Random Forest",
                y_test,
                forest_predictions,
                forest_probabilities,
            ),
        ]
    )

    confusion = confusion_matrix(y_test, forest_predictions)
    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probabilities)
    forest_fpr, forest_tpr, _ = roc_curve(y_test, forest_probabilities)
    roc_curve_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "false_positive_rate": logistic_fpr,
                    "true_positive_rate": logistic_tpr,
                    "model": "Logistic Regression",
                }
            ),
            pd.DataFrame(
                {
                    "false_positive_rate": forest_fpr,
                    "true_positive_rate": forest_tpr,
                    "model": "Random Forest",
                }
            ),
        ],
        ignore_index=True,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": random_forest_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "roc_curve": roc_curve_data,
        "feature_importance": feature_importance,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
