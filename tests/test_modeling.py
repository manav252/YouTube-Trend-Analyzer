from src.data_processing import load_clean_featured_data
from src.ml_model import train_engagement_models


def test_model_training_returns_required_outputs():
    df = load_clean_featured_data()
    results = train_engagement_models(df)

    assert {"metrics", "confusion_matrix", "roc_curve", "feature_importance"}.issubset(
        results.keys()
    )
    assert "roc_auc" in results["metrics"].columns
    assert results["confusion_matrix"].shape == (2, 2)
    assert len(results["feature_importance"]) > 0
