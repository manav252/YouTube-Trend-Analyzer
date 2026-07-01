from src.data_processing import load_clean_featured_data


def test_clean_featured_data_has_expected_columns():
    df = load_clean_featured_data()

    expected_columns = {
        "category_name",
        "title_length",
        "publish_hour",
        "publish_day",
        "engagement_rate",
        "like_ratio",
        "comment_ratio",
        "high_engagement",
    }

    assert expected_columns.issubset(df.columns)
    assert len(df) > 0


def test_engagement_features_are_non_negative():
    df = load_clean_featured_data()

    assert (df["engagement_rate"] >= 0).all()
    assert (df["like_ratio"] >= 0).all()
    assert (df["comment_ratio"] >= 0).all()
