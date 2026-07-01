# Architecture

The project separates dashboard code from reusable data science logic.

## Components

- `app.py`: Streamlit interface, tabs, filters, KPI cards, and charts.
- `src/data_processing.py`: loading, cleaning, category mapping, date conversion, and feature creation.
- `src/feature_engineering.py`: feature-building wrapper for engagement features.
- `src/ml_model.py`: Logistic Regression and Random Forest training/evaluation.
- `src/modeling.py`: modeling wrapper for future extension.
- `src/visualization.py`: reusable Plotly chart helpers.
- `src/utils.py`: formatting utilities.
- `tests/`: pytest checks for data processing, modeling, and app imports.

## Data Flow

1. Load raw CSV and category JSON from `data/`.
2. Clean missing values, dates, duplicates, and numeric columns.
3. Engineer engagement features.
4. Render Streamlit insights.
5. Train and evaluate high-engagement classification models.
