import pandas as pd


def format_number(value: float) -> str:
    """Format large numbers for dashboard KPI cards."""
    if pd.isna(value):
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"
