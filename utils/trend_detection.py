import pandas as pd


def detect_negative_spikes(
    df: pd.DataFrame,
    window: int = 7,
    multiplier: float = 1.5,
    min_reviews: int = 5,
) -> pd.DataFrame:
    """Detect periods where negative rate spikes above a rolling baseline."""
    if df.empty or "date" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["date", "negative_rate", "baseline", "is_spike"])

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work = work.dropna(subset=["date"])

    if work.empty:
        return pd.DataFrame(columns=["date", "negative_rate", "baseline", "is_spike"])

    daily = (
        work.groupby("date")
        .agg(
            total_reviews=("sentiment", "count"),
            negative_reviews=("sentiment", lambda s: (s.astype(str).str.lower() == "negative").sum()),
        )
        .reset_index()
    )

    daily["negative_rate"] = daily["negative_reviews"] / daily["total_reviews"]
    daily = daily.sort_values("date")
    daily["baseline"] = daily["negative_rate"].rolling(window=window, min_periods=1).mean().shift(1)
    daily["baseline"] = daily["baseline"].fillna(daily["negative_rate"].expanding().mean())

    threshold = daily["baseline"] * multiplier
    daily["is_spike"] = (daily["total_reviews"] >= min_reviews) & (daily["negative_rate"] > threshold)

    return daily[["date", "negative_rate", "baseline", "is_spike"]]
