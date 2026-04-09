import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import analyze_review

load_dotenv()

DEFAULT_REVIEWS_PATH = Path("data/reviews.json")
REQUIRED_REVIEW_FIELDS = {"text"}


def load_reviews(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Load real reviews from JSON and skip invalid records."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("reviews.json must be a JSON array of review objects")

    valid_reviews: List[Dict[str, Any]] = []
    skipped = 0

    for record in payload:
        if not isinstance(record, dict):
            skipped += 1
            continue

        if not REQUIRED_REVIEW_FIELDS.issubset(record.keys()):
            skipped += 1
            continue

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue

        normalized = dict(record)
        normalized["platform"] = record.get("platform") or record.get("source") or "Unknown"
        valid_reviews.append(normalized)

    return valid_reviews, skipped


def analyze_reviews(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    analyzed: List[Dict[str, Any]] = []
    for review in reviews:
        result = analyze_review(review["text"])
        enriched = dict(review)
        enriched.update(result)
        analyzed.append(enriched)
    return analyzed


def style_negative_rows(row: pd.Series) -> List[str]:
    if str(row.get("sentiment", "")).lower() == "negative":
        return ["background-color: #ffe6e6"] * len(row)
    return [""] * len(row)


def main() -> None:
    st.set_page_config(page_title="SignalAgents: Brand Image Monitor", layout="wide")
    st.title("SignalAgents: Brand Image Monitor")

    st.caption("Loads real reviews from JSON, analyzes sentiment with OpenAI, and shows simple brand-health signals.")

    reviews_path_input = st.sidebar.text_input("Reviews JSON path", str(DEFAULT_REVIEWS_PATH))
    run_analysis = st.sidebar.button("Run Analysis", type="primary")

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if run_analysis:
        path = Path(reviews_path_input)
        try:
            reviews, skipped = load_reviews(path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load reviews: {exc}")
            st.stop()

        if not reviews:
            st.warning("No valid real reviews found. Add real records to the JSON file and try again.")
            st.stop()

        with st.spinner("Analyzing reviews with OpenAI..."):
            results = analyze_reviews(reviews)

        st.session_state.analysis_results = {
            "rows": results,
            "skipped": skipped,
        }

    state = st.session_state.analysis_results
    if not state:
        st.info("Click Run Analysis to load and analyze real reviews.")
        st.stop()

    rows = state["rows"]
    skipped = state["skipped"]
    df = pd.DataFrame(rows)

    if skipped > 0:
        st.warning(f"Skipped {skipped} invalid review record(s) due to missing/invalid required fields.")

    negative_count = int((df.get("sentiment", pd.Series(dtype=str)).str.lower() == "negative").sum())
    st.metric("Negative Reviews", negative_count)

    sentiment_counts = (
        df.get("sentiment", pd.Series(dtype=str))
        .str.lower()
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    if not sentiment_counts.empty:
        st.subheader("Sentiment Distribution")
        sentiment_chart = sentiment_counts.set_index("sentiment")
        st.bar_chart(sentiment_chart["count"])

    platform_counts = (
        df.get("platform", pd.Series(dtype=str))
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename_axis("platform")
        .reset_index(name="count")
    )
    if not platform_counts.empty:
        st.subheader("Platform Breakdown")
        platform_chart = platform_counts.set_index("platform")
        st.bar_chart(platform_chart["count"])

    st.subheader("Reviews")
    table_columns = [col for col in ["platform", "text", "sentiment", "category", "severity", "confidence", "reason", "error"] if col in df.columns]
    display_df = df[table_columns].copy()

    styled = display_df.style.apply(style_negative_rows, axis=1)
    st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()
