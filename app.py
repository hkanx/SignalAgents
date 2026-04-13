import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from analyzer import analyze_review
from utils.reddit_affiliate_filter import is_affiliate_relevant

load_dotenv()

DEFAULT_COMPANY_NAME = "Giftcards.com"
DEFAULT_COMPANY_SYNONYMS = "giftcards.com, bhn, blackhawk network, giftcardmall, CashStar, tango card"
DEFAULT_REDDIT_USER_AGENT = "signalagents-brand-monitor/0.1"

REDDIT_SEARCH_ENDPOINT = "https://www.reddit.com/search.json"
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def _build_company_terms(company_name: str, synonyms_raw: str) -> List[str]:
    terms: List[str] = []
    if company_name.strip():
        terms.append(company_name.strip())

    for item in synonyms_raw.split(","):
        candidate = item.strip()
        if candidate:
            terms.append(candidate)

    deduped: List[str] = []
    seen = set()
    for term in terms:
        lowered = term.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(term)

    return deduped


def _normalize_text_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def _dedupe_reviews(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()

    for record in records:
        source_ref = str(record.get("source_ref") or "").strip().lower()
        title = str(record.get("title") or "").strip().lower()
        text = str(record.get("text") or "")

        text_hash = hashlib.sha256(_normalize_text_for_hash(text).encode("utf-8")).hexdigest() if text else ""
        key = f"{source_ref}|{title}|{text_hash}"

        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped.append(record)

    return deduped


def fetch_reddit_reviews(
    company_name: str,
    synonyms_raw: str,
    subreddit_name: str,
    limit_per_term: int,
    years_back: int,
    max_pages: int = 5,
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    terms = _build_company_terms(company_name, synonyms_raw)
    if not terms:
        return [], 0, "Provide a company name or at least one synonym"

    headers = {
        "User-Agent": os.getenv("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT),
    }

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=365 * years_back)

    records: List[Dict[str, Any]] = []
    skipped = 0
    seen_submission_ids = set()

    try:
        for term in terms:
            after_token: Optional[str] = None

            for _ in range(max_pages):
                if subreddit_name.strip() and subreddit_name.strip().lower() != "all":
                    endpoint = f"https://www.reddit.com/r/{subreddit_name.strip()}/search.json"
                    params = {
                        "q": term,
                        "limit": limit_per_term,
                        "sort": "new",
                        "t": "all",
                        "restrict_sr": 1,
                    }
                else:
                    endpoint = REDDIT_SEARCH_ENDPOINT
                    params = {
                        "q": term,
                        "limit": limit_per_term,
                        "sort": "new",
                        "t": "all",
                    }

                if after_token:
                    params["after"] = after_token

                response = requests.get(endpoint, headers=headers, params=params, timeout=20)
                if response.status_code != 200:
                    return [], skipped, f"Reddit public API error: {response.status_code} {response.text}"

                payload = response.json()
                listing_data = payload.get("data", {})
                children = listing_data.get("children", [])
                if not children:
                    break

                oldest_in_page_in_window = False

                for child in children:
                    submission = child.get("data", {})
                    submission_id = str(submission.get("id") or "").strip()
                    if not submission_id or submission_id in seen_submission_ids:
                        continue

                    created_utc = submission.get("created_utc")
                    if created_utc is None:
                        skipped += 1
                        continue

                    created_dt = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                    if created_dt < cutoff:
                        continue

                    oldest_in_page_in_window = True
                    seen_submission_ids.add(submission_id)

                    title = str(submission.get("title") or "").strip()
                    body = str(submission.get("selftext") or "").strip()

                    relevant, reason = is_affiliate_relevant(title=title, body=body)
                    if not relevant:
                        skipped += 1
                        continue

                    text = f"{title}\n\n{body}".strip()
                    if not text:
                        skipped += 1
                        continue

                    permalink = str(submission.get("permalink") or "").strip()
                    source_ref = f"https://reddit.com{permalink}" if permalink else ""

                    records.append(
                        {
                            "review_id": f"reddit_{submission_id}",
                            "platform": "reddit",
                            "source": "reddit-public",
                            "source_ref": source_ref,
                            "date": created_dt.isoformat(),
                            "title": title,
                            "text": text,
                            "author": str(submission.get("author") or "[deleted]"),
                            "subreddit": str(submission.get("subreddit") or ""),
                            "score": int(submission.get("score") or 0),
                            "matched_term": term,
                            "filter_reason": reason,
                        }
                    )

                # If this page already has no items in our time window, later pages will be older.
                if not oldest_in_page_in_window:
                    break

                after_token = listing_data.get("after")
                if not after_token:
                    break

    except Exception as exc:  # noqa: BLE001
        return [], skipped, f"Reddit public request failed: {exc}"

    return records, skipped, None


def fetch_web_reviews(company_name: str, synonyms_raw: str, count_per_term: int) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    terms = _build_company_terms(company_name, synonyms_raw)
    if not terms:
        return [], 0, "Provide a company name or at least one synonym"

    bing_api_key = os.getenv("BING_API_KEY")
    if not bing_api_key:
        return [], 0, "Missing BING_API_KEY"

    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    records: List[Dict[str, Any]] = []
    skipped = 0

    try:
        for term in terms:
            query = f"{term} reviews complaints pros cons"
            response = requests.get(
                BING_ENDPOINT,
                headers=headers,
                params={"q": query, "count": count_per_term},
                timeout=20,
            )
            if response.status_code != 200:
                return [], skipped, f"Bing API error: {response.status_code} {response.text}"

            payload = response.json()
            for result in payload.get("webPages", {}).get("value", []):
                name = str(result.get("name") or "").strip()
                snippet = str(result.get("snippet") or "").strip()
                url = str(result.get("url") or "").strip()
                if not snippet and not name:
                    skipped += 1
                    continue

                text = f"{name}: {snippet}".strip(": ")
                records.append(
                    {
                        "review_id": f"web_{hashlib.sha1((url + text).encode('utf-8')).hexdigest()}",
                        "platform": "web",
                        "source": "bing",
                        "source_ref": url,
                        "date": datetime.now(timezone.utc).isoformat(),
                        "title": name,
                        "text": text,
                        "matched_term": term,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return [], skipped, f"Web search request failed: {exc}"

    return records, skipped, None


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

    st.caption("Monitor brand image and sentiment from real Reddit and web-search data, then analyze.")

    data_source = st.sidebar.selectbox("Data source", ["Reddit", "Web", "Reddit + Web"])

    st.sidebar.markdown("### Company Terms")
    company_name = st.sidebar.text_input("Company name", value=DEFAULT_COMPANY_NAME)
    company_synonyms = st.sidebar.text_input("Company synonyms (comma-separated)", value=DEFAULT_COMPANY_SYNONYMS)

    st.sidebar.markdown("### Reddit Search")
    subreddit_name = st.sidebar.text_input("Subreddit", "all")
    reddit_limit_per_term = st.sidebar.slider("Reddit posts per term", min_value=5, max_value=100, value=20, step=5)
    years_back = st.sidebar.selectbox("Reddit lookback", [1, 2, 3, 4, 5], index=0, format_func=lambda y: f"Last {y} year{'s' if y > 1 else ''}")
    max_pages = st.sidebar.slider("Reddit pages per term", min_value=1, max_value=10, value=5, step=1)

    if data_source in {"Web", "Reddit + Web"}:
        st.sidebar.markdown("### Web Search (Bing)")
        web_count_per_term = st.sidebar.slider("Web results per term", min_value=3, max_value=20, value=5, step=1)
    else:
        web_count_per_term = 0

    run_analysis = st.sidebar.button("Run Analysis", type="primary")

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if run_analysis:
        combined_reviews: List[Dict[str, Any]] = []
        skipped = 0

        if data_source in {"Reddit", "Reddit + Web"}:
            reddit_reviews, reddit_skipped, reddit_error = fetch_reddit_reviews(
                company_name=company_name,
                synonyms_raw=company_synonyms,
                subreddit_name=subreddit_name,
                limit_per_term=reddit_limit_per_term,
                years_back=int(years_back),
                max_pages=int(max_pages),
            )
            skipped += reddit_skipped
            if reddit_error:
                st.error(f"Failed to load Reddit data: {reddit_error}")
                st.stop()
            combined_reviews.extend(reddit_reviews)

        if data_source in {"Web", "Reddit + Web"}:
            web_reviews, web_skipped, web_error = fetch_web_reviews(
                company_name=company_name,
                synonyms_raw=company_synonyms,
                count_per_term=web_count_per_term,
            )
            skipped += web_skipped
            if web_error:
                st.error(f"Failed to load Web data: {web_error}")
                st.stop()
            combined_reviews.extend(web_reviews)

        deduped_reviews = _dedupe_reviews(combined_reviews)

        if not deduped_reviews:
            st.warning("No valid real reviews found from selected sources. Update inputs and try again.")
            st.stop()

        with st.spinner("Analyzing reviews with OpenAI..."):
            results = analyze_reviews(deduped_reviews)

        st.session_state.analysis_results = {
            "rows": results,
            "skipped": skipped,
            "source": data_source,
            "raw_count": len(combined_reviews),
            "deduped_count": len(deduped_reviews),
        }

    state = st.session_state.analysis_results
    if not state:
        st.info("Choose a source mode and click Run Analysis.")
        st.stop()

    rows = state["rows"]
    skipped = state["skipped"]
    raw_count = state.get("raw_count", len(rows))
    deduped_count = state.get("deduped_count", len(rows))

    df = pd.DataFrame(rows)

    st.caption(f"Collected {raw_count} records, deduped to {deduped_count} records.")

    if skipped > 0:
        st.warning(f"Skipped {skipped} invalid, out-of-window, or irrelevant record(s) from selected source(s).")

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
        st.bar_chart(sentiment_counts.set_index("sentiment")["count"])

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
        st.bar_chart(platform_counts.set_index("platform")["count"])

    st.subheader("Reviews")
    table_columns = [
        col
        for col in [
            "platform",
            "source",
            "subreddit",
            "title",
            "text",
            "sentiment",
            "category",
            "severity",
            "confidence",
            "reason",
            "source_ref",
            "error",
            "filter_reason",
        ]
        if col in df.columns
    ]
    st.dataframe(df[table_columns].style.apply(style_negative_rows, axis=1), use_container_width=True)


if __name__ == "__main__":
    main()
