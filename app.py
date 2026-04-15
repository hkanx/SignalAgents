import hashlib
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from analyzer import analyze_review
from utils.keyword_diagnostics import build_response_playbook, compute_brand_health_summary, compute_keyword_diagnostics
from utils.reddit_affiliate_filter import score_affiliate_relevance

load_dotenv()

DEFAULT_COMPANY_NAME = "Giftcards.com"
DEFAULT_COMPANY_SYNONYMS = "giftcards.com, bhn, blackhawk network, giftcardmall, CashStar, tango card"
DEFAULT_REDDIT_USER_AGENT = "signalagents-brand-monitor/0.1"

REDDIT_SEARCH_ENDPOINT = "https://www.reddit.com/search.json"
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_RELEVANCE_THRESHOLD = 2.5

COLOR_GOOD = "#2ca02c"
COLOR_BAD = "#d62728"
COLOR_NEUTRAL = "#9e9e9e"
COLOR_INFO = "#1f77b4"


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


def _pretty_reason(reason: str) -> str:
    cleaned = reason.replace("excluded:", "").replace("matched:", "")
    cleaned = cleaned.replace("score=", "").replace("_", " ")
    return cleaned.strip().title()


def _build_response_template(row: pd.Series) -> str:
    category = str(row.get("category", "General Feedback")).strip() or "General Feedback"
    issue_reason = str(row.get("reason", "")).strip()
    issue_reason = issue_reason if issue_reason else "the issue you reported"
    severity = str(row.get("severity", "medium")).strip().lower() or "medium"

    urgency = "high-priority" if severity == "high" else ("time-sensitive" if severity == "medium" else "important")
    return (
        f"Thanks for flagging this. We are sorry about {issue_reason}. "
        f"Our {category} team is reviewing this as a {urgency} case and will share next steps soon."
    )


def _one_based_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = range(1, len(out) + 1)
    return out


def _render_priority_case_queue(df: pd.DataFrame, key_prefix: str, show_draft_workspace: bool) -> None:
    st.markdown("**Priority Case Queue (Highest-Risk Posts)**")
    queue_df = df.copy()
    queue_df["severity_rank"] = queue_df.get("severity", pd.Series(dtype=str)).astype(str).str.lower().map(
        {"high": 0, "medium": 1, "low": 2}
    ).fillna(3)
    queue_df["sentiment_score_num"] = pd.to_numeric(queue_df.get("sentiment_score"), errors="coerce").fillna(0.0)
    queue_df["date_dt"] = pd.to_datetime(queue_df.get("date"), errors="coerce")
    queue_df = queue_df.sort_values(
        by=["severity_rank", "sentiment_score_num", "date_dt"],
        ascending=[True, True, False],
    )
    queue_df = queue_df[queue_df.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower() == "negative"].head(15)
    queue_df["response_template"] = queue_df.apply(_build_response_template, axis=1)

    if queue_df.empty:
        st.info("No negative cases available for queueing.")
        return

    queue_view_df = queue_df.reset_index(drop=True).copy()
    queue_cols = [
        col
        for col in [
            "posted_at",
            "subreddit",
            "title",
            "star_rating",
            "sentiment_score",
            "category",
            "severity",
            "reason",
            "response_template",
            "source_ref",
        ]
        if col in queue_view_df.columns
    ]
    st.dataframe(_one_based_index(queue_view_df[queue_cols]), use_container_width=True)

    if not show_draft_workspace:
        return

    st.markdown("**Copy-Ready Response Draft**")
    queue_view_df["case_label"] = queue_view_df.apply(
        lambda row: (
            f"[{row.get('severity', 'n/a')}] "
            f"{str(row.get('title', 'Untitled'))[:80]} | "
            f"r/{row.get('subreddit', 'unknown')}"
        ),
        axis=1,
    )
    case_idx = st.selectbox(
        "Select case",
        options=queue_view_df.index.tolist(),
        format_func=lambda idx: str(queue_view_df.loc[idx, "case_label"]),
        key=f"{key_prefix}_case_select",
    )
    base_reply = str(queue_view_df.loc[case_idx, "response_template"])
    optional_detail = st.text_input(
        "Optional company-specific detail to append",
        value="",
        placeholder="Example: Please DM your order reference so we can investigate.",
        key=f"{key_prefix}_detail_text",
    ).strip()
    final_reply = base_reply if not optional_detail else f"{base_reply} {optional_detail}"
    st.text_area("Reply draft", value=final_reply, height=140, key=f"{key_prefix}_reply_draft")
    st.code(final_reply)


def fetch_reddit_reviews(
    company_name: str,
    synonyms_raw: str,
    subreddit_name: str,
    limit_per_term: int,
    years_back: int,
    relevance_threshold: float,
    max_pages: int = 5,
) -> Tuple[List[Dict[str, Any]], int, Optional[str], Dict[str, Any]]:
    terms = _build_company_terms(company_name, synonyms_raw)
    if not terms:
        return [], 0, "Provide a company name or at least one synonym", {}

    headers = {
        "User-Agent": os.getenv("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT),
    }

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=365 * years_back)

    records: List[Dict[str, Any]] = []
    skipped = 0
    seen_submission_ids = set()
    fetched_total = 0
    excluded_by_reason: Dict[str, int] = {}
    included_by_reason: Dict[str, int] = {}
    included_scores: List[float] = []
    request_retries = 0
    request_failures = 0
    request_failure_messages: List[str] = []

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

                payload: Optional[Dict[str, Any]] = None
                last_error = ""
                max_attempts = 4
                for attempt in range(1, max_attempts + 1):
                    try:
                        response = requests.get(endpoint, headers=headers, params=params, timeout=20)
                        if response.status_code == 200:
                            payload = response.json()
                            break

                        if response.status_code in {429, 500, 502, 503, 504}:
                            last_error = f"HTTP {response.status_code}"
                        else:
                            last_error = f"HTTP {response.status_code}: {response.text[:160]}"
                            break
                    except requests.exceptions.Timeout:
                        last_error = "timeout"
                    except requests.exceptions.RequestException as exc:
                        last_error = f"request_error: {exc}"

                    if attempt < max_attempts:
                        request_retries += 1
                        time.sleep(1.2 * attempt)

                if payload is None:
                    request_failures += 1
                    failure_reason = "excluded:reddit_request_failed"
                    excluded_by_reason[failure_reason] = excluded_by_reason.get(failure_reason, 0) + 1
                    request_failure_messages.append(
                        f"term={term}, after={after_token or 'start'}, error={last_error}"
                    )
                    # Move to next term if this page repeatedly failed.
                    break

                listing_data = payload.get("data", {})
                children = listing_data.get("children", [])
                if not children:
                    break

                oldest_in_page_in_window = False

                for child in children:
                    fetched_total += 1
                    submission = child.get("data", {})
                    submission_id = str(submission.get("id") or "").strip()
                    if not submission_id or submission_id in seen_submission_ids:
                        continue

                    created_utc = submission.get("created_utc")
                    if created_utc is None:
                        skipped += 1
                        reason = "excluded:missing_created_utc"
                        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
                        continue

                    created_dt = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                    if created_dt < cutoff:
                        reason = "excluded:out_of_window"
                        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
                        continue

                    oldest_in_page_in_window = True
                    seen_submission_ids.add(submission_id)

                    title = str(submission.get("title") or "").strip()
                    body = str(submission.get("selftext") or "").strip()

                    relevance_score, reason = score_affiliate_relevance(title=title, body=body)
                    if relevance_score < relevance_threshold:
                        skipped += 1
                        reason_key = reason if reason.startswith("excluded:") else "excluded:below_threshold"
                        excluded_by_reason[reason_key] = excluded_by_reason.get(reason_key, 0) + 1
                        continue

                    text = f"{title}\n\n{body}".strip()
                    if not text:
                        skipped += 1
                        reason = "excluded:empty_text"
                        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
                        continue

                    permalink = str(submission.get("permalink") or "").strip()
                    source_ref = f"https://reddit.com{permalink}" if permalink else ""
                    included_by_reason[reason] = included_by_reason.get(reason, 0) + 1
                    included_scores.append(float(relevance_score))

                    records.append(
                        {
                            "review_id": f"reddit_{submission_id}",
                            "platform": "reddit",
                            "source": "reddit-public",
                            "source_ref": source_ref,
                            "date": created_dt.isoformat(),
                            "posted_at": created_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                            "title": title,
                            "text": text,
                            "author": str(submission.get("author") or "[deleted]"),
                            "subreddit": str(submission.get("subreddit") or ""),
                            "score": int(submission.get("score") or 0),
                            "matched_term": term,
                            "filter_reason": reason,
                            "relevance_score": round(relevance_score, 2),
                        }
                    )

                # If this page already has no items in our time window, later pages will be older.
                if not oldest_in_page_in_window:
                    break

                after_token = listing_data.get("after")
                if not after_token:
                    break

    except Exception as exc:  # noqa: BLE001
        return [], skipped, f"Reddit public request failed: {exc}", {}

    if not records and request_failures > 0:
        return (
            [],
            skipped,
            "Reddit requests timed out repeatedly. Try fewer pages/terms, or run again in a minute.",
            {
                "request_failures": request_failures,
                "request_retries": request_retries,
                "request_failure_messages": request_failure_messages[:8],
            },
        )

    diagnostics = {
        "fetched_total": fetched_total,
        "included_total": len(records),
        "excluded_total": max(0, fetched_total - len(records)),
        "excluded_by_reason": excluded_by_reason,
        "included_by_reason": included_by_reason,
        "included_relevance_stats": {
            "min": min(included_scores) if included_scores else 0.0,
            "avg": (sum(included_scores) / len(included_scores)) if included_scores else 0.0,
            "max": max(included_scores) if included_scores else 0.0,
        },
        "request_failures": request_failures,
        "request_retries": request_retries,
        "request_failure_messages": request_failure_messages[:8],
    }
    return records, skipped, None, diagnostics


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


def _rank_reviews_for_analysis(reviews: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    sorted_reviews = sorted(
        reviews,
        key=lambda r: (
            float(r.get("relevance_score", 0.0)),
            int(r.get("score", 0)),
            str(r.get("date", "")),
        ),
        reverse=True,
    )
    return sorted_reviews[:top_n]


def style_negative_rows(row: pd.Series) -> List[str]:
    if str(row.get("sentiment", "")).lower() == "negative":
        return ["background-color: #ffe6e6"] * len(row)
    return [""] * len(row)


def main() -> None:
    st.set_page_config(page_title="SignalAgents: Brand Image Monitor", layout="wide")
    st.title("SignalAgents: Brand Image Monitor")

    st.caption("Brand image and sentiment monitoring platform from real user content and web-search data. Surfaces early negative trends, generate insights, and accelerates brand responses.")

    data_source = st.sidebar.selectbox("Data source", ["Reddit", "Web", "Reddit + Web"])

    st.sidebar.markdown("### Company Terms")
    company_name = st.sidebar.text_input("Company name", value=DEFAULT_COMPANY_NAME)
    company_synonyms = st.sidebar.text_input("Company synonyms (comma-separated)", value=DEFAULT_COMPANY_SYNONYMS)

    st.sidebar.markdown("### Reddit Search")
    subreddit_name = st.sidebar.text_input("Subreddit", "all")
    reddit_limit_per_term = st.sidebar.slider("Reddit posts per term", min_value=5, max_value=100, value=20, step=5)
    years_back = st.sidebar.selectbox("Reddit lookback", [1, 2, 3, 4, 5], index=0, format_func=lambda y: f"Last {y} year{'s' if y > 1 else ''}")
    max_pages = st.sidebar.slider("Reddit pages per term", min_value=1, max_value=10, value=5, step=1)
    relevance_threshold = st.sidebar.slider(
        "Affiliate relevance threshold",
        min_value=1.0,
        max_value=4.0,
        value=DEFAULT_RELEVANCE_THRESHOLD,
        step=0.5,
    )
    max_analyzed_reviews = st.sidebar.slider("Max reviews to analyze", min_value=10, max_value=100, value=30, step=5)

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
            reddit_reviews, reddit_skipped, reddit_error, reddit_diagnostics = fetch_reddit_reviews(
                company_name=company_name,
                synonyms_raw=company_synonyms,
                subreddit_name=subreddit_name,
                limit_per_term=reddit_limit_per_term,
                years_back=int(years_back),
                relevance_threshold=float(relevance_threshold),
                max_pages=int(max_pages),
            )
            skipped += reddit_skipped
            if reddit_error:
                st.error(f"Failed to load Reddit data: {reddit_error}")
                st.stop()
            combined_reviews.extend(reddit_reviews)
        else:
            reddit_diagnostics = None

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
        selected_reviews = _rank_reviews_for_analysis(deduped_reviews, top_n=int(max_analyzed_reviews))

        if not selected_reviews:
            st.warning("No valid real reviews found from selected sources. Update inputs and try again.")
            st.stop()

        with st.spinner("Analyzing reviews..."):
            results = analyze_reviews(selected_reviews)

        st.session_state.analysis_results = {
            "rows": results,
            "skipped": skipped,
            "source": data_source,
            "raw_count": len(combined_reviews),
            "deduped_count": len(deduped_reviews),
            "analyzed_count": len(selected_reviews),
            "reddit_diagnostics": reddit_diagnostics,
            "relevance_threshold": float(relevance_threshold),
            "max_analyzed_reviews": int(max_analyzed_reviews),
            "brand_health_summary": compute_brand_health_summary(pd.DataFrame(results)),
            "keyword_diagnostics": compute_keyword_diagnostics(
                pd.DataFrame(results),
                top_n=15,
                min_keyword_mentions=5,
                min_negative_mentions=2,
                risk_lift_threshold=1.6,
            ),
        }

    state = st.session_state.analysis_results
    if not state:
        st.info("Choose a source mode and click Run Analysis.")
        st.stop()

    rows = state["rows"]
    skipped = state["skipped"]
    raw_count = state.get("raw_count", len(rows))
    deduped_count = state.get("deduped_count", len(rows))
    analyzed_count = state.get("analyzed_count", len(rows))
    reddit_diagnostics = state.get("reddit_diagnostics")

    df = pd.DataFrame(rows)
    brand_health_summary = state.get("brand_health_summary") or compute_brand_health_summary(df)
    keyword_diagnostics = state.get("keyword_diagnostics") or compute_keyword_diagnostics(
        df,
        top_n=15,
        min_keyword_mentions=5,
        min_negative_mentions=2,
        risk_lift_threshold=1.6,
    )
    response_playbook = build_response_playbook(
        issue_diagnosis=keyword_diagnostics.get("issue_diagnosis", []),
        brand_health_summary=brand_health_summary,
        subreddit_risk=keyword_diagnostics.get("subreddit_risk", []),
        max_actions=5,
    )

    st.caption(
        f"Collected {raw_count} records, deduped to {deduped_count} records, analyzed top {analyzed_count} records."
    )
    st.caption(f"Current source mode: {state.get('source', 'Reddit')}")

    if skipped > 0:
        st.warning(f"Skipped {skipped} invalid, out-of-window, or irrelevant record(s) from selected source(s).")
    if reddit_diagnostics and int(reddit_diagnostics.get("request_failures", 0)) > 0:
        st.warning(
            f"Reddit had {int(reddit_diagnostics.get('request_failures', 0))} request failure(s); "
            f"retried {int(reddit_diagnostics.get('request_retries', 0))} time(s). Results shown are partial."
        )

    metric_df = df.copy()
    metric_df["sentiment_norm"] = metric_df.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower()
    metric_df["sentiment_score_num"] = pd.to_numeric(metric_df.get("sentiment_score"), errors="coerce").fillna(0.0)
    metric_df["date_dt"] = pd.to_datetime(metric_df.get("date"), errors="coerce", utc=True)

    negative_count = int((metric_df["sentiment_norm"] == "negative").sum())
    total_analyzed = len(metric_df)
    avg_sentiment = float(brand_health_summary["avg_sentiment_score"])
    brand_image_score = max(0, min(100, round((avg_sentiment + 1.0) * 50)))
    negative_delta_pts = brand_health_summary["negative_share_delta"] * 100

    positive_share = float((metric_df["sentiment_norm"] == "positive").mean()) if total_analyzed else 0.0
    negative_share = float((metric_df["sentiment_norm"] == "negative").mean()) if total_analyzed else 0.0
    net_sentiment = positive_share - negative_share

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=30)
    if metric_df["date_dt"].notna().sum() == 0:
        recent_df = metric_df
        baseline_df = metric_df
    else:
        recent_df = metric_df[metric_df["date_dt"] >= cutoff_dt]
        baseline_df = metric_df[metric_df["date_dt"] < cutoff_dt]
        if recent_df.empty:
            recent_df = metric_df
        if baseline_df.empty:
            baseline_df = metric_df

    recent_avg_sent = float(recent_df["sentiment_score_num"].mean()) if not recent_df.empty else 0.0
    baseline_avg_sent = float(baseline_df["sentiment_score_num"].mean()) if not baseline_df.empty else 0.0
    recent_pos_share = float((recent_df["sentiment_norm"] == "positive").mean()) if not recent_df.empty else 0.0
    baseline_pos_share = float((baseline_df["sentiment_norm"] == "positive").mean()) if not baseline_df.empty else 0.0
    recent_neg_share = float((recent_df["sentiment_norm"] == "negative").mean()) if not recent_df.empty else 0.0
    baseline_neg_share = float((baseline_df["sentiment_norm"] == "negative").mean()) if not baseline_df.empty else 0.0
    recent_net_sent = recent_pos_share - recent_neg_share
    baseline_net_sent = baseline_pos_share - baseline_neg_share

    avg_sentiment_improvement = recent_avg_sent - baseline_avg_sent
    positive_share_improvement = recent_pos_share - baseline_pos_share
    negative_share_change = recent_neg_share - baseline_neg_share
    net_sentiment_improvement = recent_net_sent - baseline_net_sent

    if brand_health_summary["negative_share"] >= 0.45 or negative_delta_pts >= 8:
        status_label = "At Risk"
    elif brand_health_summary["negative_share"] >= 0.30 or negative_delta_pts >= 4:
        status_label = "Watch"
    else:
        status_label = "Stable"

    sentiment_counts = (
        metric_df.get("sentiment", pd.Series(dtype=str))
        .str.lower()
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    timeline_df = pd.DataFrame()
    if "date" in metric_df.columns and not metric_df.empty:
        tmp = metric_df.copy()
        tmp = tmp.dropna(subset=["date_dt"])
        if not tmp.empty:
            tmp["period"] = tmp["date_dt"].dt.to_period("M").dt.to_timestamp()
            trend = tmp.groupby(["period", "sentiment_norm"]).size().reset_index(name="count")
            pivot = trend.pivot(index="period", columns="sentiment_norm", values="count").fillna(0).sort_index()
            for col in ["positive", "neutral", "negative"]:
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot[["positive", "neutral", "negative"]]
            timeline_df = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0).reset_index()
            timeline_df.rename(columns={"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}, inplace=True)

    tabs = st.tabs(["Executive Brand Image Summary", "Customer Response Generator", "Data Analytics"])

    with tabs[0]:
        st.subheader("Brand Health Summary")
        if status_label == "At Risk":
            st.error("At Risk: negative sentiment is elevated or worsening quickly. Triage response and root-cause fixes now.")
        elif status_label == "Watch":
            st.warning("Watch: pressure is building. Respond quickly and monitor problem themes daily.")
        else:
            st.success("Stable: current brand sentiment is manageable. Maintain response quality and monitor weekly.")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Brand Image Score", f"{brand_image_score}/100")
        m2.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}", delta=f"{avg_sentiment_improvement:+.2f}", delta_color="normal")
        m3.metric("Positive Mentions %", f"{positive_share * 100:.1f}%", delta=f"{positive_share_improvement * 100:+.1f} pts", delta_color="normal")
        m4.metric("Negative Mentions %", f"{negative_share * 100:.1f}%", delta=f"{negative_share_change * 100:+.1f} pts", delta_color="inverse")
        m5.metric("Net Sentiment Score", f"{net_sentiment:.2f}", delta=f"{net_sentiment_improvement:+.2f}", delta_color="normal")
        if positive_share == 0:
            st.warning("No positive mentions were detected in this run. This can happen when the current query mix is complaint-heavy.")

        st.markdown("**Sentiment Trend Over Time**")
        if not timeline_df.empty:
            trend_long = timeline_df.melt(
                id_vars=["period"],
                value_vars=["Positive", "Neutral", "Negative"],
                var_name="sentiment",
                value_name="share",
            )
            st.vega_lite_chart(
                trend_long,
                {
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {"field": "period", "type": "temporal", "title": "Month"},
                        "y": {"field": "share", "type": "quantitative", "title": "Share"},
                        "color": {
                            "field": "sentiment",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Positive", "Neutral", "Negative"],
                                "range": [COLOR_GOOD, COLOR_NEUTRAL, COLOR_BAD],
                            },
                        },
                        "tooltip": [{"field": "period"}, {"field": "sentiment"}, {"field": "share"}],
                    },
                },
                use_container_width=True,
            )
        else:
            st.info("Not enough timestamped data for trend chart.")

        viz_left, viz_right = st.columns(2)
        with viz_left:
            st.markdown("**Sentiment Distribution**")
            if not sentiment_counts.empty:
                pie_df = sentiment_counts.copy()
                pie_df["sentiment"] = pie_df["sentiment"].str.title()
                st.vega_lite_chart(
                    pie_df,
                    {
                        "mark": {"type": "arc", "innerRadius": 45},
                        "encoding": {
                            "theta": {"field": "count", "type": "quantitative"},
                            "color": {
                                "field": "sentiment",
                                "type": "nominal",
                                "scale": {"domain": ["Positive", "Neutral", "Negative"], "range": [COLOR_GOOD, COLOR_NEUTRAL, COLOR_BAD]},
                            },
                            "tooltip": [{"field": "sentiment"}, {"field": "count"}],
                        },
                    },
                    use_container_width=True,
                )
            else:
                st.info("No sentiment data available.")
        with viz_right:
            st.markdown("**Which Channels/Subreddits To Look Out For**")
            subreddit_risk_df = pd.DataFrame(keyword_diagnostics.get("subreddit_risk", []))
            if not subreddit_risk_df.empty:
                st.dataframe(_one_based_index(subreddit_risk_df.head(10)), use_container_width=True)
                st.vega_lite_chart(
                    subreddit_risk_df.head(10),
                    {
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "subreddit", "type": "nominal", "sort": "-y"},
                            "y": {"field": "negative_rate", "type": "quantitative"},
                            "color": {"value": COLOR_BAD},
                            "tooltip": [{"field": "subreddit"}, {"field": "negative_rate"}, {"field": "volume"}],
                        },
                    },
                    use_container_width=True,
                )
            else:
                st.info("Not enough subreddit volume for risk comparison.")

        st.markdown("**Sentiment By Topic**")
        issue_df = pd.DataFrame(keyword_diagnostics.get("issue_diagnosis", []))
        if not issue_df.empty:
            topic_df = issue_df.copy()
            topic_df["topic_score"] = (1 - (2 * topic_df["negative_rate"])).round(2)
            st.dataframe(
                _one_based_index(topic_df[["issue", "topic_score", "mentions", "negative_rate", "what_to_answer", "what_to_fix"]]),
                use_container_width=True,
            )

            st.markdown("**Volume vs Sentiment: Topic Priority Map**")
            st.vega_lite_chart(
                topic_df,
                {
                    "mark": {"type": "circle", "opacity": 0.8},
                    "encoding": {
                        "x": {"field": "mentions", "type": "quantitative", "title": "Volume (Mentions)"},
                        "y": {"field": "topic_score", "type": "quantitative", "title": "Sentiment Score (-1 to 1)"},
                        "size": {"field": "severity_score", "type": "quantitative"},
                        "color": {
                            "field": "negative_rate",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1], "range": [COLOR_GOOD, COLOR_NEUTRAL, COLOR_BAD]},
                        },
                        "tooltip": [
                            {"field": "issue"},
                            {"field": "mentions"},
                            {"field": "topic_score"},
                            {"field": "negative_rate"},
                            {"field": "what_to_answer"},
                        ],
                    },
                },
                use_container_width=True,
            )
        else:
            st.info("No topic diagnosis available yet.")

        st.markdown("**Actionable Insights**")
        if not issue_df.empty:
            st.markdown("Top Problem Themes (What To Answer / Fix)")
            st.dataframe(
                _one_based_index(
                    issue_df.head(8)[
                        ["issue", "mentions", "negative_rate", "what_to_answer", "what_to_fix", "sample_posts"]
                    ]
                ),
                use_container_width=True,
            )
        if response_playbook:
            st.markdown("Response Playbook")
            playbook_df = pd.DataFrame(response_playbook)
            st.dataframe(
                _one_based_index(
                    playbook_df[
                        [
                            "priority",
                            "issue",
                            "why_now",
                            "what_to_answer",
                            "what_to_fix",
                            "owner_team",
                            "recommended_sla",
                            "watch_channel",
                        ]
                    ]
                ),
                use_container_width=True,
            )

        _render_priority_case_queue(df, key_prefix="exec_queue", show_draft_workspace=False)

        st.subheader("Review Feed")
        filter_col1, filter_col2 = st.columns(2)
        all_sentiments = sorted(metric_df["sentiment_norm"].dropna().astype(str).unique().tolist())
        selected_sentiments = filter_col1.multiselect(
            "Filter by sentiment",
            options=all_sentiments,
            default=all_sentiments,
            key="exec_review_sentiments",
        )
        all_subreddits = sorted(metric_df.get("subreddit", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        selected_subreddits = filter_col2.multiselect(
            "Filter by subreddit",
            options=all_subreddits,
            default=all_subreddits[:20],
            key="exec_review_subreddits",
        )

        filtered_df = metric_df.copy()
        if selected_sentiments:
            filtered_df = filtered_df[filtered_df["sentiment_norm"].isin(selected_sentiments)]
        if selected_subreddits and "subreddit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["subreddit"].astype(str).isin(selected_subreddits)]

        st.caption(f"Showing {len(filtered_df)} of {len(metric_df)} analyzed reviews.")
        table_columns = [
            col
            for col in [
                "platform",
                "source",
                "subreddit",
                "posted_at",
                "date",
                "title",
                "text",
                "sentiment",
                "sentiment_score",
                "star_rating",
                "category",
                "severity",
                "confidence",
                "relevance_score",
                "reason",
                "source_ref",
                "error",
                "filter_reason",
            ]
            if col in filtered_df.columns
        ]
        st.dataframe(_one_based_index(filtered_df[table_columns]).style.apply(style_negative_rows, axis=1), use_container_width=True)

    with tabs[1]:
        st.subheader("Customer Response")
        _render_priority_case_queue(df, key_prefix="response_queue", show_draft_workspace=True)

    with tabs[2]:
        st.subheader("Data Quality And Filter Diagnostics")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Raw Collected", raw_count)
        q2.metric("After Dedupe", deduped_count)
        q3.metric("Analyzed", analyzed_count)
        q4.metric("Skipped", skipped)

        data_quality_df = df.copy()
        missing_text = int(data_quality_df.get("text", pd.Series(dtype=str)).astype(str).str.strip().eq("").sum()) if "text" in data_quality_df.columns else 0
        missing_date = int(pd.to_datetime(data_quality_df.get("date"), errors="coerce").isna().sum()) if "date" in data_quality_df.columns else 0
        dq1, dq2 = st.columns(2)
        dq1.metric("Missing Text In Analyzed", missing_text)
        dq2.metric("Missing/Invalid Date In Analyzed", missing_date)

        if reddit_diagnostics:
            with st.expander("Filter Diagnosis", expanded=True):
                d1, d2, d3, d4 = st.columns(4)
                fetched_total = int(reddit_diagnostics.get("fetched_total", 0))
                included_total = int(reddit_diagnostics.get("included_total", 0))
                excluded_total = int(reddit_diagnostics.get("excluded_total", 0))
                include_rate = (included_total / fetched_total * 100.0) if fetched_total else 0.0
                d1.metric("Fetched", fetched_total)
                d2.metric("Included", included_total)
                d3.metric("Excluded", excluded_total)
                d4.metric("Include Rate", f"{include_rate:.1f}%")
                st.caption(
                    f"Threshold: {state.get('relevance_threshold', DEFAULT_RELEVANCE_THRESHOLD)} | "
                    f"Max analyzed: {state.get('max_analyzed_reviews', 30)}"
                )

                excluded_by_reason = reddit_diagnostics.get("excluded_by_reason", {})
                included_by_reason = reddit_diagnostics.get("included_by_reason", {})

                if excluded_by_reason:
                    ex_df = pd.DataFrame(
                        [
                            {"reason": _pretty_reason(reason), "raw_reason": reason, "count": count}
                            for reason, count in sorted(excluded_by_reason.items(), key=lambda x: x[1], reverse=True)
                        ]
                    )
                    st.markdown("**Excluded By Reason**")
                    st.dataframe(_one_based_index(ex_df), use_container_width=True)
                    st.vega_lite_chart(
                        ex_df,
                        {
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "reason", "type": "nominal", "sort": "-y"},
                                "y": {"field": "count", "type": "quantitative"},
                                "color": {"value": COLOR_INFO},
                            },
                        },
                        use_container_width=True,
                    )

                if included_by_reason:
                    in_df = pd.DataFrame(
                        [
                            {"reason": _pretty_reason(reason), "raw_reason": reason, "count": count}
                            for reason, count in sorted(included_by_reason.items(), key=lambda x: x[1], reverse=True)
                        ]
                    )
                    st.markdown("**Included By Reason**")
                    st.dataframe(_one_based_index(in_df), use_container_width=True)
                    st.vega_lite_chart(
                        in_df,
                        {
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "reason", "type": "nominal", "sort": "-y"},
                                "y": {"field": "count", "type": "quantitative"},
                                "color": {"value": COLOR_INFO},
                            },
                        },
                        use_container_width=True,
                    )

                rel_stats = reddit_diagnostics.get("included_relevance_stats", {})
                rs1, rs2, rs3 = st.columns(3)
                rs1.metric("Min Relevance", f"{float(rel_stats.get('min', 0.0)):.2f}")
                rs2.metric("Avg Relevance", f"{float(rel_stats.get('avg', 0.0)):.2f}")
                rs3.metric("Max Relevance", f"{float(rel_stats.get('max', 0.0)):.2f}")
                if int(reddit_diagnostics.get("request_failures", 0)) > 0:
                    st.caption(
                        f"Request failures: {int(reddit_diagnostics.get('request_failures', 0))} | "
                        f"Retries: {int(reddit_diagnostics.get('request_retries', 0))}"
                    )
                    failure_details = reddit_diagnostics.get("request_failure_messages", [])
                    if failure_details:
                        st.markdown("**Recent Request Failures**")
                        st.dataframe(
                            _one_based_index(
                                pd.DataFrame([{"detail": msg} for msg in failure_details])
                            ),
                            use_container_width=True,
                        )

        with st.expander("Keyword And Risk Diagnostics", expanded=False):
            risk_cfg = keyword_diagnostics.get("risk_thresholds", {})
            if risk_cfg:
                st.caption(
                    "Risk keyword thresholds: "
                    f"min mentions {risk_cfg.get('min_keyword_mentions', 0)}, "
                    f"min negative mentions {risk_cfg.get('min_negative_mentions', 0)}, "
                    f"lift >= {risk_cfg.get('risk_lift_threshold', 0)}"
                )
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Emerging Risk Keywords**")
                risk_df = pd.DataFrame(keyword_diagnostics.get("negative_lift_keywords", []))
                if not risk_df.empty:
                    st.dataframe(_one_based_index(risk_df), use_container_width=True)
                    st.vega_lite_chart(
                        risk_df,
                        {
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "keyword", "type": "nominal", "sort": "-y"},
                                "y": {"field": "negative_lift", "type": "quantitative"},
                                "color": {"value": COLOR_BAD},
                            },
                        },
                        use_container_width=True,
                    )
                else:
                    st.info("No emerging risk keywords found.")

                st.markdown("**Top Negative Keywords**")
                neg_df = pd.DataFrame(keyword_diagnostics.get("top_keywords_by_sentiment", {}).get("negative", []))
                if not neg_df.empty:
                    st.dataframe(_one_based_index(neg_df), use_container_width=True)
                    st.vega_lite_chart(
                        neg_df,
                        {
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "keyword", "type": "nominal", "sort": "-y"},
                                "y": {"field": "count", "type": "quantitative"},
                                "color": {"value": COLOR_BAD},
                            },
                        },
                        use_container_width=True,
                    )
                else:
                    st.info("No negative keyword data available.")

            with col_b:
                st.markdown("**Top Positive Keywords**")
                pos_df = pd.DataFrame(keyword_diagnostics.get("top_keywords_by_sentiment", {}).get("positive", []))
                if not pos_df.empty:
                    st.dataframe(_one_based_index(pos_df), use_container_width=True)
                    st.vega_lite_chart(
                        pos_df,
                        {
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "keyword", "type": "nominal", "sort": "-y"},
                                "y": {"field": "count", "type": "quantitative"},
                                "color": {"value": COLOR_GOOD},
                            },
                        },
                        use_container_width=True,
                    )
                else:
                    st.info("No positive keyword data available.")

                st.markdown("**Subreddit Risk View**")
                subreddit_risk_df = pd.DataFrame(keyword_diagnostics.get("subreddit_risk", []))
                if not subreddit_risk_df.empty:
                    st.dataframe(_one_based_index(subreddit_risk_df), use_container_width=True)
                    st.vega_lite_chart(
                        subreddit_risk_df,
                        {
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "subreddit", "type": "nominal", "sort": "-y"},
                                "y": {"field": "negative_rate", "type": "quantitative"},
                                "color": {"value": COLOR_BAD},
                            },
                        },
                        use_container_width=True,
                    )
                else:
                    st.info("Not enough subreddit volume for risk comparison.")


if __name__ == "__main__":
    main()
