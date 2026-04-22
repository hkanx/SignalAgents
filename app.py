import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from analyzer import analyze_review
from utils.keyword_diagnostics import build_response_playbook, compute_brand_health_summary, compute_keyword_diagnostics
# from utils.opensearch_kb import build_opensearch_client, search_kb
from utils.reddit_affiliate_filter import score_affiliate_relevance
from utils.response_generator import generate_kb_response
from utils.synthetic_kb import lookup_synthetic_kb_hits
from triage_agent import (
   fetch_reddit_post,
   analyze_complaint,
   decide_triage_action,
   compute_criticality_score,
   CRITICALITY_THRESHOLD,
)
from utils.jira_client import build_form_data, fill_intake_form

load_dotenv()

logger = logging.getLogger(__name__)

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

SENTIMENT_SCORE_THRESHOLD = 0.6


# Company comparison palette
COLOR_COMPANY_A = "#1f77b4"   # blue
COLOR_COMPANY_B = "#ff7f0e"   # orange


#EXCLUDED_BRANDS = {"rakuten"}
EXCLUDED_BRANDS = {"n/a"}

def _extract_comparison_kpis(bhs: Dict[str, float], rows_df: pd.DataFrame) -> Dict[str, Any]:
   """Extract standard KPIs from a brand health summary and DataFrame for comparison views."""
   avg_sent = bhs.get("avg_sentiment_score", 0.0)
   brand_score = max(0, min(100, round((avg_sent + 1.0) * 50)))
   total = len(rows_df)
   sent_norm = rows_df.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower()
   pos_share = float((sent_norm == "positive").mean()) if total else 0.0
   neg_share = float((sent_norm == "negative").mean()) if total else 0.0
   return {
       "brand_image_score": brand_score,
       "avg_sentiment": avg_sent,
       "positive_share": pos_share,
       "negative_share": neg_share,
       "net_sentiment": pos_share - neg_share,
       "total_reviews": total,
   }

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
    # issue_reason = issue_reason if issue_reason else "the issue you reported"
    severity = str(row.get("severity", "medium")).strip().lower() or "medium"
    sentiment = str(row.get("sentiment", "negative")).strip().lower()

    if sentiment == "positive":
        issue_reason = issue_reason if issue_reason else "your positive experience"
        return (
            f"Thank you for sharing your experience! We're glad to hear about {issue_reason}. "
            f"Our {category} team appreciates your kind words — feedback like yours keeps us going!"
        )

    issue_reason = issue_reason if issue_reason else "the issue you reported"



    urgency = "high-priority" if severity == "high" else ("time-sensitive" if severity == "medium" else "important")
    return (
        f"Thanks for flagging this. We are sorry about {issue_reason}. "
        f"Our {category} team is reviewing this as a {urgency} case and will share next steps soon."
    )


def _one_based_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = range(1, len(out) + 1)
    return out


# def _render_priority_case_queue(df: pd.DataFrame, key_prefix: str, show_draft_workspace: bool) -> None:
#     st.markdown("**Priority Case Queue (Highest-Risk Posts)**")
def _render_priority_case_queue(df: pd.DataFrame, key_prefix: str, show_draft_workspace: bool, sentiment_filter: str = "negative") -> None:
    if sentiment_filter == "negative":
        st.markdown("**Priority Case Queue (Highest-Risk Posts)**")
    else:
        st.markdown("**Positive Mentions Queue (Top Advocates)**")

    queue_df = df.copy()
    queue_df["severity_rank"] = queue_df.get("severity", pd.Series(dtype=str)).astype(str).str.lower().map(
        {"high": 0, "medium": 1, "low": 2}
    ).fillna(3)
    queue_df["sentiment_score_num"] = pd.to_numeric(queue_df.get("sentiment_score"), errors="coerce").fillna(0.0)
    queue_df["date_dt"] = pd.to_datetime(queue_df.get("date"), errors="coerce")
    # queue_df = queue_df.sort_values(
    #     by=["severity_rank", "sentiment_score_num", "date_dt"],
    #     ascending=[True, True, False],
    # )
    # queue_df = queue_df[queue_df.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower() == "negative"].head(15)
    if sentiment_filter == "negative":
       queue_df = queue_df.sort_values(
           by=["sentiment_score_num", "date_dt"],
           ascending=[True, False],
       )
    else:
        queue_df = queue_df.sort_values(
            by=["sentiment_score_num", "date_dt"],
            ascending=[False, False],
        )
    queue_df = queue_df[queue_df.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower() == sentiment_filter]
    queue_df = queue_df.head(15)

    queue_df["response_template"] = queue_df.apply(_build_response_template, axis=1)

    if queue_df.empty:
        # st.info("No negative cases available for queueing.")
        st.info(f"No {sentiment_filter} cases available for queueing.")
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
    # base_reply = str(queue_view_df.loc[case_idx, "response_template"])

    row = queue_view_df.loc[case_idx]
    post_title = str(row.get("title") or "")
    post_text = str(row.get("text") or "")
    category = str(row.get("category") or "General Feedback")
    severity = str(row.get("severity") or "medium")
    reason = str(row.get("reason") or "the issue you reported")
    row_sentiment = str(row.get("sentiment") or "negative").strip().lower()

    # ── Case detail panel ──────────────────────────────────────────────────
    st.markdown("---")
    d1, d2, d3 = st.columns(3)
    d1.metric("Posted", str(row.get("posted_at") or row.get("date") or "—"))
    d2.metric("Subreddit", f"r/{row.get('subreddit', '—')}")
    d3.metric("Sentiment Score", str(row.get("sentiment_score") or "—"))
    if reason:
        st.caption(f"**Analysis:** {reason}")
    with st.expander(f"📄 Post: {post_title or '(no title)'}", expanded=True):
        st.write(post_text or "*(no post body)*")
    st.markdown("---")

    # # Build a KB search cache key so we only re-query when the case changes.
    cache_key = f"{key_prefix}_kb_{hashlib.md5((post_title + reason).encode()).hexdigest()}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = lookup_synthetic_kb_hits(
            post_title=post_title,
            post_text=post_text,
            category=category,
            severity=severity,
            reason=reason or category,
            top_k=3,
        )
    kb_hits = st.session_state.get(cache_key, [])

    # Generate (or retrieve cached) response draft.
    draft_key = f"{key_prefix}_draft_{cache_key}"
    if draft_key not in st.session_state:
        # include kb_hits in the func call after reason later
        with st.spinner("Generating response draft…"):
            st.session_state[draft_key] = generate_kb_response(
                post_title, post_text, category, severity, reason, kb_hits, sentiment=row_sentiment
            )

    base_reply = st.session_state[draft_key]

    optional_detail = st.text_input(
        "Optional company-specific detail to append",
        value="",
        placeholder="Example: Please DM your order reference so we can investigate.",
        # key=f"{key_prefix}_detail_text",
        key=f"{key_prefix}_detail_text{cache_key}",
    ).strip()
    final_reply = base_reply if not optional_detail else f"{base_reply} {optional_detail}"

    # st.text_area("Reply draft", value=final_reply, height=140, key=f"{key_prefix}_reply_draft")
    # st.code(final_reply)

    # Use a case-specific key so the textarea resets whenever the selected case changes.
    reply_key = f"{key_prefix}_reply_draft_{cache_key}"
    st.text_area("Reply draft", value=final_reply, height=160, key=reply_key)
    # Read back from session_state so st.code reflects any manual edits the user made.
    st.code(st.session_state.get(reply_key, final_reply))

    st.markdown("---")
    st.markdown("**Related Knowledge Base Articles (Synthetic)**")
    if kb_hits:
        has_direct = any(bool(h.get("is_direct")) for h in kb_hits)
        if not has_direct:
            st.caption("No exact synthetic match found. Showing related references.")
        for i, hit in enumerate(kb_hits):
            label = str(hit.get("title", "Untitled KB Article"))
            if not hit.get("is_direct"):
                label = f"{label} (related reference)"
            with st.expander(label, expanded=(i == 0)):
                st.markdown(str(hit.get("snippet", "")))
                steps = hit.get("resolution_steps", [])
                if isinstance(steps, list) and steps:
                    st.markdown("Suggested steps:")
                    for step in steps:
                        st.markdown(f"- {step}")
                url_stub = str(hit.get("url_stub", "")).strip()
                if url_stub:
                    st.caption(f"Reference: {url_stub}")
    else:
        st.caption("No synthetic KB articles matched this case; draft generated from issue context only.")

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

                    combined_lower = f"{title} {body}".lower()
                    if any(brand in combined_lower for brand in EXCLUDED_BRANDS):
                        skipped += 1
                        excluded_by_reason["excluded:brand_blocklist"] = excluded_by_reason.get("excluded:brand_blocklist", 0) + 1
                        continue

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
    # analyzed: List[Dict[str, Any]] = []
    # for review in reviews:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _analyze_one(review: Dict[str, Any]) -> Dict[str, Any]:
        result = analyze_review(review["text"])
        enriched = dict(review)
        enriched.update(result)
        return enriched
    analyzed: List[Dict[str, Any]] = [None] * len(reviews)
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_idx = {pool.submit(_analyze_one, r): i for i, r in enumerate(reviews)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            analyzed[idx] = future.result()
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
        # Force readable contrast in dark mode when highlighting negative rows.
        return ["background-color: #ffd6d6; color: #111111"] * len(row)
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
    reddit_limit_per_term = st.sidebar.slider("Reddit posts per term", min_value=5, max_value=100, value=15, step=5)
    years_back = st.sidebar.selectbox("Reddit lookback", [1, 2, 3, 4, 5], index=0, format_func=lambda y: f"Last {y} year{'s' if y > 1 else ''}")
    max_pages = st.sidebar.slider("Reddit pages per term", min_value=1, max_value=10, value=5, step=1)
    relevance_threshold = st.sidebar.slider(
        "Affiliate relevance threshold",
        min_value=1.0,
        max_value=4.0,
        value=DEFAULT_RELEVANCE_THRESHOLD,
        step=0.5,
    )
    max_analyzed_reviews = st.sidebar.slider("Max reviews to analyze", min_value=10, max_value=100, value=100, step=5)

    if data_source in {"Web", "Reddit + Web"}:
        st.sidebar.markdown("### Web Search (Bing)")
        web_count_per_term = st.sidebar.slider("Web results per term", min_value=3, max_value=20, value=5, step=1)
    else:
        web_count_per_term = 0

    run_analysis = st.sidebar.button("Run Analysis", type="primary")

    # ── Competitor comparison (opt-in) ────────────────────────────────
    enable_comparison = st.sidebar.checkbox("Enable competitor comparison", value=False, key="enable_comparison")
    run_comparison = False
    comp_company_name = ""
    comp_company_synonyms = ""
    comp_subreddit_name = "all"
    if enable_comparison:
        st.sidebar.markdown("### Competitor Company")
        comp_company_name = st.sidebar.text_input("Competitor name", value="Gift Card Granny", key="comp_company_name")
        comp_company_synonyms = st.sidebar.text_input(
            "Competitor synonyms (comma-separated)",
            value="giftcardgranny, vanillagift, giftya, egifter, perfectgift.com",
            key="comp_company_synonyms",
        )
        comp_subreddit_name = st.sidebar.text_input("Competitor subreddit", "all", key="comp_subreddit")
        run_comparison = st.sidebar.button("Run Competitor Analysis", type="secondary", key="run_comparison")

    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if run_analysis:
        combined_reviews: List[Dict[str, Any]] = []
        skipped = 0
        reddit_error: Optional[str] = None

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
                # st.error(f"Failed to load Reddit data: {reddit_error}")
                # st.stop()
                cached_path = Path(__file__).parent / "data" / "scheduled_results.json"
                if cached_path.exists():
                    st.warning(f"Live Reddit fetch failed ({reddit_error}). Using cached data.")
                    cached = json.loads(cached_path.read_text())
                    st.session_state.analysis_results = cached
                    run_analysis = False
                else:
                    st.error(f"Failed to load Reddit data: {reddit_error}")
                    st.stop()
            combined_reviews.extend(reddit_reviews)
        else:
            reddit_diagnostics = None
        
        if not st.session_state.analysis_results or not reddit_error:
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


    # ── Company B fetch + analyze ───────────────────────────────────
    if run_comparison and comp_company_name.strip():
        comp_reviews: List[Dict[str, Any]] = []


        if data_source in {"Reddit", "Reddit + Web"}:
            comp_reddit, _, comp_reddit_err, _ = fetch_reddit_reviews(
                company_name=comp_company_name,
                synonyms_raw=comp_company_synonyms,
                subreddit_name=comp_subreddit_name,
                limit_per_term=reddit_limit_per_term,
                years_back=int(years_back),
                relevance_threshold=float(relevance_threshold),
                max_pages=int(max_pages),
            )
            if comp_reddit_err:
                st.error(f"Competitor Reddit fetch failed: {comp_reddit_err}")
            else:
                comp_reviews.extend(comp_reddit)

            if data_source in {"Web", "Reddit + Web"}:
                comp_web, comp_web, _, comp_web_err = fetch_web_reviews(
                    company_name=comp_company_name,
                    synonyms_raw=comp_company_synonyms,
                    count_per_term=web_count_per_term,
                )
                if comp_web_err:
                    st.error(f"Competitor Web fetch failed: {comp_web_err}")
                else:
                    comp_reviews.extend(comp_web)


        comp_deduped = _dedupe_reviews(comp_reviews)
        comp_selected = _rank_reviews_for_analysis(comp_deduped, top_n=int(max_analyzed_reviews))


        if not comp_selected:
            st.warning(f"No valid reviews found for {comp_company_name}. Adjust inputs and try again.")
        else:
            with st.spinner(f"Analyzing {comp_company_name} reviews..."):
                comp_results = analyze_reviews(comp_selected)
            comp_df_tmp = pd.DataFrame(comp_results)
            st.session_state.comparison_results = {
                "company_name": comp_company_name,
                "rows": comp_results,
                "analyzed_count": len(comp_selected),
                "brand_health_summary": compute_brand_health_summary(comp_df_tmp),
                "keyword_diagnostics": compute_keyword_diagnostics(
                    comp_df_tmp,
                    top_n=15,
                    min_keyword_mentions=5,
                    min_negative_mentions=2,
                    risk_lift_threshold=1.6,
                ),
            }

    state = st.session_state.get("analysis_results")
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

    tabs = st.tabs(["How to Use", "Executive Brand Image Summary", "Competitive Comparison", "Customer Response Generator", "Data Analytics", "Jira Intake Triage"])


    # ── Tab 0: How to Use ─────────────────────────────────────────────
    with tabs[0]:
       st.markdown("## Brand Intelligence Monitor & Automated Issue Resolution")
       st.markdown("> *Turn customer complaints into actionable insights — automatically.*")

       st.markdown(
           "SignalAgents is an AI-driven brand monitoring platform that continuously scans public social media "
           "and web sources for customer sentiment, detects emerging issues before they escalate, and automates "
           "the handoff between customer support, product, and engineering teams. What used to take hours of "
           "manual triage now happens in seconds."
       )


       st.divider()
       st.markdown("### Why SignalAgents?")
       col_prob, col_sol = st.columns(2)
       with col_prob:
           st.markdown("**The Problem**")
           st.markdown(
               "- Customer complaints are scattered across Reddit, forums, and review sites\n"
               "- Manual triage is slow and inconsistent\n"
               "- Product teams miss emerging trends until they become P1s\n"
               "- Engineering learns about bugs from escalations, not data\n"
               "- No visibility into how competitors are performing"
           )
       with col_sol:
           st.markdown("**SignalAgents' Solution**")
           st.markdown(
               "- Centralized ingestion with real-time AI sentiment analysis\n"
               "- AI-powered triage automatically routes issues to support or engineering\n"
               "- Keyword diagnostics and trend detection surface patterns early\n"
               "- Automated Jira intake form creation from complaint analysis\n"
               "- Multi-brand analysis enables competitor benchmarking"
           )


       st.divider()
       st.markdown("### Who Is This For?")
       who1, who2, who3, who4 = st.columns(4)
       with who1:
           st.markdown("**Business & Marketing**")
           st.markdown(
               "Monitor brand health with a single score. "
               "Track sentiment shifts by topic and keyword. "
               "Benchmark against competitors in the gift card and e-commerce landscape."
           )
       with who2:
           st.markdown("**Product**")
           st.markdown(
               "Discover emerging issue trends before they escalate. "
               "View prioritized playbooks with severity scores. "
               "Build your backlog from customer signal data."
           )
       with who3:
           st.markdown("**Engineering**")
           st.markdown(
               "Get Jira intake tickets auto-generated from confirmed bugs. "
               "See affected components (checkout, activation, balance). "
               "Criticality scoring ensures only real issues hit the queue."
           )
       with who4:
           st.markdown("**Customer Service**")
           st.markdown(
               "AI-generated response drafts informed by your knowledge base. "
               "Priority case queue ranks the most urgent complaints. "
               "One-click Jira handoff when an issue needs engineering."
           )


       st.divider()
       st.markdown("### How to Use This Dashboard")


       st.markdown("#### 1. Configure the Sidebar")
       st.markdown(
           "Use the sidebar on the left to set your data source (Reddit, Web, or both), "
           "enter the company name and synonyms, and adjust parameters like subreddit, "
           "lookback years, and relevance threshold. Then click **Run Analysis**."
       )


       st.markdown("#### 2. Executive Brand Image Summary")
       st.markdown(
           "The top-level view of brand health. See your **Brand Image Score** (0-100), "
           "sentiment trends over time, distribution charts, and a prioritized **Response Playbook** "
           "with P1/P2/P3 action items and recommended SLAs."
       )


       st.markdown("#### 3. Competitive Comparison")
       st.markdown(
           "Compare your brand against a competitor side-by-side. Enable **competitor comparison** "
           "in the sidebar checkbox, enter the competitor's name and synonyms, then click **Run Competitor Analysis**. "
           "See overlaid sentiment trends, grouped distribution charts, and side-by-side problem theme tables."
       )


       st.markdown("#### 4. Customer Response Generator")
       st.markdown(
           "A priority queue of the most urgent customer complaints. Expand any case to see the "
           "full post and analysis, then generate an **AI response draft** powered by your knowledge base articles. "
           "Drafts are ready to copy and send."
       )


       st.markdown("#### 5. Data Analytics")
       st.markdown(
           "Deep-dive analytics: issue diagnosis tables, topic scatter plots, emerging risk keywords, "
           "positive/negative keyword breakdowns, subreddit risk views, and raw data export. "
           "Use this to identify trends, build backlogs, and inform product decisions."
       )


       st.markdown("#### 6. Jira Intake Triage")
       st.markdown(
           "Paste a Reddit URL and click **Analyze Post**. The AI analyzes sentiment, decides whether "
           "it warrants a customer response or a Jira ticket, and computes a **criticality score** (0-10). "
           "For critical issues, review the full intake form preview and click **Fill Out Jira Intake Form** "
           "to auto-fill the D2C Intake form via browser automation. All submissions are logged with screenshots."
       )


       st.divider()
       st.markdown("### Multi-Brand & Competitor Analysis")
       st.markdown(
           "SignalAgents is not locked to a single brand. Change the company name and synonyms in the sidebar "
           "to analyze **any brand or competitor** in the gift card, e-commerce, or fintech space. Use it for:"
       )
       st.markdown(
           "- **Competitive benchmarking** — compare sentiment scores across brands\n"
           "- **Market landscape analysis** — understand pain points across the category\n"
           "- **Opportunity identification** — find gaps where competitors are failing"
       )


       st.divider()
       st.markdown("### Impact")
       imp1, imp2, imp3 = st.columns(3)
       with imp1:
           st.metric("Response Time", "Seconds", delta="vs. hours manually", delta_color="normal")
           st.caption("AI drafts cut triage-to-reply from hours to seconds")
       with imp2:
           st.metric("Issue Coverage", "Automated", delta="vs. manual scanning", delta_color="normal")
           st.caption("Catches complaints that manual monitoring would miss")
       with imp3:
           st.metric("Cross-Team Alignment", "1 Source", delta="Support + Product + Eng", delta_color="off")
           st.caption("Single source of truth across all stakeholder teams")


       st.divider()
       st.caption(
           "**Setup note:** Create a `.env` file from `.env.example` and add your API keys. "
           "Reddit and OpenAI APIs may rate-limit frequent requests — if you see timeouts, "
           "wait 30-60 seconds before running another analysis."
       )


    with tabs[1]:
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


       st.markdown("**Average Sentiment Trend Over Time**")
       if "date" in metric_df.columns and not metric_df.empty:
           _trend_tmp = metric_df.copy()
           _trend_tmp["date_dt"] = pd.to_datetime(_trend_tmp["date"], errors="coerce", utc=True)
           _trend_tmp = _trend_tmp.dropna(subset=["date_dt"])
           if not _trend_tmp.empty:
               _trend_tmp["sentiment_score_num"] = pd.to_numeric(_trend_tmp.get("sentiment_score"), errors="coerce").fillna(0.0)
               _trend_tmp["period"] = _trend_tmp["date_dt"].dt.to_period("W").apply(lambda p: p.start_time)
               _exec_trend = _trend_tmp.groupby("period").agg(
                   avg_sentiment=("sentiment_score_num", "mean"),
                   reviews=("sentiment_score_num", "count"),
               ).reset_index()
               st.vega_lite_chart(
                   _exec_trend,
                   {
                       "mark": {"type": "line", "point": True},
                       "encoding": {
                           "x": {"field": "period", "type": "temporal", "title": "Week"},
                           "y": {"field": "avg_sentiment", "type": "quantitative", "title": "Avg Sentiment Score"},
                           "color": {"value": COLOR_COMPANY_A},
                           "tooltip": [{"field": "period"}, {"field": "avg_sentiment", "format": ".2f"}, {"field": "reviews", "title": "# Reviews"}],
                       },
                   },
                   use_container_width=True,
               )
           else:
               st.info("Not enough timestamped data for trend chart.")
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
       all_categories = sorted(metric_df.get("category", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
       selected_categories = filter_col2.multiselect(
           "Filter by problem theme",
           options=all_categories,
           default=all_categories,
           key="exec_review_categories",
       )


       filtered_df = metric_df.copy()
       if selected_sentiments:
           filtered_df = filtered_df[filtered_df["sentiment_norm"].isin(selected_sentiments)]
       if selected_categories and "category" in filtered_df.columns:
           filtered_df = filtered_df[filtered_df["category"].astype(str).isin(selected_categories)]


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


    # ── Tab 2: Competitive Comparison ─────────────────────────────────
    with tabs[2]:
       st.subheader("Competitive Comparison")
       comp_state = st.session_state.comparison_results
       company_a_name = company_name
       company_b_name = comp_state["company_name"] if comp_state else "Competitor"


       # KPI Cards — Company A
       st.markdown(f"#### {company_a_name}")
       a_kpis = _extract_comparison_kpis(brand_health_summary, metric_df)
       a1, a2, a3, a4, a5 = st.columns(5)
       a1.metric("Brand Image Score", f"{a_kpis['brand_image_score']}/100")
       a2.metric("Avg Sentiment", f"{a_kpis['avg_sentiment']:.2f}")
       a3.metric("Positive %", f"{a_kpis['positive_share'] * 100:.1f}%")
       a4.metric("Negative %", f"{a_kpis['negative_share'] * 100:.1f}%")
       a5.metric("Net Sentiment", f"{a_kpis['net_sentiment']:.2f}")


       if comp_state:
           comp_df_b = pd.DataFrame(comp_state["rows"])
           comp_bhs = comp_state["brand_health_summary"]
           comp_kd = comp_state["keyword_diagnostics"]
           b_kpis = _extract_comparison_kpis(comp_bhs, comp_df_b)


           # KPI Cards — Company B
           st.markdown(f"#### {company_b_name}")
           b1, b2, b3, b4, b5 = st.columns(5)
           b1.metric("Brand Image Score", f"{b_kpis['brand_image_score']}/100",
                      delta=f"{a_kpis['brand_image_score'] - b_kpis['brand_image_score']:+d} vs {company_a_name}")
           b2.metric("Avg Sentiment", f"{b_kpis['avg_sentiment']:.2f}",
                      delta=f"{a_kpis['avg_sentiment'] - b_kpis['avg_sentiment']:+.2f} vs {company_a_name}")
           b3.metric("Positive %", f"{b_kpis['positive_share'] * 100:.1f}%")
           b4.metric("Negative %", f"{b_kpis['negative_share'] * 100:.1f}%")
           b5.metric("Net Sentiment", f"{b_kpis['net_sentiment']:.2f}")


           st.caption(
               f"Compared {a_kpis['total_reviews']} {company_a_name} reviews vs "
               f"{b_kpis['total_reviews']} {company_b_name} reviews."
           )


           # ── Overlaid Sentiment Trend ──────────────────────────────
           st.divider()
           st.markdown("**Average Sentiment Trend Over Time**")


           def _build_comp_timeline(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
               if df_in.empty or "date" not in df_in.columns:
                   return pd.DataFrame()
               tmp = df_in.copy()
               tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce", utc=True)
               tmp = tmp.dropna(subset=["date_dt"])
               if tmp.empty:
                   return pd.DataFrame()
               tmp["sentiment_norm"] = tmp.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower()
               tmp["sentiment_score_num"] = pd.to_numeric(tmp.get("sentiment_score"), errors="coerce").fillna(0.0)
               # Use weekly grouping for finer granularity (monthly collapses sparse data)
               tmp["period"] = tmp["date_dt"].dt.to_period("W").apply(lambda p: p.start_time)
               trend = tmp.groupby("period").agg(
                   avg_sentiment=("sentiment_score_num", "mean"),
                   negative_share=("sentiment_norm", lambda s: (s == "negative").mean()),
                   reviews=("sentiment_norm", "count"),
               ).reset_index()
               trend["company"] = label
               return trend


           timeline_a = _build_comp_timeline(df, company_a_name)
           timeline_b = _build_comp_timeline(comp_df_b, company_b_name)
           combined_timeline = pd.concat([timeline_a, timeline_b], ignore_index=True)


           if not combined_timeline.empty:
               st.vega_lite_chart(
                   combined_timeline,
                   {
                       "mark": {"type": "line", "point": True},
                       "encoding": {
                           "x": {"field": "period", "type": "temporal", "title": "Week"},
                           "y": {"field": "avg_sentiment", "type": "quantitative", "title": "Avg Sentiment Score"},
                           "color": {
                               "field": "company", "type": "nominal",
                               "scale": {"domain": [company_a_name, company_b_name], "range": [COLOR_COMPANY_A, COLOR_COMPANY_B]},
                           },
                           "strokeDash": {"field": "company", "type": "nominal"},
                           "tooltip": [{"field": "period"}, {"field": "company"}, {"field": "avg_sentiment", "format": ".2f"}, {"field": "reviews", "title": "# Reviews"}],
                       },
                   },
                   use_container_width=True,
               )
           else:
               st.info("Not enough timestamped data for trend overlay.")


           # ── Grouped Sentiment Distribution ────────────────────────
           st.markdown("**Sentiment Distribution**")


           def _build_sentiment_dist(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
               if df_in.empty:
                   return pd.DataFrame()
               counts = df_in.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower().value_counts()
               total = counts.sum()
               return pd.DataFrame({
                   "sentiment": [s.title() for s in counts.index],
                   "share": [c / total for c in counts.values],
                   "company": label,
               })


           dist_a = _build_sentiment_dist(df, company_a_name)
           dist_b = _build_sentiment_dist(comp_df_b, company_b_name)
           combined_dist = pd.concat([dist_a, dist_b], ignore_index=True)


           if not combined_dist.empty:
               st.vega_lite_chart(
                   combined_dist,
                   {
                       "mark": "bar",
                       "encoding": {
                           "x": {"field": "sentiment", "type": "nominal", "title": "Sentiment"},
                           "y": {"field": "share", "type": "quantitative", "title": "Share"},
                           "color": {
                               "field": "company", "type": "nominal",
                               "scale": {"domain": [company_a_name, company_b_name], "range": [COLOR_COMPANY_A, COLOR_COMPANY_B]},
                           },
                           "xOffset": {"field": "company"},
                           "tooltip": [{"field": "sentiment"}, {"field": "company"}, {"field": "share", "format": ".1%"}],
                       },
                   },
                   use_container_width=True,
               )


           # ── Negative Share Trend Overlay ──────────────────────────
           if not combined_timeline.empty and "negative_share" in combined_timeline.columns:
               st.markdown("**Negative Share Trend**")
               st.vega_lite_chart(
                   combined_timeline,
                   {
                       "mark": {"type": "area", "opacity": 0.3, "line": True},
                       "encoding": {
                           "x": {"field": "period", "type": "temporal", "title": "Week"},
                           "y": {"field": "negative_share", "type": "quantitative", "title": "Negative Share"},
                           "color": {
                               "field": "company", "type": "nominal",
                               "scale": {"domain": [company_a_name, company_b_name], "range": [COLOR_COMPANY_A, COLOR_COMPANY_B]},
                           },
                           "tooltip": [{"field": "period"}, {"field": "company"}, {"field": "negative_share", "format": ".1%"}, {"field": "reviews", "title": "# Reviews"}],
                       },
                   },
                   use_container_width=True,
               )


           # ── Side-by-Side Problem Themes ───────────────────────────
           st.divider()
           st.markdown("**Top Problem Themes**")
           theme_left, theme_right = st.columns(2)


           with theme_left:
               st.markdown(f"**{company_a_name}**")
               a_issues = pd.DataFrame(keyword_diagnostics.get("issue_diagnosis", []))
               if not a_issues.empty:
                   show_cols = [c for c in ["issue", "mentions", "negative_rate", "what_to_fix"] if c in a_issues.columns]
                   st.dataframe(_one_based_index(a_issues.head(5)[show_cols]), use_container_width=True)
               else:
                   st.info("No issue data.")


           with theme_right:
               st.markdown(f"**{company_b_name}**")
               b_issues = pd.DataFrame(comp_kd.get("issue_diagnosis", []))
               if not b_issues.empty:
                   show_cols = [c for c in ["issue", "mentions", "negative_rate", "what_to_fix"] if c in b_issues.columns]
                   st.dataframe(_one_based_index(b_issues.head(5)[show_cols]), use_container_width=True)
               else:
                   st.info("No issue data.")


           # # ── Side-by-Side Risk Keywords ────────────────────────────
           # st.markdown("**Risk Keywords**")
           # rk_left, rk_right = st.columns(2)


           # with rk_left:
           #     st.markdown(f"**{company_a_name}**")
           #     a_risk = pd.DataFrame(keyword_diagnostics.get("negative_lift_keywords", []))
           #     if not a_risk.empty:
           #         st.dataframe(_one_based_index(a_risk.head(8)), use_container_width=True)
           #     else:
           #         st.info("No risk keywords.")


           # with rk_right:
           #     st.markdown(f"**{company_b_name}**")
           #     b_risk = pd.DataFrame(comp_kd.get("negative_lift_keywords", []))
           #     if not b_risk.empty:
           #         st.dataframe(_one_based_index(b_risk.head(8)), use_container_width=True)
           #     else:
           #         st.info("No risk keywords.")


       else:
           st.info(
               "Enable **competitor comparison** in the sidebar, configure the competitor company, "
               "and click **Run Competitor Analysis** to see a side-by-side comparison."
           )


    with tabs[3]:
       st.subheader("Customer Response")
       response_mode = st.radio(
           "Response type",
           ["Negative — Issue Resolution", "Positive — Advocate Engagement"],
           horizontal=True,
           key="response_mode",
       )
       if response_mode.startswith("Negative"):
           _render_priority_case_queue(df, key_prefix="response_queue_neg", show_draft_workspace=True, sentiment_filter="negative")
       else:
           _render_priority_case_queue(df, key_prefix="response_queue_pos", show_draft_workspace=True, sentiment_filter="positive")


    with tabs[4]:
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


    # ── Tab 4: Jira Intake Triage ─────────────────────────────────────
    # ── Tab 4: Jira Intake Triage ─────────────────────────────────────
    # ── Tab 5: Jira Intake Triage ─────────────────────────────────────
    with tabs[5]:
       st.subheader("Jira Intake Triage")
       st.caption(
           "Analyze a Reddit complaint and auto-fill the D2C Intake: Issue Resolution Request Form. "
           f"Posts with a criticality score >= {CRITICALITY_THRESHOLD}/10 are flagged as critical."
       )


       triage_url = st.text_input(
           "Reddit post URL",
           placeholder="https://www.reddit.com/r/CreditCards/comments/...",
           key="triage_url",
       )


       if st.button("Analyze Post", key="triage_analyze") and triage_url.strip():
           with st.spinner("Fetching Reddit post..."):
               triage_post = fetch_reddit_post(triage_url.strip())


           if "error" in triage_post:
               st.error(f"Failed to fetch post: {triage_post['error']}")
           else:
               st.session_state["triage_post"] = triage_post


               with st.expander("Reddit Post", expanded=True):
                   st.markdown(f"**{triage_post['title']}**")
                   st.caption(f"u/{triage_post['author']} · r/{triage_post['subreddit']} · score {triage_post['score']}")
                   st.text(triage_post.get("body", "")[:1000])


               complaint_text = f"{triage_post.get('title', '')} {triage_post.get('body', '')}".strip()


               with st.spinner("Analyzing complaint..."):
                   triage_analysis = analyze_complaint(complaint_text)
               st.session_state["triage_analysis"] = triage_analysis


               if "error" in triage_analysis:
                   st.error(f"Analysis failed: {triage_analysis['error']}")
               else:
                   with st.spinner("Running triage decision..."):
                       triage_decision = decide_triage_action(triage_analysis, triage_post)
                   st.session_state["triage_decision"] = triage_decision


                   if "error" in triage_decision:
                       st.error(f"Triage failed: {triage_decision['error']}")
                   else:
                       crit = compute_criticality_score(triage_analysis, triage_decision)
                       st.session_state["triage_crit_score"] = crit


       # Display results if analysis is complete
       if "triage_analysis" in st.session_state and "error" not in st.session_state.get("triage_analysis", {}):
           triage_analysis = st.session_state["triage_analysis"]
           triage_decision = st.session_state.get("triage_decision", {})
           triage_post = st.session_state.get("triage_post", {})
           crit = st.session_state.get("triage_crit_score", 0.0)


           # Analysis metrics
           a1, a2, a3, a4 = st.columns(4)
           a1.metric("Sentiment", f"{triage_analysis['sentiment']} ({triage_analysis['sentiment_score']:+.2f})")
           a2.metric("Category", triage_analysis["category"])
           a3.metric("Severity", triage_analysis["severity"].upper())
           a4.metric("Confidence", f"{triage_analysis['confidence']:.0%}")
           st.caption(f"**Reason:** {triage_analysis['reason']}")


           if "error" not in triage_decision:
               st.divider()
               action_label = "Jira Ticket" if triage_decision["action"] == "jira_ticket" else "Customer Response"
               is_critical = crit >= CRITICALITY_THRESHOLD


               t1, t2, t3 = st.columns(3)
               t1.metric("Recommended Action", action_label)
               t2.metric("Criticality Score", f"{crit}/10")
               t3.metric("Status", "CRITICAL" if is_critical else "Below Threshold")


               if is_critical:
                   st.error(f"Criticality {crit}/10 exceeds threshold ({CRITICALITY_THRESHOLD}). Jira intake form recommended.")
               else:
                   st.info(f"Criticality {crit}/10 is below threshold ({CRITICALITY_THRESHOLD}).")


               st.caption(f"**Triage rationale:** {triage_decision['rationale']}")


               if triage_decision["action"] == "jira_ticket":
                   st.divider()
                   st.markdown("#### Intake Form Preview")


                   form_data = build_form_data(triage_post, triage_analysis, triage_decision)


                   col_left, col_right = st.columns(2)
                   with col_left:
                       st.text_input("Email", value=form_data["email"], disabled=True, key="form_email")
                       st.text_input("Summary", value=form_data["summary"], disabled=True, key="form_summary")
                       st.text_input("Requesting Team", value=form_data["requesting_team"], disabled=True, key="form_team")
                       st.text_input("Target Audience", value=form_data["target_audience"], disabled=True, key="form_audience")
                       st.text_input("Urgency", value=form_data["urgency"], disabled=True, key="form_urgency")
                       st.text_input("Delivery Timeframe", value=form_data["desired_delivery_timeframe"], disabled=True, key="form_timeframe")
                       st.text_input("Contract Status", value=form_data["contract_status"], disabled=True, key="form_contract")
                       st.text_input("Requirements Defined?", value=form_data["requirements_defined"], disabled=True, key="form_reqs")


                   with col_right:
                       st.text_area("Problem Statement", value=form_data["problem_statement"], height=150, disabled=True, key="form_problem")
                       st.text_area("Proposed Solutions", value=form_data["proposed_solutions"], height=100, disabled=True, key="form_solutions")
                       st.text_area("Expected Benefits", value=form_data["expected_benefits"], height=80, disabled=True, key="form_benefits")
                       st.text_area("Urgency Reason", value=form_data["urgency_reason"], height=80, disabled=True, key="form_urgency_reason")
                       st.text_area("Risks and Dependencies", value=form_data["risks_and_dependencies"], height=80, disabled=True, key="form_risks")


                   _HISTORY_DIR = os.path.join(os.path.dirname(__file__), "triage_history")


                   if st.button("Fill Out Jira Intake Form", type="primary", key="triage_fill_form"):
                       with st.spinner("Filling out the Jira intake form..."):
                           result = fill_intake_form(
                               form_data,
                               headless=True,
                               screenshot_dir=_HISTORY_DIR,
                           )
                       if result["success"]:
                           st.success("Form filled successfully!")
                           st.image(result["screenshot"], caption="Filled Jira Intake Form")
                       else:
                           st.error(f"Form filling failed: {result['error']}")


               elif triage_decision["action"] == "customer_response":
                   st.divider()
                   st.markdown("#### Draft Customer Response")
                   draft_key = f"triage_draft_{hash(triage_post.get('url', ''))}"
                   if draft_key not in st.session_state:
                       with st.spinner("Generating response draft..."):
                           from triage_agent import generate_response_draft
                           st.session_state[draft_key] = generate_response_draft(triage_post, triage_analysis)
                   st.text_area("Response Draft", value=st.session_state[draft_key], height=200, disabled=True, key="form_draft")


       # ── Submission History ────────────────────────────────────────
       st.divider()
       st.markdown("#### Submission History")


       _HISTORY_DIR_DISPLAY = os.path.join(os.path.dirname(__file__), "triage_history")
       if os.path.isdir(_HISTORY_DIR_DISPLAY):
           import json as _json_mod
           history_files = sorted(
               [f for f in os.listdir(_HISTORY_DIR_DISPLAY) if f.endswith(".json")],
               reverse=True,
           )
           if history_files:
               for hf in history_files:
                   meta_path = os.path.join(_HISTORY_DIR_DISPLAY, hf)
                   with open(meta_path) as _f:
                       meta = _json_mod.load(_f)
                   ts = meta.get("timestamp", "")
                   display_ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}" if len(ts) >= 15 else ts
                   label = f"{display_ts}  |  {meta.get('urgency', '')}  |  {meta.get('summary', '')[:80]}"
                   with st.expander(label, expanded=False):
                       sc1, sc2 = st.columns(2)
                       sc1.markdown(f"**Summary:** {meta.get('summary', '')}")
                       sc1.markdown(f"**Urgency:** {meta.get('urgency', '')}")
                       sc1.markdown(f"**Email:** {meta.get('email', '')}")
                       sc2.markdown(f"**Team:** {meta.get('requesting_team', '')}")
                       sc2.markdown(f"**Timestamp:** {display_ts}")
                       img_path = os.path.join(_HISTORY_DIR_DISPLAY, meta.get("screenshot", ""))
                       if os.path.isfile(img_path):
                           st.image(img_path, caption="Form Screenshot", use_container_width=True)
           else:
               st.info("No submissions yet. Fill out an intake form above to see history here.")
       else:
           st.info("No submissions yet. Fill out an intake form above to see history here.")

if __name__ == "__main__":
    main()
