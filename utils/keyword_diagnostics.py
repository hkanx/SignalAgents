import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "is",
    "it",
    "this",
    "that",
    "with",
    "from",
    "by",
    "at",
    "as",
    "be",
    "are",
    "was",
    "were",
    "i",
    "you",
    "we",
    "they",
    "my",
    "our",
    "their",
    "your",
    "have",
    "has",
    "had",
    "but",
    "not",
    "so",
    "if",
    "can",
    "just",
    "about",
    "all",
    "more",
    "very",
    "really",
    "one",
    "get",
    "got",
    "would",
    "could",
    "should",
    "also",
    "there",
    "here",
}

BRAND_STOP_TERMS = {
    "etsy",
    "etsycom",
    "etsy.com",
    "shop",
    "listing",
    "seller",
    "buyer",
    "marketplace",
}

WORD_RE = re.compile(r"[a-z0-9']+")
URL_RE = re.compile(r"https?://\S+")

ISSUE_RULES = [
    {
        "key": "order_not_received",
        "label": "Order Not Received",
        "patterns": [r"\bnever arrived\b", r"\bdidn'?t arrive\b", r"\bnot received\b", r"\bmissing package\b", r"\bwhere is my order\b"],
        "questions": "Where is the order now, and what recovery timeline can the customer expect?",
        "actions": "Tighten shipment tracking visibility, trigger proactive delay outreach, and expedite replacement/refund decisions.",
    },
    {
        "key": "item_not_as_described",
        "label": "Item Not As Described",
        "patterns": [r"\bnot as described\b", r"\bwrong item\b", r"\bdefective\b", r"\bquality issue\b", r"\bdamaged\b", r"\bmisleading\b"],
        "questions": "What was different from the listing and what proof is needed to resolve quickly?",
        "actions": "Improve listing quality checks, standardize evidence requirements, and shorten dispute-resolution loops.",
    },
    {
        "key": "refund_delay_or_denial",
        "label": "Refund Delay or Denial",
        "patterns": [r"\brefund\b", r"\bno refund\b", r"\brefund denied\b", r"\bwaiting for refund\b", r"\bchargeback\b", r"\bcase closed\b"],
        "questions": "What is the refund decision status and when will funds be returned, if approved?",
        "actions": "Define clear refund SLAs, automate status notifications, and add escalation for delayed payouts.",
    },
    {
        "key": "seller_communication_gap",
        "label": "Seller Communication Gap",
        "patterns": [r"\bno response\b", r"\bignored\b", r"\bnot replying\b", r"\bunresponsive\b", r"\bghosted\b", r"\bmessage seller\b"],
        "questions": "Who owns communication next and what response timeline can be guaranteed?",
        "actions": "Set first-response SLA targets, enforce response accountability, and escalate stalled threads faster.",
    },
    {
        "key": "account_or_case_hold",
        "label": "Account or Case Hold",
        "patterns": [r"\baccount suspended\b", r"\baccount hold\b", r"\bcase on hold\b", r"\bverification\b", r"\breview pending\b", r"\bappeal\b"],
        "questions": "Why was the account/case held and what exact steps unblock it?",
        "actions": "Clarify hold reasons, publish unblock checklists, and reduce manual-review turnaround times.",
    },
    {
        "key": "shipping_cost_or_delay",
        "label": "Shipping Cost or Delay",
        "patterns": [r"\bshipping\b", r"\btracking\b", r"\bdelayed\b", r"\blate\b", r"\bpostage\b", r"\bshipping cost\b"],
        "questions": "Are shipping costs/timelines clear upfront and are delays handled proactively?",
        "actions": "Improve shipping-fee transparency, monitor carrier-delay hotspots, and message customers before SLAs slip.",
    },
]


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = URL_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _extract_tokens(text: str) -> List[str]:
    text = _normalize_text(text)
    tokens = WORD_RE.findall(text)
    cleaned: List[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        if token in BRAND_STOP_TERMS:
            continue
        cleaned.append(token)
    return cleaned


def _top_keywords(counter: Counter, top_n: int = 15) -> List[Dict[str, Any]]:
    return [{"keyword": word, "count": count} for word, count in counter.most_common(top_n)]


def _detect_issues(text: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for issue in ISSUE_RULES:
        for pattern in issue["patterns"]:
            if re.search(pattern, text):
                found.append(issue)
                break
    return found


def compute_keyword_diagnostics(
    df: pd.DataFrame,
    top_n: int = 15,
    min_keyword_mentions: int = 5,
    min_negative_mentions: int = 2,
    risk_lift_threshold: float = 1.6,
) -> Dict[str, Any]:
    if df.empty or "text" not in df.columns:
        return {
            "top_keywords_overall": [],
            "top_keywords_by_sentiment": {"negative": [], "neutral": [], "positive": []},
            "negative_lift_keywords": [],
            "subreddit_risk": [],
        }

    work = df.copy()
    work["sentiment_norm"] = work.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower()

    overall_counter: Counter = Counter()
    negative_counter: Counter = Counter()
    neutral_counter: Counter = Counter()
    positive_counter: Counter = Counter()

    keyword_doc_counts: Counter = Counter()
    negative_keyword_doc_counts: Counter = Counter()
    issue_counts: Counter = Counter()
    issue_negative_counts: Counter = Counter()
    issue_example_titles: Dict[str, List[str]] = {}

    for _, row in work.iterrows():
        tokens = _extract_tokens(str(row.get("text", "")))
        if not tokens:
            continue

        overall_counter.update(tokens)
        sentiment = row.get("sentiment_norm", "")
        if sentiment == "negative":
            negative_counter.update(tokens)
        elif sentiment == "positive":
            positive_counter.update(tokens)
        else:
            neutral_counter.update(tokens)

        normalized_text = _normalize_text(str(row.get("text", "")))
        matched_issues = _detect_issues(normalized_text)
        for issue in matched_issues:
            key = issue["key"]
            issue_counts[key] += 1
            if sentiment == "negative":
                issue_negative_counts[key] += 1

            title = str(row.get("title", "")).strip()
            if title:
                issue_example_titles.setdefault(key, [])
                if len(issue_example_titles[key]) < 3 and title not in issue_example_titles[key]:
                    issue_example_titles[key].append(title)

        unique_tokens = set(tokens)
        keyword_doc_counts.update(unique_tokens)
        if sentiment == "negative":
            negative_keyword_doc_counts.update(unique_tokens)

    total_docs = max(len(work), 1)
    negative_docs = max(int((work["sentiment_norm"] == "negative").sum()), 1)
    base_negative_rate = negative_docs / total_docs

    negative_lift: List[Dict[str, Any]] = []
    for keyword, neg_doc_count in negative_keyword_doc_counts.items():
        total_doc_count = keyword_doc_counts.get(keyword, 0)
        if total_doc_count < min_keyword_mentions:
            continue
        if neg_doc_count < min_negative_mentions:
            continue
        keyword_negative_rate = neg_doc_count / total_doc_count
        lift = keyword_negative_rate / base_negative_rate if base_negative_rate > 0 else 0.0
        if lift >= risk_lift_threshold:
            negative_lift.append(
                {
                    "keyword": keyword,
                    "keyword_negative_rate": round(keyword_negative_rate, 3),
                    "overall_negative_rate": round(base_negative_rate, 3),
                    "negative_lift": round(lift, 3),
                    "mentions": int(total_doc_count),
                    "negative_mentions": int(neg_doc_count),
                }
            )

    negative_lift = sorted(
        negative_lift,
        key=lambda x: (x["negative_lift"], x["negative_mentions"], x["mentions"]),
        reverse=True,
    )[:top_n]

    subreddit_risk: List[Dict[str, Any]] = []
    if "subreddit" in work.columns:
        grouped = work.groupby("subreddit", dropna=True)
        for subreddit, gdf in grouped:
            volume = len(gdf)
            if volume < 3:
                continue
            neg_rate = float((gdf["sentiment_norm"] == "negative").mean())
            avg_sent = float(gdf.get("sentiment_score", pd.Series(dtype=float)).fillna(0.0).mean())
            subreddit_risk.append(
                {
                    "subreddit": str(subreddit),
                    "volume": int(volume),
                    "negative_rate": round(neg_rate, 3),
                    "avg_sentiment_score": round(avg_sent, 3),
                }
            )

        subreddit_risk = sorted(
            subreddit_risk,
            key=lambda x: (x["negative_rate"], x["volume"]),
            reverse=True,
        )[:20]

    issue_diagnosis: List[Dict[str, Any]] = []
    issue_lookup = {issue["key"]: issue for issue in ISSUE_RULES}
    for issue_key, mentions in issue_counts.items():
        issue_meta = issue_lookup.get(issue_key)
        if not issue_meta:
            continue
        negative_mentions = issue_negative_counts.get(issue_key, 0)
        negative_rate = (negative_mentions / mentions) if mentions else 0.0
        severity_score = mentions * (0.5 + negative_rate)
        issue_diagnosis.append(
            {
                "issue_key": issue_key,
                "issue": issue_meta["label"],
                "mentions": int(mentions),
                "negative_mentions": int(negative_mentions),
                "negative_rate": round(negative_rate, 3),
                "severity_score": round(severity_score, 3),
                "what_to_answer": issue_meta["questions"],
                "what_to_fix": issue_meta["actions"],
                "sample_posts": " | ".join(issue_example_titles.get(issue_key, [])),
            }
        )

    issue_diagnosis = sorted(
        issue_diagnosis,
        key=lambda x: (x["severity_score"], x["negative_mentions"], x["mentions"]),
        reverse=True,
    )

    return {
        "top_keywords_overall": _top_keywords(overall_counter, top_n=top_n),
        "top_keywords_by_sentiment": {
            "negative": _top_keywords(negative_counter, top_n=top_n),
            "neutral": _top_keywords(neutral_counter, top_n=top_n),
            "positive": _top_keywords(positive_counter, top_n=top_n),
        },
        "negative_lift_keywords": negative_lift,
        "subreddit_risk": subreddit_risk,
        "issue_diagnosis": issue_diagnosis,
        "risk_thresholds": {
            "min_keyword_mentions": min_keyword_mentions,
            "min_negative_mentions": min_negative_mentions,
            "risk_lift_threshold": risk_lift_threshold,
        },
    }


def build_response_playbook(
    issue_diagnosis: List[Dict[str, Any]],
    brand_health_summary: Dict[str, float],
    subreddit_risk: List[Dict[str, Any]],
    max_actions: int = 5,
) -> List[Dict[str, Any]]:
    """Create a prioritized response playbook from current diagnosis outputs."""
    playbook: List[Dict[str, Any]] = []
    if not issue_diagnosis:
        return playbook

    negative_share = float(brand_health_summary.get("negative_share", 0.0))
    negative_delta = float(brand_health_summary.get("negative_share_delta", 0.0))
    top_subreddit = subreddit_risk[0]["subreddit"] if subreddit_risk else "N/A"

    for row in issue_diagnosis[:max_actions]:
        mentions = int(row.get("mentions", 0))
        neg_rate = float(row.get("negative_rate", 0.0))
        severity_score = float(row.get("severity_score", 0.0))

        if severity_score >= 8 or neg_rate >= 0.8:
            priority = "P1"
        elif severity_score >= 4 or neg_rate >= 0.6:
            priority = "P2"
        else:
            priority = "P3"

        why_now = (
            f"{mentions} mentions, {neg_rate * 100:.0f}% negative; "
            f"overall negative share {negative_share * 100:.0f}% "
            f"({negative_delta * 100:+.0f} pts vs baseline)."
        )

        playbook.append(
            {
                "priority": priority,
                "issue": row.get("issue", "Unknown issue"),
                "why_now": why_now,
                "what_to_answer": row.get("what_to_answer", ""),
                "what_to_fix": row.get("what_to_fix", ""),
                "owner_team": "Support + Ops",
                "recommended_sla": "24h customer-facing response, 72h remediation plan",
                "watch_channel": f"Monitor subreddit: {top_subreddit}",
            }
        )

    priority_rank = {"P1": 0, "P2": 1, "P3": 2}
    return sorted(playbook, key=lambda x: priority_rank.get(x["priority"], 9))


def compute_brand_health_summary(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "negative_share": 0.0,
            "avg_sentiment_score": 0.0,
            "avg_star_rating": 0.0,
            "net_sentiment": 0.0,
            "recent_negative_share": 0.0,
            "baseline_negative_share": 0.0,
            "negative_share_delta": 0.0,
        }

    work = df.copy()
    work["sentiment_norm"] = work.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower()

    negative_share = float((work["sentiment_norm"] == "negative").mean())
    positive_share = float((work["sentiment_norm"] == "positive").mean())
    net_sentiment = positive_share - negative_share

    avg_sentiment_score = float(work.get("sentiment_score", pd.Series(dtype=float)).fillna(0.0).mean())
    avg_star_rating = float(work.get("star_rating", pd.Series(dtype=float)).fillna(0.0).mean())

    recent_negative_share = negative_share
    baseline_negative_share = negative_share

    if "date" in work.columns:
        work["date_dt"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
        work = work.dropna(subset=["date_dt"])
        if not work.empty:
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            recent = work[work["date_dt"] >= cutoff]
            baseline = work[work["date_dt"] < cutoff]

            if len(recent) >= 1:
                recent_negative_share = float((recent["sentiment_norm"] == "negative").mean())
            if len(baseline) >= 1:
                baseline_negative_share = float((baseline["sentiment_norm"] == "negative").mean())

    return {
        "negative_share": negative_share,
        "avg_sentiment_score": avg_sentiment_score,
        "avg_star_rating": avg_star_rating,
        "net_sentiment": net_sentiment,
        "recent_negative_share": recent_negative_share,
        "baseline_negative_share": baseline_negative_share,
        "negative_share_delta": recent_negative_share - baseline_negative_share,
    }
