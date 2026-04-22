import json
import re
from pathlib import Path
from typing import Any, Dict, List

WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(_normalize(text)))


def load_synthetic_kb(path: Path | None = None) -> List[Dict[str, Any]]:
    kb_path = path or (Path(__file__).resolve().parents[1] / "kb" / "synthetic_kb_articles.json")
    with kb_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return []
    return [p for p in payload if isinstance(p, dict)]


def lookup_synthetic_kb_hits(
    post_title: str,
    post_text: str,
    category: str,
    severity: str,
    reason: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    articles = load_synthetic_kb()
    if not articles:
        return []

    query_text = " ".join([post_title or "", post_text or "", reason or ""])
    query_tokens = _tokens(query_text)
    category_norm = _normalize(category)
    severity_norm = _normalize(severity)

    scored: List[Dict[str, Any]] = []
    for article in articles:
        art_category = _normalize(str(article.get("category", "")))
        art_keywords = {str(k).lower() for k in article.get("keywords", []) if isinstance(k, str)}
        art_severities = {str(s).lower() for s in article.get("severity_tags", []) if isinstance(s, str)}

        score = 0.0
        if art_category and art_category == category_norm:
            score += 3.0
        if severity_norm in art_severities:
            score += 1.5

        keyword_overlap = len(query_tokens.intersection(art_keywords))
        if keyword_overlap:
            score += min(4.0, float(keyword_overlap))

        if score <= 0:
            continue

        is_direct = score >= 5.0
        scored.append(
            {
                "id": article.get("id", ""),
                "title": article.get("title", ""),
                "category": article.get("category", ""),
                "snippet": article.get("snippet", ""),
                "resolution_steps": article.get("resolution_steps", []),
                "url_stub": article.get("url_stub", ""),
                "score": round(score, 2),
                "is_direct": is_direct,
            }
        )

    scored.sort(key=lambda x: (x["score"], x["is_direct"]), reverse=True)
    return scored[:top_k]
