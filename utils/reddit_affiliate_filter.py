import re
from typing import Dict, List, Tuple

STRONG_INCLUDE_TERMS = {
    "blackhawk network",
    "giftcards.com",
    "giftcardmall",
    "cardpool",
    "omnicard",
    "cashstar",
    "tango card",
}

WEAK_INCLUDE_TERMS = {
    "giftcards",
    "gift cards",
    "gift card",
}

EXCLUDE_PATTERNS = {
    "hockey_context": re.compile(r"\b(blackhawks|nhl|hockey)\b", re.IGNORECASE),
    "job_posting": re.compile(r"\b(job|jobs|career|careers|hiring|internship|recruiter|recruiting)\b", re.IGNORECASE),
}

BHN_CONTEXT_TERMS = {
    "blackhawk",
    "network",
    "gift card",
    "gift cards",
    "giftcard",
    "giftcardmall",
    "prepaid",
    "issuer",
    "merchant",
    "reload",
    "balance",
    "activation",
    "redemption",
}

BUSINESS_CONTEXT_TERMS = {"issuer", "merchant", "prepaid", "activation", "balance", "reload", "redemption"}

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
BHN_WORD_PATTERN = re.compile(r"\bbhn\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _normalize_text(title: str, body: str) -> str:
    text = f"{title} {body}".lower()
    text = URL_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def evaluate_affiliate_relevance(title: str, body: str, threshold: float = 2.5) -> Dict[str, object]:
    """Two-stage relevance decision for BHN/Giftcards affiliate context."""
    raw_text = f"{title} {body}".lower()
    text = _normalize_text(title=title, body=body)
    tokens = set(_tokenize(text))

    for reason_key, pattern in EXCLUDE_PATTERNS.items():
        if pattern.search(text):
            return {
                "include": False,
                "stage": "context_gate",
                "score": 0.0,
                "reason": f"excluded:{reason_key}",
                "signals": [],
            }

    strong_hits = [term for term in STRONG_INCLUDE_TERMS if term in text]
    weak_hits = [term for term in WEAK_INCLUDE_TERMS if term in text]
    has_bhn = bool(BHN_WORD_PATTERN.search(text))
    has_bhn_in_raw = "bhn" in raw_text
    has_bhn_context = any(ctx in text for ctx in BHN_CONTEXT_TERMS)

    if has_bhn_in_raw and not has_bhn:
        return {
            "include": False,
            "stage": "context_gate",
            "score": 0.0,
            "reason": "excluded:url_token_noise",
            "signals": [],
        }

    if has_bhn and not has_bhn_context:
        return {
            "include": False,
            "stage": "context_gate",
            "score": 0.0,
            "reason": "excluded:bhn_without_context",
            "signals": [],
        }

    passes_context = bool(strong_hits) or (has_bhn and has_bhn_context) or (weak_hits and (bool(strong_hits) or has_bhn_context))
    if not passes_context:
        if weak_hits:
            return {
                "include": False,
                "stage": "context_gate",
                "score": 0.0,
                "reason": "excluded:weak_term_without_brand_context",
                "signals": [],
            }
        return {
            "include": False,
            "stage": "context_gate",
            "score": 0.0,
            "reason": "excluded:no_brand_context",
            "signals": [],
        }

    score = 0.0
    signals: List[str] = []
    if strong_hits:
        score += 2.0 + min(2.0, float(len(strong_hits)))
        signals.append(f"strong:{len(strong_hits)}")
    if has_bhn and has_bhn_context:
        score += 2.0
        signals.append("bhn:context")
    if weak_hits:
        score += 0.5
        signals.append("weak:gift_card")
    if tokens.intersection(BUSINESS_CONTEXT_TERMS):
        score += 0.5
        signals.append("business_context")

    if score < threshold:
        return {
            "include": False,
            "stage": "score_gate",
            "score": round(score, 3),
            "reason": "excluded:below_threshold",
            "signals": signals,
        }

    return {
        "include": True,
        "stage": "score_gate",
        "score": round(score, 3),
        "reason": f"matched:score={score:.2f}",
        "signals": signals,
    }


def score_affiliate_relevance(title: str, body: str, threshold: float = 2.5) -> Tuple[float, str]:
    result = evaluate_affiliate_relevance(title=title, body=body, threshold=threshold)
    return float(result["score"]), str(result["reason"])


def is_affiliate_relevant(title: str, body: str, threshold: float = 2.5) -> Tuple[bool, str]:
    result = evaluate_affiliate_relevance(title=title, body=body, threshold=threshold)
    return bool(result["include"]), str(result["reason"])
