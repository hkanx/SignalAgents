import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}
ALLOWED_CATEGORIES = {
    "Product Issue",
    "Order Issue",
    "UX Issue",
    "Customer Support",
    "Pricing",
    "General Feedback",
}
ALLOWED_SEVERITIES = {"low", "medium", "high"}

PROMPT_TEMPLATE = """You are a review analysis assistant.
Analyze the review text and return ONLY valid JSON with this exact schema:
{{
  "sentiment": "positive | neutral | negative",
  "confidence": float,
  "category": "Product Issue | Order Issue | UX Issue | Customer Support | Pricing | General Feedback",
  "severity": "low | medium | high",
  "reason": "short explanation"
}}
Rules:
- Output must be JSON only. No markdown. No extra keys.
- confidence must be between 0.0 and 1.0.
- Use one category only.
- Keep reason concise (1-2 sentences).

Review text:
<review>
{review_text}
</review>
"""


def _error_result(message: str) -> Dict[str, Any]:
    return {
        "sentiment": "neutral",
        "confidence": 0.0,
        "category": "General Feedback",
        "severity": "low",
        "reason": "Analysis failed.",
        "error": message,
    }


def _validate_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = str(payload.get("sentiment", "")).lower().strip()
    category = str(payload.get("category", "")).strip()
    severity = str(payload.get("severity", "")).lower().strip()
    reason = str(payload.get("reason", "")).strip()

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise ValueError("confidence must be a float")

    if sentiment not in ALLOWED_SENTIMENTS:
        raise ValueError("invalid sentiment")
    if category not in ALLOWED_CATEGORIES:
        raise ValueError("invalid category")
    if severity not in ALLOWED_SEVERITIES:
        raise ValueError("invalid severity")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence out of range")
    if not reason:
        raise ValueError("reason is required")

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "category": category,
        "severity": severity,
        "reason": reason,
    }


def _build_prompt(review_text: str) -> str:
    return PROMPT_TEMPLATE.format(review_text=review_text)


def analyze_review(text: str) -> Dict[str, Any]:
    """Analyze a single review with OpenAI and return structured JSON."""
    if not isinstance(text, str) or not text.strip():
        return _error_result("Empty or invalid review text")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _error_result("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    try:
        prompt = _build_prompt(text.strip())
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Prompt build failed: {exc}")

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"OpenAI request failed: {exc}")

    try:
        raw_text = response.output_text
        parsed = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Failed to parse model JSON: {exc}")

    try:
        return _validate_output(parsed)
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Model output validation failed: {exc}")
