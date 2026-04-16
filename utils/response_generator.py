"""
LLM-powered customer response draft generator.

Uses LLM to synthesise a polished reply from the customer's post context
and relevant KB article snippets retrieved from OpenSearch.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
   "You are a professional customer support specialist for an e-commerce company "
   "Write empathetic, concise, and helpful customer-facing "
   "reply drafts. Keep responses to 3–4 sentences. Do not include placeholders like "
   "'[Name]' — write a complete, copy-ready reply."
)

_USER_PROMPT_TEMPLATE = """\
A customer posted the following on social media:


Title: {post_title}
Message: {post_text}


Analysis:
- Category: {category}
- Severity: {severity}
- Issue summary: {reason}


{kb_section}


Write a professional, empathetic reply draft the support team can send directly to this customer.
"""


_KB_SECTION_TEMPLATE = """\
Relevant knowledge-base articles that may address this issue:
{articles}
Use the KB information to make the response more specific and helpful where appropriate."""


_NO_KB_SECTION = "No specific KB articles were found for this issue."


def _build_kb_section(kb_hits: list[dict[str, str]]) -> str:
   if not kb_hits:
       return _NO_KB_SECTION
   articles = "\n".join(
       f"- [{hit['title']}]: {hit['snippet']}" for hit in kb_hits if hit.get("snippet")
   )
   return _KB_SECTION_TEMPLATE.format(articles=articles) if articles else _NO_KB_SECTION


def _fallback_template(category: str, severity: str, reason: str) -> str:
   urgency = "high-priority" if severity == "high" else ("time-sensitive" if severity == "medium" else "important")
   return (
       f"Thanks for flagging this. We are sorry about {reason}. "
       f"Our {category} team is reviewing this as a {urgency} case and will share next steps soon."
   )


def generate_kb_response(
   post_title: str,
   post_text: str,
   category: str,
   severity: str,
   reason: str,
   kb_hits: list[dict[str, str]],
) -> str:
   """
   Generate a KB-informed response draft using LLM.


   Falls back to the static template if LLM is unavailable or not configured.
   """
   api_key = os.getenv("OPENAI_API_KEY", "").strip()
   if not api_key:
       logger.warning("OPENAI_API_KEY not set; using fallback template.")
       return _fallback_template(category, severity, reason)


   model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
   kb_section = _build_kb_section(kb_hits)


   user_prompt = _USER_PROMPT_TEMPLATE.format(
       post_title=post_title or "(no title)",
       post_text=post_text or "(no message body)",
       category=category,
       severity=severity,
       reason=reason or "the issue you reported",
       kb_section=kb_section,
   )

   try:
       client = OpenAI(api_key=api_key)
       completion = client.chat.completions.create(
           model=model,
           messages=[
               {"role": "system", "content": _SYSTEM_PROMPT},
               {"role": "user", "content": user_prompt},
           ],
           temperature=0.4,
           max_tokens=300,
       )
       return completion.choices[0].message.content.strip()
   except Exception as exc:
       logger.warning("OpenAI call failed: %s", exc)
       return _fallback_template(category, severity, reason)