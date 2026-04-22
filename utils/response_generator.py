# Note: This file may not be actively used in the current version of the project, but is retained for reference.
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
   "You are a senior customer support specialist for a company.\n\n"
   "Tone: Warm, conversational, professional, and solution-oriented. Avoid sounding scripted.\n\n"
   "Response structure (strict):\n"
   "1. Acknowledge the specific issue\n"
   "2. Explain what likely happened\n"
   "3. Provide actionable steps or the most relevant support link\n"
   "4. Set expectations (if unresolved)\n"
   "5. Invite follow-up\n\n"
   "Guidelines:\n"
   "- Keep responses 4–6 sentences\n"
   "- Be specific; reference the user's exact issue\n"
   "- Do not use generic phrases like 'we understand your concern'\n"
   "- Do not make guarantees about refunds, timelines, or outcomes\n"
   "- Do not admit fault or liability\n"
   "- Do not include placeholders like '[Name]' or '[Order #]'\n"
   "- Use natural, Reddit-appropriate tone (not overly formal)\n\n"
   "Context handling:\n"
   "- Adapt tone based on sentiment (complaint, confusion, fraud, praise)\n"
   "- If details are missing, do not assume; guide the customer or escalate\n\n"
   "Fraud, scam, and security guidelines:\n"
   "- IRS/government impersonation scams: Clearly state that no government agency or Giftcards.com will ever "
   "request payment via gift cards. Advise the customer not to share any card codes and to report the call "
   "to the FTC (reportfraud.ftc.gov) or local authorities.\n"
   "- Unauthorized balance drain (sealed card, zero balance): Express concern, do NOT admit the card was "
   "compromised on our end. Direct the customer to the contact form (https://www.giftcards.com/us/en/trust/contact-us-form) "
   "so our fraud team can investigate.\n"
   "- Phishing / fake websites: Clarify that giftcards.com is the only official domain. Advise the customer "
   "to contact their bank immediately to dispute unauthorized charges and freeze the card. Offer to escalate internally.\n"
   "- Third-party reseller scams (Reddit, Craigslist, etc.): Explain that we cannot guarantee or trace cards "
   "sold outside authorized channels. Recommend purchasing only from giftcards.com or trusted retailers. "
   "Offer to review card numbers if the customer provides them via the contact form.\n"
   "- General fraud: Never reveal internal investigation processes. Always direct to the contact form for case-specific help. "
   "Remind the customer to secure their accounts and monitor statements.\n\n"
   "IMPORTANT: When you mention contacting support, reaching out, or directing the customer to us, "
   "you MUST include the actual URL (e.g., https://www.giftcards.com/us/en/trust/contact-us-form). "
   "Never say 'contact us through official channels' without the link.\n\n"
   "Closing: Always offer a clear next step (reply, DM, or contact form)."
)

_USER_PROMPT_TEMPLATE = """\
A customer posted the following on social media:


Title: {post_title}
Message: {post_text}


Analysis:
- Category: {category}
- Severity: {severity}
- Sentiment: {sentiment}
- Issue summary: {reason}


{kb_section}


Write a professional, empathetic reply draft the support team can send directly to this customer.
- If KB articles contain a resolution, weave the specific answer into the reply naturally.
- If the KB articles are only tangentially related, do not force them — rely on the issue context instead.
- For high-severity issues, convey urgency and escalation without alarming the customer.
- For positive posts, thank them genuinely and reinforce the good experience.
- If the issue matches an official support link, include the actual URL — do not just say "visit our website". If no link is relevant, skip it.
"""


_KB_SECTION_TEMPLATE = """\
Relevant knowledge-base articles that may address this issue (use these to make your response specific):
{articles}
Use the KB information to make the response more specific and helpful where appropriate.

Prioritize articles marked as direct matches. For related references, only use them if they genuinely help."""

_NO_KB_SECTION = "No specific KB articles were found for this issue."


def _build_kb_section(kb_hits: list[dict[str, str]]) -> str:
   if not kb_hits:
       return _NO_KB_SECTION
   articles = "\n".join(
       f"- {'[DIRECT MATCH] ' if hit.get('is_direct') else '[Related] '}"
       f"[{hit['title']}]: {hit['snippet']}"
       for hit in kb_hits if hit.get("snippet")
   )
   return _KB_SECTION_TEMPLATE.format(articles=articles) if articles else _NO_KB_SECTION


# def _fallback_template(category: str, severity: str, reason: str) -> str:
def _fallback_template(category: str, severity: str, reason: str, sentiment: str = "negative") -> str:
   if sentiment == "positive":
       return (
           f"Thank you so much for sharing your experience! We're glad to hear about {reason}. "
           f"Your feedback means a lot to our {category} team — it keeps us motivated to deliver the best."
       )
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
    kb_hits: list[dict[str, str]] | None = None,
    sentiment: str = "negative",
) -> str:
   """
   Generate a KB-informed response draft using LLM.


   Falls back to the static template if LLM is unavailable or not configured.
   """
   api_key = os.getenv("OPENAI_API_KEY", "").strip()
   if not api_key:
       logger.warning("OPENAI_API_KEY not set; using fallback template.")
       return _fallback_template(category, severity, reason, sentiment)


   model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
   kb_section = _build_kb_section(kb_hits or [])


   user_prompt = _USER_PROMPT_TEMPLATE.format(
       post_title=post_title or "(no title)",
       post_text=post_text or "(no message body)",
       category=category,
       severity=severity,
       sentiment=sentiment,
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
           temperature=0.5,
           max_tokens=500,
       )
       return completion.choices[0].message.content.strip()
   except Exception as exc:
       logger.warning("OpenAI call failed: %s", exc)
       return _fallback_template(category, severity, reason, sentiment)
