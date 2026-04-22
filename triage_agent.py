"""
Complaint Triage Agent — standalone CLI script.


Reads a Reddit complaint post, analyzes it, and either:
 A) Drafts a customer response (using KB articles), or
 B) Creates a Jira ticket for giftcards.com developers.


Always confirms with the user before submitting anything.


Usage:
   python triage_agent.py
"""


import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional


import requests
from dotenv import load_dotenv
from openai import OpenAI


# Reuse existing project utilities
from analyzer import analyze_review
from utils.jira_client import build_form_data, submit_mock_intake_form


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


REDDIT_URL_PATTERN = re.compile(
   r"https?://(www\.)?reddit\.com/r/\w+/comments/\w+"
)


# ── Triage LLM prompt ────────────────────────────────────────────────


TRIAGE_SYSTEM_PROMPT = (
   "You are a complaint triage specialist for giftcards.com (operated by Blackhawk Network). "
   "Your job is to decide whether a customer complaint should receive a direct customer response "
   "or should be escalated as a bug/issue ticket for the development team."
)


TRIAGE_USER_PROMPT = """\
A customer posted the following complaint:


Title: {post_title}
Message: {post_text}
Source: {post_url}


Automated analysis results:
- Sentiment: {sentiment} (score: {sentiment_score})
- Category: {category}
- Severity: {severity}
- Issue summary: {reason}


Based on this information, decide the appropriate action:


ACTION A — "customer_response": The issue can be addressed by customer support with
information, apology, or guidance. Examples: confusion about fees, delivery timeline
questions, general dissatisfaction, policy questions.


ACTION B — "jira_ticket": The issue indicates a technical bug, system failure, or
product defect that requires developer investigation. Examples: activation failures,
checkout errors, balance display bugs, broken redemption flows, 500 errors.


Return ONLY valid JSON:
{{
 "action": "customer_response" or "jira_ticket",
 "confidence": <float between 0.0 and 1.0>,
 "rationale": "<1-2 sentence explanation>",
 "suggested_summary": "<short Jira ticket title if jira_ticket, else empty string>",
 "suggested_priority": "<highest | high | medium | low | lowest>",
 "affected_component": "<checkout | activation | delivery | balance | redemption | support | website | other>"
}}
"""


ALLOWED_ACTIONS = {"customer_response", "jira_ticket"}
ALLOWED_PRIORITIES = {"highest", "high", "medium", "low", "lowest"}
ALLOWED_COMPONENTS = {
   "checkout", "activation", "delivery", "balance",
   "redemption", "support", "website", "other",
}


# ── Criticality scoring ──────────────────────────────────────────────
# Mirrors the P1/P2/P3 severity_score pattern from keyword_diagnostics.py
# (severity_score >= 8 → P1, >= 4 → P2). Scale: 0-10.


CRITICALITY_THRESHOLD = 7.0  # only auto-fill Jira intake form above this


_DEFAULT_CATEGORY = "General Feedback"
_SEVERITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}
_CATEGORY_WEIGHT = {
   "Product Issue": 1.5,
   "UX Issue": 1.5,
   "Order Issue": 1.2,
   "Customer Support": 0.8,
   "Pricing": 0.7,
   _DEFAULT_CATEGORY: 0.5,
}




def compute_criticality_score(
   analysis: Dict[str, Any], triage: Dict[str, Any]
) -> float:
   """
   Compute a 0-10 criticality score for a complaint.


   Factors:
     - severity weight (high=3, medium=2, low=1)
     - category weight (Product/UX Issue=1.5 … General Feedback=0.5)
     - negative sentiment strength (0-1, derived from sentiment_score)
     - triage confidence (0-1)


   Returns a float on a 0-10 scale where higher = more critical.
   Threshold of 7.0 aligns with keyword_diagnostics P1 (severity_score >= 8).
   """
   sev = _SEVERITY_WEIGHT.get(analysis.get("severity", "low"), 1.0)
   cat = _CATEGORY_WEIGHT.get(analysis.get("category", _DEFAULT_CATEGORY), 1.0)
   sent = max(0.0, -float(analysis.get("sentiment_score", 0.0)))  # 0 to 1
   conf = float(triage.get("confidence", 0.5))


   raw = sev * cat * (sent * 0.4 + conf * 0.3 + 0.3)
   # Normalize: max theoretical = 3 * 1.5 * 1.0 = 4.5
   return min(10.0, round(raw * 10 / 4.5, 1))




# ── Reddit API fetch ──────────────────────────────────────────────────────


def fetch_reddit_post(url: str) -> Dict[str, Any]:
   """Fetch a single Reddit post via its public JSON endpoint."""
   clean = url.split("?")[0].rstrip("/")
   if not clean.endswith(".json"):
       clean += ".json"


   headers = {
       "User-Agent": os.getenv("REDDIT_USER_AGENT", "signalagents-triage-agent/0.1"),
   }


   try:
       resp = requests.get(clean, headers=headers, timeout=15)
       resp.raise_for_status()
       data = resp.json()
       post_data = data[0]["data"]["children"][0]["data"]
       return {
           "title": post_data.get("title", ""),
           "body": post_data.get("selftext", ""),
           "author": post_data.get("author", ""),
           "subreddit": post_data.get("subreddit", ""),
           "url": url,
           "score": post_data.get("score", 0),
           "created_utc": post_data.get("created_utc", 0),
       }
   except Exception as exc:
       logger.warning("Failed to fetch Reddit post: %s", exc)
       return {"error": str(exc)}


# ── Analysis ──────────────────────────────────────────────────────────


def analyze_complaint(text: str) -> Dict[str, Any]:
   """Run sentiment/category/severity analysis on the complaint text."""
   return analyze_review(text)


# ── Triage decision ──────────────────────────────────────────────────


def _validate_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
   """Validate and normalize triage LLM output."""
   action = str(payload.get("action", "")).strip().lower()
   if action not in ALLOWED_ACTIONS:
       raise ValueError(f"Invalid action: {action}")


   confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
   rationale = str(payload.get("rationale", "")).strip()
   summary = str(payload.get("suggested_summary", "")).strip()
   priority = str(payload.get("suggested_priority", "medium")).strip().lower()
   component = str(payload.get("affected_component", "other")).strip().lower()


   if priority not in ALLOWED_PRIORITIES:
       priority = "medium"
   if component not in ALLOWED_COMPONENTS:
       component = "other"


   return {
       "action": action,
       "confidence": confidence,
       "rationale": rationale,
       "suggested_summary": summary,
       "suggested_priority": priority,
       "affected_component": component,
   }


def decide_triage_action(
   analysis: Dict[str, Any], post: Dict[str, Any]
) -> Dict[str, Any]:
   """Use OpenAI to decide: customer_response or jira_ticket."""
   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       return {"error": "OPENAI_API_KEY not set"}


   model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
   client = OpenAI(api_key=api_key)


   post_text = post.get("body") or post.get("text") or ""
   user_prompt = TRIAGE_USER_PROMPT.format(
       post_title=post.get("title", "(no title)"),
       post_text=post_text[:3000],
       post_url=post.get("url", "(unknown)"),
       sentiment=analysis.get("sentiment", "neutral"),
       sentiment_score=analysis.get("sentiment_score", 0.0),
       category=analysis.get("category", _DEFAULT_CATEGORY),
       severity=analysis.get("severity", "low"),
       reason=analysis.get("reason", ""),
   )


   try:
       response = client.responses.create(
           model=model,
           instructions=TRIAGE_SYSTEM_PROMPT,
           input=user_prompt,
           temperature=0,
       )
       parsed = json.loads(response.output_text)
       return _validate_triage(parsed)
   except Exception as exc:
       logger.warning("Triage LLM call failed: %s", exc)
       return {"error": str(exc)}


# ── Customer response (Knowledge DataBase-powered) ───────────────────────────────────


def generate_response_draft(
   post: Dict[str, Any], analysis: Dict[str, Any]
) -> str:
   """Generate a KB-informed (knowledge base) customer response draft."""
   from utils.synthetic_kb import lookup_synthetic_kb_hits

   kb_hits = lookup_synthetic_kb_hits(
       post_title=post.get("title", ""),
       post_text=post.get("body") or post.get("text") or "",
       category=analysis.get("category", _DEFAULT_CATEGORY),
       severity=analysis.get("severity", "low"),
       reason=analysis.get("reason", ""),
       top_k=3,
   )


   from utils.response_generator import generate_kb_response
   return generate_kb_response(
       post_title=post.get("title", ""),
       post_text=post.get("body") or post.get("text") or "",
       category=analysis.get("category", _DEFAULT_CATEGORY),
       severity=analysis.get("severity", "low"),
       reason=analysis.get("reason", ""),
       kb_hits=kb_hits,
   )


# ── Display helpers ───────────────────────────────────────────────────


def _print_section(title: str, content: str) -> None:
   width = max(len(title) + 6, 50)
   print(f"\n{'=' * width}")
   print(f"   {title}")
   print(f"{'=' * width}")
   print(content)


def _print_analysis(analysis: Dict[str, Any]) -> None:
   if "error" in analysis:
       print(f"\n  [Analysis error: {analysis['error']}]")
       return
   lines = [
       f"  Sentiment:  {analysis['sentiment']} (score: {analysis['sentiment_score']})",
       f"  Category:   {analysis['category']}",
       f"  Severity:   {analysis['severity']}",
       f"  Confidence: {analysis['confidence']}",
       f"  Reason:     {analysis['reason']}",
   ]
   _print_section("COMPLAINT ANALYSIS", "\n".join(lines))


def _print_triage(triage: Dict[str, Any], crit_score: float) -> None:
   action_label = (
       "Draft Customer Response" if triage["action"] == "customer_response"
       else "Create Jira Ticket"
   )
   crit_label = (
       "CRITICAL" if crit_score >= CRITICALITY_THRESHOLD
       else "Below threshold"
   )
   lines = [
       f"  Action:       {action_label}",
       f"  Confidence:   {triage['confidence']:.0%}",
       f"  Criticality:  {crit_score}/10 ({crit_label}, threshold={CRITICALITY_THRESHOLD})",
       f"  Rationale:    {triage['rationale']}",
   ]
   if triage["action"] == "jira_ticket":
       lines.append(f"  Summary:      {triage['suggested_summary']}")
       lines.append(f"  Priority:     {triage['suggested_priority']}")
       lines.append(f"  Component:    {triage['affected_component']}")
   _print_section("TRIAGE DECISION", "\n".join(lines))


def _print_form_preview(form_data: Dict[str, str]) -> None:
   problem_preview = form_data.get("problem_statement", "")[:200]
   if len(form_data.get("problem_statement", "")) > 200:
       problem_preview += "..."
   lines = [
       f"  Email:       {form_data.get('email', 'N/A')}",
       f"  Summary:     {form_data.get('summary', 'N/A')}",
       f"  Team:        {form_data.get('requesting_team', 'N/A')}",
       f"  Audience:    {form_data.get('target_audience', 'N/A')}",
       f"  Urgency:     {form_data.get('urgency', 'N/A')}",
       f"  Timeframe:   {form_data.get('desired_delivery_timeframe', 'N/A')}",
       f"  Contract:    {form_data.get('contract_status', 'N/A')}",
       f"  Problem:     {problem_preview}",
   ]
   _print_section("INTAKE FORM PREVIEW", "\n".join(lines))


# ── Confirmation ──────────────────────────────────────────────────────


def confirm(prompt_text: str) -> bool:
   """Ask user for yes/no confirmation."""
   try:
       answer = input(f"\n{prompt_text} [y/N]: ").strip().lower()
       return answer in ("y", "yes")
   except (EOFError, KeyboardInterrupt):
       print()
       return False


# ── Main flow ─────────────────────────────────────────────────────────


def main() -> None:
   print("=" * 50)
   print("   Complaint Triage Agent")
   print("=" * 50)


   # Step 1: Get input
   user_input = input("\nEnter Reddit URL or type 'text' for manual input: ").strip()


   if not user_input:
       print("No input provided. Exiting.")
       return


   post: Dict[str, Any]
   if user_input.lower() == "text":
       print("Paste the complaint text below (press Enter twice to finish):")
       lines = []
       while True:
           line = input()
           if line == "":
               if lines and lines[-1] == "":
                   break
               lines.append(line)
           else:
               lines.append(line)
       text = "\n".join(lines).strip()
       if not text:
           print("No text provided. Exiting.")
           return
       post = {"title": "", "body": text, "url": "(manual input)", "author": "unknown"}
   elif REDDIT_URL_PATTERN.match(user_input):
       print("\nFetching Reddit post...")
       post = fetch_reddit_post(user_input)
       if "error" in post:
           print(f"Failed to fetch post: {post['error']}")
           fallback = input("Enter complaint text manually instead? [y/N]: ").strip().lower()
           if fallback not in ("y", "yes"):
               return
           print("Paste the complaint text (press Enter twice to finish):")
           lines = []
           while True:
               line = input()
               if line == "":
                   if lines and lines[-1] == "":
                       break
                   lines.append(line)
               else:
                   lines.append(line)
           post = {"title": "", "body": "\n".join(lines).strip(), "url": user_input, "author": "unknown"}
       else:
           print(f"\n  Title:     {post['title']}")
           print(f"  Author:    u/{post['author']}")
           print(f"  Subreddit: r/{post['subreddit']}")
           print(f"  Score:     {post['score']}")
           body_preview = (post['body'][:300] + "...") if len(post.get('body', '')) > 300 else post.get('body', '')
           print(f"  Body:      {body_preview}")
   else:
       # Treat as raw text input
       post = {"title": "", "body": user_input, "url": "(direct input)", "author": "unknown"}


   # Step 2: Analyze complaint
   complaint_text = f"{post.get('title', '')} {post.get('body', '')}".strip()
   if not complaint_text:
       print("No complaint text to analyze. Exiting.")
       return


   print("\nAnalyzing complaint...")
   analysis = analyze_complaint(complaint_text)
   _print_analysis(analysis)


   if "error" in analysis:
       print("\nAnalysis failed. Cannot proceed with triage.")
       return


   # Step 3: Triage decision
   print("\nRunning triage decision...")
   triage = decide_triage_action(analysis, post)


   if "error" in triage:
       print(f"\nTriage failed: {triage['error']}")
       print("Choose action manually:")
       choice = input("  [1] Draft customer response\n  [2] Create Jira ticket\n  Choice: ").strip()
       if choice == "1":
           triage = {
               "action": "customer_response",
               "confidence": 0.0,
               "rationale": "Manual override — triage LLM unavailable",
               "suggested_summary": "",
               "suggested_priority": "medium",
               "affected_component": "other",
           }
       elif choice == "2":
           summary = input("  Ticket summary: ").strip() or f"[Reddit] {post.get('title', 'Untitled')[:80]}"
           triage = {
               "action": "jira_ticket",
               "confidence": 0.0,
               "rationale": "Manual override — triage LLM unavailable",
               "suggested_summary": summary,
               "suggested_priority": "medium",
               "affected_component": "other",
           }
       else:
           print("Invalid choice. Exiting.")
           return


   # Step 3b: Compute criticality score
   crit_score = compute_criticality_score(analysis, triage)
   _print_triage(triage, crit_score)


   # Step 4: Allow override
   override = input("\nAccept this decision? [Y/n/switch]: ").strip().lower()
   if override == "n":
       print("Cancelled.")
       return
   elif override == "switch":
       if triage["action"] == "customer_response":
           triage["action"] = "jira_ticket"
           if not triage["suggested_summary"]:
               triage["suggested_summary"] = input("  Ticket summary: ").strip() or f"[Reddit] {post.get('title', 'Untitled')[:80]}"
       else:
           triage["action"] = "customer_response"
       crit_score = compute_criticality_score(analysis, triage)
       _print_triage(triage, crit_score)


   # Step 5: Execute action
   if triage["action"] == "customer_response":
       print("\nGenerating customer response draft...")
       draft = generate_response_draft(post, analysis)
       _print_section("DRAFT RESPONSE", f"  {draft}")


       if confirm("Finalize this response draft?"):
           print("\nResponse draft finalized. (Reddit posting is not yet enabled.)")
           print(f"\n{draft}")
       else:
           print("\nDraft discarded.")


   elif triage["action"] == "jira_ticket":
       if crit_score < CRITICALITY_THRESHOLD:
           print(f"\n  Criticality {crit_score}/10 is below threshold "
                 f"({CRITICALITY_THRESHOLD}). This may not warrant an intake form.")
           if not confirm("Proceed with the Jira intake form anyway?"):
               print("\nSkipped. Consider drafting a customer response instead.")
               return


       print("\nBuilding intake form data...")
       form_data = build_form_data(post, analysis, triage)
       _print_form_preview(form_data)


       if confirm("Save synthetic Jira intake artifact locally?"):
           print("\nSubmitting synthetic intake (mock mode)...")
           result = submit_mock_intake_form(form_data)
           if result["success"]:
               print(
                   f"\nMock intake submitted: {result['mock_ticket_id']} "
                   f"at {result['submitted_at']}\nArtifact: {result['artifact_path']}"
               )
           else:
               print(f"\nMock submit failed: {result['error']}")
       else:
           print("\nIntake form cancelled.")


if __name__ == "__main__":
   main()
