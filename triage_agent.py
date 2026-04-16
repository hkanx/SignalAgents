#!/usr/bin/env python3
# Note: This file may not be actively used in the current version of the project, but is retained for reference.
"""
Complaint Triage Agent — standalone script.

Reads a Reddit complaint post, analyzes it, and either:
 A) Drafts a customer response (using KB articles), or
 B) Creates a Jira ticket for developers.

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
from utils.jira_client import build_form_data, fill_intake_form

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REDDIT_URL_PATTERN = re.compile(
   r"https?://(www\.)?reddit\.com/r/\w+/comments/\w+"
)

# ── Triage LLM prompt ────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = (
   "You are a complaint triage specialist for a company. "
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

# ── Reddit fetch ──────────────────────────────────────────────────────

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
       category=analysis.get("category", "General Feedback"),
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

# ── Customer response (KB-powered) ───────────────────────────────────


def generate_response_draft(
   post: Dict[str, Any], analysis: Dict[str, Any]
) -> str:
   """Generate a KB-informed customer response draft."""
   # Try to use OpenSearch KB if available
   kb_hits: list = []
   try:
       endpoint = os.getenv("OPENSEARCH_ENDPOINT", "")
    #    index = os.getenv("OPENSEARCH_INDEX", "ENTER THE INDEX NAME HERE")
       index = os.getenv("OPENSEARCH_INDEX", "")
    #    region = os.getenv("AWS_REGION", "us-west-2")
       region = os.getenv("AWS_REGION", "")
    #    if endpoint:
        #    from utils.opensearch_kb import build_opensearch_client, search_kb
        #    client = build_opensearch_client(endpoint, region)
        #    query = f"{post.get('title', '')} {analysis.get('reason', '')}"
        #    kb_hits = search_kb(client, query, index)
   except Exception as exc:
       logger.warning("KB search unavailable: %s", exc)


   from utils.response_generator import generate_kb_response
   return generate_kb_response(
       post_title=post.get("title", ""),
       post_text=post.get("body") or post.get("text") or "",
       category=analysis.get("category", "General Feedback"),
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




def _print_triage(triage: Dict[str, Any]) -> None:
   action_label = (
       "Draft Customer Response" if triage["action"] == "customer_response"
       else "Create Jira Ticket"
   )
   lines = [
       f"  Action:     {action_label}",
       f"  Confidence: {triage['confidence']:.0%}",
       f"  Rationale:  {triage['rationale']}",
   ]
   if triage["action"] == "jira_ticket":
       lines.append(f"  Summary:    {triage['suggested_summary']}")
       lines.append(f"  Priority:   {triage['suggested_priority']}")
       lines.append(f"  Component:  {triage['affected_component']}")
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
   _print_section("TICKET INTAKE FORM PREVIEW", "\n".join(lines))


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


   _print_triage(triage)


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
       _print_triage(triage)


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
       print("\nBuilding intake form data...")
       form_data = build_form_data(post, analysis, triage)
       _print_form_preview(form_data)


       if confirm("Open browser and fill out the intake form?"):
           print("\nOpening browser to fill out the Intake form...")
           print("(Review the form in the browser, then come back here)")
           result = fill_intake_form(form_data)
           if result["success"]:
               print(f"\n{result['message']}")
           else:
               print(f"\nForm filling failed: {result['error']}")
       else:
           print("\nIntake form cancelled.")

if __name__ == "__main__":
   main()