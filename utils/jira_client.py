# Note: This file may not be actively used in the current version of the project, but is retained for reference.
"""
Jira client for creating issues from triage decisions.


Supports two modes:
 1. REST API v3 — direct issue creation via Basic auth
 2. Intake form — fills out an Intake form via Playwright browser automation


Configuration is read from environment variables (see .env.example).
"""


import logging
import os
from typing import Any, Dict, Optional


import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


load_dotenv()


logger = logging.getLogger(__name__)


_JSON_CONTENT_TYPE = "application/json"


JIRA_PRIORITY_MAP = {
   "highest": "Highest",
   "high": "High",
   "medium": "Medium",
   "low": "Low",
   "lowest": "Lowest",
}

def _get_jira_config() -> Dict[str, str]:
   """Read Jira configuration from environment variables."""
   base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
   email = os.getenv("JIRA_USER_EMAIL", "")
   token = os.getenv("JIRA_API_TOKEN", "")
   project_key = os.getenv("JIRA_PROJECT_KEY", "")
   issue_type = os.getenv("JIRA_ISSUE_TYPE", "Task")


   missing = []
   if not base_url:
       missing.append("JIRA_BASE_URL")
   if not email:
       missing.append("JIRA_USER_EMAIL")
   if not token:
       missing.append("JIRA_API_TOKEN")
   if not project_key:
       missing.append("JIRA_PROJECT_KEY")


   if missing:
       raise ValueError(f"Missing required Jira env vars: {', '.join(missing)}")


   return {
       "base_url": base_url,
       "email": email,
       "token": token,
       "project_key": project_key,
       "issue_type": issue_type,
   }

def _build_adf_description(
   post: Dict[str, Any],
   analysis: Dict[str, Any],
   triage: Dict[str, Any],
) -> Dict[str, Any]:
   """Build an Atlassian Document Format description for the Jira issue."""

   def _text_node(text: str) -> Dict[str, Any]:
       return {"type": "text", "text": text}


   def _heading(level: int, text: str) -> Dict[str, Any]:
       return {
           "type": "heading",
           "attrs": {"level": level},
           "content": [_text_node(text)],
       }


   def _paragraph(text: str) -> Dict[str, Any]:
       return {"type": "paragraph", "content": [_text_node(text)]}


   def _list_item(text: str) -> Dict[str, Any]:
       return {
           "type": "listItem",
           "content": [{"type": "paragraph", "content": [_text_node(text)]}],
       }


   return {
       "version": 1,
       "type": "doc",
       "content": [
           _heading(3, "Customer Complaint"),
           _paragraph(post.get("body") or post.get("text") or "(no message body)"),
           _heading(3, "Analysis"),
           {
               "type": "bulletList",
               "content": [
                   _list_item(f"Category: {analysis.get('category', 'N/A')}"),
                   _list_item(f"Severity: {analysis.get('severity', 'N/A')}"),
                   _list_item(f"Sentiment: {analysis.get('sentiment', 'N/A')} ({analysis.get('sentiment_score', 'N/A')})"),
                   _list_item(f"Reason: {analysis.get('reason', 'N/A')}"),
                   _list_item(f"Component: {triage.get('affected_component', 'N/A')}"),
               ],
           },
           _heading(3, "Source"),
           _paragraph(f"Reddit: {post.get('url', 'N/A')}"),
           _heading(3, "Triage Rationale"),
           _paragraph(triage.get("rationale", "")),
       ],
   }

def build_issue_fields(
   post: Dict[str, Any],
   analysis: Dict[str, Any],
   triage: Dict[str, Any],
   custom_field_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
   """
   Build the Jira issue fields dict from triage results.


   custom_field_map: optional mapping of semantic names to Jira custom field IDs,
   e.g. {"source_url": "customfield_10050"}. Updated after form field discovery.
   """
   config = _get_jira_config()


   summary = triage.get("suggested_summary") or f"[Reddit] {post.get('title', 'Untitled')[:80]}"
   priority_name = JIRA_PRIORITY_MAP.get(
       triage.get("suggested_priority", "medium"), "Medium"
   )


   fields: Dict[str, Any] = {
       "project": {"key": config["project_key"]},
       "issuetype": {"name": config["issue_type"]},
       "summary": summary,
       "priority": {"name": priority_name},
       "description": _build_adf_description(post, analysis, triage),
       "labels": ["hannah-llmpathy"],
   }


   # Steps to Reproduce — map the complaint details
   steps = (
       f"Source: {post.get('url', 'N/A')}\n"
       f"Author: u/{post.get('author', 'unknown')}\n\n"
       f"{post.get('body') or post.get('text') or '(no details)'}"
   )
   fields["customfield_10642"] = steps


   # Environment — include the affected component from triage
   component = triage.get("affected_component", "")
   if component and component != "other":
       fields["environment"] = f"Affected area: {component}"


   # Map any additional custom fields
   if custom_field_map:
       custom_values = {
           "source_url": post.get("url", ""),
           "affected_component": triage.get("affected_component", ""),
           "customer_sentiment": analysis.get("sentiment", ""),
           "severity": analysis.get("severity", ""),
       }
       for semantic_name, field_id in custom_field_map.items():
           if semantic_name in custom_values and custom_values[semantic_name]:
               fields[field_id] = custom_values[semantic_name]


   return fields


def create_jira_issue(fields: Dict[str, Any]) -> Dict[str, Any]:
   """
   Create a Jira issue via REST API v3.


   Returns:
       On success: {"success": True, "key": "GC-123", "url": "...", "id": "12345"}
       On failure: {"success": False, "error": "...", "status_code": ..., "response_body": "..."}
   """
   try:
       config = _get_jira_config()
   except ValueError as exc:
       return {"success": False, "error": str(exc)}


   url = f"{config['base_url']}/rest/api/3/issue"
   headers = {
       "Content-Type": _JSON_CONTENT_TYPE,
       "Accept": _JSON_CONTENT_TYPE,
   }

   try:
       resp = requests.post(
           url,
           json={"fields": fields},
           headers=headers,
           auth=(config["email"], config["token"]),
           timeout=30,
       )

       if resp.status_code in (200, 201):
           data = resp.json()
           issue_key = data.get("key", "")
           return {
               "success": True,
               "key": issue_key,
               "url": f"{config['base_url']}/browse/{issue_key}",
               "id": data.get("id", ""),
           }

       return {
           "success": False,
           "error": f"Jira API returned {resp.status_code}",
           "status_code": resp.status_code,
           "response_body": resp.text[:1000],
       }

   except requests.RequestException as exc:
       logger.warning("Jira API request failed: %s", exc)
       return {"success": False, "error": f"Request failed: {exc}"}


def get_create_meta(project_key: str) -> Dict[str, Any]:
   """
   Diagnostic helper: fetch issue creation metadata to discover available fields.


   Run this once to learn which fields the project/issue-type supports,
   then update JIRA_CUSTOM_FIELD_MAP in triage_agent.py.
   """
   try:
       config = _get_jira_config()
   except ValueError as exc:
       return {"error": str(exc)}


   url = f"{config['base_url']}/rest/api/3/issue/createmeta/{project_key}/issuetypes"
   headers = {"Accept": _JSON_CONTENT_TYPE}

   try:
       resp = requests.get(
           url,
           headers=headers,
           auth=(config["email"], config["token"]),
           timeout=30,
       )
       if resp.status_code == 200:
           return resp.json()
       return {"error": f"HTTP {resp.status_code}", "body": resp.text[:1000]}
   except requests.RequestException as exc:
       return {"error": str(exc)}

# ── Intake form via Playwright ────────────────────────────────────────


# FORM_URL = (
#    "ENTER INTAKE FORM URL HERE"
# )
FORM_URL = (
   ""
)


# Maps triage priority to the form's Urgency dropdown values
_URGENCY_MAP = {
   "highest": "Critical",
   "high": "High",
   "medium": "Medium",
   "low": "Low",
   "lowest": "Low",
}

def build_form_data(
   post: Dict[str, Any],
   analysis: Dict[str, Any],
   triage: Dict[str, Any],
   email: str = "",
) -> Dict[str, str]:
   """
   Build a dict of form field values from triage results.


   Keys match the Intake: Issue Resolution Request Form fields.
   """
   summary = (
       triage.get("suggested_summary")
       or f"[Reddit] {post.get('title', 'Untitled')[:80]}"
   )

   post_body = post.get("body") or post.get("text") or "(no details)"
   source_url = post.get("url", "N/A")


   problem_statement = (
       f"A customer reported the following issue on Reddit:\n\n"
       f"{post_body}\n\n"
       f"Source: {source_url}\n"
       f"Author: u/{post.get('author', 'unknown')}"
   )

   proposed_solutions = (
       f"Investigate the reported {triage.get('affected_component', 'issue')} "
       f"based on the customer complaint.\n\n"
       f"Triage rationale: {triage.get('rationale', 'N/A')}"
   )


   expected_benefits = (
       f"Resolving this issue will improve customer experience and reduce "
       f"negative sentiment (current: {analysis.get('sentiment', 'N/A')}, "
       f"score: {analysis.get('sentiment_score', 'N/A')})."
   )


   urgency = _URGENCY_MAP.get(
       triage.get("suggested_priority", "medium"), "Medium"
   )


   urgency_reason = (
       f"Category: {analysis.get('category', 'N/A')}\n"
       f"Severity: {analysis.get('severity', 'N/A')}\n"
       f"Reason: {analysis.get('reason', 'N/A')}"
   )


   risks = (
       f"Customer has indicated intent to escalate publicly "
       f"(Reddit post with {post.get('score', 0)} upvotes). "
       f"Delayed resolution may lead to additional negative reviews."
   )


   return {
       "email": email or os.getenv("JIRA_USER_EMAIL", ""),
       "summary": summary,
       # INSERT TEAM BEING REQUESTED NAME IF APPLICABLE
       "requesting_team": "Customer Support",
       "target_audience": "External Customers",
       "problem_statement": problem_statement,
       "proposed_solutions": proposed_solutions,
       "expected_benefits": expected_benefits,
       "urgency": urgency,
       "urgency_reason": urgency_reason,
       "requirements_defined": "No",
       "risks_and_dependencies": risks,
       "desired_delivery_timeframe": "30 Days",  # exact match: <30, 30, 60, 90, 120 Days / 6 Months / >6 Months
       "contract_status": "Not Applicable",
   }

def _select_dropdown(page, combobox_index: int, value: str) -> None:
   """Click a combobox by index and select an option by visible text."""
   comboboxes = page.query_selector_all('[role="combobox"]')
   if combobox_index >= len(comboboxes):
       logger.warning("Combobox index %d out of range", combobox_index)
       return
   comboboxes[combobox_index].click()
   page.wait_for_timeout(500)
   option = page.get_by_role("option", name=value, exact=True)
   if option.count() > 0:
       option.first.click()
   else:
       logger.warning("Option '%s' not found in dropdown %d", value, combobox_index)
       page.keyboard.press("Escape")
   page.wait_for_timeout(300)

def _fill_rich_text(page, field_index: int, text: str) -> None:
   """Fill a rich-text editor field by its index among [role='textbox'] elements."""
   editors = page.query_selector_all('[role="textbox"]')
   if field_index >= len(editors):
       logger.warning("Rich text editor index %d out of range", field_index)
       return
   editors[field_index].click()
   page.wait_for_timeout(200)
   page.keyboard.type(text, delay=5)
   page.wait_for_timeout(200)

def fill_intake_form(
   form_data: Dict[str, str],
   headless: bool = False,
   auto_submit: bool = False,
) -> Dict[str, Any]:
   """
   Open the Intake form in a browser and fill it out.

   By default opens a visible browser and pauses before submit so
   the user can review. Set auto_submit=True to submit automatically
   (still requires reCAPTCHA to pass).

   Returns:
       {"success": True, "message": "..."} or {"success": False, "error": "..."}
   """
   with sync_playwright() as p:
       browser = p.chromium.launch(headless=headless)
       page = browser.new_context().new_page()

       try:
           logger.info("Navigating to intake form...")
           page.goto(FORM_URL, timeout=30000)
           page.wait_for_timeout(3000)


           # Dismiss cookie banner if present
           try:
               only_necessary = page.get_by_text("Only necessary")
               if only_necessary.count() > 0:
                   only_necessary.first.click()
                   page.wait_for_timeout(500)
           except Exception:
               pass


           # 1. Email (text input)
           page.fill('input[name="email"]', form_data["email"])


           # 2. Summary (text input)
           page.fill('input[name="summary"]', form_data["summary"])


           # 3. Requesting Team (text input)
           page.fill(
               'input[name="customfield_12071"]',
               form_data["requesting_team"],
           )


           # 4. Target Audience (dropdown index 0)
           _select_dropdown(page, 0, form_data["target_audience"])


           # 5. Problem Statement (rich text index 0)
           _fill_rich_text(page, 0, form_data["problem_statement"])


           # 6. Proposed Solutions (rich text index 1)
           _fill_rich_text(page, 1, form_data["proposed_solutions"])


           # 7. Expected Benefits (rich text index 2)
           _fill_rich_text(page, 2, form_data["expected_benefits"])


           # 8. Urgency (dropdown index 1)
           _select_dropdown(page, 1, form_data["urgency"])


           # 9. Urgency Reason (rich text index 3)
           _fill_rich_text(page, 3, form_data["urgency_reason"])


           # 10. Requirements Defined? (dropdown index 2)
           _select_dropdown(page, 2, form_data["requirements_defined"])


           # 11. Risks and Dependencies (rich text index 4)
           _fill_rich_text(page, 4, form_data["risks_and_dependencies"])


           # 12. Desired Delivery Timeframe (dropdown index 3)
           _select_dropdown(page, 3, form_data["desired_delivery_timeframe"])


           # 13. Contract Status (dropdown index 4)
           _select_dropdown(page, 4, form_data["contract_status"])


           logger.info("Form filled successfully.")


           # Screenshot the filled form
           page.screenshot(path="/tmp/jira_form_filled.png", full_page=True)


           if auto_submit:
               page.get_by_role("button", name="Submit").click()
               page.wait_for_timeout(5000)
               page.screenshot(
                   path="/tmp/jira_form_submitted.png", full_page=True
               )
               browser.close()
               return {
                   "success": True,
                   "message": "Form submitted. Screenshot at /tmp/jira_form_submitted.png",
               }

           # Pause for user review
           print("\n  Form has been filled out in the browser.")
           print("  Review it, then come back here.")
           print("  Screenshot saved to /tmp/jira_form_filled.png")
           input("  Press Enter to close the browser...")
           browser.close()
           return {
               "success": True,
               "message": "Form filled. User reviewed in browser.",
           }

       except Exception as exc:
           logger.warning("Form filling failed: %s", exc)
           try:
               page.screenshot(
                   path="/tmp/jira_form_error.png", full_page=True
               )
           except Exception:
               pass
           browser.close()
           return {"success": False, "error": str(exc)}