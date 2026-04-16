# Note: This file may not be actively used in the current version of the project, but is retained for reference.
"""
OpenSearch KB client with AWS SigV4 authentication.

Uses the current boto3 credential chain (SSO, env vars, instance profile, etc.)
so no hardcoded credentials are required.

Index schema for kb_pages_gift_cards_us_en:
 - text        (str)  Q&A or product-catalog content
 - chunk_index (int)  position within original document
 - metadata    (obj)  currently empty
 - embedding   (vec)  dense vector — excluded from text searches

 # INSERT HOW MANY DOCS TOTAL IN THE KNOWLEDGE BASE (KB) HERE
Content breakdown (XX docs):
 - X FAQ articles  (Q: / A: format) 
 - Y brand/product catalog pages  (start with brand name + KEYWORDS:)
"""

import logging

# import boto3
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)


_SNIPPET_LENGTH = 500
# Minimum pure-match score for a FAQ doc to be considered "directly relevant"
_DIRECT_SCORE_THRESHOLD = 0.5


# def build_opensearch_client(endpoint: str, region: str) -> OpenSearch:
#    """Return an OpenSearch client signed with the current AWS session credentials."""
#    session = boto3.Session()
#    credentials = session.get_credentials().get_frozen_credentials()
#    auth = AWS4Auth(
#        credentials.access_key,
#        credentials.secret_key,
#        region,
#        "es",
#        session_token=credentials.token,
#    )
#    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
#    return OpenSearch(
#        hosts=[{"host": host, "port": 443}],
#        http_auth=auth,
#        use_ssl=True,
#        verify_certs=True,
#        connection_class=RequestsHttpConnection,
#        timeout=10,
#    )

def _derive_title(text: str) -> str:
   """Extract a short title from the first line of the KB text chunk."""
   first_line = text.strip().splitlines()[0] if text.strip() else ""
   first_line = first_line.removeprefix("Q:").strip()
   if len(first_line) > 100:
       first_line = first_line[:97] + "…"
   return first_line or "KB Article"


def _is_faq_doc(text: str) -> bool:
   """Return True if this document is a Q&A support article."""
   stripped = text.strip()
   return stripped.startswith("Q:") and "\nA:" in stripped


def _make_hit(text: str, url: str, is_direct: bool) -> dict:
   snippet = text[:_SNIPPET_LENGTH] + ("…" if len(text) > _SNIPPET_LENGTH else "")
   return {
       "title": _derive_title(text),
       "snippet": snippet,
       "url": url,
       "is_direct": is_direct,
   }

# def _fetch_scored_faq_docs(client: OpenSearch, index: str, query: str) -> list[dict]:
#    """
#    Fetch all FAQ docs and score them against the query.
#    Returns list of dicts with keys: text, url, score.
#    """
#    body = {
#        "size": 10,
#        "_source": ["text", "metadata"],
#        "query": {
#            "bool": {
#                "must": [
#                    {"match": {"text": {"query": query, "fuzziness": "AUTO"}}},
#                    {"bool": {
#                        "must_not": {"match_phrase": {"text": "KEYWORDS:"}},
#                    }},
#                ]
#            }
#        },
#    }
#    try:
#        response = client.search(index=index, body=body)
#        hits = response.get("hits", {}).get("hits", [])
#        results = []
#        for hit in hits:
#            text = str(hit.get("_source", {}).get("text") or "").strip()
#            if _is_faq_doc(text):
#                url = str(hit.get("_source", {}).get("metadata", {}).get("url") or "").strip()
#                results.append({"text": text, "url": url, "score": hit.get("_score") or 0})
#        return results
#    except Exception as exc:
#        logger.warning("FAQ scored fetch failed: %s", exc)
#        return []

# def _fetch_all_faq_docs(client: OpenSearch, index: str) -> list[dict]:
#    """Return all FAQ docs as (text, url) pairs without scoring."""
#    body = {
#        "size": 10,
#        "_source": ["text", "metadata"],
#        "query": {
#            "bool": {
#                "must_not": {"match_phrase": {"text": "KEYWORDS:"}},
#            }
#        },
#    }
#    try:
#        response = client.search(index=index, body=body)
#        results = []
#        for hit in response.get("hits", {}).get("hits", []):
#            text = str(hit.get("_source", {}).get("text") or "").strip()
#            if _is_faq_doc(text):
#                url = str(hit.get("_source", {}).get("metadata", {}).get("url") or "").strip()
#                results.append({"text": text, "url": url})
#        return results
#    except Exception as exc:
#        logger.warning("FAQ fetch failed: %s", exc)
#        return []

def search_kb(
#    client: OpenSearch,
   query: str,
   index: str,
   top_k: int = 3,
) -> list[dict]:
   """
   Search the KB and return up to top_k results.

   Strategy:
   - Always include all FAQ docs (there are only 3); mark is_direct=True when
     their pure-match score exceeds the threshold for this query.
   - Fill remaining slots (up to top_k) with highest-scoring catalog pages.


   Each result dict: title, snippet, url, is_direct.
   Returns an empty list only on connection errors.
   """
   if not query or not query.strip():
       return []


   # 1. Score FAQ docs against the query
   scored_faqs = _fetch_scored_faq_docs(client, index, query)
   # Get unscored FAQ docs to ensure we always have all 3 even if none matched
   all_faqs = _fetch_all_faq_docs(client, index)


   # Build FAQ result set: scored ones first, then unscored ones as fallback
   faq_results: list[dict] = []
   seen_titles: set[str] = set()


   for item in scored_faqs:
       entry = _make_hit(item["text"], item["url"], is_direct=(item["score"] >= _DIRECT_SCORE_THRESHOLD))
       if entry["title"] not in seen_titles:
           seen_titles.add(entry["title"])
           faq_results.append(entry)


   for item in all_faqs:
       entry = _make_hit(item["text"], item["url"], is_direct=False)
       if entry["title"] not in seen_titles:
           seen_titles.add(entry["title"])
           faq_results.append(entry)


   # 2. Fill remaining slots with best-matching catalog pages
   catalog_needed = max(0, top_k - len(faq_results))
   catalog_results: list[dict] = []


   if catalog_needed > 0:
       body = {
           "size": catalog_needed + 5,  # overfetch to allow dedup
           "_source": ["text", "metadata"],
           "query": {
               "bool": {
                   "must": {"match": {"text": {"query": query, "fuzziness": "AUTO"}}},
                   "must_not": [
                       # Exclude FAQ docs (already included above)
                       {"bool": {"must_not": {"match_phrase": {"text": "KEYWORDS:"}}}},
                   ],
               }
           },
       }
       try:
           response = client.search(index=index, body=body)
           for hit in response.get("hits", {}).get("hits", []):
               text = str(hit.get("_source", {}).get("text") or "").strip()
               url = str(hit.get("_source", {}).get("metadata", {}).get("url") or "").strip()
               entry = _make_hit(text, url, is_direct=False)
               if entry["snippet"] and entry["title"] not in seen_titles:
                   seen_titles.add(entry["title"])
                   catalog_results.append(entry)
               if len(catalog_results) >= catalog_needed:
                   break
       except Exception as exc:
           logger.warning("Catalog search failed: %s", exc)

   return (faq_results + catalog_results)[:top_k]