from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import (
    DEFAULT_COMPANY_NAME,
    DEFAULT_COMPANY_SYNONYMS,
    DEFAULT_RELEVANCE_THRESHOLD,
    _dedupe_reviews,
    _rank_reviews_for_analysis,
    analyze_reviews,
    fetch_reddit_reviews,
    fetch_web_reviews,
)
from utils.keyword_diagnostics import build_response_playbook, compute_brand_health_summary, compute_keyword_diagnostics
from triage_agent import (
    CRITICALITY_THRESHOLD,
    analyze_complaint,
    compute_criticality_score,
    decide_triage_action,
    fetch_reddit_post,
    generate_response_draft,
)

load_dotenv()

app = FastAPI(title="SignalAgents API", version="0.1.0")

origins_raw = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
)
origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS: Dict[str, Dict[str, Any]] = {}


class AnalyzeRunRequest(BaseModel):
    company_name: str = Field(default=DEFAULT_COMPANY_NAME)
    synonyms: str = Field(default=DEFAULT_COMPANY_SYNONYMS)
    source_mode: Literal["Reddit", "Web", "Reddit + Web"] = "Reddit"
    subreddit: str = "all"
    years_back: int = Field(default=1, ge=1, le=5)
    reddit_posts_per_term: int = Field(default=15, ge=5, le=100)
    reddit_pages_per_term: int = Field(default=5, ge=1, le=10)
    relevance_threshold: float = Field(default=DEFAULT_RELEVANCE_THRESHOLD, ge=1.0, le=4.0)
    max_reviews: int = Field(default=100, ge=10, le=200)
    web_results_per_term: int = Field(default=5, ge=3, le=20)
    strictness_preset: Literal["Strict", "Balanced", "Broad"] = "Balanced"


class AnalyzeRunResponse(BaseModel):
    run_id: str
    created_at: str
    cached: bool


class CompareRunRequest(BaseModel):
    company_a: AnalyzeRunRequest
    company_b: AnalyzeRunRequest


class TriageUrlRequest(BaseModel):
    reddit_url: str


PRESET_MAP: Dict[str, Dict[str, Any]] = {
    "Strict": {"relevance_threshold": 3.0, "max_reviews": 80},
    "Balanced": {"relevance_threshold": 2.5, "max_reviews": 100},
    "Broad": {"relevance_threshold": 2.0, "max_reviews": 120},
}


def _cache_key(payload: AnalyzeRunRequest) -> str:
    data = payload.model_dump()
    packed = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _run_analysis(payload: AnalyzeRunRequest) -> Dict[str, Any]:
    preset = PRESET_MAP.get(payload.strictness_preset, PRESET_MAP["Balanced"])
    relevance_threshold = float(payload.relevance_threshold or preset["relevance_threshold"])
    max_reviews = int(payload.max_reviews or preset["max_reviews"])

    combined_reviews: List[Dict[str, Any]] = []
    skipped = 0
    reddit_diagnostics: Optional[Dict[str, Any]] = None

    if payload.source_mode in {"Reddit", "Reddit + Web"}:
        reddit_reviews, reddit_skipped, reddit_error, reddit_diagnostics = fetch_reddit_reviews(
            company_name=payload.company_name,
            synonyms_raw=payload.synonyms,
            subreddit_name=payload.subreddit,
            limit_per_term=payload.reddit_posts_per_term,
            years_back=payload.years_back,
            relevance_threshold=relevance_threshold,
            max_pages=payload.reddit_pages_per_term,
        )
        skipped += reddit_skipped
        if reddit_error:
            cached_path = Path(__file__).resolve().parent.parent / "data" / "scheduled_results.json"
            if cached_path.exists():
                cached = json.loads(cached_path.read_text())
                return {
                    "rows": cached.get("rows", []),
                    "skipped": cached.get("skipped", 0),
                    "source": payload.source_mode,
                    "raw_count": cached.get("raw_count", 0),
                    "deduped_count": cached.get("deduped_count", 0),
                    "analyzed_count": cached.get("analyzed_count", len(cached.get("rows", []))),
                    "reddit_diagnostics": cached.get("reddit_diagnostics") or reddit_diagnostics,
                    "relevance_threshold": relevance_threshold,
                    "max_analyzed_reviews": max_reviews,
                    "brand_health_summary": cached.get("brand_health_summary", {}),
                    "keyword_diagnostics": cached.get("keyword_diagnostics", {}),
                    "response_playbook": cached.get("response_playbook", []),
                    "used_cached_results": True,
                    "warning": f"Reddit fetch failed and cached data was used: {reddit_error}",
                }
            raise HTTPException(status_code=502, detail=f"Failed to load Reddit data: {reddit_error}")

        combined_reviews.extend(reddit_reviews)

    if payload.source_mode in {"Web", "Reddit + Web"}:
        web_reviews, web_skipped, web_error = fetch_web_reviews(
            company_name=payload.company_name,
            synonyms_raw=payload.synonyms,
            count_per_term=payload.web_results_per_term,
        )
        skipped += web_skipped
        if web_error:
            raise HTTPException(status_code=502, detail=f"Failed to load Web data: {web_error}")
        combined_reviews.extend(web_reviews)

    deduped_reviews = _dedupe_reviews(combined_reviews)
    selected_reviews = _rank_reviews_for_analysis(deduped_reviews, top_n=max_reviews)

    if not selected_reviews:
        raise HTTPException(
            status_code=400,
            detail="No valid real reviews found from selected sources. Update inputs and try again.",
        )

    results = analyze_reviews(selected_reviews)
    df = pd.DataFrame(results)

    brand_health_summary = compute_brand_health_summary(df)
    keyword_diagnostics = compute_keyword_diagnostics(
        df,
        top_n=15,
        min_keyword_mentions=5,
        min_negative_mentions=2,
        risk_lift_threshold=1.6,
    )
    response_playbook = build_response_playbook(
        issue_diagnosis=keyword_diagnostics.get("issue_diagnosis", []),
        brand_health_summary=brand_health_summary,
        subreddit_risk=keyword_diagnostics.get("subreddit_risk", []),
        max_actions=5,
    )

    return {
        "rows": results,
        "skipped": skipped,
        "source": payload.source_mode,
        "raw_count": len(combined_reviews),
        "deduped_count": len(deduped_reviews),
        "analyzed_count": len(selected_reviews),
        "reddit_diagnostics": reddit_diagnostics,
        "relevance_threshold": relevance_threshold,
        "max_analyzed_reviews": max_reviews,
        "brand_health_summary": brand_health_summary,
        "keyword_diagnostics": keyword_diagnostics,
        "response_playbook": response_playbook,
        "used_cached_results": False,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "signalagents-api"}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "signalagents-api",
        "status": "ok",
        "routes": ["/health", "/analyze/run", "/analyze/results/{run_id}", "/compare/run", "/triage/analyze_url"],
    }


@app.post("/analyze/run", response_model=AnalyzeRunResponse)
def analyze_run(payload: AnalyzeRunRequest) -> AnalyzeRunResponse:
    key = _cache_key(payload)

    for rid, data in RUNS.items():
        if data.get("cache_key") == key:
            return AnalyzeRunResponse(run_id=rid, created_at=data["created_at"], cached=True)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    result = _run_analysis(payload)
    RUNS[run_id] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": key,
        "request": payload.model_dump(),
        "result": result,
    }

    return AnalyzeRunResponse(run_id=run_id, created_at=RUNS[run_id]["created_at"], cached=False)


@app.get("/analyze/results/{run_id}")
def analyze_results(run_id: str) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run_id,
        "created_at": run["created_at"],
        "request": run["request"],
        "result": run["result"],
    }


@app.post("/compare/run")
def compare_run(payload: CompareRunRequest) -> Dict[str, Any]:
    result_a = _run_analysis(payload.company_a)
    result_b = _run_analysis(payload.company_b)

    a_summary = result_a.get("brand_health_summary", {})
    b_summary = result_b.get("brand_health_summary", {})

    delta = {
        "avg_sentiment_score_delta": float(a_summary.get("avg_sentiment_score", 0.0)) - float(b_summary.get("avg_sentiment_score", 0.0)),
        "negative_share_delta": float(a_summary.get("negative_share", 0.0)) - float(b_summary.get("negative_share", 0.0)),
        "positive_share_delta": float(a_summary.get("positive_share", 0.0)) - float(b_summary.get("positive_share", 0.0)),
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "company_a": {
            "request": payload.company_a.model_dump(),
            "result": result_a,
        },
        "company_b": {
            "request": payload.company_b.model_dump(),
            "result": result_b,
        },
        "delta": delta,
    }


@app.post("/triage/analyze_url")
def triage_analyze_url(payload: TriageUrlRequest) -> Dict[str, Any]:
    post = fetch_reddit_post(payload.reddit_url.strip())
    if "error" in post:
        raise HTTPException(status_code=400, detail=f"Failed to fetch post: {post['error']}")

    complaint_text = f"{post.get('title', '')} {post.get('body', '')}".strip()
    analysis = analyze_complaint(complaint_text)
    if "error" in analysis:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {analysis['error']}")

    triage_decision = decide_triage_action(analysis, post)
    if "error" in triage_decision:
        raise HTTPException(status_code=502, detail=f"Triage failed: {triage_decision['error']}")

    criticality = compute_criticality_score(analysis, triage_decision)
    response_draft = generate_response_draft(post, analysis)

    return {
        "post": post,
        "analysis": analysis,
        "triage_decision": triage_decision,
        "criticality_score": criticality,
        "criticality_threshold": CRITICALITY_THRESHOLD,
        "response_draft": response_draft,
    }
