export type SourceMode = "Reddit" | "Web" | "Reddit + Web";
export type StrictnessPreset = "Strict" | "Balanced" | "Broad";

export type AnalyzeRunRequest = {
  company_name: string;
  synonyms: string;
  source_mode: SourceMode;
  subreddit: string;
  years_back: 1 | 2 | 3 | 4 | 5;
  reddit_posts_per_term: number;
  reddit_pages_per_term: number;
  relevance_threshold: number;
  max_reviews: number;
  web_results_per_term: number;
  strictness_preset: StrictnessPreset;
};

export type AnalyzeRunResponse = {
  run_id: string;
  created_at: string;
  cached: boolean;
};

export type AnalyzeResultEnvelope = {
  run_id: string;
  created_at: string;
  request: AnalyzeRunRequest;
  result: {
    rows: Record<string, unknown>[];
    skipped: number;
    source: SourceMode;
    raw_count: number;
    deduped_count: number;
    analyzed_count: number;
    reddit_diagnostics?: Record<string, unknown> | null;
    relevance_threshold: number;
    max_analyzed_reviews: number;
    brand_health_summary: Record<string, unknown>;
    keyword_diagnostics: Record<string, unknown>;
    response_playbook: Record<string, unknown>[];
    used_cached_results?: boolean;
    warning?: string;
  };
};

export type CompareRunRequest = {
  company_a: AnalyzeRunRequest;
  company_b: AnalyzeRunRequest;
};

export type CompareRunResponse = {
  created_at: string;
  company_a: { request: AnalyzeRunRequest; result: AnalyzeResultEnvelope["result"] };
  company_b: { request: AnalyzeRunRequest; result: AnalyzeResultEnvelope["result"] };
  delta: Record<string, number>;
};

export type TriageResult = {
  post: Record<string, unknown>;
  analysis: Record<string, unknown>;
  triage_decision: Record<string, unknown>;
  criticality_score: number;
  criticality_threshold: number;
  response_draft: string;
};
