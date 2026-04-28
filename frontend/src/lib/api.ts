import type {
  AnalyzeResultEnvelope,
  AnalyzeRunRequest,
  AnalyzeRunResponse,
  CompareRunRequest,
  CompareRunResponse,
  TriageResult,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed with status ${res.status}`);
  }

  return (await res.json()) as T;
}

export async function runAnalysis(payload: AnalyzeRunRequest): Promise<AnalyzeRunResponse> {
  return fetchJson<AnalyzeRunResponse>(`${API_BASE_URL}/analyze/run`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getAnalysisResults(runId: string): Promise<AnalyzeResultEnvelope> {
  return fetchJson<AnalyzeResultEnvelope>(`${API_BASE_URL}/analyze/results/${runId}`);
}

export async function runCompare(payload: CompareRunRequest): Promise<CompareRunResponse> {
  return fetchJson<CompareRunResponse>(`${API_BASE_URL}/compare/run`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runTriageByUrl(redditUrl: string): Promise<TriageResult> {
  return fetchJson<TriageResult>(`${API_BASE_URL}/triage/analyze_url`, {
    method: "POST",
    body: JSON.stringify({ reddit_url: redditUrl }),
  });
}

export async function getHealth(): Promise<{ status: string; service: string }> {
  return fetchJson<{ status: string; service: string }>(`${API_BASE_URL}/health`);
}
