"use client";

import { useMemo, useState } from "react";
import { getAnalysisResults, runAnalysis, runCompare, runTriageByUrl } from "../lib/api";
import type { AnalyzeResultEnvelope, AnalyzeRunRequest, CompareRunResponse, TriageResult } from "../lib/types";

type TabId = "howto" | "executive" | "comparison" | "response" | "analytics" | "triage";

type Row = Record<string, unknown>;

const defaultRequest: AnalyzeRunRequest = {
  company_name: "Giftcards.com",
  synonyms: "giftcards.com, bhn, blackhawk network, giftcardmall, CashStar, tango card",
  source_mode: "Reddit",
  subreddit: "all",
  years_back: 1,
  reddit_posts_per_term: 15,
  reddit_pages_per_term: 5,
  relevance_threshold: 2.5,
  max_reviews: 100,
  web_results_per_term: 5,
  strictness_preset: "Balanced",
};

const defaultCompetitor: AnalyzeRunRequest = {
  ...defaultRequest,
  company_name: "Gift Card Granny",
  synonyms: "giftcardgranny, vanillagift, giftya, egifter, perfectgift.com",
};

function pct(v: number): string {
  if (!Number.isFinite(v)) return "0.0%";
  return `${(v * 100).toFixed(1)}%`;
}

function n(v: unknown, d = 2): number {
  const x = Number(v);
  if (!Number.isFinite(x)) return 0;
  return Number(x.toFixed(d));
}

function sentimentClass(sentiment: string): string {
  const s = sentiment.toLowerCase();
  if (s === "negative") return "bad";
  if (s === "positive") return "good";
  return "neutral";
}

function renderSimpleBar(label: string, value: number, colorClass: "good" | "bad" | "neutral" | "info") {
  const pctValue = Math.max(0, Math.min(100, value));
  return (
    <div style={{ marginBottom: 8 }}>
      <div className="label" style={{ display: "flex", justifyContent: "space-between" }}>
        <span>{label}</span><span>{pctValue.toFixed(1)}%</span>
      </div>
      <div style={{ width: "100%", background: "#e5e7eb", borderRadius: 999, height: 8 }}>
        <div className={colorClass} style={{ width: `${pctValue}%`, height: 8, borderRadius: 999, background: "currentColor" }} />
      </div>
    </div>
  );
}

function responseTemplate(row: Row): string {
  const sentiment = String(row.sentiment || "negative").toLowerCase();
  const reason = String(row.reason || "the issue you reported");
  const category = String(row.category || "General Feedback");
  if (sentiment === "positive") {
    return `Thank you for sharing this positive experience about ${reason}. Our ${category} team appreciates your feedback.`;
  }
  return `Thanks for flagging this. We are sorry about ${reason}. Our ${category} team is reviewing this and will share next steps.`;
}

export default function HomePage() {
  const [tab, setTab] = useState<TabId>("howto");
  const [request, setRequest] = useState<AnalyzeRunRequest>(defaultRequest);
  const [competitor, setCompetitor] = useState<AnalyzeRunRequest>(defaultCompetitor);
  const [runId, setRunId] = useState("");
  const [analysis, setAnalysis] = useState<AnalyzeResultEnvelope | null>(null);
  const [comparison, setComparison] = useState<CompareRunResponse | null>(null);
  const [triageUrl, setTriageUrl] = useState("");
  const [triage, setTriage] = useState<TriageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const rows = (analysis?.result.rows ?? []) as Row[];
  const bhs = (analysis?.result.brand_health_summary ?? {}) as Row;
  const kd = (analysis?.result.keyword_diagnostics ?? {}) as Row;
  const playbook = (analysis?.result.response_playbook ?? []) as Row[];
  const diagnostics = (analysis?.result.reddit_diagnostics ?? {}) as Row;

  const negativeRows = useMemo(
    () => rows.filter((r) => String(r.sentiment || "").toLowerCase() === "negative")
      .sort((a, b) => n(a.sentiment_score) - n(b.sentiment_score)),
    [rows],
  );

  const positiveRows = useMemo(
    () => rows.filter((r) => String(r.sentiment || "").toLowerCase() === "positive")
      .sort((a, b) => n(b.sentiment_score) - n(a.sentiment_score)),
    [rows],
  );

  const sentimentCounts = useMemo(() => {
    const counts = { positive: 0, neutral: 0, negative: 0 };
    for (const row of rows) {
      const s = String(row.sentiment || "neutral").toLowerCase();
      if (s in counts) counts[s as keyof typeof counts] += 1;
    }
    return counts;
  }, [rows]);

  const issueDx = ((kd.issue_diagnosis as Row[]) || []);
  const subredditRisk = ((kd.subreddit_risk as Row[]) || []);

  const avgSent = n(bhs.avg_sentiment_score, 3);
  const posShare = n(bhs.positive_share, 4);
  const negShare = n(bhs.negative_share, 4);
  const netSent = posShare - negShare;
  const brandImageScore = Math.max(0, Math.min(100, Math.round((avgSent + 1) * 50)));
  const negDeltaPts = n(bhs.negative_share_delta, 4) * 100;

  async function onRunAnalysis() {
    setLoading(true);
    setError("");
    try {
      const run = await runAnalysis(request);
      setRunId(run.run_id);
      const full = await getAnalysisResults(run.run_id);
      setAnalysis(full);
      setTab("executive");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  async function onRunComparison() {
    setLoading(true);
    setError("");
    try {
      const resp = await runCompare({ company_a: request, company_b: competitor });
      setComparison(resp);
      setTab("comparison");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  async function onRunTriage() {
    if (!triageUrl.trim()) return;
    setLoading(true);
    setError("");
    try {
      const resp = await runTriageByUrl(triageUrl.trim());
      setTriage(resp);
      setTab("triage");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <section className="header">
        <h1 style={{ margin: 0 }}>SignalAgents: Brand Command Center</h1>
        <div className="label">Streamlit-parity frontend: executive, comparison, response, analytics, and triage workflows.</div>
        <div className="controls" style={{ marginTop: 12 }}>
          <input value={request.company_name} onChange={(e) => setRequest({ ...request, company_name: e.target.value })} placeholder="Company name" />
          <input value={request.synonyms} onChange={(e) => setRequest({ ...request, synonyms: e.target.value })} placeholder="Company synonyms" />
          <select value={request.source_mode} onChange={(e) => setRequest({ ...request, source_mode: e.target.value as AnalyzeRunRequest["source_mode"] })}>
            <option>Reddit</option><option>Web</option><option>Reddit + Web</option>
          </select>
          <select value={request.years_back} onChange={(e) => setRequest({ ...request, years_back: Number(e.target.value) as 1 | 2 | 3 | 4 | 5 })}>
            {[1,2,3,4,5].map((y) => <option key={y} value={y}>Last {y} year{y > 1 ? "s" : ""}</option>)}
          </select>
          <select value={request.strictness_preset} onChange={(e) => setRequest({ ...request, strictness_preset: e.target.value as AnalyzeRunRequest["strictness_preset"] })}>
            <option>Strict</option><option>Balanced</option><option>Broad</option>
          </select>
          <button className="primary" onClick={onRunAnalysis} disabled={loading}>{loading ? "Running..." : "Run Analysis"}</button>
        </div>
        <div className="controls" style={{ marginTop: 10 }}>
          <input value={competitor.company_name} onChange={(e) => setCompetitor({ ...competitor, company_name: e.target.value })} placeholder="Competitor company" />
          <input value={competitor.synonyms} onChange={(e) => setCompetitor({ ...competitor, synonyms: e.target.value })} placeholder="Competitor synonyms" />
          <button onClick={onRunComparison} disabled={loading}>Run Competitor Analysis</button>
          <input value={triageUrl} onChange={(e) => setTriageUrl(e.target.value)} placeholder="Reddit URL for triage" />
          <button onClick={onRunTriage} disabled={loading}>Analyze Post</button>
        </div>
        {runId && <div className="label" style={{ marginTop: 8 }}>Run ID: {runId}</div>}
        {error && <div className="bad" style={{ marginTop: 8 }}>{error}</div>}
      </section>

      <div className="tabs">
        <button className={`tab ${tab === "howto" ? "active" : ""}`} onClick={() => setTab("howto")}>How To Use</button>
        <button className={`tab ${tab === "executive" ? "active" : ""}`} onClick={() => setTab("executive")}>Executive Brand Image Summary</button>
        <button className={`tab ${tab === "comparison" ? "active" : ""}`} onClick={() => setTab("comparison")}>Competitive Comparison</button>
        <button className={`tab ${tab === "response" ? "active" : ""}`} onClick={() => setTab("response")}>Customer Response Generator</button>
        <button className={`tab ${tab === "analytics" ? "active" : ""}`} onClick={() => setTab("analytics")}>Data Analytics</button>
        <button className={`tab ${tab === "triage" ? "active" : ""}`} onClick={() => setTab("triage")}>Jira Intake Triage</button>
      </div>

      {tab === "howto" && (
        <section className="grid grid-2">
          <article className="card">
            <h3 style={{ marginTop: 0 }}>1. Run Brand Analysis</h3>
            <p className="label">Configure company terms, source mode, lookback, and strictness. Click Run Analysis.</p>
            <h3>2. Read Executive Health</h3>
            <p className="label">Use Brand Image Score, sentiment distribution, and response playbook to assess current risk.</p>
            <h3>3. Operate Response Queue</h3>
            <p className="label">Use customer response tab for highest-risk negative posts and draft-ready responses.</p>
          </article>
          <article className="card">
            <h3 style={{ marginTop: 0 }}>4. Compare Competitors</h3>
            <p className="label">Run side-by-side analysis against a competitor using the same rules and lookback window.</p>
            <h3>5. Monitor Data Quality</h3>
            <p className="label">Review filter diagnostics (included/excluded) and keyword risk signals.</p>
            <h3>6. Triage Individual Incidents</h3>
            <p className="label">Paste a Reddit URL to generate triage decision, criticality score, and response draft.</p>
          </article>
        </section>
      )}

      {tab === "executive" && (
        <section className="grid" style={{ gap: 14 }}>
          <div className="grid grid-4">
            <article className="card"><div className="label">Brand Image Score</div><div className="kpi info">{brandImageScore}</div></article>
            <article className="card"><div className="label">Avg Sentiment Score</div><div className={`kpi ${avgSent >= 0 ? "good" : "bad"}`}>{avgSent.toFixed(3)}</div></article>
            <article className="card"><div className="label">Positive Mentions</div><div className="kpi good">{pct(posShare)}</div></article>
            <article className="card"><div className="label">Negative Mentions</div><div className={`kpi ${negDeltaPts > 0 ? "bad" : "good"}`}>{pct(negShare)}</div></article>
          </div>

          <div className="grid grid-2">
            <article className="card">
              <h3 style={{ marginTop: 0 }}>Sentiment Distribution</h3>
              {renderSimpleBar("Positive", rows.length ? (sentimentCounts.positive / rows.length) * 100 : 0, "good")}
              {renderSimpleBar("Neutral", rows.length ? (sentimentCounts.neutral / rows.length) * 100 : 0, "neutral")}
              {renderSimpleBar("Negative", rows.length ? (sentimentCounts.negative / rows.length) * 100 : 0, "bad")}
            </article>
            <article className="card">
              <h3 style={{ marginTop: 0 }}>Which Channels/Subreddits To Watch</h3>
              <table>
                <thead><tr><th>#</th><th>Subreddit</th><th>Negative Rate</th><th>Volume</th></tr></thead>
                <tbody>
                  {subredditRisk.slice(0, 10).map((row, i) => (
                    <tr key={i}><td>{i + 1}</td><td>{String(row.subreddit || "")}</td><td>{String(row.negative_rate || "")}</td><td>{String(row.volume || "")}</td></tr>
                  ))}
                </tbody>
              </table>
            </article>
          </div>

          <article className="card">
            <h3 style={{ marginTop: 0 }}>Top Problem Themes (What To Answer / Fix)</h3>
            <table>
              <thead><tr><th>#</th><th>Issue</th><th>Mentions</th><th>Negative Rate</th><th>What to answer</th><th>What to fix</th></tr></thead>
              <tbody>
                {issueDx.slice(0, 8).map((x, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{String(x.issue || x.theme || "-")}</td>
                    <td>{String(x.mentions || "-")}</td>
                    <td>{String(x.negative_rate || "-")}</td>
                    <td>{String(x.what_to_answer || "-")}</td>
                    <td>{String(x.what_to_fix || "-")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>

          <article className="card">
            <h3 style={{ marginTop: 0 }}>Response Playbook</h3>
            <table>
              <thead><tr><th>#</th><th>Priority</th><th>Issue</th><th>Why now</th><th>Owner</th><th>SLA</th></tr></thead>
              <tbody>
                {playbook.slice(0, 8).map((p, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{String(p.priority || "")}</td>
                    <td>{String(p.issue || "")}</td>
                    <td>{String(p.why_now || "")}</td>
                    <td>{String(p.owner_team || "")}</td>
                    <td>{String(p.recommended_sla || "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
        </section>
      )}

      {tab === "comparison" && (
        <section className="grid grid-2">
          <article className="card">
            <h3 style={{ marginTop: 0 }}>{comparison?.company_a.request.company_name ?? request.company_name}</h3>
            <p className="label">Brand Image: {Math.round((n(comparison?.company_a.result.brand_health_summary.avg_sentiment_score, 3) + 1) * 50)}</p>
            <p className="label">Avg Sentiment: {n(comparison?.company_a.result.brand_health_summary.avg_sentiment_score, 3)}</p>
            <p className="label">Positive %: {pct(n(comparison?.company_a.result.brand_health_summary.positive_share, 4))}</p>
            <p className="label">Negative %: {pct(n(comparison?.company_a.result.brand_health_summary.negative_share, 4))}</p>
          </article>
          <article className="card">
            <h3 style={{ marginTop: 0 }}>{comparison?.company_b.request.company_name ?? competitor.company_name}</h3>
            <p className="label">Brand Image: {Math.round((n(comparison?.company_b.result.brand_health_summary.avg_sentiment_score, 3) + 1) * 50)}</p>
            <p className="label">Avg Sentiment: {n(comparison?.company_b.result.brand_health_summary.avg_sentiment_score, 3)}</p>
            <p className="label">Positive %: {pct(n(comparison?.company_b.result.brand_health_summary.positive_share, 4))}</p>
            <p className="label">Negative %: {pct(n(comparison?.company_b.result.brand_health_summary.negative_share, 4))}</p>
          </article>
          <article className="card" style={{ gridColumn: "1 / -1" }}>
            <h3 style={{ marginTop: 0 }}>Delta (Company A - Company B)</h3>
            <table>
              <tbody>
                <tr><th>Avg Sentiment Delta</th><td>{n(comparison?.delta.avg_sentiment_score_delta, 3)}</td></tr>
                <tr><th>Negative Share Delta</th><td>{pct(n(comparison?.delta.negative_share_delta, 4))}</td></tr>
                <tr><th>Positive Share Delta</th><td>{pct(n(comparison?.delta.positive_share_delta, 4))}</td></tr>
              </tbody>
            </table>
          </article>
        </section>
      )}

      {tab === "response" && (
        <section className="grid grid-2">
          <article className="card">
            <h3 style={{ marginTop: 0 }}>Priority Case Queue (Negative)</h3>
            <table>
              <thead><tr><th>#</th><th>Posted</th><th>Subreddit</th><th>Title</th><th>Severity</th><th>Reason</th></tr></thead>
              <tbody>
                {negativeRows.slice(0, 12).map((row, i) => (
                  <tr key={i} className="negative">
                    <td>{i + 1}</td><td>{String(row.posted_at || row.date || "")}</td><td>{String(row.subreddit || "")}</td>
                    <td>{String(row.title || "")}</td><td>{String(row.severity || "")}</td><td>{String(row.reason || "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
          <article className="card">
            <h3 style={{ marginTop: 0 }}>Positive Mentions Queue</h3>
            <table>
              <thead><tr><th>#</th><th>Posted</th><th>Subreddit</th><th>Title</th><th>Reason</th></tr></thead>
              <tbody>
                {positiveRows.slice(0, 12).map((row, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td><td>{String(row.posted_at || row.date || "")}</td><td>{String(row.subreddit || "")}</td>
                    <td>{String(row.title || "")}</td><td>{String(row.reason || "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
          <article className="card" style={{ gridColumn: "1 / -1" }}>
            <h3 style={{ marginTop: 0 }}>Copy-ready Response Drafts</h3>
            {negativeRows.slice(0, 6).map((row, i) => (
              <div key={i} style={{ borderTop: i ? "1px solid var(--border)" : "none", paddingTop: i ? 10 : 0, marginTop: i ? 10 : 0 }}>
                <div className={`badge ${sentimentClass(String(row.sentiment || "negative"))}`}>{String(row.sentiment || "negative")}</div>
                <p><strong>{String(row.title || "Untitled")}</strong></p>
                <p className="label">{responseTemplate(row)}</p>
              </div>
            ))}
          </article>
        </section>
      )}

      {tab === "analytics" && (
        <section className="grid grid-2">
          <article className="card">
            <h3 style={{ marginTop: 0 }}>Data Quality</h3>
            <table>
              <tbody>
                <tr><th>Raw Collected</th><td>{String(analysis?.result.raw_count ?? 0)}</td></tr>
                <tr><th>After Dedupe</th><td>{String(analysis?.result.deduped_count ?? 0)}</td></tr>
                <tr><th>Analyzed</th><td>{String(analysis?.result.analyzed_count ?? 0)}</td></tr>
                <tr><th>Skipped</th><td>{String(analysis?.result.skipped ?? 0)}</td></tr>
              </tbody>
            </table>
          </article>

          <article className="card">
            <h3 style={{ marginTop: 0 }}>Filter Diagnosis</h3>
            <table>
              <tbody>
                <tr><th>Fetched</th><td>{String(diagnostics.fetched_total ?? 0)}</td></tr>
                <tr><th>Included</th><td>{String(diagnostics.included_total ?? 0)}</td></tr>
                <tr><th>Excluded</th><td>{String(diagnostics.excluded_total ?? 0)}</td></tr>
                <tr><th>Threshold</th><td>{String(analysis?.result.relevance_threshold ?? "-")}</td></tr>
              </tbody>
            </table>
          </article>

          <article className="card" style={{ gridColumn: "1 / -1" }}>
            <h3 style={{ marginTop: 0 }}>Excluded By Reason</h3>
            <table>
              <thead><tr><th>#</th><th>Reason</th><th>Count</th></tr></thead>
              <tbody>
                {Object.entries((diagnostics.excluded_by_reason as Record<string, unknown>) || {}).map(([reason, count], idx) => (
                  <tr key={reason}><td>{idx + 1}</td><td>{reason}</td><td>{String(count)}</td></tr>
                ))}
              </tbody>
            </table>
          </article>

          <article className="card" style={{ gridColumn: "1 / -1" }}>
            <h3 style={{ marginTop: 0 }}>Review Feed</h3>
            <table>
              <thead><tr><th>#</th><th>Platform</th><th>Subreddit</th><th>Posted</th><th>Title</th><th>Sentiment</th><th>Score</th><th>Category</th></tr></thead>
              <tbody>
                {rows.slice(0, 100).map((r, i) => (
                  <tr key={i} className={String(r.sentiment || "").toLowerCase() === "negative" ? "negative" : ""}>
                    <td>{i + 1}</td><td>{String(r.platform || "")}</td><td>{String(r.subreddit || "")}</td><td>{String(r.posted_at || r.date || "")}</td>
                    <td>{String(r.title || "")}</td><td className={sentimentClass(String(r.sentiment || "neutral"))}>{String(r.sentiment || "")}</td>
                    <td>{String(r.sentiment_score || "")}</td><td>{String(r.category || "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
        </section>
      )}

      {tab === "triage" && (
        <section className="grid grid-2">
          <article className="card">
            <h3 style={{ marginTop: 0 }}>Incident Triage Result</h3>
            <p className="label">Paste a Reddit URL in the header section and click Analyze Post.</p>
            <table>
              <tbody>
                <tr><th>Action</th><td>{String(triage?.triage_decision?.action || "-")}</td></tr>
                <tr><th>Confidence</th><td>{String(triage?.triage_decision?.confidence || "-")}</td></tr>
                <tr><th>Criticality</th><td>{String(triage?.criticality_score || "-")} / {String(triage?.criticality_threshold || "-")}</td></tr>
                <tr><th>Category</th><td>{String(triage?.analysis?.category || "-")}</td></tr>
                <tr><th>Severity</th><td>{String(triage?.analysis?.severity || "-")}</td></tr>
              </tbody>
            </table>
          </article>
          <article className="card">
            <h3 style={{ marginTop: 0 }}>Reddit Post</h3>
            <p><strong>{String(triage?.post?.title || "")}</strong></p>
            <p className="label">r/{String(triage?.post?.subreddit || "")} · score {String(triage?.post?.score || "")}</p>
            <p className="label">{String(triage?.post?.body || "")}</p>
          </article>
          <article className="card" style={{ gridColumn: "1 / -1" }}>
            <h3 style={{ marginTop: 0 }}>Draft Customer Response</h3>
            <textarea readOnly value={String(triage?.response_draft || "")} style={{ minHeight: 220 }} />
          </article>
        </section>
      )}
    </main>
  );
}
