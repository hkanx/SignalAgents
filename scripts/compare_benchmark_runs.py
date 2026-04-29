from __future__ import annotations

"""compare_benchmark_runs.py — Combine multiple retrieval eval JSON reports into a side-by-side
delta table and emit a recommendation for the default retrieval mode.

Usage:
    python scripts/compare_benchmark_runs.py \\
        --reports data/benchmark_bm25_*.json data/benchmark_vector_*.json data/benchmark_hybrid_*.json \\
        --output-md data/benchmark_comparison_latest.md \\
        --baseline bm25

The first report matching --baseline (by its 'mode' field in the weighted_decision block, or
the filename stem) is used as the reference; all others show deltas relative to it.

Exit codes:
    0 — comparison produced successfully
    1 — fewer than 2 reports provided (nothing to compare)
"""

import argparse
import json
import sys
from pathlib import Path


def _load_report(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _detect_mode(report: dict, path: str) -> str:
    """Infer the retrieval mode from the report or fall back to the filename stem."""
    wd = report.get("weighted_decision") or {}
    scores = wd.get("scores") or []
    if scores:
        # The top-ranked mode in weighted_decision is the winner for this run,
        # but we want the *backend* mode — derive it from filename stem instead.
        pass
    # Try filename stem: benchmark_<mode>_<ts>.json
    stem = Path(path).stem  # e.g. "benchmark_hybrid_20260428T213000"
    parts = stem.split("_")
    for known in ("bm25", "vector", "hybrid"):
        if known in parts:
            return known
    # Fallback: vector_backend field in report
    return report.get("vector_backend", stem)


def _find_row(rows: list[dict], mode: str, k: int) -> dict | None:
    for row in rows:
        if str(row.get("mode")) == mode and int(row.get("k", -1)) == int(k):
            return row
    return None


def _ci_str(ci: dict | None) -> str:
    if not ci:
        return "—"
    return f"{ci.get('lower', 0):.4f}–{ci.get('upper', 0):.4f} (μ={ci.get('mean', 0):.4f})"


def _build_comparison(
    reports: list[tuple[str, dict]],
    baseline_name: str,
    k: int = 10,
) -> str:
    lines: list[str] = []
    lines.append("# Retrieval Benchmark Comparison")
    lines.append("")
    lines.append(f"- **Baseline mode**: `{baseline_name}`")
    lines.append(f"- **k**: {k}")
    lines.append(f"- **Reports**: {', '.join(name for name, _ in reports)}")
    lines.append("")

    # ── Collect overall@k metrics for each run ───────────────────────────────
    # For each report we need: recall@k, mrr@k, ndcg@k (overall), weighted composite, CI
    data: list[dict] = []
    baseline_overall: dict | None = None

    for name, report in reports:
        overall_row = _find_row(report.get("results", []), mode="hybrid", k=k)
        # For bm25-only and vector-only runs the top mode won't be "hybrid";
        # take the first result at k instead.
        if not overall_row:
            results_at_k = [r for r in report.get("results", []) if int(r.get("k", -1)) == k]
            overall_row = results_at_k[0] if results_at_k else {}

        wd = report.get("weighted_decision") or {}
        wd_scores = wd.get("scores") or []
        top_weighted = wd_scores[0] if wd_scores else {}

        ci_overall = report.get("confidence_intervals", {}).get("overall", [])
        ci_row = next(
            (r for r in ci_overall if int(r.get("k", -1)) == k),
            None,
        )

        entry = {
            "name": name,
            "recall": float(overall_row.get("recall_at_k", 0.0)),
            "mrr": float(overall_row.get("mrr_at_k", 0.0)),
            "ndcg": float(overall_row.get("ndcg_at_k", 0.0)),
            "weighted_composite": float(top_weighted.get("weighted_composite", 0.0)),
            "ci_recall": ci_row.get("recall_at_k_ci") if ci_row else None,
            "ci_mrr": ci_row.get("mrr_at_k_ci") if ci_row else None,
            "ci_ndcg": ci_row.get("ndcg_at_k_ci") if ci_row else None,
        }
        data.append(entry)
        if name == baseline_name:
            baseline_overall = entry

    if not baseline_overall:
        # Use first entry as baseline
        baseline_overall = data[0]

    # ── Overall metrics table ─────────────────────────────────────────────────
    lines.append(f"## Overall Metrics @k={k}")
    lines.append("")
    lines.append("| Mode | Recall@K | MRR@K | NDCG@K | Weighted Composite |")
    lines.append("|---|---:|---:|---:|---:|")
    for entry in data:
        lines.append(
            "| {name} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {wc:.4f} |".format(
                name=entry["name"],
                recall=entry["recall"],
                mrr=entry["mrr"],
                ndcg=entry["ndcg"],
                wc=entry["weighted_composite"],
            )
        )
    lines.append("")

    # ── Delta table vs baseline ───────────────────────────────────────────────
    lines.append(f"## Deltas vs Baseline (`{baseline_name}`) @k={k}")
    lines.append("")
    lines.append("| Mode | ΔRecall@K | ΔMRR@K | ΔNDCG@K | ΔWeighted Composite |")
    lines.append("|---|---:|---:|---:|---:|")
    for entry in data:
        if entry["name"] == baseline_overall["name"]:
            continue
        dr = entry["recall"] - baseline_overall["recall"]
        dm = entry["mrr"] - baseline_overall["mrr"]
        dn = entry["ndcg"] - baseline_overall["ndcg"]
        dw = entry["weighted_composite"] - baseline_overall["weighted_composite"]
        lines.append(
            "| {name} | {dr:+.4f} | {dm:+.4f} | {dn:+.4f} | {dw:+.4f} |".format(
                name=entry["name"], dr=dr, dm=dm, dn=dn, dw=dw
            )
        )
    lines.append("")

    # ── Confidence interval table ─────────────────────────────────────────────
    lines.append(f"## Confidence Intervals @k={k} (95%)")
    lines.append("")
    lines.append("| Mode | Recall CI | MRR CI | NDCG CI |")
    lines.append("|---|---|---|---|")
    for entry in data:
        lines.append(
            "| {name} | {rc} | {mc} | {nc} |".format(
                name=entry["name"],
                rc=_ci_str(entry["ci_recall"]),
                mc=_ci_str(entry["ci_mrr"]),
                nc=_ci_str(entry["ci_ndcg"]),
            )
        )
    lines.append("")

    # ── Intent + paraphrase tag breakdown ────────────────────────────────────
    lines.append(f"## Intent + Paraphrase Tag Metrics @k={k}")
    lines.append("*(These are the hardest query types and most predictive of production quality.)*")
    lines.append("")
    lines.append("| Mode | Tag | Recall@K | MRR@K | NDCG@K |")
    lines.append("|---|---|---:|---:|---:|")
    for name, report in reports:
        for tag in ("intent", "paraphrase"):
            tag_rows = (report.get("results_by_tag") or {}).get(tag, [])
            # Use the first mode available for this run at k
            tag_row = _find_row(tag_rows, mode="hybrid", k=k)
            if not tag_row:
                available = [r for r in tag_rows if int(r.get("k", -1)) == k]
                tag_row = available[0] if available else {}
            if tag_row:
                lines.append(
                    "| {name} | {tag} | {r:.4f} | {m:.4f} | {n:.4f} |".format(
                        name=name,
                        tag=tag,
                        r=float(tag_row.get("recall_at_k", 0.0)),
                        m=float(tag_row.get("mrr_at_k", 0.0)),
                        n=float(tag_row.get("ndcg_at_k", 0.0)),
                    )
                )
    lines.append("")

    # ── Recommendation ────────────────────────────────────────────────────────
    best = max(data, key=lambda x: x["weighted_composite"])
    runner_up = sorted(data, key=lambda x: x["weighted_composite"], reverse=True)
    ci_overlap = False
    if len(runner_up) >= 2:
        b = runner_up[0]
        r = runner_up[1]
        b_ci = b.get("ci_ndcg") or {}
        r_ci = r.get("ci_ndcg") or {}
        if b_ci and r_ci:
            # CIs overlap if best lower < runner-up upper and runner-up lower < best upper
            ci_overlap = (
                b_ci.get("lower", 0) < r_ci.get("upper", 1)
                and r_ci.get("lower", 0) < b_ci.get("upper", 1)
            )

    lines.append("## Recommendation")
    lines.append("")
    if ci_overlap:
        lines.append(
            f"> ⚠️ **Confidence intervals overlap** between `{runner_up[0]['name']}` and `{runner_up[1]['name']}` "
            f"on NDCG@{k}. The difference is not statistically reliable with the current corpus size. "
            f"Consider expanding the eval seed before finalising the default mode."
        )
    else:
        lines.append(
            f"> ✅ **Recommended default mode: `{best['name']}`** "
            f"(highest weighted composite: {best['weighted_composite']:.4f}, "
            f"CI non-overlapping with runner-up on NDCG@{k})."
        )
    lines.append("")
    lines.append("### Interpretation guide")
    lines.append("")
    lines.append(
        "- **Δ ≥ 0.03 in NDCG@10** is a meaningful improvement for corpora of this size (100–500 docs). "
        "Smaller deltas are likely within noise — check CI overlap before acting on them."
    )
    lines.append(
        "- **Overlapping 95% CIs** mean the two modes are statistically indistinguishable on the current seed. "
        "Expand the seed (more queries or documents) to increase power."
    )
    lines.append(
        "- **Intent + paraphrase recall** are the best proxy for production quality because those queries "
        "cannot be answered by exact token match — they require semantic understanding."
    )
    lines.append(
        "- **Cost/latency tradeoff**: `hybrid` uses OpenAI embeddings and adds ~100–150ms p50 and ~$0.0001 per query. "
        "If that cost or latency is a constraint, `bm25` is the next-best option for high-throughput monitoring."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple retrieval eval JSON reports.")
    parser.add_argument(
        "--reports", nargs="+", required=True,
        help="Paths to retrieval_eval_report JSON files (at least 2).",
    )
    parser.add_argument("--output-md", default="", help="Path to write combined markdown comparison.")
    parser.add_argument("--baseline", default="bm25", help="Mode name to use as baseline for deltas.")
    parser.add_argument("--k", type=int, default=10, help="K value to compare at.")
    args = parser.parse_args()

    if len(args.reports) < 2:
        print("ERROR: provide at least 2 report paths to compare.", file=sys.stderr)
        sys.exit(1)

    reports: list[tuple[str, dict]] = []
    for path in args.reports:
        report = _load_report(path)
        name = _detect_mode(report, path)
        reports.append((name, report))

    comparison_md = _build_comparison(reports, baseline_name=args.baseline, k=args.k)
    print(comparison_md)

    if args.output_md:
        Path(args.output_md).write_text(comparison_md, encoding="utf-8")
        print(f"\nSaved comparison: {Path(args.output_md).resolve()}")


if __name__ == "__main__":
    main()
