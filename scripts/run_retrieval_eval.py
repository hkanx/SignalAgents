from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.retrieval_eval import evaluate, load_eval_file


def _find_row(rows: list[dict], mode: str, k: int) -> dict | None:
    for row in rows:
        if str(row.get("mode")) == mode and int(row.get("k", -1)) == int(k):
            return row
    return None


def _compute_weighted_scores(report: dict, k: int = 10) -> dict:
    results_by_tag = report.get("results_by_tag", {}) or {}
    if not isinstance(results_by_tag, dict):
        return {"k": k, "weights": {}, "scores": []}

    # Prioritize harder semantic query styles over lexical/noisy matches.
    weights = {
        "intent": 0.4,
        "paraphrase": 0.4,
        "lexical": 0.1,
        "noisy": 0.05,
        "acronym": 0.05,
    }
    scores = []
    for mode in ("bm25", "vector", "hybrid"):
        recall = 0.0
        mrr = 0.0
        ndcg = 0.0
        used_weight = 0.0
        for tag, w in weights.items():
            row = _find_row(results_by_tag.get(tag, []), mode=mode, k=k)
            if not row:
                continue
            recall += float(row.get("recall_at_k", 0.0)) * w
            mrr += float(row.get("mrr_at_k", 0.0)) * w
            ndcg += float(row.get("ndcg_at_k", 0.0)) * w
            used_weight += w

        norm = used_weight if used_weight > 0 else 1.0
        scores.append(
            {
                "mode": mode,
                "k": int(k),
                "weighted_recall_at_k": round(recall / norm, 6),
                "weighted_mrr_at_k": round(mrr / norm, 6),
                "weighted_ndcg_at_k": round(ndcg / norm, 6),
                "weighted_composite": round(((recall + mrr + ndcg) / 3.0) / norm, 6),
                "weights_used_sum": round(used_weight, 3),
            }
        )

    scores.sort(key=lambda x: x["weighted_composite"], reverse=True)
    return {"k": int(k), "weights": weights, "scores": scores}


def _evaluate_tag_gates(report: dict, min_recall: float, min_ndcg: float, k: int = 10) -> list[dict]:
    failures: list[dict] = []
    results_by_tag = report.get("results_by_tag", {}) or {}
    gated_tags = ("intent", "paraphrase")
    for mode in ("bm25", "vector", "hybrid"):
        for tag in gated_tags:
            row = _find_row(results_by_tag.get(tag, []), mode=mode, k=k)
            if not row:
                failures.append(
                    {
                        "mode": mode,
                        "tag": tag,
                        "k": k,
                        "reason": "missing_tag_metrics",
                    }
                )
                continue
            recall = float(row.get("recall_at_k", 0.0))
            ndcg = float(row.get("ndcg_at_k", 0.0))
            if recall < min_recall:
                failures.append(
                    {
                        "mode": mode,
                        "tag": tag,
                        "k": k,
                        "reason": "recall_below_threshold",
                        "actual": round(recall, 6),
                        "threshold": min_recall,
                    }
                )
            if ndcg < min_ndcg:
                failures.append(
                    {
                        "mode": mode,
                        "tag": tag,
                        "k": k,
                        "reason": "ndcg_below_threshold",
                        "actual": round(ndcg, 6),
                        "threshold": min_ndcg,
                    }
                )
    return failures


def _build_markdown_summary(report: dict) -> str:
    lines = []
    lines.append("# Retrieval Evaluation Summary")
    lines.append("")
    lines.append(f"- Documents: {report.get('num_documents', 0)}")
    lines.append(f"- Queries: {report.get('num_queries', 0)}")
    lines.append("")
    lines.append("| Mode | K | Recall@K | MRR@K | NDCG@K |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report.get("results", []):
        lines.append(
            "| {mode} | {k} | {recall_at_k:.4f} | {mrr_at_k:.4f} | {ndcg_at_k:.4f} |".format(
                mode=row["mode"],
                k=row["k"],
                recall_at_k=row["recall_at_k"],
                mrr_at_k=row["mrr_at_k"],
                ndcg_at_k=row["ndcg_at_k"],
            )
        )
    lines.append("")

    split_results = report.get("results_by_split", {})
    if split_results:
        lines.append("## Split-Level Results")
        lines.append("")
        for split_name, rows in split_results.items():
            lines.append(f"### Split: {split_name}")
            lines.append("")
            lines.append("| Mode | K | Queries | Recall@K | MRR@K | NDCG@K |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in rows:
                lines.append(
                    "| {mode} | {k} | {num_queries} | {recall_at_k:.4f} | {mrr_at_k:.4f} | {ndcg_at_k:.4f} |".format(
                        mode=row["mode"],
                        k=row["k"],
                        num_queries=row.get("num_queries", 0),
                        recall_at_k=row["recall_at_k"],
                        mrr_at_k=row["mrr_at_k"],
                        ndcg_at_k=row["ndcg_at_k"],
                    )
                )
            lines.append("")
    weighted = report.get("weighted_decision")
    if weighted and weighted.get("scores"):
        lines.append("## Weighted Decision Score")
        lines.append("")
        lines.append(
            f"- k: {weighted.get('k')} | weights: `{json.dumps(weighted.get('weights', {}), sort_keys=True)}`"
        )
        lines.append("")
        lines.append("| Rank | Mode | Weighted Recall@K | Weighted MRR@K | Weighted NDCG@K | Composite |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, row in enumerate(weighted.get("scores", []), start=1):
            lines.append(
                "| {rank} | {mode} | {wr:.4f} | {wmrr:.4f} | {wndcg:.4f} | {wc:.4f} |".format(
                    rank=i,
                    mode=row["mode"],
                    wr=row["weighted_recall_at_k"],
                    wmrr=row["weighted_mrr_at_k"],
                    wndcg=row["weighted_ndcg_at_k"],
                    wc=row["weighted_composite"],
                )
            )
        lines.append("")

    gate = report.get("tag_gate")
    if gate:
        lines.append("## Tag Gate")
        lines.append("")
        lines.append(
            f"- k: {gate.get('k')} | min recall: {gate.get('min_recall')} | min ndcg: {gate.get('min_ndcg')}"
        )
        failures = gate.get("failures", [])
        if failures:
            lines.append(f"- status: **failed** ({len(failures)} issues)")
        else:
            lines.append("- status: **passed**")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation (bm25/vector/hybrid).")
    parser.add_argument("--input", required=True, help="Path to eval json file with documents+queries.")
    parser.add_argument("--output-json", default="retrieval_eval_report.json", help="Output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional markdown summary path.")
    parser.add_argument("--decision-k", type=int, default=10, help="K used for weighted decision and tag gating.")
    parser.add_argument("--min-intent-paraphrase-recall", type=float, default=0.06)
    parser.add_argument("--min-intent-paraphrase-ndcg", type=float, default=0.04)
    parser.add_argument("--fail-on-tag-gate", action="store_true", help="Exit non-zero if tag gate fails.")
    args = parser.parse_args()

    documents, queries = load_eval_file(args.input)
    report = evaluate(documents=documents, queries=queries, modes=("bm25", "vector", "hybrid"), ks=(5, 10, 20))
    report["weighted_decision"] = _compute_weighted_scores(report, k=int(args.decision_k))
    tag_failures = _evaluate_tag_gates(
        report=report,
        min_recall=float(args.min_intent_paraphrase_recall),
        min_ndcg=float(args.min_intent_paraphrase_ndcg),
        k=int(args.decision_k),
    )
    report["tag_gate"] = {
        "k": int(args.decision_k),
        "min_recall": float(args.min_intent_paraphrase_recall),
        "min_ndcg": float(args.min_intent_paraphrase_ndcg),
        "failures": tag_failures,
        "passed": len(tag_failures) == 0,
    }

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = _build_markdown_summary(report)
    print(summary)
    print(f"Saved JSON report: {output_json.resolve()}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.write_text(summary, encoding="utf-8")
        print(f"Saved Markdown summary: {output_md.resolve()}")

    if args.fail_on_tag_gate and tag_failures:
        print(f"Tag gate failed with {len(tag_failures)} issue(s).", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
