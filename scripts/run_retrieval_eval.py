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
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation (bm25/vector/hybrid).")
    parser.add_argument("--input", required=True, help="Path to eval json file with documents+queries.")
    parser.add_argument("--output-json", default="retrieval_eval_report.json", help="Output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional markdown summary path.")
    args = parser.parse_args()

    documents, queries = load_eval_file(args.input)
    report = evaluate(documents=documents, queries=queries, modes=("bm25", "vector", "hybrid"), ks=(5, 10, 20))

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = _build_markdown_summary(report)
    print(summary)
    print(f"Saved JSON report: {output_json.resolve()}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.write_text(summary, encoding="utf-8")
        print(f"Saved Markdown summary: {output_md.resolve()}")


if __name__ == "__main__":
    main()
