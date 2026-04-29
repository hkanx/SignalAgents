from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.retrieval_eval import evaluate, load_eval_file
ALL_MODES = ("bm25", "vector", "hybrid", "bm25_vector_rerank", "hybrid_weighted")


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
    for mode in ALL_MODES:
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
    for mode in ALL_MODES:
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


def _bootstrap_ci(values: list[float], num_samples: int, alpha: float, seed: int) -> dict:
    if not values:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(max(1, num_samples)):
        sample_mean = sum(values[rng.randrange(0, n)] for _ in range(n)) / n
        means.append(sample_mean)
    means.sort()
    lo_idx = int(max(0, (alpha / 2.0) * (len(means) - 1)))
    hi_idx = int(min(len(means) - 1, (1.0 - alpha / 2.0) * (len(means) - 1)))
    return {
        "lower": round(means[lo_idx], 6),
        "upper": round(means[hi_idx], 6),
        "mean": round(sum(values) / len(values), 6),
    }


def _compute_confidence_intervals(report: dict, num_samples: int, alpha: float, seed: int) -> dict:
    out: dict = {"overall": [], "by_tag": {}, "settings": {"samples": num_samples, "alpha": alpha, "seed": seed}}

    for row in report.get("results", []):
        per_query = row.get("per_query", []) or []
        recalls = [float(x.get("recall_at_k", 0.0)) for x in per_query]
        mrrs = [float(x.get("mrr_at_k", 0.0)) for x in per_query]
        ndcgs = [float(x.get("ndcg_at_k", 0.0)) for x in per_query]
        out["overall"].append(
            {
                "mode": row.get("mode"),
                "k": row.get("k"),
                "recall_at_k_ci": _bootstrap_ci(recalls, num_samples, alpha, seed + 1),
                "mrr_at_k_ci": _bootstrap_ci(mrrs, num_samples, alpha, seed + 2),
                "ndcg_at_k_ci": _bootstrap_ci(ndcgs, num_samples, alpha, seed + 3),
            }
        )

    for tag, rows in (report.get("results_by_tag", {}) or {}).items():
        out["by_tag"][tag] = []
        for row in rows:
            per_query = row.get("per_query", []) or []
            recalls = [float(x.get("recall_at_k", 0.0)) for x in per_query]
            mrrs = [float(x.get("mrr_at_k", 0.0)) for x in per_query]
            ndcgs = [float(x.get("ndcg_at_k", 0.0)) for x in per_query]
            out["by_tag"][tag].append(
                {
                    "mode": row.get("mode"),
                    "k": row.get("k"),
                    "recall_at_k_ci": _bootstrap_ci(recalls, num_samples, alpha, seed + 11),
                    "mrr_at_k_ci": _bootstrap_ci(mrrs, num_samples, alpha, seed + 12),
                    "ndcg_at_k_ci": _bootstrap_ci(ndcgs, num_samples, alpha, seed + 13),
                }
            )
    return out


def _evaluate_weighted_gate(report: dict, min_weighted_composite: float) -> dict:
    weighted = report.get("weighted_decision", {}) or {}
    scores = weighted.get("scores", []) or []
    top = scores[0] if scores else None
    top_score = float(top.get("weighted_composite", 0.0)) if top else 0.0
    passed = bool(top) and top_score >= float(min_weighted_composite)
    return {
        "min_weighted_composite": float(min_weighted_composite),
        "top_mode": top.get("mode") if top else None,
        "top_weighted_composite": round(top_score, 6),
        "passed": passed,
    }


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

    # ── Weighted decision score (standalone — independent of baseline_comparison) ──
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

    # ── Baseline comparison (optional) ──────────────────────────────────────────
    comparison = report.get("baseline_comparison")
    if comparison and comparison.get("rows"):
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append(f"- Baseline mode: `{comparison.get('baseline_mode')}` @k={comparison.get('k')}")
        lines.append("")
        lines.append("| Mode | dRecall@K | dMRR@K | dNDCG@K |")
        lines.append("|---|---:|---:|---:|")
        for row in comparison.get("rows", []):
            lines.append(
                "| {mode} | {dr:.4f} | {dmrr:.4f} | {dndcg:.4f} |".format(
                    mode=row["mode"],
                    dr=row["delta_recall_at_k"],
                    dmrr=row["delta_mrr_at_k"],
                    dndcg=row["delta_ndcg_at_k"],
                )
            )
        lines.append("")

    # ── Tag gate ────────────────────────────────────────────────────────────────
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
            lines.append("")
            lines.append("| Mode | Tag | Reason | Actual | Threshold |")
            lines.append("|---|---|---|---:|---:|")
            for f in failures:
                actual = f.get("actual", "—")
                threshold = f.get("threshold", "—")
                actual_str = f"{actual:.4f}" if isinstance(actual, float) else str(actual)
                threshold_str = f"{threshold:.4f}" if isinstance(threshold, float) else str(threshold)
                lines.append(
                    f"| {f.get('mode')} | {f.get('tag')} | {f.get('reason')} | {actual_str} | {threshold_str} |"
                )
        else:
            lines.append("- status: **passed**")
        lines.append("")

    # ── Weighted gate ────────────────────────────────────────────────────────────
    weighted_gate = report.get("weighted_gate")
    if weighted_gate:
        lines.append("## Weighted Gate")
        lines.append("")
        lines.append(
            f"- min composite: {weighted_gate.get('min_weighted_composite')} | "
            f"top mode: {weighted_gate.get('top_mode')} | "
            f"top composite: {weighted_gate.get('top_weighted_composite')}"
        )
        lines.append(f"- status: **{'passed' if weighted_gate.get('passed') else 'failed'}**")
        lines.append("")

    # ── Confidence intervals ─────────────────────────────────────────────────────
    ci = report.get("confidence_intervals")
    if ci:
        settings = ci.get("settings", {})
        lines.append("## Confidence Intervals")
        lines.append("")
        lines.append(
            f"- bootstrap samples: {settings.get('samples')} | alpha: {settings.get('alpha')} | seed: {settings.get('seed')}"
        )
        lines.append("")
        overall = ci.get("overall", [])
        if overall:
            lines.append("### Overall CI")
            lines.append("")
            lines.append("| Mode | K | Recall CI (lo–hi) | MRR CI (lo–hi) | NDCG CI (lo–hi) |")
            lines.append("|---|---:|---|---|---|")
            for row in overall:
                r_ci = row.get("recall_at_k_ci", {})
                m_ci = row.get("mrr_at_k_ci", {})
                n_ci = row.get("ndcg_at_k_ci", {})
                lines.append(
                    "| {mode} | {k} | {rl:.4f}–{rh:.4f} | {ml:.4f}–{mh:.4f} | {nl:.4f}–{nh:.4f} |".format(
                        mode=row.get("mode", ""),
                        k=row.get("k", ""),
                        rl=r_ci.get("lower", 0.0),
                        rh=r_ci.get("upper", 0.0),
                        ml=m_ci.get("lower", 0.0),
                        mh=m_ci.get("upper", 0.0),
                        nl=n_ci.get("lower", 0.0),
                        nh=n_ci.get("upper", 0.0),
                    )
                )
            lines.append("")

        by_tag = ci.get("by_tag", {})
        if by_tag:
            lines.append("### CI by Tag")
            lines.append("")
            for tag, tag_rows in by_tag.items():
                lines.append(f"#### Tag: {tag}")
                lines.append("")
                lines.append("| Mode | K | Recall CI (lo–hi) | MRR CI (lo–hi) | NDCG CI (lo–hi) |")
                lines.append("|---|---:|---|---|---|")
                for row in tag_rows:
                    r_ci = row.get("recall_at_k_ci", {})
                    m_ci = row.get("mrr_at_k_ci", {})
                    n_ci = row.get("ndcg_at_k_ci", {})
                    lines.append(
                        "| {mode} | {k} | {rl:.4f}–{rh:.4f} | {ml:.4f}–{mh:.4f} | {nl:.4f}–{nh:.4f} |".format(
                            mode=row.get("mode", ""),
                            k=row.get("k", ""),
                            rl=r_ci.get("lower", 0.0),
                            rh=r_ci.get("upper", 0.0),
                            ml=m_ci.get("lower", 0.0),
                            mh=m_ci.get("upper", 0.0),
                            nl=n_ci.get("lower", 0.0),
                            nh=n_ci.get("upper", 0.0),
                        )
                    )
                lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation (bm25/vector/hybrid).")
    parser.add_argument("--input", required=True, help="Path to eval json file with documents+queries.")
    parser.add_argument("--output-json", default="retrieval_eval_report.json", help="Output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional markdown summary path.")
    parser.add_argument("--decision-k", type=int, default=10, help="K used for weighted decision and tag gating.")
    # Strengthened defaults: intent+paraphrase are the hardest and most predictive query types;
    # the previous defaults (0.06 recall, 0.04 ndcg) were too permissive to catch real regressions.
    parser.add_argument("--min-intent-paraphrase-recall", type=float, default=0.10)
    parser.add_argument("--min-intent-paraphrase-ndcg", type=float, default=0.06)
    parser.add_argument("--min-weighted-composite", type=float, default=0.18)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--strict-quality-gate", action="store_true", help="Fail if any gate (tag/weighted) fails.")
    parser.add_argument("--fail-on-tag-gate", action="store_true", help="Exit non-zero if tag gate fails.")
    args = parser.parse_args()

    documents, queries = load_eval_file(args.input)
    report = evaluate(documents=documents, queries=queries, modes=ALL_MODES, ks=(5, 10, 20))
    report["weighted_decision"] = _compute_weighted_scores(report, k=int(args.decision_k))
    baseline_mode = "hybrid"
    base_row = _find_row(report.get("results", []), mode=baseline_mode, k=int(args.decision_k))
    baseline_rows = []
    if base_row:
        for mode in ALL_MODES:
            if mode == baseline_mode:
                continue
            row = _find_row(report.get("results", []), mode=mode, k=int(args.decision_k))
            if not row:
                continue
            baseline_rows.append(
                {
                    "mode": mode,
                    "delta_recall_at_k": round(float(row.get("recall_at_k", 0.0)) - float(base_row.get("recall_at_k", 0.0)), 6),
                    "delta_mrr_at_k": round(float(row.get("mrr_at_k", 0.0)) - float(base_row.get("mrr_at_k", 0.0)), 6),
                    "delta_ndcg_at_k": round(float(row.get("ndcg_at_k", 0.0)) - float(base_row.get("ndcg_at_k", 0.0)), 6),
                }
            )
    report["baseline_comparison"] = {
        "baseline_mode": baseline_mode,
        "k": int(args.decision_k),
        "rows": baseline_rows,
    }
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
    report["weighted_gate"] = _evaluate_weighted_gate(
        report=report,
        min_weighted_composite=float(args.min_weighted_composite),
    )
    report["confidence_intervals"] = _compute_confidence_intervals(
        report=report,
        num_samples=int(args.bootstrap_samples),
        alpha=float(args.bootstrap_alpha),
        seed=int(args.bootstrap_seed),
    )

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
    if args.strict_quality_gate:
        gate_failed = bool(tag_failures) or not bool(report.get("weighted_gate", {}).get("passed", False))
        if gate_failed:
            print("Strict quality gate failed (tag gate and/or weighted gate).", file=sys.stderr)
            sys.exit(3)


if __name__ == "__main__":
    main()
