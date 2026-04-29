from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "is", "it", "this", "that",
    "i", "we", "you", "they", "our", "my", "was", "are", "be", "as", "at", "from", "by", "have", "has",
}


@dataclass
class RowDoc:
    doc_id: str
    text: str
    category: str
    reason: str
    subreddit: str
    source_ref: str
    tokens: List[str]


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOP]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return payload["rows"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("documents"), list):
            docs = payload.get("documents", [])
            converted: List[Dict[str, Any]] = []
            for i, doc in enumerate(docs):
                meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
                converted.append(
                    {
                        "review_id": str(doc.get("doc_id") or f"doc_{i}") if isinstance(doc, dict) else f"doc_{i}",
                        "title": "",
                        "text": str(doc.get("text") or "") if isinstance(doc, dict) else "",
                        "category": str(meta.get("category") or "General Feedback"),
                        "reason": str(meta.get("reason") or "general feedback"),
                        "subreddit": str(meta.get("subreddit") or "unknown"),
                        "source_ref": str(meta.get("source_ref") or ""),
                    }
                )
            return converted
    if isinstance(payload, list):
        return payload
    raise ValueError("Input must be a JSON list of rows or object with key 'rows'.")


def _auto_input_path() -> Path:
    candidates = [
        Path("data/analysis_results.json"),
        Path("data/scheduled_results.json"),
        Path("data/reddit_analysis_rows.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No default analyzed-rows file found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _row_to_doc(row: Dict[str, Any], idx: int) -> RowDoc | None:
    title = _normalize_text(str(row.get("title") or ""))
    text = _normalize_text(str(row.get("text") or ""))
    if not text and not title:
        return None

    full = text if text else title
    if title and title not in text:
        full = f"{title}. {full}"

    category = _normalize_text(str(row.get("category") or "General Feedback"))
    reason = _normalize_text(str(row.get("reason") or "general feedback"))
    subreddit = _normalize_text(str(row.get("subreddit") or "unknown"))
    source_ref = _normalize_text(str(row.get("source_ref") or ""))
    doc_id = _normalize_text(str(row.get("review_id") or f"doc_{idx}"))
    toks = _tokenize(full)
    if not toks:
        return None
    return RowDoc(
        doc_id=doc_id,
        text=full,
        category=category,
        reason=reason,
        subreddit=subreddit,
        source_ref=source_ref,
        tokens=toks,
    )


def _group_docs(docs: Sequence[RowDoc], min_group_size: int) -> Dict[str, List[RowDoc]]:
    groups: Dict[str, List[RowDoc]] = defaultdict(list)
    for d in docs:
        reason_core = " ".join(d.reason.lower().split()[:8])
        key = f"{d.category.lower()}|{reason_core}"
        groups[key].append(d)

    filtered: Dict[str, List[RowDoc]] = {}
    for key, gdocs in groups.items():
        if len(gdocs) >= min_group_size:
            filtered[key] = gdocs
    return filtered


def _noisy_variant(base: str) -> str:
    s = base.lower()
    replacements = {
        "etsy": "ets",
        "marketplace": "mkt",
        "listing": "lst",
        "seller": "slr",
        "buyer": "byr",
        "balance": "bal",
        "refund": "rfd",
        "shipping": "ship",
        "website": "site",
        "customer": "cust",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def _intent_variant(category: str, reason: str) -> str:
    cat = category.lower()
    r = reason.lower()
    if "order" in cat:
        return f"customer cannot complete purchase because {r}"
    if "ux" in cat:
        return f"user flow problem causing friction: {r}"
    if "support" in cat:
        return f"support escalation needed due to {r}"
    if "pricing" in cat:
        return f"pricing confusion or fee concern: {r}"
    if "product" in cat:
        return f"product defect pattern: {r}"
    return f"customer issue trend: {r}"


def _lexical_variant(text: str) -> str:
    toks = _tokenize(text)
    return " ".join(toks[:10]) if toks else text[:80]


def _paraphrase_variant(category: str, reason: str) -> str:
    return f"{category} issue reported by customers: {reason}"


def _query_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _split_for(query_id: str) -> str:
    h = int(hashlib.md5(query_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "dev"
    return "test"


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _pick_relevant_docs(group_docs: Sequence[RowDoc], max_relevant: int) -> List[RowDoc]:
    # Deterministic: sort by doc_id and clip to avoid trivial all-doc relevance.
    ordered = sorted(group_docs, key=lambda d: d.doc_id)
    return ordered[:max_relevant]


def _hard_negatives(
    anchor: RowDoc,
    rel_ids: set[str],
    all_docs: Sequence[RowDoc],
    min_negatives: int,
    max_negatives: int,
) -> List[str]:
    candidates = [d for d in all_docs if d.doc_id not in rel_ids]

    scored = []
    for d in candidates:
        overlap = _jaccard(anchor.tokens, d.tokens)
        same_sub = 1 if d.subreddit == anchor.subreddit else 0
        # Prefer overlap to create difficult negatives. Slight boost for same subreddit noise.
        score = overlap + (0.05 * same_sub)
        scored.append((score, d))

    scored.sort(key=lambda x: (x[0], x[1].doc_id), reverse=True)

    picked: List[str] = [d.doc_id for s, d in scored if s > 0][:max_negatives]
    if len(picked) < min_negatives:
        fallback = [d.doc_id for _, d in scored if d.doc_id not in picked]
        picked.extend(fallback[: max(0, min_negatives - len(picked))])

    return picked[:max_negatives]


def _build_seed(
    docs: Sequence[RowDoc],
    min_group_size: int = 2,
    max_relevant_per_query: int = 3,
    min_hard_negatives_per_query: int = 5,
    max_hard_negatives_per_query: int = 12,
    include_acronym_every_n: int = 2,
    include_noisy_every_n: int = 1,
) -> Dict[str, Any]:
    groups = _group_docs(docs, min_group_size=min_group_size)

    documents = [
        {
            "doc_id": d.doc_id,
            "text": d.text,
            "meta": {
                "category": d.category,
                "reason": d.reason,
                "subreddit": d.subreddit,
                "source_ref": d.source_ref,
            },
        }
        for d in docs
    ]

    queries: List[Dict[str, Any]] = []
    for key, gdocs in groups.items():
        for anchor in sorted(gdocs, key=lambda d: d.doc_id):
            # Build relevant set centered around the anchor to increase query count
            # while keeping each query's relevance set selective.
            near_same_group = sorted(
                (
                    (_jaccard(anchor.tokens, cand.tokens), cand)
                    for cand in gdocs
                    if cand.doc_id != anchor.doc_id
                ),
                key=lambda x: (x[0], x[1].doc_id),
                reverse=True,
            )
            chosen_rel_docs = [anchor] + [cand for _, cand in near_same_group[: max(0, max_relevant_per_query - 1)]]
            rel_ids = [d.doc_id for d in chosen_rel_docs]
            rel_id_set = set(rel_ids)
            if len(rel_ids) < min_group_size:
                continue

            lex = _lexical_variant(anchor.text)
            variants = [
                ("lexical", lex),
                ("paraphrase", _paraphrase_variant(anchor.category, anchor.reason)),
                ("intent", _intent_variant(anchor.category, anchor.reason)),
            ]
            if include_noisy_every_n > 0 and (len(queries) % include_noisy_every_n == 0):
                variants.append(("noisy", _noisy_variant(lex)))
            if include_acronym_every_n > 0 and (len(queries) % include_acronym_every_n == 0):
                variants.append(("acronym", f"etsy mkt {lex}".strip()))

            hard_neg_ids = _hard_negatives(
                anchor=anchor,
                rel_ids=rel_id_set,
                all_docs=docs,
                min_negatives=min_hard_negatives_per_query,
                max_negatives=max_hard_negatives_per_query,
            )

            for tag, qtext in variants:
                qtext = _normalize_text(qtext)
                if not qtext:
                    continue
                qid = f"q_{_query_id(key + anchor.doc_id + tag + qtext)}"
                tags = [tag]
                if tag == "acronym" or re.search(r"\bmkt\b", qtext.lower()):
                    tags.append("acronym")
                tags = list(dict.fromkeys(tags))

                queries.append(
                    {
                        "query_id": qid,
                        "text": qtext,
                        "relevant_doc_ids": rel_ids,
                        "hard_negative_doc_ids": hard_neg_ids,
                        "tags": tags,
                        "split": _split_for(qid),
                        "group_key": key,
                    }
                )

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "build_retrieval_eval_seed.py",
            "min_group_size": min_group_size,
            "max_relevant_per_query": max_relevant_per_query,
            "min_hard_negatives_per_query": min_hard_negatives_per_query,
            "max_hard_negatives_per_query": max_hard_negatives_per_query,
            "include_acronym_every_n": include_acronym_every_n,
            "include_noisy_every_n": include_noisy_every_n,
        },
        "documents": documents,
        "queries": queries,
    }


def _stats(seed: Dict[str, Any]) -> Dict[str, Any]:
    docs = seed.get("documents", [])
    queries = seed.get("queries", [])
    rel_counts = [len(q.get("relevant_doc_ids", [])) for q in queries]
    split_counts = Counter(q.get("split", "unknown") for q in queries)
    tag_counts = Counter(tag for q in queries for tag in q.get("tags", []))
    hard_neg_counts = [len(q.get("hard_negative_doc_ids", [])) for q in queries]

    return {
        "num_documents": len(docs),
        "num_queries": len(queries),
        "median_relevant_docs_per_query": float(median(rel_counts)) if rel_counts else 0.0,
        "max_relevant_docs_per_query": max(rel_counts) if rel_counts else 0,
        "min_relevant_docs_per_query": min(rel_counts) if rel_counts else 0,
        "split_counts": dict(split_counts),
        "tag_counts": dict(tag_counts),
        "median_hard_negatives_per_query": float(median(hard_neg_counts)) if hard_neg_counts else 0.0,
        "min_hard_negatives_per_query": min(hard_neg_counts) if hard_neg_counts else 0,
    }


def _validate(
    stats: Dict[str, Any],
    min_docs: int,
    min_queries: int,
    min_median_rel: float,
    min_hard_neg: int,
    min_dev_queries: int,
    min_test_queries: int,
) -> List[str]:
    errs: List[str] = []
    if stats["num_documents"] < min_docs:
        errs.append(f"docs<{min_docs} (got {stats['num_documents']})")
    if stats["num_queries"] < min_queries:
        errs.append(f"queries<{min_queries} (got {stats['num_queries']})")
    if stats["median_relevant_docs_per_query"] < min_median_rel:
        errs.append(
            f"median_relevant_docs_per_query<{min_median_rel} (got {stats['median_relevant_docs_per_query']})"
        )
    if stats["max_relevant_docs_per_query"] > 5:
        errs.append(f"max_relevant_docs_per_query>5 (got {stats['max_relevant_docs_per_query']})")
    if stats["min_hard_negatives_per_query"] < min_hard_neg:
        errs.append(f"min_hard_negatives<{min_hard_neg} (got {stats['min_hard_negatives_per_query']})")
    if stats["tag_counts"].get("noisy", 0) == 0 or stats["tag_counts"].get("intent", 0) == 0:
        errs.append("missing noisy or intent query tags")
    if stats["tag_counts"].get("acronym", 0) == 0:
        errs.append("missing acronym-tag queries")
    if "dev" not in stats["split_counts"] or "test" not in stats["split_counts"]:
        errs.append("missing dev/test split coverage")
    if int(stats["split_counts"].get("dev", 0)) < min_dev_queries:
        errs.append(f"dev_queries<{min_dev_queries} (got {int(stats['split_counts'].get('dev', 0))})")
    if int(stats["split_counts"].get("test", 0)) < min_test_queries:
        errs.append(f"test_queries<{min_test_queries} (got {int(stats['split_counts'].get('test', 0))})")
    return errs


def _summary_md(stats: Dict[str, Any], out_seed: Path) -> str:
    return "\n".join(
        [
            "# Retrieval Eval Seed Summary",
            "",
            f"- Seed file: {out_seed}",
            f"- Documents: {stats['num_documents']}",
            f"- Queries: {stats['num_queries']}",
            f"- Median relevant docs/query: {stats['median_relevant_docs_per_query']}",
            f"- Max relevant docs/query: {stats['max_relevant_docs_per_query']}",
            f"- Median hard negatives/query: {stats['median_hard_negatives_per_query']}",
            "",
            "## Split Counts",
            json.dumps(stats.get("split_counts", {}), indent=2),
            "",
            "## Tag Counts",
            json.dumps(stats.get("tag_counts", {}), indent=2),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build realistic retrieval eval seed from analyzed review rows.")
    parser.add_argument("--input", default="", help="Path to rows json or object with rows[].")
    parser.add_argument("--out-seed", default="data/retrieval_eval_seed.json")
    parser.add_argument("--out-stats", default="data/retrieval_eval_seed_stats.json")
    parser.add_argument("--out-summary", default="", help="Optional markdown summary path.")
    parser.add_argument("--min-docs", type=int, default=100)
    parser.add_argument("--min-queries", type=int, default=30)
    parser.add_argument("--min-median-rel", type=float, default=2.0)
    parser.add_argument("--min-hard-neg", type=int, default=5)
    parser.add_argument("--min-dev-queries", type=int, default=5)
    parser.add_argument("--min-test-queries", type=int, default=5)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--max-relevant-per-query", type=int, default=3)
    parser.add_argument("--include-acronym-every-n", type=int, default=2)
    parser.add_argument("--include-noisy-every-n", type=int, default=1)
    parser.add_argument("--allow-small", action="store_true", help="Allow writing even when quality gates fail.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else _auto_input_path()
    rows = _read_rows(input_path)
    docs: List[RowDoc] = []
    for i, row in enumerate(rows):
        d = _row_to_doc(row, i)
        if d:
            docs.append(d)

    seed = _build_seed(
        docs,
        min_group_size=args.min_group_size,
        max_relevant_per_query=args.max_relevant_per_query,
        min_hard_negatives_per_query=args.min_hard_neg,
        include_acronym_every_n=args.include_acronym_every_n,
        include_noisy_every_n=args.include_noisy_every_n,
    )
    stats = _stats(seed)
    errs = _validate(
        stats,
        args.min_docs,
        args.min_queries,
        args.min_median_rel,
        args.min_hard_neg,
        args.min_dev_queries,
        args.min_test_queries,
    )

    if errs and not args.allow_small:
        raise SystemExit("Quality gates failed: " + "; ".join(errs))

    if errs:
        stats["quality_gate_status"] = "failed_but_allowed"
        stats["quality_gate_errors"] = errs
    else:
        stats["quality_gate_status"] = "passed"

    out_seed = Path(args.out_seed)
    out_stats = Path(args.out_stats)
    out_summary = Path(args.out_summary) if args.out_summary else None
    out_seed.parent.mkdir(parents=True, exist_ok=True)

    out_seed.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if out_summary is not None:
        out_summary.write_text(_summary_md(stats, out_seed), encoding="utf-8")

    print(f"Wrote seed: {out_seed.resolve()}")
    print(f"Wrote stats: {out_stats.resolve()}")
    if out_summary is not None:
        print(f"Wrote summary: {out_summary.resolve()}")
    print(f"Input rows source: {input_path.resolve()}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
