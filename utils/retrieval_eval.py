from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence

Mode = Literal["bm25", "vector", "hybrid"]


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    score: float


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    text: str
    relevant_doc_ids: Sequence[str]
    split: str = "all"
    tags: Sequence[str] = ()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _build_doc_tokens(documents: Dict[str, str]) -> Dict[str, List[str]]:
    return {doc_id: _tokenize(text) for doc_id, text in documents.items()}


def _idf(doc_tokens: Dict[str, List[str]]) -> Dict[str, float]:
    n_docs = len(doc_tokens)
    df: Dict[str, int] = {}
    for toks in doc_tokens.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n_docs + 1) / (f + 1)) + 1.0 for t, f in df.items()}


def _bm25_scores(
    query_tokens: Sequence[str],
    doc_tokens: Dict[str, List[str]],
    idf: Dict[str, float],
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[str, float]:
    lengths = {doc_id: len(toks) for doc_id, toks in doc_tokens.items()}
    avgdl = (sum(lengths.values()) / max(1, len(lengths))) or 1.0

    scores: Dict[str, float] = {}
    for doc_id, toks in doc_tokens.items():
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        dl = max(1, lengths[doc_id])
        for t in query_tokens:
            if t not in tf:
                continue
            term_idf = idf.get(t, 0.0)
            freq = tf[t]
            num = freq * (k1 + 1.0)
            den = freq + k1 * (1.0 - b + b * (dl / avgdl))
            score += term_idf * (num / den)

        if score > 0:
            scores[doc_id] = score
    return scores


def _tfidf_vector(tokens: Sequence[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    if not tf:
        return {}
    max_tf = max(tf.values())
    vec = {t: (f / max_tf) * idf.get(t, 0.0) for t, f in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def _cosine_scores(
    query_tokens: Sequence[str],
    idf: Dict[str, float],
    doc_vecs: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    qv = _tfidf_vector(query_tokens, idf)
    if not qv:
        return {}

    scores: Dict[str, float] = {}
    for doc_id, dv in doc_vecs.items():
        if not dv:
            continue
        dot = 0.0
        for t, qval in qv.items():
            dot += qval * dv.get(t, 0.0)
        if dot > 0:
            scores[doc_id] = dot
    return scores


_EMBED_CACHE: Dict[str, List[float]] = {}
_RETRIEVE_CACHE: Dict[tuple[str, str, str, int], List[RetrievedDoc]] = {}
_EMBED_MAX_WORDS = 6000
_EMBED_MAX_CHARS = 12000
_CORPUS_CACHE: Dict[str, tuple[Dict[str, List[str]], Dict[str, float]]] = {}
_DOC_TFIDF_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


def _cosine_dense(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embedding_scores(query: str, documents: Dict[str, str]) -> Dict[str, float]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        from openai import OpenAI
    except Exception:
        return {}

    client = OpenAI(api_key=api_key)

    def _truncate_for_embedding(text: str, max_words: int = _EMBED_MAX_WORDS, max_chars: int = _EMBED_MAX_CHARS) -> str:
        words = (text or "").split()
        out = text if len(words) <= max_words else " ".join(words[:max_words])
        return out[:max_chars]

    def get_embedding(text: str) -> List[float]:
        safe_text = _truncate_for_embedding(text)
        key = f"{model}::{safe_text}"
        if key in _EMBED_CACHE:
            return _EMBED_CACHE[key]
        attempt = safe_text
        for _ in range(6):
            try:
                emb = client.embeddings.create(model=model, input=attempt).data[0].embedding
                _EMBED_CACHE[key] = emb
                return emb
            except Exception as exc:
                msg = str(exc).lower()
                if "maximum context length" not in msg and "invalid 'input'" not in msg:
                    raise
                if len(attempt) <= 512:
                    raise
                attempt = attempt[: max(512, len(attempt) // 2)]
        # Should only be reachable if all retries failed with context errors.
        emb = client.embeddings.create(model=model, input=attempt).data[0].embedding
        _EMBED_CACHE[key] = emb
        return emb

    qv = get_embedding(query)
    scores: Dict[str, float] = {}
    for doc_id, text in documents.items():
        dv = get_embedding(text)
        score = _cosine_dense(qv, dv)
        if score > 0:
            scores[doc_id] = score
    return scores


def _rank(scores: Dict[str, float], k: int) -> List[RetrievedDoc]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [RetrievedDoc(doc_id=doc_id, score=float(score)) for doc_id, score in ranked]


def _rrf_fuse(rankings: Iterable[List[RetrievedDoc]], k_rrf: int = 60) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for idx, item in enumerate(ranking, start=1):
            fused[item.doc_id] = fused.get(item.doc_id, 0.0) + (1.0 / (k_rrf + idx))
    return fused


def retrieve(documents: Dict[str, str], query: str, mode: Mode, k: int) -> List[RetrievedDoc]:
    cache_key = (mode, str(k), query, str(len(documents)))
    if cache_key in _RETRIEVE_CACHE:
        return _RETRIEVE_CACHE[cache_key]

    corpus_key = str(hash(tuple(sorted(documents.keys()))))
    if corpus_key in _CORPUS_CACHE:
        doc_tokens, idf = _CORPUS_CACHE[corpus_key]
    else:
        doc_tokens = _build_doc_tokens(documents)
        idf = _idf(doc_tokens)
        _CORPUS_CACHE[corpus_key] = (doc_tokens, idf)

    if corpus_key in _DOC_TFIDF_CACHE:
        doc_vecs = _DOC_TFIDF_CACHE[corpus_key]
    else:
        doc_vecs = {doc_id: _tfidf_vector(toks, idf) for doc_id, toks in doc_tokens.items()}
        _DOC_TFIDF_CACHE[corpus_key] = doc_vecs

    q_tokens = _tokenize(query)
    vector_backend = os.getenv("RETRIEVAL_VECTOR_BACKEND", "auto").strip().lower()

    if mode == "bm25":
        result = _rank(_bm25_scores(q_tokens, doc_tokens, idf), k)
        _RETRIEVE_CACHE[cache_key] = result
        return result

    if mode == "vector":
        if vector_backend in {"auto", "openai"}:
            emb_scores = _embedding_scores(query, documents)
            if emb_scores:
                result = _rank(emb_scores, k)
                _RETRIEVE_CACHE[cache_key] = result
                return result
            if vector_backend == "openai":
                raise RuntimeError(
                    "RETRIEVAL_VECTOR_BACKEND=openai set, but embeddings were unavailable. "
                    "Check OPENAI_API_KEY/openai package/model."
                )
        result = _rank(_cosine_scores(q_tokens, idf, doc_vecs), k)
        _RETRIEVE_CACHE[cache_key] = result
        return result

    if mode == "hybrid":
        bm25_r = _rank(_bm25_scores(q_tokens, doc_tokens, idf), k)
        if vector_backend in {"auto", "openai"}:
            emb_scores = _embedding_scores(query, documents)
            if emb_scores:
                vec_r = _rank(emb_scores, k)
            elif vector_backend == "openai":
                raise RuntimeError(
                    "RETRIEVAL_VECTOR_BACKEND=openai set, but embeddings were unavailable. "
                    "Check OPENAI_API_KEY/openai package/model."
                )
            else:
                vec_r = _rank(_cosine_scores(q_tokens, idf, doc_vecs), k)
        else:
            vec_r = _rank(_cosine_scores(q_tokens, idf, doc_vecs), k)
        fused = _rrf_fuse([bm25_r, vec_r])
        result = _rank(fused, k)
        _RETRIEVE_CACHE[cache_key] = result
        return result

    raise ValueError(f"Unknown retrieval mode: {mode}")


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & rel)
    return hit / len(rel)


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    rel = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    rel = set(relevant_ids)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        gain = 1.0 if doc_id in rel else 0.0
        if gain:
            dcg += gain / math.log2(i + 1)

    ideal_hits = min(k, len(rel))
    idcg = sum((1.0 / math.log2(i + 1)) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate(
    documents: Dict[str, str],
    queries: Sequence[EvalQuery],
    modes: Sequence[Mode] = ("bm25", "vector", "hybrid"),
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, object]:
    _RETRIEVE_CACHE.clear()
    _CORPUS_CACHE.clear()
    _DOC_TFIDF_CACHE.clear()

    cache_stats = {
        "retrieve_hits": 0,
        "retrieve_misses": 0,
    }
    latency_samples_ms: List[float] = []

    def _percentile(values: Sequence[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
        return round(sorted_vals[idx], 3)

    def _aggregate(query_subset: Sequence[EvalQuery], mode: Mode, k: int) -> Dict[str, object]:
        r_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        per_query: List[Dict[str, object]] = []

        for q in query_subset:
            cache_key = (mode, str(k), q.text, str(len(documents)))
            was_cached = cache_key in _RETRIEVE_CACHE
            t0 = time.perf_counter()
            ranked = retrieve(documents, q.text, mode=mode, k=k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latency_samples_ms.append(elapsed_ms)
            if was_cached:
                cache_stats["retrieve_hits"] += 1
            else:
                cache_stats["retrieve_misses"] += 1
            ranked_ids = [d.doc_id for d in ranked]

            r = recall_at_k(ranked_ids, q.relevant_doc_ids, k)
            m = mrr_at_k(ranked_ids, q.relevant_doc_ids, k)
            n = ndcg_at_k(ranked_ids, q.relevant_doc_ids, k)

            r_sum += r
            mrr_sum += m
            ndcg_sum += n
            per_query.append(
                {
                    "query_id": q.query_id,
                    "split": q.split,
                    "tags": list(q.tags),
                    "recall_at_k": round(r, 6),
                    "mrr_at_k": round(m, 6),
                    "ndcg_at_k": round(n, 6),
                    "top_doc_ids": ranked_ids,
                }
            )

        denom = max(1, len(query_subset))
        return {
            "mode": mode,
            "k": k,
            "num_queries": len(query_subset),
            "recall_at_k": round(r_sum / denom, 6),
            "mrr_at_k": round(mrr_sum / denom, 6),
            "ndcg_at_k": round(ndcg_sum / denom, 6),
            "per_query": per_query,
        }

    report: Dict[str, object] = {
        "num_documents": len(documents),
        "num_queries": len(queries),
        "vector_backend": os.getenv("RETRIEVAL_VECTOR_BACKEND", "auto").strip().lower() or "auto",
        "results": [],
        "results_by_split": {},
        "results_by_tag": {},
    }

    split_map: Dict[str, List[EvalQuery]] = {}
    tag_map: Dict[str, List[EvalQuery]] = {}
    for q in queries:
        split_map.setdefault(q.split or "all", []).append(q)
        if not q.tags:
            tag_map.setdefault("untagged", []).append(q)
        else:
            for tag in q.tags:
                tag_map.setdefault(tag, []).append(q)

    for mode in modes:
        for k in ks:
            report["results"].append(_aggregate(queries, mode, k))

            for split, split_queries in split_map.items():
                report["results_by_split"].setdefault(split, [])
                report["results_by_split"][split].append(_aggregate(split_queries, mode, k))

            for tag, tag_queries in tag_map.items():
                report["results_by_tag"].setdefault(tag, [])
                report["results_by_tag"][tag].append(_aggregate(tag_queries, mode, k))

    total_calls = cache_stats["retrieve_hits"] + cache_stats["retrieve_misses"]
    hit_rate = (cache_stats["retrieve_hits"] / total_calls) if total_calls else 0.0
    report["cache_stats"] = {
        **cache_stats,
        "retrieve_total_calls": total_calls,
        "retrieve_hit_rate": round(hit_rate, 6),
    }
    report["latency"] = {
        "samples": len(latency_samples_ms),
        "p50_ms": _percentile(latency_samples_ms, 50),
        "p95_ms": _percentile(latency_samples_ms, 95),
        "max_ms": round(max(latency_samples_ms), 3) if latency_samples_ms else 0.0,
    }

    return report


def load_eval_file(path: str) -> tuple[Dict[str, str], List[EvalQuery]]:
    payload = json.loads(open(path, "r", encoding="utf-8").read())

    docs_raw = payload.get("documents", [])
    queries_raw = payload.get("queries", [])
    if not docs_raw or not queries_raw:
        raise ValueError("Eval input must include non-empty 'documents' and 'queries'.")

    documents: Dict[str, str] = {}
    for doc in docs_raw:
        doc_id = str(doc.get("doc_id", "")).strip()
        text = str(doc.get("text", "")).strip()
        if doc_id and text:
            documents[doc_id] = text

    queries: List[EvalQuery] = []
    for q in queries_raw:
        query_id = str(q.get("query_id", "")).strip()
        text = str(q.get("text", "")).strip()
        rel = [str(x).strip() for x in q.get("relevant_doc_ids", []) if str(x).strip()]
        split = str(q.get("split", "all")).strip() or "all"
        tags = [str(t).strip() for t in q.get("tags", []) if str(t).strip()]
        if query_id and text and rel:
            queries.append(EvalQuery(query_id=query_id, text=text, relevant_doc_ids=rel, split=split, tags=tags))

    if not documents or not queries:
        raise ValueError("No valid documents/queries found in eval input.")

    return documents, queries
