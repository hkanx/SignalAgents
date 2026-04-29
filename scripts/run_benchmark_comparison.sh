#!/usr/bin/env bash
# run_benchmark_comparison.sh — Repeatable retrieval benchmark across bm25, vector, and hybrid.
#
# Usage:
#   bash scripts/run_benchmark_comparison.sh [path/to/eval_seed.json]
#
# Defaults to data/retrieval_eval_seed.json if no argument given.
# All three runs use the same bootstrap seed for reproducibility.
# Results are written to data/benchmark_<mode>_<timestamp>.json/.md
# A combined comparison is written by compare_benchmark_runs.py afterwards.
#
# Requirements:
#   - OPENAI_API_KEY in env (or .env) for vector/hybrid runs
#   - python scripts/run_retrieval_eval.py must be runnable from project root
#   - jq (optional, for pretty-printing the recommendation)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load .env if present (for OPENAI_API_KEY etc.)
if [ -f ".env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

SEED=42
INPUT="${1:-data/retrieval_eval_seed.json}"
OUTDIR="data"
TS=$(date +%Y%m%dT%H%M%S)

mkdir -p "$OUTDIR"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: Eval seed file not found: $INPUT" >&2
  echo "Generate one first with:" >&2
  echo "  python scripts/build_retrieval_eval_seed.py --input <analyzed_rows.json> --out-seed $INPUT" >&2
  exit 1
fi

echo "=== Retrieval Benchmark Comparison ==="
echo "Input:  $INPUT"
echo "Seed:   $SEED"
echo "Output: $OUTDIR/benchmark_*_${TS}.{json,md}"
echo ""

REPORTS=()

for mode in bm25 vector hybrid; do
  echo "--- Running mode: $mode ---"
  case $mode in
    bm25)
      export RETRIEVAL_VECTOR_BACKEND=tfidf
      ;;
    vector)
      export RETRIEVAL_VECTOR_BACKEND=openai
      ;;
    hybrid)
      export RETRIEVAL_VECTOR_BACKEND=auto
      ;;
  esac

  OUT_JSON="$OUTDIR/benchmark_${mode}_${TS}.json"
  OUT_MD="$OUTDIR/benchmark_${mode}_${TS}.md"

  python scripts/run_retrieval_eval.py \
    --input "$INPUT" \
    --output-json "$OUT_JSON" \
    --output-md "$OUT_MD" \
    --bootstrap-seed "$SEED" \
    --decision-k 10 \
    --min-intent-paraphrase-recall 0.10 \
    --min-intent-paraphrase-ndcg 0.06 \
    || true   # don't abort on gate failures — we want all three reports

  REPORTS+=("$OUT_JSON")
  echo "  Saved: $OUT_JSON"
done

echo ""
echo "=== Generating comparison report ==="

COMPARISON_MD="$OUTDIR/benchmark_comparison_latest.md"
python scripts/compare_benchmark_runs.py \
  --reports "${REPORTS[@]}" \
  --output-md "$COMPARISON_MD" \
  --baseline bm25

echo ""
echo "Done. Comparison report: $COMPARISON_MD"
