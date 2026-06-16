#!/usr/bin/env bash
# End-to-end LongMemEval QA via the OFFICIAL harness (vendor/LongMemEval/src), with
# MiniMax-M3 as reader+judge over its OpenAI-compatible endpoint. Our theme-memory BM25
# is plugged in by bm25_to_official.py, which emits the official retrieval_results schema.
#
#   bash run_official.sh CONFIG [LIMIT] [STRATIFIED]
#     CONFIG     = ours | no-mem | oracle | official-bm25
#     LIMIT      = number of questions (0/empty = all 500)
#     STRATIFIED = "strat" to balance across the 6 question types
#
# Configs:
#   ours          our BM25 over the real 115k-token haystack, top-10 sessions   (headline)
#   no-mem        official no-retrieval: bare question, no history               (lower bound)
#   oracle        evidence-only haystack, all sessions shown                     (upper bound)
#   official-bm25 rank_bm25.BM25Okapi baseline (needs: pip install rank_bm25)    (BM25 baseline)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GEN="$REPO/vendor/LongMemEval/src/generation/run_generation.py"
EVAL="$REPO/vendor/LongMemEval/src/evaluation/evaluate_qa.py"
KEY="$(cat ~/.minimax_key)"
BASE="https://api.minimaxi.com/v1"
S_CLEANED="$HERE/data/longmemeval_s_cleaned.json"   # real haystack (reader + judge ref)
ORACLE="$HERE/data/longmemeval_oracle.json"          # evidence-only variant
OUT="$HERE/official_out"; mkdir -p "$OUT/gen"

CONFIG="${1:?need CONFIG: ours|no-mem|oracle|official-bm25}"
LIMIT="${2:-0}"
STRAT="${3:-}"; [ "$STRAT" = "strat" ] && STRAT_FLAG="--stratified" || STRAT_FLAG=""
TAG="${CONFIG}_n${LIMIT:-all}${STRAT:+_strat}"

# ---- 1. build the input file + retriever_type for run_generation ----
case "$CONFIG" in
  ours|official-bm25)
    IN="$OUT/retr_${TAG}.jsonl"
    $PY "$HERE/bm25_to_official.py" --out "$IN" --retriever "$CONFIG" \
        ${LIMIT:+--limit "$LIMIT"} $STRAT_FLAG
    RETR_TYPE="flat-session" ;;
  no-mem)
    IN="$OUT/in_${TAG}.json"
    $PY - "$S_CLEANED" "$IN" "$LIMIT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[3] or 0)
json.dump(d[:n] if n else d, open(sys.argv[2],"w"))
PY
    RETR_TYPE="no-retrieval" ;;
  oracle)
    IN="$OUT/in_${TAG}.json"
    $PY - "$ORACLE" "$IN" "$LIMIT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[3] or 0)
json.dump(d[:n] if n else d, open(sys.argv[2],"w"))
PY
    RETR_TYPE="orig-session" ;;
  *) echo "unknown CONFIG: $CONFIG"; exit 1 ;;
esac

# ---- 2. official generation (MiniMax-M3) ----
$PY "$GEN" \
  --in_file "$IN" --out_dir "$OUT/gen" --out_file_suffix "_$TAG" \
  --model_name MiniMax-M3 --model_alias minimax-m3 \
  --openai_base_url "$BASE" --openai_key "$KEY" \
  --retriever_type "$RETR_TYPE" --topk_context 10 \
  --history_format json --useronly false --cot false --gen_length 2048

HYP="$(ls -t "$OUT"/gen/*"_$TAG" | head -1)"

# ---- 3. official judge (MiniMax-M3) ----
OPENAI_API_KEY="$KEY" OPENAI_BASE_URL="$BASE" \
  $PY "$EVAL" minimax-m3 "$HYP" "$S_CLEANED"

echo "== done [$TAG] : hyp=$HYP =="
