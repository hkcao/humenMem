#!/usr/bin/env bash
# Parallel THEME-MEMORY ingest: one process per question_id (stores are fully independent),
# then a single merged generation + judge over all shards. Speeds up the ~8h sequential
# 12-question run to wall-clock ~= slowest single question, API rate limits permitting
# (_chat has exponential backoff, so over-concurrency self-throttles).
#
#   bash run_theme_parallel.sh "qid1,qid2,..."   TAG
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GEN="$REPO/vendor/LongMemEval/src/generation/run_generation.py"
EVAL="$REPO/vendor/LongMemEval/src/evaluation/evaluate_qa.py"
# Model is swappable via env (defaults = MiniMax-M3). For DeepSeek-V4-Pro:
#   MODEL_NAME=deepseek-v4-pro MODEL_ALIAS=deepseek-v4-pro \
#   API_BASE=https://api.deepseek.com/v1 KEY_FILE=~/.deepseek_key bash run_theme_parallel.sh ...
MODEL_NAME="${MODEL_NAME:-MiniMax-M3}"
MODEL_ALIAS="${MODEL_ALIAS:-minimax-m3}"
API_BASE="${API_BASE:-https://api.minimaxi.com/v1}"
KEY_FILE="${KEY_FILE:-$HOME/.minimax_key}"
KEY="$(cat "${KEY_FILE/#\~/$HOME}")"
BASE="$API_BASE"
S_CLEANED="$HERE/data/longmemeval_s_cleaned.json"
OUT="$HERE/official_out"; mkdir -p "$OUT/gen" "$OUT/shards"

QIDS="$1"; TAG="${2:-theme_n12_strat}"
IFS=',' read -ra ARR <<< "$QIDS"
echo "== model=$MODEL_NAME  base=$API_BASE  key=$KEY_FILE =="

echo "== parallel ingest: ${#ARR[@]} questions =="
PIDS=()
for q in "${ARR[@]}"; do
  $PY "$HERE/theme_to_official.py" --out "$OUT/shards/retr_${q}.jsonl" \
      --qids "$q" --stores "$OUT/theme_stores" --topk 10 \
      --model "$MODEL_NAME" --base_url "$API_BASE" --key_file "${KEY_FILE/#\~/$HOME}" \
      > "$OUT/shards/ingest_${q}.log" 2>&1 &
  PIDS+=($!)
done
echo "launched pids: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL+1)); done
echo "== ingest done (failed procs: $FAIL) =="

# merge shards in the given qid order
IN="$OUT/retr_${TAG}.jsonl"
: > "$IN"
for q in "${ARR[@]}"; do
  [ -s "$OUT/shards/retr_${q}.jsonl" ] && cat "$OUT/shards/retr_${q}.jsonl" >> "$IN"
done
echo "merged -> $IN ($(wc -l < "$IN") lines)"

# single official generation + judge over the merged log
$PY "$GEN" \
  --in_file "$IN" --out_dir "$OUT/gen" --out_file_suffix "_$TAG" \
  --model_name "$MODEL_NAME" --model_alias "$MODEL_ALIAS" \
  --openai_base_url "$BASE" --openai_key "$KEY" \
  --retriever_type flat-session --topk_context 10 \
  --merge_key_expansion_into_value replace \
  --history_format json --useronly false --cot false --gen_length 2048

HYP="$(ls -t "$OUT"/gen/*"_$TAG" | head -1)"
OPENAI_API_KEY="$KEY" OPENAI_BASE_URL="$BASE" \
  $PY "$EVAL" "$MODEL_ALIAS" "$HYP" "$S_CLEANED"
echo "== done [$TAG] : hyp=$HYP =="
