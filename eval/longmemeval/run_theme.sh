#!/usr/bin/env bash
# End-to-end THEME-MEMORY (the real scheme) on LongMemEval via the OFFICIAL harness.
# Unlike run_official.sh's `ours` (BM25 baseline, topic layer OFF), this BUILDS memory:
# theme_to_official.py ingests each haystack into a per-question theme store (MiniMax does
# topic routing + fact extraction), recalls topic-aware memories, and emits the official
# retrieval log. We then read with run_generation.py `--merge_key_expansion_into_value
# replace` (feeds our memory text under the official "facts extracted" prompt) and judge
# with the unchanged official evaluate_qa.py — same judge as every other config.
#
#   bash run_theme.sh [LIMIT] [strat]      LIMIT=0/empty -> all 500
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GEN="$REPO/vendor/LongMemEval/src/generation/run_generation.py"
EVAL="$REPO/vendor/LongMemEval/src/evaluation/evaluate_qa.py"
KEY="$(cat ~/.minimax_key)"
BASE="https://api.minimaxi.com/v1"
S_CLEANED="$HERE/data/longmemeval_s_cleaned.json"
OUT="$HERE/official_out"; mkdir -p "$OUT/gen"

LIMIT="${1:-30}"
STRAT="${2:-}"; [ "$STRAT" = "strat" ] && STRAT_FLAG="--stratified" || STRAT_FLAG=""
TAG="theme_n${LIMIT:-all}${STRAT:+_strat}"
IN="$OUT/retr_${TAG}.jsonl"

# ---- 1. build theme memory + topic-aware recall -> official retrieval log (resumable) ----
$PY "$HERE/theme_to_official.py" --out "$IN" --limit "$LIMIT" $STRAT_FLAG \
    --stores "$OUT/theme_stores" --topk 10

# ---- 2. official generation (MiniMax-M3), feeding OUR memory text via `replace` ----
$PY "$GEN" \
  --in_file "$IN" --out_dir "$OUT/gen" --out_file_suffix "_$TAG" \
  --model_name MiniMax-M3 --model_alias minimax-m3 \
  --openai_base_url "$BASE" --openai_key "$KEY" \
  --retriever_type flat-session --topk_context 10 \
  --merge_key_expansion_into_value replace \
  --history_format json --useronly false --cot false --gen_length 2048

HYP="$(ls -t "$OUT"/gen/*"_$TAG" | head -1)"

# ---- 3. official judge (MiniMax-M3, unchanged) ----
OPENAI_API_KEY="$KEY" OPENAI_BASE_URL="$BASE" \
  $PY "$EVAL" minimax-m3 "$HYP" "$S_CLEANED"

echo "== done [$TAG] : hyp=$HYP =="
