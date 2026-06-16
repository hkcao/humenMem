#!/usr/bin/env bash
# mem0 横向对比：跑 mem0 OFFICIAL LongMemEval harness（vendor/memory-benchmarks），
# 但 LLM（抽取+答题+评判）全换成 MiniMax-M3，embedder 用本地 all-MiniLM-L6-v2。
#
# 前置：mem0_shim.py 必须在 :8888 跑着（见下方 start-shim）。run.py 带断点续跑。
#
#   bash run_mem0.sh start-shim          # 启动本地 mem0 REST 桥（前台；另开窗口）
#   bash run_mem0.sh run [PROJECT] [PER_TYPE] [WORKERS]
#     PROJECT  = 结果子目录名（默认 mm30）
#     PER_TYPE = 每题型抽样数（默认 5 -> 30 题）
#     WORKERS  = 并发题数（默认 3；MiniMax 限流下别太高）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv-mem0/bin/python"
MB="$REPO/vendor/memory-benchmarks"
KEY="$(cat ~/.minimax_key)"
BASE="https://api.minimaxi.com/v1"
DATA="$HERE/data/longmemeval_s_cleaned.json"

case "${1:?start-shim|run}" in
  start-shim)
    export MEM0_TELEMETRY=false
    exec "$REPO/.venv-mem0/bin/uvicorn" --app-dir "$HERE" mem0_shim:app \
      --host 127.0.0.1 --port 8888 --log-level warning ;;
  run)
    PROJECT="${2:-mm30}"; PER_TYPE="${3:-5}"; WORKERS="${4:-3}"
    OUT="$HERE/official_out/mem0"; mkdir -p "$OUT"
    # sanity: shim up?
    curl -s --noproxy '*' -m5 localhost:8888/health >/dev/null \
      || { echo "ERROR: mem0 shim not reachable on :8888. Start it: bash run_mem0.sh start-shim"; exit 1; }
    cd "$MB"
    OPENAI_API_KEY="$KEY" OPENAI_BASE_URL="$BASE" MEM0_TELEMETRY=false PYTHONUNBUFFERED=1 \
      "$PY" -m benchmarks.longmemeval.run \
        --project-name "$PROJECT" --provider openai \
        --answerer-model MiniMax-M3 --judge-model MiniMax-M3 \
        --backend oss --mem0-host http://localhost:8888 \
        --per-type "$PER_TYPE" --top-k 30 --top-k-cutoffs 10,20,30 \
        --max-workers "$WORKERS" --dataset-path "$DATA" --output-dir "$OUT"
    ;;
  *) echo "usage: bash run_mem0.sh start-shim | run [PROJECT] [PER_TYPE] [WORKERS]"; exit 1 ;;
esac
