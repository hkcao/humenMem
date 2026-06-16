#!/usr/bin/env bash
# Reproducibly fetch the OFFICIAL harnesses we run against and apply our minimal
# MiniMax adaptation patch. Clones are gitignored; this script is the source of truth.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENDOR="$REPO/vendor"; mkdir -p "$VENDOR"

# Pinned commits the patch + our results were produced against.
LME_COMMIT="9e0b455f4ef0e2ab8f2e582289761153549043fc"   # xiaowu0162/LongMemEval
MB_COMMIT="4b61c5d31b9c668a12b4f5e78064248a02c82d2b"     # mem0ai/memory-benchmarks

clone_at() { # url dir commit
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then git clone "$url" "$dir"; fi
  git -C "$dir" fetch --depth 1 origin "$commit" 2>/dev/null || git -C "$dir" fetch origin
  git -C "$dir" checkout -q "$commit"
}

clone_at https://github.com/xiaowu0162/LongMemEval.git      "$VENDOR/LongMemEval"        "$LME_COMMIT"
clone_at https://github.com/mem0ai/memory-benchmarks.git    "$VENDOR/memory-benchmarks"  "$MB_COMMIT"

# Apply our MiniMax adaptation to LongMemEval (idempotent: skip if already applied).
if ! git -C "$VENDOR/LongMemEval" diff --quiet; then
  echo "LongMemEval already patched (working tree dirty); skipping."
else
  git -C "$VENDOR/LongMemEval" apply "$HERE/official_patches.diff" \
    && echo "Applied official_patches.diff to vendor/LongMemEval"
fi

# Python deps for the LongMemEval reader+judge path (light; no torch/transformers).
"$REPO/.venv/bin/pip" install -q openai tiktoken backoff numpy
echo "Setup complete. See README.md for run commands."
