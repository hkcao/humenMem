# LongMemEval Phase B — Results & Status

Config: our BM25 session retrieval (`config=bm25`, `topk=10`) → answer with **MiniMax-M3**
→ judge with **MiniMax-M3** using the **verbatim official LongMemEval prompts**
(per-type + abstention). Dataset: `LongMemEval_S` (500 questions, ~115k-token haystack).

## Status: partially complete — blocked on MiniMax quota

| Item | State |
|---|---|
| Harness (faithful replication) | ✅ done & validated |
| 12-question smoke sample (gen + judge) | ✅ complete |
| Full 500 — **generation** | ⚠️ **412/500 saved** (`results/hyp_bm25_k10_nall.jsonl`) |
| Full 500 — **judging** | ❌ 0/500 — MiniMax Token Plan quota exhausted |

The MiniMax Token Plan hit its usage cap mid-run:

> `rate_limit_error: 已达到 Token Plan 用量上限：请升级 Token Plan 套餐或购买积分补充用量。(2056)`

Each generation call carries a ~30k-token retrieved context, so the 500-question sweep
exceeded the plan's quota. This is an external account limit, not a harness issue.

## What we have: 12-question smoke sample (balanced, 2 per type)

Validates the end-to-end pipeline; **not** the benchmark result (n is tiny).

| question_type | acc | n |
|---|---|---|
| single-session-user | 1.00 | 2 |
| single-session-assistant | 1.00 | 2 |
| single-session-preference | 0.00 | 2 |
| multi-session | 0.00 | 2 |
| temporal-reasoning | 1.00 | 2 |
| knowledge-update | 1.00 | 2 |
| **Task-averaged** | **0.667** | — |
| **Overall (micro)** | **0.667** | 12 |

Spot-checked judgments were faithful: failures are genuine retrieval/answer misses
(e.g. multi-session counting — gold 3 vs answered 2; preference — model failed to recall
the user's stated Sony/Premiere-Pro setup), exactly what LongMemEval targets.

## Finishing the full run (resumable, ~3.5–4M tokens)

Once the MiniMax quota is restored (top-up/upgrade) or a working key is set in
`~/.minimax_key`, one command resumes from the 412 saved answers — it backfills the
remaining ~88 generations and judges all 500, skipping everything already done:

```bash
cd eval/longmemeval
MINIMAX_MIN_INTERVAL=1.0 python3 run_phaseb.py all --config bm25 --topk 10 --workers 3
```

Then read the printed Task-averaged / Overall / Abstention metrics. Update this file with
the full-500 numbers.

> Note: keep `--workers` low and `MINIMAX_MIN_INTERVAL` ≥ 1.0s to avoid the per-minute
> 429 storms seen at higher concurrency.
