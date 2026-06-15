# LongMemEval Phase B — Results

**Config:** our BM25 session retrieval (`config=bm25`, `topk=10`) → answer with
**MiniMax-M3** → judge with **MiniMax-M3** using the **verbatim official LongMemEval
prompts** (per-type + abstention). **Dataset:** `LongMemEval_S` (500 questions,
~115k-token haystack per question). **Complete: 500/500 generated and judged.**

## Headline

| Metric | Score |
|---|---|
| **Overall accuracy** (micro, non-abstention, n=470) | **0.834** |
| **Task-averaged accuracy** (mean of 6 types) | **0.817** |
| **Abstention accuracy** (n=30) | **0.833** |

## Per question type

| question_type | accuracy | n |
|---|---|---|
| single-session-user | 0.984 | 64 |
| single-session-assistant | 0.929 | 56 |
| temporal-reasoning | 0.866 | 127 |
| knowledge-update | 0.847 | 72 |
| multi-session | 0.744 | 121 |
| single-session-preference | 0.533 | 30 |

(Per-type n excludes the 30 abstention questions, which are scored separately;
64+56+127+72+121+30 + 30 abstention = 500.)

## Reading the results

- **Single-session recall is near-ceiling** (0.98 / 0.93): when the answer lives in one
  session, our BM25 reliably surfaces it and MiniMax-M3 answers correctly. This is the
  core value of Step 1 (retrieval & recall) and it lands.
- **Multi-session (0.744)** is the main gap: these need *all* relevant sessions in the
  top-k, and flat BM25's `recall_all@k` is the bottleneck (spot-checks showed genuine
  miscounts from a missing session, not judging errors). This is exactly where a better
  retriever / index expansion would help.
- **Preference (0.533)** is hardest: the question rarely shares keywords with the persona
  session, so BM25 misses it — a known weakness of lexical retrieval for preference recall.
- **Abstention (0.833)** is healthy: the additive/fail-safe design (never hide, let the
  model see it lacks evidence) does not push the model into over-confident hallucination.

## Caveats

- This measures **Step 1 (retrieval → QA)** on cross-session questions. It does **not**
  measure Step 2's in-session topic-switch interference (covered by `eval/run_routing.py`).
- Answerer and judge are the **same** model (MiniMax-M3), matching the budget constraint;
  LongMemEval's reference judge is GPT-4o, so absolute numbers aren't 1:1 comparable to the
  paper's leaderboard, but the methodology (prompts, metrics) is replicated verbatim.

## Reproduce

```bash
cd eval/longmemeval
MINIMAX_MIN_INTERVAL=1.2 python3 run_phaseb.py all --config bm25 --topk 10 --workers 3
```

Keep `--workers` low and `MINIMAX_MIN_INTERVAL` ≥ 1.0s to stay under MiniMax's per-minute
quota. Artifacts: `results/hyp_bm25_k10_nall.jsonl` (answers),
`results/eval_bm25_k10_nall.jsonl` (judgments).
