# LongMemEval — Phase B (end-to-end QA) for theme-memory

Evaluates the theme-memory **retrieval/recall engine** on the
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark, judged the **official
LongMemEval way** (verbatim per-type / abstention judge prompts), with **MiniMax-M3** as
both the answering and the judging model.

## Why LongMemEval (vs LOCOMO)

LongMemEval was designed to fix LOCOMO's weaknesses for memory evaluation: it has
per-question **evidence labels**, a **controllable distractor haystack** (so selective
recall actually matters), and dedicated **knowledge-update** and **abstention** categories
— which map directly onto our retrieval metric and our biggest reliability risks.

## What this measures (and what it doesn't)

- **Measures:** whether our BM25 recall (`theme_memory/retrieve.py`) surfaces the right
  sessions to answer cross-session questions, and how that translates to QA accuracy under
  the official judge. This is the value of **Step 1 (retrieval & recall)**.
- **Does not measure:** Step 2's in-session topic-switch interference (these are
  cross-session QA tasks, not single-window topic switching). That is covered by
  `eval/run_routing.py`.

## Faithful replication notes

- **Judge prompts** (`harness.py`): verbatim from LongMemEval `get_anscheck_prompt`, with
  the per-type variants — standard / temporal (off-by-one tolerance) / knowledge-update /
  preference (rubric) — and abstention selected by `_abs` in `question_id`. Label =
  `"yes" in response.lower()` (after stripping MiniMax-M3's `<think>` block).
- **Answer prompt** (`harness.py`): verbatim non-CoT "direct" reading template, with the
  `### Session i / Session Date / Session Content` history block format.
- **Retrieval:** session granularity; each session doc = its **user** turns only (matches
  the official flat index). Ranked by **our** BM25 (`theme_memory.retrieve.bm25`); the
  session id (which encodes `answer`/`noans`) is kept out of the scored text so ground
  truth can't leak.
- **Metrics** (`run_phaseb.py`): **Task-averaged** (mean of per-type accuracies),
  **Overall** (micro over non-abstention), and **Abstention accuracy** — matching
  `print_qa_metrics.py`.

## Reproduce

```bash
# 1. data (gitignored; ~292MB) — downloaded automatically on first run, or:
#    curl -sL -o data/longmemeval_oracle.json      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
#    curl -sL -o data/longmemeval_s_cleaned.json   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 2. API key (kept OUTSIDE the repo)
echo "sk-cp-..." > ~/.minimax_key && chmod 600 ~/.minimax_key   # MiniMax CN, model MiniMax-M3

# 3. quick smoke (12 questions, balanced across types)
python3 run_phaseb.py all --config bm25 --topk 10 --limit 12 --stratified

# 4. full benchmark (500 questions)
python3 run_phaseb.py all --config bm25 --topk 10 --workers 8

# configs: bm25 (our retriever) | oracle (perfect-retrieval upper bound) | no-mem (lower bound)
```

Outputs land in `results/`: `hyp_<tag>.jsonl` (answers), `eval_<tag>.jsonl` (judgments),
and the printed metrics.

## Results

See `RESULTS.md` (written after the full run).
