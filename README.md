# humanMem — topic-partitioned memory for long conversations

An external, topic-partitioned memory for AI coding/chat agents, built to fight the thing
that actually breaks long single-window conversations: **cross-topic confusion when you
switch subjects** (entity/attribute bleed, reference ambiguity, stale-topic anchoring,
source misattribution).

The design is staged from low-risk/additive to high-risk/aggressive. The principle: the
reliable, valuable part is **topic-structured retrieval + on-demand recall** (fail-safe —
it only ever *adds* context, so a routing mistake is never catastrophic); true *eviction*
is deferred because it carries the routing + storage risk.

## What's here

### Step 1 — retrieval & recall (skills) — `theme_memory/`, `.claude/skills/theme-memory/`
A topic-partitioned store under `~/.claude/hank_memory/` (override `HANK_MEMORY_DIR`):
per-topic **append-only `log.md`** (source of truth), a **rebuildable `summary.md`**
cache, and a `MEMORY_INDEX.md`. Four tools (`overview / retrieve / append / summarize`)
expose pure-Python BM25 recall (English + Chinese). Verified by `eval/run_recall.py`.

### Step 2 — per-turn structure injection (hook) — `hooks/`, `theme_memory/topic_state.py`
A `UserPromptSubmit` hook injects, before each turn: the **current-topic anchor**, a
**tagged one-line description of every known topic**, and a *don't-misattribute* reminder.
Routing is **sticky** (stay unless another topic clearly wins) and **fail-safe** (all
topics are always listed, so a misroute can't hide anything). Verified by
`eval/run_routing.py`.

### Roadmap (not yet built)
Step 3 — soft isolation (fold the off-topic into tagged summaries on switch). Step 4 —
true hard eviction (requires owning the message array via Agent SDK / a custom harness).

## Evaluation — LongMemEval Phase B

End-to-end QA on [LongMemEval](https://github.com/xiaowu0162/LongMemEval)_S (500 questions),
judged the **official way** (verbatim per-type + abstention prompts), with **MiniMax-M3**
as both answerer and judge. Our BM25 retrieval (topk=10):

| Metric | Score |
|---|---|
| Overall accuracy (micro, non-abstention, n=470) | **0.834** |
| Task-averaged (mean of 6 types) | **0.817** |
| Abstention (n=30) | **0.833** |

Single-session recall is near-ceiling (0.98 / 0.93); the gaps are multi-session (0.744)
and preference (0.533). Full per-type table, analysis, and reproduction:
[`eval/longmemeval/README.md`](eval/longmemeval/README.md) and
[`eval/longmemeval/RESULTS.md`](eval/longmemeval/RESULTS.md).

> Why LongMemEval over LOCOMO: it has per-question evidence labels, a controllable
> distractor haystack (so selective recall actually matters), and dedicated
> knowledge-update / abstention categories — which map onto our retrieval metric and our
> reliability risks.

## Layout

```
theme_memory/            Step 1 store + BM25 + CLI; Step 2 topic_state
hooks/                   UserPromptSubmit injection hook
.claude/                 skill + settings.json (hook registration)
eval/run_recall.py       Step 1 recall@k verification
eval/run_routing.py      Step 2 routing + fail-safe verification
eval/longmemeval/        LongMemEval Phase B harness, results, docs
```
