# humanMem — 面向长对话的主题分区记忆

一个为 AI 编程/聊天 agent 设计的、按**主题分区**的外部记忆系统。它要解决的，是真正
会拖垮长单窗口对话的那个问题：**切换话题时的跨主题混淆**（实体/属性串味、指代歧义、
旧话题锚定、来源张冠李戴）。

整个设计从"低风险/只增量"到"高风险/激进"分阶段推进。核心原则是：可靠且有价值的部分
是**主题化检索 + 按需召回**（fail-safe——它永远只会*追加*上下文，所以一次路由错误绝不
会是灾难性的）；而真正的*淘汰/驱逐*被推后，因为它带有路由 + 存储的风险。

## 这里有什么

### 第 1 步 —— 检索与召回（skills）—— `theme_memory/`、`.claude/skills/theme-memory/`
一个位于 `~/.claude/hank_memory/`（可用 `HANK_MEMORY_DIR` 覆盖）的主题分区存储：
每个主题一份**只追加的 `log.md`**（事实来源）、一份**可重建的 `summary.md`** 缓存，
以及一份 `MEMORY_INDEX.md`。四个工具（`overview / retrieve / append / summarize`）
对外暴露纯 Python 的 BM25 召回（支持中英文）。由 `eval/run_recall.py` 验证。

### 第 2 步 —— 每轮的结构注入（hook）—— `hooks/`、`theme_memory/topic_state.py`
一个 `UserPromptSubmit` hook 会在每一轮之前注入：**当前话题锚点**、**对每个已知话题
打标签的一行描述**，以及一条*不要张冠李戴*的提醒。路由是**粘性的**（除非另一个话题
明显胜出，否则保持不变）且**fail-safe 的**（所有话题始终都被列出，所以一次路由错误
藏不住任何东西）。由 `eval/run_routing.py` 验证。

### 路线图（尚未构建）
第 3 步 —— 软隔离（切换话题时把跑题内容折叠进打标签的摘要里）。第 4 步 —— 真正的
硬淘汰（需要通过 Agent SDK / 自定义 harness 来掌控消息数组本身）。

## 评测 —— LongMemEval（官方 harness）

在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)_S（500 道题）上做端到端 QA。
**规范跑法是直接复用官方 `src/` harness**（`run_generation` + `evaluate_qa`，逐题型 /
弃答 verbatim prompt），我们的 theme-memory BM25 按官方 `retrieval_results` 契约插进去；
读题与评判都用 **MiniMax-M3**（OpenAI 兼容端点）。这样数字才能和 LongMemEval 榜单、以及
mem0 等框架横向对比。跑法见 [`eval/longmemeval/README_official.md`](eval/longmemeval/README_official.md)。

四个 config 一键跑（`bash run_official.sh CONFIG`）：

- **ours** —— 我们的 BM25 在真实 ~115k-token haystack 上取 top-10 session。
- **no-mem** —— 官方 `no-retrieval`：prompt 为**裸问题**，无任何历史、无 "no relevant
  history" 框架（与官方闭卷跑法一致）。检索下界。
- **oracle** —— evidence-only haystack，完美检索上界。
- **official-bm25** —— 官方 `rank_bm25` 基线。

**错因归因**：检索适配器每题落一份 `*.trace.jsonl`，记录每个证据 session 的排名、
recall@k 与 `failure_stage`（`retrieval_ok` / `retrieval_miss` / …），答错时一眼区分
**检索没捞到** vs **生成没用好**。

> 早期用过一套**手写复刻**的 harness（`harness.py` / `run_phaseb.py`，逐字复刻官方
> prompt 但流程自建），跑出过 整体 0.834 / 题型平均 0.817 / 弃答 0.833。现已被官方
> harness 取代——历史结果与说明见
> [`eval/longmemeval/README.md`](eval/longmemeval/README.md) 和
> [`RESULTS.md`](eval/longmemeval/RESULTS.md)。

### mem0 横向对比（官方 harness + MiniMax）

用 **mem0 自己的官方 LongMemEval 评测**（[`memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks)，
即其 README 0.9+ 同一套），把 LLM 换成 MiniMax-M3、embedder 用本地 all-MiniLM——看在我们
模型下 mem0 能到多少，与我们的 BM25 做"各自官方 harness、同模型同题库"的系统级对照。跑法
（含一个绕开 Docker 的本地 REST 桥 `mem0_shim.py`，后端是官方 `mem0.Memory` 库）见
[`eval/longmemeval/README_mem0.md`](eval/longmemeval/README_mem0.md)。

> 为什么选 LongMemEval 而非 LOCOMO：它有逐题的证据标注、一个可控的干扰项 haystack
> （让选择性召回真正起作用），以及专门的知识更新 / 弃答类别——这些恰好对应到我们的
> 检索指标和我们的可靠性风险点。

## 目录结构

```
theme_memory/                      第 1 步存储 + BM25 + CLI；第 2 步 topic_state
hooks/                             UserPromptSubmit 注入 hook
.claude/                           skill + settings.json（hook 注册）
eval/run_recall.py                 第 1 步 recall@k 验证
eval/run_routing.py                第 2 步路由 + fail-safe 验证
eval/longmemeval/
  setup_official.sh                clone 官方仓库（pinned）+ 打 MiniMax 补丁 + 装依赖
  bm25_to_official.py              BM25 → 官方 retrieval_results + 错因归因 trace
  run_official.sh                  ours / no-mem / oracle / official-bm25 一键跑
  official_patches.diff            对官方脚本的最小 MiniMax 适配
  mem0_shim.py / run_mem0.sh       mem0 官方 eval（MiniMax）的本地 REST 桥 + 跑法
  README_official.md / README_mem0.md   官方 harness 跑法文档
  harness.py / run_phaseb.py       （已取代）早期手写复刻 harness
```
