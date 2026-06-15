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

## 评测 —— LongMemEval Phase B

在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)_S（500 道题）上做端到端
QA，按**官方方式**评判（逐题型 verbatim prompt + 弃答 prompt），用 **MiniMax-M3**
同时作为答题模型和评判模型。我们的 BM25 检索（topk=10）：

| 指标 | 分数 |
|---|---|
| 整体准确率（micro，非弃答，n=470） | **0.834** |
| 题型平均（6 种题型的均值） | **0.817** |
| 弃答准确率（n=30） | **0.833** |

单会话召回接近天花板（0.98 / 0.93）；差距出在多会话（0.744）和偏好（0.533）。
完整的逐题型表格、分析与复现方法见
[`eval/longmemeval/README.md`](eval/longmemeval/README.md) 和
[`eval/longmemeval/RESULTS.md`](eval/longmemeval/RESULTS.md)。

### 对比基线（无记忆）

为了量化记忆带来的增益，我们加入一个**基线组**：用同一个 MiniMax-M3 模型、**不提供
任何历史会话**直接作答（`config=no-mem`，即 harness 给模型的历史是"未找到相关历史"）。
这是检索的下界——模型只能靠常识/猜测，无法回看任何会话。完整数字见
[`eval/longmemeval/RESULTS.md`](eval/longmemeval/RESULTS.md)。

> 为什么选 LongMemEval 而非 LOCOMO：它有逐题的证据标注、一个可控的干扰项 haystack
> （让选择性召回真正起作用），以及专门的知识更新 / 弃答类别——这些恰好对应到我们的
> 检索指标和我们的可靠性风险点。

## 目录结构

```
theme_memory/            第 1 步存储 + BM25 + CLI；第 2 步 topic_state
hooks/                   UserPromptSubmit 注入 hook
.claude/                 skill + settings.json（hook 注册）
eval/run_recall.py       第 1 步 recall@k 验证
eval/run_routing.py      第 2 步路由 + fail-safe 验证
eval/longmemeval/        LongMemEval Phase B harness、结果、文档
```
