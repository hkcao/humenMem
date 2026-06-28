# humanMem — 面向长对话的主题分区记忆

一个为 AI 编程/聊天 agent 设计的、按**主题分区**的外部记忆系统。它要解决的，是真正会
拖垮长单窗口对话的那个问题：**切换话题时的跨主题混淆**（实体/属性串味、指代歧义、旧话题
锚定、来源张冠李戴），以及**滚出窗口的事实如何被可靠召回**。

核心原则贯穿始终：

- **Fail-safe**：记忆只会*追加*上下文，从不静默删事实——一次路由错误绝不会是灾难性的。
- **theme ⊇ bm25**：召回最底层始终是一份无损的全 session 语料（floor），其 BM25 检索等价
  于扁平 session 基线，所以主题层再怎么出错，召回也**永不弱于**朴素 BM25。
- **无损更新**：wiki 用"局部增删改"而非全量重写，模型只动它点名的行，其余逐字保留，杜绝
  重写时静默丢事实。

## 架构总览

系统分两层，共用同一套存储引擎（`theme_memory/scripts/`）：

| 层 | 是什么 | 入口 |
|---|---|---|
| **运行时（agent 侧）** | Claude Code skill（按需召回/写入）+ 每轮注入话题状态的 hook | `theme_memory/SKILL.md` + `scripts/`、`hooks/` |
| **完整方案（eval 侧）** | LongMemEval 上的端到端 theme-memory：LLM 主题路由 + 入库 + 主题归并 + 级联召回 | `eval/longmemeval/theme_to_official.py` |

运行时层用的是稳定、可靠的子集（主题化 BM25 召回 + 模型路由的话题状态注入）；更激进的机制
（跨主题根 wiki、主题归并、级联召回、充分性探针）先在 eval 侧验证，证明有效后再下放运行时。

## 存储布局

存储根目录为 `~/.claude/hank_memory/`（用 `HANK_MEMORY_DIR` 覆盖；eval 里每题一个隔离子库）：

```
<root>/
  MEMORY_INDEX.md            主题清单：每行 "主题名 — 一行描述"
  log.md                     全局时间线：所有主题的条目按写入时序（带 topic= 标签）
  wiki.md                    跨主题根 wiki：pending 待办/关键事实/跨主题数据（eval 侧维护）
  <topic>/
    logs/<YYYY-MM-DD>.md     按天切分的只追加日志 —— 事实来源（source of truth）
    wiki.md                  该主题的知识库（局部增删改维护的事实/数据）
  _sessions/logs/<day>.md    无损 session floor：每个原始 session 全文（bm25 等价语料，eval 侧）
```

`logs/` 是真相源，`wiki.md` 是可重建缓存，`_sessions/` 是召回兜底的无损底座。运行时 CLI 会
写 `logs/`、`wiki.md`、`log.md`、`MEMORY_INDEX.md`；根 `wiki.md`、`_sessions/` floor、主题归并
目前由 eval 侧的 ingest 产生。

## 快速开始 —— 在 agent 里集成

记忆引擎是**纯 Python、零依赖**。集成有两条腿，可单独用也可一起用。

**A. Skill（模型按需召回/写入）** —— skill 是标准结构（`theme_memory/SKILL.md` + `scripts/`）。
拷进目标项目时放到它的 `.claude/skills/` 下供框架发现：

```bash
cp -r theme_memory/   <你的项目>/.claude/skills/theme_memory/
```

之后 agent 会在"用户提到之前聊过的东西 / 切换话题 / 问当前窗口外的上下文"时自动触发 skill，
调用四个工具：

```bash
python3 theme_memory/scripts/cli.py overview                              # 索引 + 根 wiki + 各主题 wiki
python3 theme_memory/scripts/cli.py retrieve --query "staging 数据库 host" --limit 5   # BM25 跨主题召回
python3 theme_memory/scripts/cli.py append --topic deploy-prod --source user \
  --content "Prod DB host 是 db-prod.internal:5432" --desc "生产部署配置"     # 持久化一条事实
# 局部维护 wiki（增删改，未触及的行逐字保留；--root 改跨主题根 wiki）：
python3 theme_memory/scripts/cli.py wiki --topic deploy-prod                          # 带行号查看
python3 theme_memory/scripts/cli.py wiki --topic deploy-prod --append "Prod region eu-west-1" \
  --update 2 "Prod DB host 现为 db2.internal" --delete 5
python3 theme_memory/scripts/cli.py merge --into farm --from "farm-ops,farm-maint"    # 归并近义主题
python3 theme_memory/scripts/cli.py summarize --topic deploy-prod                     # 整篇重写（兜底）
```

`retrieve` / `append` **只追加、从不删除**（召回永远安全 fail-safe）；`wiki` / `merge` 是你显式
发起的维护操作，`wiki` 的局部编辑不会动你没点名的行。

**B. Hook（每轮自动注入话题状态）** —— 拷 hook 并在 `.claude/settings.json` 注册：

```bash
cp -r hooks/   <你的项目>/hooks/
```

```jsonc
// .claude/settings.json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR/hooks/inject_topic_state.py\"" } ] }
    ]
  }
}
```

每轮提交前，hook 注入**当前话题锚点 + 每个已知话题的一行标签 + 一条"别张冠李戴"的提醒**。
当前话题由**模型判断**（无 key/失败时回落到带粘性的 BM25），所有话题始终列出（fail-safe）。
hook 出任何错都静默 exit 0，绝不挡住或破坏用户的 prompt。可用 `THEME_MEMORY_MODEL_ROUTING=0`
关掉模型路由、退回纯 BM25（零延迟）。

> 不用 Claude Code 也能集成：把 `theme_memory/scripts/cli.py` 当子进程调，或 `import store /
> retrieve / topic_state` 在你自己的 harness 里用——都是普通 Python 模块。

## 完整方案（eval 侧的端到端设计）

`eval/longmemeval/theme_to_official.py` 是 theme-memory 设计的完整实现。对每道题的 ~115k haystack：

**Ingest（入库）** —— 按时间顺序逐 session：

1. **无损 floor**：每个 session 全文先存进 `_sessions/`（与扁平 BM25 基线同语料，保底覆盖）。
2. **主题路由**：一次 LLM 调用把该 session 的 turns 分组到主题（能复用已有主题就复用），verbatim
   摘录写进各主题的 `logs/<day>.md`。
3. **局部更新 wiki**：模型对每个主题的 wiki 做**增删改**（append 新事实 / update 被取代的行 /
   delete 已完成或过时的行），未点名的行逐字保留——无损累积。
4. **根 wiki（按天）**：把跨主题关键事实（pending 待办、个人核心事实、跨主题数据）局部更新进根
   `wiki.md`——**同一天的 session 全处理完才整理一次**（模拟"每天结束时归整"），而非每 session 一次。
5. **主题归并**：ingest 后跑一次 LLM 聚类，把近义/碎片化的小主题合并成大主题（`merge_topics`）。

**Recall（召回）** —— 级联升级，每层由充分性探针 gate，根 wiki 始终在视野内：

1. **模型选相关主题**（BM25 兜底）。
2. **wiki 层**：**模型判断**该题走 `wiki_bm25`（检索若干相关行——单点查找，可扩展）还是
   `wiki_full`（整篇加载——计数/聚合/概览），不强制；不够则继续下探。
3. `topic_raw`（选中主题的 raw 日志）→ `full_raw`（**floor 兜底**，全 session BM25，等价 bm25 基线）。
4. floor 保证 theme 召回**永不弱于 bm25**。

召回结果按官方 `retrieval_results` 契约喂给官方 reader，由官方 judge 评分（详见下文评测）。

## 评测 —— LongMemEval（官方 harness）

在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)_S（500 题，每题 ~115k-token haystack）
上做端到端 QA，**全程复用官方 `src/` harness**（`run_generation` + `evaluate_qa`，逐题型/弃答
verbatim prompt）；读题与评判都用 **MiniMax-M3**（OpenAI 兼容端点），数字可与 LongMemEval 榜单、
mem0 等横向对比。完整跑法见 [`eval/longmemeval/README.md`](eval/longmemeval/README.md)。

```bash
cd eval/longmemeval
bash setup_official.sh                       # clone 官方仓库(pinned) + 打 MiniMax 补丁 + 建 .venv
echo "sk-..." > ~/.minimax_key && chmod 600 ~/.minimax_key
# 数据（gitignored，约 292MB）：
curl -sL -o data/longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

bash run_theme.sh 6 strat                     # theme 完整方案：6 题分层（小样本验证）
bash run_official.sh ours                      # 对照：BM25 session 检索基线
bash run_official.sh no-mem                    # 下界：裸问题
```

配置说明、错因归因（`*.trace.jsonl` / `*.themes.jsonl`）、mem0 横向对比，均见
[`eval/longmemeval/README.md`](eval/longmemeval/README.md)。

> 为什么选 LongMemEval 而非 LOCOMO：它有逐题证据标注、可控的干扰项 haystack（让选择性召回真正
> 起作用），以及知识更新 / 弃答类别——恰好对应到我们的检索指标与可靠性风险点。

## 目录结构

```
theme_memory/
  SKILL.md                         记忆 skill 定义（拷进目标项目的 .claude/skills/ 用）
  scripts/
    store.py                       存储：按天日志 / wiki / 根 wiki / floor / 主题归并
    retrieve.py                    纯 Python BM25 召回（中英文）
    topic_state.py                 每轮话题状态：模型路由（BM25 兜底）+ 粘性 + fail-safe
    cli.py                         overview / retrieve / append / wiki(局部增删改) / merge / summarize
hooks/inject_topic_state.py        UserPromptSubmit 话题状态注入 hook
.claude/settings.json              本仓 hook 注册
eval/run_recall.py                 recall@k 验证
eval/run_routing.py                路由 + fail-safe 验证
eval/longmemeval/
  setup_official.sh                clone 官方仓库（pinned）+ 打 MiniMax 补丁 + 建 venv
  theme_to_official.py             完整方案：ingest（路由+局部更新 wiki+根 wiki+归并）+ 级联召回
  bm25_to_official.py              BM25 基线 → 官方 retrieval_results + 错因归因 trace
  run_theme.sh / run_official.sh   theme 完整方案 / 对照配置 一键跑
  official_patches.diff            对官方脚本的最小 MiniMax 适配
  mem0_shim.py / run_mem0.sh       mem0 官方 eval（MiniMax）的本地 REST 桥 + 跑法
  README.md                        评测跑法文档（含 mem0 横向对比）
```
