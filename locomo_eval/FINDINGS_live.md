# Live-mode + 模型切换实验 (MiniMax-M2.7)

本次实验做了两件事：
1. 在新增的 `eval.live`（DialSim 风格 turn-by-turn 注入问题）模式下跑各方案；
2. 把查询/构建模型从 **DeepSeek-v4-pro** 换成 **MiniMax-M2.7**，看主题化方案是否 model-agnostic。

> ⚠️ **重要对照说明**：下文表格里同时含两个变量（模型 + 评测模式），任何"涨/跌"都不能归到单一因素：
> - **DeepSeek 列**：来自原 `FINDINGS.md`（cold-start，DeepSeek-v4-pro），原始记忆库用 DeepSeek 建。
> - **MiniMax cold-start 列**：本次跑，**记忆库已用 MiniMax-M2.7 重建**（`eval.build` + `eval.build --mem0`），查询也用 MiniMax-M2.7。
> - **MiniMax live 列**：本次跑，记忆库由 agent 在 timeline 中**边走边建**（每个 turn 喂进去），全程 MiniMax-M2.7。

数据集都是 `conv-26` 前 50 题，budget 4000，judge 用 LLM-judge（同模型）。

---

## 主表：准确率 (overall_acc)

| Scheme | DeepSeek-v4-pro<br>cold-start | MiniMax-M2.7<br>cold-start | MiniMax-M2.7<br>live (DialSim) |
|---|---:|---:|---:|
| full-context | 0.70 | **0.44** | **0.66** |
| bm25-rag | 0.40 | 0.34 | 0.32 |
| theme-mem-sf | **0.54** | 0.24 | 0.10 |
| theme-mem-evict | 0.52 | 0.16 | 0.10 |
| theme-mem-accum | 0.48 | 0.22 | 0.06 |
| mem0 | 0.38 | 0.06 | 0.00 |

（DeepSeek 列的 50 题与 MiniMax 列的 50 题是同一组前 50 题；类别构成在 MiniMax 子表里只剩 multi-hop / temporal / open-domain — 这 50 题里没出现 single-hop 和 adversarial。）

## 平均上下文 token

| Scheme | DeepSeek cold | MiniMax cold | MiniMax live |
|---|---:|---:|---:|
| full-context | 18535 | 18535 | 5887 |
| bm25-rag | 3978 | 3969 | 1128 |
| theme-mem-sf | 3966 | 4109 | 2283 |
| theme-mem-evict | 3904 | 4086 | 2336 |
| theme-mem-accum | 4288 | 4342 | 2359 |
| mem0 | 871 | 938 | 0 |

---

## 观察

### 1. **full-context 在 MiniMax 上 live > cold（0.66 > 0.44）** —— 不是"live 模式更好"，是 live 自然把上下文裁短了

Live 模式按 evidence 的最末位置注入问题（`build_timeline._latest`），所以问题被问到时**只看见 evidence 之前的对话**，平均 5887 token；cold-start 永远塞全文 18535 token。
**说明 MiniMax-M2.7 在 ~18K 长上下文上对细节检索能力下降**（DeepSeek 同样的 18K 拿到 0.70 没问题）；裁到 ~6K 反而更准。

### 2. **主题化方案全线塌方（sf/evict/accum）**

| | DeepSeek cold | MiniMax cold | MiniMax live |
|---|---:|---:|---:|
| theme-mem-sf | 0.54 | 0.24 | 0.10 |
| theme-mem-evict | 0.52 | 0.16 | 0.10 |
| theme-mem-accum | 0.48 | 0.22 | 0.06 |

两个变量同时变：
- **模型换 MiniMax** 已经把 cold-start 从 0.54 拉到 0.24 —— 主题分段 (`_segment`)、core mem 蒸馏 (`_update_core`)、Stage-1 路由全部依赖结构化 JSON 输出 + 语义判别。MiniMax-M2.7 是 thinking 模型，剥掉 `<think>` 后实际答题部分对这类带摘要 + 列名 + ranking 的多步任务表现明显弱于 DeepSeek。
- **再叠加 live 模式** 跌到 0.10。两个加重因素：
  - **建库时机不对称**：cold-start 是离线建库（看完整 session），live 是边走边建（每个 session 结束才能 `_segment`/`_update_core`）；任何前几个 session 主题分得不好，后续 session 就在错误的主题集合里继续追加。
  - **路由 stage-1 必须在所有摘要里点名相关主题**，摘要数量从 5 涨到 14，MiniMax-M2.7 的命中率比 DeepSeek 差更多。

排除"模型 + 模式"哪个贡献更大，需要再做一次 *DeepSeek-v4-pro live*；本次没做（DeepSeek key 已被替换）。

### 3. **bm25-rag 在三组里最稳**（0.40 / 0.34 / 0.32）

纯关键词检索，不依赖 LLM 建库，也不依赖 LLM 在大候选集里"列名"。模型/模式切换只会从答题质量这单一维度影响它。**这进一步佐证主题化方案的塌方是因为依赖了 MiniMax 弱的环节（结构化路由 + 主题摘要），不是数据本身变难。**

### 4. **mem0 在 MiniMax 上崩：cold-start 0.06，live 直接归零**

mem0 的事实抽取（`Memory.add`）完全靠 LLM。

- **Cold-start (0.06)**：用整 session 文本抽取，MiniMax 抽出来的事实质量大幅下降（ingest token 数 182K 跟 DeepSeek 时接近 186K，说明抽取确实跑了，但内容差）。语义检索的事实越粗略，answer 就越没料。
- **Live (0.00, avg_ctx=0)**：逐 turn 喂单行 `speaker: text`，mem0 几乎抽不出任何事实存到向量库；查询时 `search()` 返回空，所有问题答 "No information available"。日志里也有 mem0 内部 JSON 解析失败（MiniMax 的 thinking 模型对 mem0 期待的 strict-JSON 格式不稳定）。`mem0-live` 在这种模型 + 增量 ingest 组合下基本不可用。

### 5. **live 模式的设计被这次实验意外验证了一个点**

`full-context-live` 的 avg_ctx 比 cold-start 少 3 倍多（5887 vs 18535），但准确率反而更高（0.66 vs 0.44）。说明 LOCOMO 的题目大多数确实在 evidence 之前就能定位，**未来真实场景的"在线问答"应该靠 evidence-时刻的局部上下文，不是整条历史**。这也是 DialSim 协议的核心动机。

---

## 结论

1. **主题化方案的 DESIGN.md 对模型有强假设**：依赖一个"能按列出的主题摘要做语义路由 + 输出结构化 JSON"的 LLM。DeepSeek-v4-pro 满足，MiniMax-M2.7 显著弱。本设计不是 model-agnostic。
2. **Bounded window 的"切主题即驱逐"假设也需要重新检验**：在 MiniMax cold-start 下 evict (0.16) 已经显著弱于 accum (0.22)，反过来了。原 DeepSeek 实验里"驱逐不损失"的结论换了模型就翻盘。
3. **bm25 是更稳健的基线**。如果上层应用要 model-agnostic，bm25 才是出发点；主题化层只有在 LLM 足够强时才贡献。
4. **要换模型再跑前请同时重建记忆库**：否则 ingest 时模型 A 切出的主题边界会拖累查询时模型 B 的路由（本次实验就是这么处理的，但被替换前如果不重建，结论会更黑）。

## 数据出处

- `results_live/summary.json` —— MiniMax live 五方案（mem0-live 在 `results_live_mem0/`）。
- `results_cold_minimax/summary.json` —— MiniMax cold-start 六方案。
- `FINDINGS.md` + `README.md`（旧的 DeepSeek 结果）—— 现保留用作对照基线。
- `memory_runtime/` —— **当前是用 MiniMax-M2.7 重建的版本**。原 DeepSeek 版本备份在 `memory_runtime_deepseek/`。
