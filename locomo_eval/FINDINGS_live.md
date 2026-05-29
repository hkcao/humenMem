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

| Scheme | DeepSeek-v4-pro<br>cold-start | MiniMax-M2.7<br>cold-start | MiniMax-M2.7<br>live (DialSim)<br>(pre-fix) | MiniMax-M2.7<br>live (DialSim)<br>**(post-fix)** |
|---|---:|---:|---:|---:|
| full-context | 0.70 | 0.44 | 0.66 | — |
| bm25-rag | 0.40 | 0.34 | 0.32 | — |
| theme-mem-sf | **0.54** | 0.24 | 0.10 | **0.32** |
| theme-mem-evict | 0.52 | 0.16 | 0.10 | **0.40** |
| theme-mem-accum | 0.48 | 0.22 | 0.06 | **0.48** |
| mem0 | 0.38 | 0.06 | 0.00 | — |

(post-fix) = `ThemeMemLive.answer` 把 in-progress 转录拼进 inner scheme 的 prompt（详见 §Q3 (a)）。`full-context-live`/`bm25-rag-live`/`mem0-live` 本来就把 in-progress turn 喂模型，不需要修，未重跑。

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

---

## 复盘：Q1/Q2/Q3 深入

### Q1：Judge 模型也跟着换了，是个未隔离变量

`eval/judge.py` 的 LLM-judge 用同一个 `LLM(scheme="judge")` 读同一组 env，所以三组对比里 judge 模型与 solver 同步切换。Judge prompt 比较宽（"prediction contains the gold fact = correct"），临界 case 受影响。要严格隔离，需要存好的 (q, gold, pred) 离线用固定 judge 重判一遍。

### Q2：`<think>` 块的去留

MiniMax-M2.7 把推理 `<think>...</think>` 内联到 content 里，**不像 DeepSeek-v4-pro 走 `completion_tokens_details.reasoning_tokens` 分离记账**（summary 里 reasoning 列大量为 0 就是这个原因）。

- **不剥**：`chat_json` 路径直接 fail（json.loads 解析 `<think>...</think>{...}` 报错），sf/evict 的 stage-1 路由、mem0 内部事实抽取、judge 全部走 `except` 静默 fallback —— 是当前实现下最有破坏性的失败模式。
- **剥**（现状）：JSON 模式恢复正常；但 `<think>` 仍然吃 `max_tokens` 额度（实测一个问题 thinking 段就占 140 tokens 里的一半多），长 prompt + 短 `max_tokens` 会把真正答案截掉。
- 推理过程本身**对题目准确率是正向**的（trace 见 §Q3）：MiniMax 用 think 正确推出 "yesterday + 2023-05-08 → 2023-05-07"。问题是只有 think 看得见正确数据时才有用。

### Q3：一道题打开看每一步——Caroline LGBTQ support group 何时去？

完整 trace 在 `scripts/trace_sf_caroline.py` + `scripts/trace_sf_caroline.json`。三个分层结论：

**(a) Live 模式下发现一个实现 bug，已修。**

`build_timeline` 按 evidence 的最末 dia_id 注入题目（这道题 evidence=`D1:3`，所以题目在 session_1 第 3 个 turn 之后立刻提问），但 `ThemeMemLive.ingest_turn` 只把 turn 缓存到 `_cur_buf`，**`MemoryStore.ingest_session()` 要 session_end 才跑**（分主题/蒸馏 core 是按 session 批做的 LLM 调用）。所以题目触发时 store 完全是空的。

更糟的是 `live_schemes.py` 旧实现把 `in_progress_tokens` 加进 `ctx_tokens` 做统计，但**没有把 in-progress 转录加到 inner scheme 的 prompt 里**：
```python
ip_text, ip_tok = self._in_progress_context()
r["ctx_tokens"] = r.get("ctx_tokens", 0) + ip_tok  # 算进去
# ip_text 从未喂给 LLM
```
对比同条件下：full-context-live 把每 turn 立刻 append 到 `_lines` 喂全文 → 答对；BM25Live 立刻入索引 → 答对；ThemeMemLive 缓存但不喂 → 答 "No information available."。这直接解释了 sf-live/evict-live/accum-live 的 temporal acc 全为 0.00（24 题 0 对）：只要 evidence 在当前 session 内、问题在当前 session 内被问，store 空 + buffer 不喂 = 必丢分。

修复（`live_schemes.py:ThemeMemLive.answer`）：monkey-patch `store.summaries_context` 把 in-progress 转录拼到末尾的 `## In-progress session (not yet committed to themes)` block，inner scheme 代码不动。

修后 smoke 验证：同样 3 个 turn 的 in-progress + 空主题，MiniMax-M2.7 stage-1 直接答 "Yesterday (7 May 2023, the day before the session on 8 May 2023)" —— 同模型、同数据，从全错变全对。**完整重跑见下表**（`results_live_fixed/`）。

**(b) Cold-start sf 的失败是另一种：summary 蒸馏吃掉了时间锚点。**

同一题在 cold-start sf 里 Stage 1 直接回答（不走 Stage 2 BM25 detail）："Caroline went to the LGBTQ support group on May 8, 2023" —— 错（应是 May 7）。看 core mem 和 theme summary：
```
- Attended an LGBTQ support group; found transgender stories particularly inspiring
[2023-05-08] Caroline catches up with Melanie and shares how attending an LGBTQ support group helped her feel accepted...
```
原文 `D1:3 Caroline: I went to a LGBTQ support group yesterday...` 中的 "yesterday" 在 `_segment` 出主题和 `_update_core` 蒸馏 core 两次 LLM 处理里都被丢掉，只剩下 session 的日期 2023-05-08。模型用了它能看见的唯一日期。

这**不是 MiniMax 推理错**——上面 Q2 的 trace 证明完整 context 下同模型可以正确算出 May 7。这是**蒸馏 lossy**。DESIGN.md 的 "core/summary/detail" 三层架构假设的"摘要保留关键事实"在 MiniMax-M2.7 的蒸馏 prompt 下没成立：时间状语/相对时间这类容易被认为是"上下文细节"的信息被丢了。

**(c) Theme 路由对 MiniMax 的依赖太强（cold-start 已弱、live 更弱）。**

trace 显示 stage-1 路由本身是 OK 的（`need_details: ["lgbtq support group"]` 选对了主题）；但跟其他题对比，14 个主题候选下 MiniMax 经常 fallback 到"打开所有主题"（见 "How long has Caroline had her current group of friends for?" 的 trace：`themes=[全部 14 个]`），等价于把 budget 摊到全部 theme detail 上做 BM25。DeepSeek 在原 README 里就报告了"主题碎片化 17→40 摘要变大"对路由命中率的代价，MiniMax 因为更弱的指令跟随把这件事放大。

### Bug 修复后的新现象

修完 `ThemeMemLive` 的 in-progress prompt bug 之后，原结论被大幅改写：

1. **三个主题化方案全部回到可用区间**：sf 0.10→0.32, evict 0.10→0.40, accum 0.06→0.48。所有方案都超过了对应的 cold-start MiniMax 版本（cold sf 0.24 / evict 0.16 / accum 0.22）—— **live 模式现在比 cold-start 准**，因为 in-progress 转录给了模型"原始 evidence"，绕过了 lossy 蒸馏带来的时间锚点丢失（见 §Q3 (b)）。
2. **排序翻转**：cold-start 是 sf(0.24) > accum(0.22) > evict(0.16)；live post-fix 是 **accum(0.48) > evict(0.40) > sf(0.32)**。流式 ingest 早期主题 detail 还少，"累积"的代价（DESIGN.md §4 担忧的"窗口膨胀"）尚未达到拐点，反而比"切就驱逐"多保留了有用上下文。
3. **"切主题即驱逐 ≈ 无状态"的结论不成立了**：原 README/FINDINGS 用 DeepSeek + cold-start 报告 evict (0.52) ≈ sf (0.54)，本次 MiniMax + live 显示 evict (0.40) > sf (0.32) 反过来——eviction 比无状态多带了"上一题剩下的主题 detail"作为隐含上下文，在 live 模式短上下文场景下是净正面。

### TODO 

- (可选) 用固定 judge re-judge 三组 pred，隔离 judge 变量。

---

## 数据出处

- `results_live/summary.json` —— MiniMax live 五方案（mem0-live 在 `results_live_mem0/`）。
- `results_cold_minimax/summary.json` —— MiniMax cold-start 六方案。
- `FINDINGS.md` + `README.md`（旧的 DeepSeek 结果）—— 现保留用作对照基线。
- `memory_runtime/` —— **当前是用 MiniMax-M2.7 重建的版本**。原 DeepSeek 版本备份在 `memory_runtime_deepseek/`。
