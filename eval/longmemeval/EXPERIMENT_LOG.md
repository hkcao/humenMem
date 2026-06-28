# 实验记录（test process log）

随手追加：每次跑测的配置、qid 集、过程数据、耗时、结果、结论。最新在上。

---

## 2026-06-28 — DeepSeek-V4-Pro 3 题重测（换模型 + 验证改进）

**背景**：MiniMax Token Plan 额度耗尽（429/2056），换 DeepSeek-V4-Pro（key `~/.deepseek_key`）。集成：
`run_theme_parallel.sh` 加 `MODEL_NAME/MODEL_ALIAS/API_BASE/KEY_FILE` env（默认仍 MiniMax）；
`run_generation.py` deepseek 走 tiktoken；`evaluate_qa.py` model_zoo 加 `deepseek-v4-pro`+is_reasoning。
patch 重生成 APPLIES CLEAN。DeepSeek 推理在独立 `reasoning_content`，content 干净（无 `<think>`）。

**结果（最新代码：按天 batch + 双向链接 + 空答案重试）**：DeepSeek-theme 1/3

| qid | type | bm25 | MiniMax-theme | DeepSeek-theme | tier |
|---|---|---|---|---|---|
| 8a2466db | preference | ❌ | ✅ | ✅ | topic_raw |
| 0a995998 | multi-session | ❌ | ❌ | ❌ | wiki_full |
| 6d550036 | multi-session | ❌ | ❌(空答案) | ❌(答"4",非空) | full_raw |

**关键验证**：
- ✅ **空答案已解决**：6d550036 MiniMax 是空字符串(`<think>`吃光预算)，DeepSeek 答 1003 字符真实答案
  （tier=full_raw，floor 兜底+检索到证据）。DeepSeek 格式天然规避 + 空答案重试双保险。
- ✅ **双向链接在真实 ingest 中生成**（每题 5-8 个主题有 `↔` 反向链接）；ingest 0 失败、0 额度错误。
- ⚠️ **两个 multi-session 仍错，坐实是计数口径**：0a995998 答"1"（"干洗店是服务非零售店不计"，gold=3）；
  6d550036 答"4"（把"领导团队/带5人/在领新功能"都算，gold=2，"领导 vs 参与"精度）。三方(bm25/MiniMax/
  DeepSeek)同错，证据都召回到了，是 reader 计数约定与 gold 不一致（gold 口径本身偏特殊/有歧义），非检索/非空答案。

**结论**：改进项(空答案重试/按天batch/双向链接)机制均生效；剩余失败是计数歧义类硬题，换模型也没翻盘，
需要的是答题口径对齐（如 prompt 明确"干洗店算店""只数本人主导"）而非记忆/检索改动。

---

## 2026-06-28 — 改进：双向 wikilinks + 空答案重试

**① 双向 wikilinks（`store.py`）**：wiki 行可用 `[[topic]]` 引用相关主题；`sync_backlinks(topic)`
在写入后扫描 `[[X]]`，给 X 的 wiki 追加反向链接 `↔ [[topic]]`（仅真实其他主题、幂等、additive）。
`update_topic_wiki` 写入后自动调用。eval `_refresh_wiki` 把"现有其他主题名"喂进 prompt（`{links}` 槽位），
模型按需加前向链接；ingest 写完调 `sync_backlinks`。e2e 验证：前向→反向自动生成、re-sync 不重复、
ghost 主题安全跳过。CLI/SKILL.md 已补文档。

**② 空答案重试（`run_generation.py` + `official_patches.diff`）**：reasoning 模型可能把 `max_tokens`
全花在 `<think>` 里，剥完为空被判错（=6d550036 的 theme 空答案）。修法：剥空则 `max_tokens` 倍增重试
最多 2 次（2x→4x）。补丁已用 `git diff` 重生成并 `git apply --check` 通过（setup 重建后存活）。

**③ 关于 0a995998 为何 bm25 检索全中仍答错**（见下方归因）：gold=3 = 干洗西装(取)+Zara 新靴(取)+
Zara 旧靴(退)；reader 用窄口径（排除干洗店、换货当 1 件）数成 1。是计数口径/聚合问题，非检索。

**待验证**：重跑 n12，看 ① 6d550036 空答案是否消失 ② 双向链接是否帮到跨主题聚合（如 0a995998 的
靴子/西装散落多主题）③ 准确率不回退。

---

## 2026-06-28 — n12 失败用例归因（错因分析）

**8a2466db（bm25 ❌ / theme ✅，两边 full_raw）—— 不是检索差异，是 wiki 蒸馏层**
- 两边都把证据 session `answer_edb03329` 召回在 rank 1（检索相同）。
- theme 的 full_raw **恒定叠加蒸馏 wiki**（`recall()` 每 tier 传 `wiki=full_wiki`）：rank-1 item 文本以
  `[MEMORY WIKI …] # GLOBAL` 开头，含"对 Adobe Premiere Pro 感兴趣"的蒸馏偏好。
- bm25 reader 只看原文 → 通用清单（没遵循偏好）→ ❌；theme reader 多了一句蒸馏偏好 → Premiere 定制 → ✅。
- 结论：`full_raw` 只描述原文升级到哪级，wiki 层始终在视野；这句蒸馏偏好是翻盘点。

**0a995998（gold=3，两边 ❌）—— 模糊计数口径，非检索**
- bm25 检索完美（3/3 证据在 top-3），仍答"1 件"；theme 停在 `wiki_full`（探针 YES），0/3 证据原文未召回，
  靠蒸馏 wiki 也答"1 件"。两边都排除干洗店西装+借出毛衣 → 1；gold=3 全算。根因=口径分歧。
- ⚠️ **潜在风险**：充分性探针会在 wiki tier 提前截断级联 → 该题 theme 检索弱于 bm25（0/3 vs 3/3）。
  floor"永不弱于 bm25"仅在升到 full_raw 时成立；探针提前 YES 就不兜底。本题同错，分数未回退。

**6d550036（gold=2，两边 ❌）—— 多跳聚合 + 生成失败**
- bm25：4/4 证据在池但仅 1/4 进 top-10，答"6 个"（把"参与"当"领导"多算）。
- theme：0/4 证据召回（topic_raw 选错主题）+ **hypothesis 空字符串**（MiniMax 只吐 `<think>` 剥成空）。

**可落地结论**：① 探针提前截断让 floor 兜底失效（计数类尤危）→ 计数问题强制不在 wiki tier 截断 / floor 始终进池；
② 空答案应重试；③ 计数/多跳是 bm25+theme 共同短板，wiki 蒸馏帮不上计数。

---

## 2026-06-26 — 优化：根 wiki 改按天 batch

**改动**：`theme_to_official.py` `ingest()` 把根 wiki 从"每 session 刷一次"改成"**按天 batch**"——
同一天的 session 全处理完、跨到下一天（或全部结束）时，用当天累积的跨主题事实做一次
`_refresh_root_wiki`（`flush_day()` 闭包，按 `store._day_key(date)` 分组）。模拟"每天结束时归整"。

**预期收益**：根 wiki 调用从 `#session` 降到 `#distinct_day`。n12 样本里单题 ~50 session 通常跨
十几天 → 根 wiki 调用降 ~3-4×，对应砍掉 ingest ~40% 里的大部分。主题 wiki 仍每 session 刷（未动）。

**待验证**：重跑 n12，确认 ① 准确率不回退（仍 ≥10/12、theme⊇bm25）② ingest 墙钟下降。


## 2026-06-21 — theme-memory n12_strat 端到端（当前 HEAD 7298661）

**目的**：用当前代码（局部更新 wiki + 模型决定 BM25/全量 + 根 wiki + 主题归并）在 12 题分层样本上跑 theme 方案，和 bm25 基线对比。

**样本**：n12 分层确定性抽样（每题型轮取），qids：
```
e47becba(ssu)  7161e7e2(ssa)  8a2466db(ssp)  0a995998(ms)
gpt4_59149c77(tr)  6a1eabeb(ku)  118b2229(ssu)  c4f10528(ssa)
06878be2(ssp)  6d550036(ms)  gpt4_f49edff3(tr)  6aeb4375(ku)
```
含两个历史修复用例 e47becba、0a995998。

**bm25 基线（对照，取自全量 retr_ours_n0 评测，同 12 qid）**：9/12 ✅
- ❌ 错：`8a2466db`(preference)、`0a995998`(multi-session)、`6d550036`(multi-session)
- ✅ 对：其余 9 题

**耗时分布**（探针 /tmp/profile_steps.py，采样 qid=8a2466db 前 4/50 session + 1 次完整 recall）：
- ingest 占 ~98%，recall ~0.5 min（很快）
- 全 50 session 外推 ingest ≈ **32 min/题**：
  | 步骤 | 单次 | 频次 | 占比 |
  |---|---|---|---|
  | `_refresh_wiki` | 15.5s | 每命中主题/session | 40% |
  | `_refresh_root_wiki` | 15.2s | 每 session | 40% |
  | `_route` | 7.7s | 每 session | 20% |
- 根因：两类 wiki 局部更新（reasoning 模型生成增删改 ops，输出长）= 80% ingest 时间
- 提速点：① 根 wiki 改批量/末尾一次（省~40%，低风险）② 主题 wiki 批量化
- 注：采样题恰 1 主题/session（温和情形）；碎片化题 `_refresh_wiki` 会 2-4×、更久

**执行方式**：题级独立 → 12 路并行 ingest（每题一进程，独立 store + 输出分片，最后合并跑一次 gen+judge）。脚本 `run_theme_parallel.sh`。墙钟从串行 ~8h 压到 ≈ 最慢单题（30-40min）。`_chat` 有指数退避，12 路并发被限流自动重试。

**执行实测**：12 路并行墙钟约 40min（含末尾单次 gen+judge）。无进程失败。各题 ingest 后 28-46 主题。

**结果：theme 10/12 vs bm25 9/12（+1，零回退）**

| qid | type | bm25 | theme | Δ | recall tier | topics |
|---|---|---|---|---|---|---|
| e47becba | ssu | ✅ | ✅ | | wiki_bm25 | 39 |
| 7161e7e2 | ssa | ✅ | ✅ | | topic_raw | 45 |
| 8a2466db | ssp | ❌ | ✅ | **+** | full_raw | 36 |
| 0a995998 | ms | ❌ | ❌ | | wiki_full | 36 |
| gpt4_59149c77 | tr | ✅ | ✅ | | wiki_bm25 | 38 |
| 6a1eabeb | ku | ✅ | ✅ | | wiki_bm25 | 36 |
| 118b2229 | ssu | ✅ | ✅ | | wiki_bm25 | 28 |
| c4f10528 | ssa | ✅ | ✅ | | topic_raw | 39 |
| 06878be2 | ssp | ✅ | ✅ | | wiki_full | 46 |
| 6d550036 | ms | ❌ | ❌ | | topic_raw | 43 |
| gpt4_f49edff3 | tr | ✅ | ✅ | | full_raw | 46 |
| 6aeb4375 | ku | ✅ | ✅ | | wiki_full | 40 |

**结论**：
- `theme ⊇ bm25` 在本样本成立——每道 bm25 答对的题 theme 都答对，**零回退**。
- theme 净 +1：修复 8a2466db（preference，走 full_raw floor）。
- 剩 2 个失败 `0a995998`/`6d550036` 都是 multi-session 多跳聚合，bm25/theme 同失（0a995998 是已知的"模糊计数"硬题）。
- tier 分布健康：wiki_bm25 / wiki_full / topic_raw / full_raw 四级都被用到，说明"模型决定 BM25/全量 + 级联探针"在真实运行、各司其职。
- 样本小（12 题），multi-session 短板与全量基线一致；结论需全量 500 验证。
