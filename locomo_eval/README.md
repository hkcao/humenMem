# LOCOMO 评测:主题化外部记忆 (DESIGN.md) 的效果验证

用 [LOCOMO](https://github.com/snap-research/locomo) 长对话 QA 基准,验证根目录 `../DESIGN.md`
提出的**主题化记忆 + 主题驱逐**方案是否真的有效,并和两个 baseline 对比
**准确率 vs token 消耗**两个维度。

> 结论速览:加上 BM25 精确检索后,主题化方案在同一批问题上**超过扁平 BM25、达到全量上下文准确率的 ~77%**;
> 而"切主题即驱逐"几乎不损准确率(evict ≈ 无状态版),验证了 DESIGN.md 的核心机制。
> 详细分析见 [`FINDINGS.md`](FINDINGS.md)。

---

## 为什么要约束窗口预算

LOCOMO 单条对话只有 12K–24K token,整条能塞进现代上下文窗口,**不约束就不会触发驱逐**。
所以除 full-context 外,所有方案都在同一个 `--budget`(默认 4000 token)工作窗口下运行,
强制"只装少量内容",这才真正测到设计的检索/驱逐机制。

## 评测的方案

### 主题化方案(本设计,共三个变体)

记忆构建方式相同(见下"记忆怎么建"),区别在**查询时怎么用**:

| 方案 | 检索逻辑 |
|---|---|
| **`theme-mem-sf`** | 摘要优先:阶段1 加载 core + **全部主题摘要**判断能否作答;不能则模型点名相关主题,阶段2 用 **BM25** 在这些主题内精确检索明细后重答。无状态。 |
| **`theme-mem-evict`** | 在 sf 基础上**有状态、按顺序**执行:窗口 = core + 全摘要 + **当前主题明细**;一旦问题切到别的主题就**驱逐**旧主题明细(DESIGN.md §4)。窗口不随主题数膨胀。 |
| **`theme-mem-accum`** | 对照组:同样有状态,但**不驱逐**,累积所有访问过的主题明细。用来量化驱逐省下的 token。 |

> 早期还有一个 `theme-mem`(top-k 盲路由)版本,已被证明是瓶颈(路由命中率低),被 `-sf` 取代。

### 基线

| 方案 | 说明 |
|---|---|
| `full-context` | 整条对话塞进窗口(准确率上界,token 最差) |
| `bm25-rag` | 同预算下扁平 BM25 轮次检索,不分主题(对照"主题分区是否有增益") |

## 记忆怎么建(ingest)

逐 session 处理,每个 session 两次 LLM 调用:
1. **主题分段** `_segment`:把 session 切成连续话题段,**完全由模型分类**(不预置主题),
   复用已有主题或新建。
2. **core mem 蒸馏** `_update_core`:抽取跨主题的稳定事实,合并进 ≤800 token 的 core。

结果落成 DESIGN.md 的磁盘布局(`index.md` + `<主题>/<日期>.md` + `summary.md`),
并 pickle 落盘 `store.pkl`(可断点续建)。

## 问题类别(LOCOMO)

1=多跳, 2=时间, 3=开放域, 4=单跳, 5=对抗。
**对抗类**的 gold 字段是 `adversarial_answer`(诱饵),正确行为是**拒答**("No information available")。

---

## 最终对比结果(conv-26,同一批 50 题,LLM-judge)

| 方案 | 总分 | 单跳 | 多跳 | 时间 | 开放域 | 对抗 | 平均上下文 token |
|---|---|---|---|---|---|---|---|
| full-context | **0.70** | 0.90 | 0.70 | 0.30 | 0.60 | 1.00 | 18,535 |
| **theme-mem-sf** | **0.54** | 0.70 | 0.20 | 0.30 | 0.60 | 0.90 | 3,966 |
| theme-mem-evict | 0.52 | 0.60 | 0.20 | 0.30 | 0.60 | 0.90 | 3,904 |
| theme-mem-accum | 0.48 | 0.50 | 0.40 | 0.00 | 0.70 | 0.80 | 4,288 |
| bm25-rag | 0.40 | 0.30 | 0.10 | 0.40 | 0.40 | 0.80 | 3,978 |

## 结论

1. **BM25 精确检索是关键**:把主题内检索从"关键词集合重叠"换成 BM25 后,单跳 0.20→**0.70**,
   总分到 0.54——主题化方案**超过扁平 bm25、达到 full-context 的 ~77%**,上下文只用其 ~1/5。
2. **"切主题即驱逐"被验证有效**:`evict`(0.52)≈ 无状态 `sf`(0.54),把窗口压到单主题**几乎不损准确率**;
   且 evict 比不驱逐的 `accum` **又准又省**(0.52>0.48,峰值 ctx 4944<6030)——累积会从无关主题
   捞进干扰轮次,净负。**bounded window 是对的。**
3. **代价**:完全由模型分类(不预置)会让主题碎片化(本例 17→40),always-on 摘要层变大,
   两阶段两次调用使总 query token 反而高于 bm25。下一步用 DESIGN.md §8 的主题合并收敛碎主题。
4. **仍弱**:多跳(跨主题本性)、时间(精确日期上 bm25 扁平检索仍略强)。

---

## 复现

```bash
# 1. 装依赖(仓库根)
python3 -m venv .venv && .venv/bin/pip install -r locomo_eval/requirements.txt

# 2. 配置 DeepSeek(OpenAI 兼容),写入仓库根 .env(已 gitignore,勿提交)
#    DEEPSEEK_API_KEY=...
#    DEEPSEEK_BASE_URL=https://api.deepseek.com
#    DEEPSEEK_MODEL=deepseek-v4-pro

# 3. 下数据集
mkdir -p locomo_eval/data && curl -fsSL \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o locomo_eval/data/locomo10.json

# 4. 建记忆(慢、可断点续建,被杀就重跑)
cd locomo_eval && ../.venv/bin/python -m eval.build --samples 0

# 5. 跑评测 + 出表
../.venv/bin/python -m eval.run --samples 0 --max-questions 50 --workers 10 \
    --schemes theme-mem-sf,theme-mem-evict,theme-mem-accum,bm25-rag,full-context
../.venv/bin/python -m eval.report results/summary.json
```

注:`deepseek-v4-pro` 是推理模型,completion_tokens 含 reasoning_tokens(已分别记账)。
LOCOMO 是冷启动 QA,而 DESIGN.md 的主题路由本是为对话连续性设计的——这个测法偏压设计的弱项,
结论需如此理解。
