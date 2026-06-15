# LongMemEval —— theme-memory 的 Phase B（端到端 QA）

在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 基准上评测 theme-memory 的
**检索/召回引擎**，按 **LongMemEval 官方方式**评判（逐题型 / 弃答的 verbatim 评判
prompt），用 **MiniMax-M3** 同时作为答题与评判模型。

## 为什么选 LongMemEval（而非 LOCOMO）

LongMemEval 在设计时就针对性地修复了 LOCOMO 在记忆评测上的弱点：它有逐题的**证据
标注**、一个**可控的干扰项 haystack**（让选择性召回真正起作用），以及专门的**知识
更新**和**弃答**类别——这些直接对应到我们的检索指标和我们最大的可靠性风险点。

## 它衡量什么（以及不衡量什么）

- **衡量：** 我们的 BM25 召回（`theme_memory/retrieve.py`）能否把回答跨会话问题所需
  的正确会话捞出来，以及这在官方评判下如何转化为 QA 准确率。这就是**第 1 步（检索与
  召回）**的价值。
- **不衡量：** 第 2 步的会话内话题切换干扰（这些是跨会话 QA 任务，不是单窗口内的话题
  切换）。那部分由 `eval/run_routing.py` 覆盖。

## 忠实复现说明

- **评判 prompt**（`harness.py`）：逐字取自 LongMemEval 的 `get_anscheck_prompt`，
  含逐题型变体——标准 / 时间推理（容忍 off-by-one）/ 知识更新 / 偏好（rubric）——
  弃答则由 `question_id` 里的 `_abs` 选中。标签 = `"yes" in response.lower()`
  （在剥掉 MiniMax-M3 的 `<think>` 块之后）。
- **答题 prompt**（`harness.py`）：逐字取自非 CoT 的"direct"阅读模板，采用
  `### Session i / Session Date / Session Content` 的历史块格式。
- **检索：** 会话粒度；每个会话文档 = 它的**用户**轮次（匹配官方的扁平索引）。
  按**我们的** BM25（`theme_memory.retrieve.bm25`）排序；会话 id（其中编码了
  `answer`/`noans`）被排除在打分文本之外，以免泄露 ground truth。
- **指标**（`run_phaseb.py`）：**题型平均**（逐题型准确率的均值）、**整体**（非弃答
  上的 micro）、**弃答准确率**——与 `print_qa_metrics.py` 一致。

## 复现

```bash
# 1. 数据（已 gitignore；约 292MB）—— 首次运行自动下载，或手动：
#    curl -sL -o data/longmemeval_oracle.json      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
#    curl -sL -o data/longmemeval_s_cleaned.json   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 2. API key（保存在仓库之外）
echo "sk-cp-..." > ~/.minimax_key && chmod 600 ~/.minimax_key   # MiniMax CN，模型 MiniMax-M3

# 3. 快速冒烟测试（12 道题，按题型均衡抽样）
python3 run_phaseb.py all --config bm25 --topk 10 --limit 12 --stratified

# 4. 完整基准（500 道题）
python3 run_phaseb.py all --config bm25 --topk 10 --workers 8

# config 取值：bm25（我们的检索器）| oracle（完美检索的上界）| no-mem（基线/下界，不带记忆）
```

输出落在 `results/`：`hyp_<tag>.jsonl`（答案）、`eval_<tag>.jsonl`（评判结果），
以及打印出来的指标。

## 结果

完整 500/500 运行（bm25，topk=10，MiniMax-M3 答题 + 评判）：

| 指标 | 分数 |
|---|---|
| 整体准确率（micro，非弃答，n=470） | **0.834** |
| 题型平均准确率（6 种题型的均值） | **0.817** |
| 弃答准确率（n=30） | **0.833** |

单会话召回接近天花板（0.98 / 0.93）；差距在多会话（0.744，需要所有证据会话都进 top-k）
和偏好（0.533，词法检索捞不到 persona 会话）。逐题型表格、分析，以及**无记忆基线
（no-mem）**的对比见 `RESULTS.md`。
