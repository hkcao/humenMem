# LongMemEval —— 官方 harness 跑法（MiniMax-M3）

这条流程**全程复用官方代码**（`vendor/LongMemEval/src/generation/run_generation.py` +
`src/evaluation/evaluate_qa.py`），我们自己的 theme-memory BM25 按官方
`retrieval_results` 契约插进去。这样数字才能和 LongMemEval 榜单、以及 mem0 横向对比。
（旧的 `harness.py` / `run_phaseb.py` 是手写复刻版，已被此流程取代，保留仅作参考。）

## 一次性准备

```bash
bash setup_official.sh        # clone 官方仓库到 vendor/（pinned commit）+ 打 MiniMax 补丁 + 装依赖
echo "sk-..." > ~/.minimax_key && chmod 600 ~/.minimax_key
# 数据（gitignored）：首次需下载到 data/
#   curl -sL -o data/longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
#   curl -sL -o data/longmemeval_oracle.json    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
```

## 跑

```bash
bash run_official.sh CONFIG [LIMIT] [STRATIFIED]
#   CONFIG     = ours | no-mem | oracle | official-bm25
#   LIMIT      = 题数（0/留空 = 全 500）
#   STRATIFIED = "strat" 按 6 种题型均衡抽样

bash run_official.sh ours 12 strat     # 冒烟：我们的 BM25，12 题分层
bash run_official.sh ours              # 完整 500：我们的 BM25（headline）
bash run_official.sh no-mem            # 基线下界：官方 no-retrieval 裸问
bash run_official.sh oracle            # 上界：evidence-only haystack
```

四个 config：
- **ours** —— 我们的 BM25 在真实 ~115k-token haystack 上取 top-10 session（`flat-session`）。
- **no-mem** —— 官方 `no-retrieval`：prompt 就是**裸问题**，无任何历史、无 "no relevant
  history" 框架（这就是修正后的第 1 点；官方本就这么跑闭卷）。
- **oracle** —— 喂 evidence-only 的 `longmemeval_oracle.json`，完美检索上界。
- **official-bm25** —— 官方 `rank_bm25.BM25Okapi` 基线（需 `pip install rank_bm25`）。

读题/评判都用 **MiniMax-M3**（OpenAI 兼容端点）；prompt、judge、指标全部官方原样。
flags 按官方推荐：`history_format=json`、`useronly=false`、`direct`（非 CoT）。

## 我们对官方代码做的最小适配（见 `official_patches.diff`）

MiniMax-M3 是 reasoning 模型，需要两处模型适配（判定逻辑/指标不变）：
1. `run_generation.py` —— `model2maxlength` 对未列出的模型兜底 128k；MiniMax 走 tiktoken
   做长度预算（避免 HF/torch）；transformers 改惰性导入；hypothesis 剥 `<think>`。
2. `evaluate_qa.py` —— `model_zoo` 加 `minimax-m3`；judge 对 reasoning 模型放开 token 预算
   （官方默认 10 太小，半截 think 出不来 yes/no）并剥 `<think>` 后再判 `'yes'`。

## 错因归因日志（第 2 点）

`bm25_to_official.py` 除了产出官方检索日志，还为每题落一份 `*.trace.jsonl`：query、候选
session 数、每个证据 session 的**排名**、recall_any/all@k，以及 `failure_stage_if_wrong`：
`retrieval_ok`（证据全进 top-10，答错=生成的锅）/ `retrieval_partial` / `retrieval_miss`
（证据没进 top-10，答错=检索的锅）/ `no_evidence_in_corpus` / `abstention`。答错时据此一眼
区分**检索没捞到** vs **生成没用好**。
```bash
# 只看检索质量（不花 API）：
python3 bm25_to_official.py --out official_out/retr.jsonl --retriever ours --limit 60 --stratified
```
