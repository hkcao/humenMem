# LongMemEval 评测

在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)_S（500 题，每题 ~115k-token
haystack）上端到端 QA，**全程复用官方 harness**：检索日志喂给官方 `run_generation.py`
读题、官方 `evaluate_qa.py` 逐题型/弃答评判。我们的 theme-memory BM25 按官方
`retrieval_results` 契约插进去；读题与评判都用 **MiniMax-M3**（OpenAI 兼容端点）。这样
数字才能和 LongMemEval 榜单、以及 mem0 等框架横向对比。

> 官方仓库（`vendor/LongMemEval`、`vendor/memory-benchmarks`）由 `setup_official.sh`
> 按 pinned commit clone 并打补丁，已 gitignore。

## 快速开始

```bash
bash setup_official.sh                    # clone 官方仓库(pinned) + 打 MiniMax 补丁 + 建 .venv 装依赖
echo "sk-..." > ~/.minimax_key && chmod 600 ~/.minimax_key
# 数据（gitignored，~292MB）
curl -sL -o data/longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
curl -sL -o data/longmemeval_oracle.json    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

bash run_official.sh ours 12 strat        # 冒烟：我们的 BM25，12 题分层
bash run_official.sh ours                  # 完整 500：我们的记忆方案（headline）
bash run_official.sh no-mem                 # 纯模型基线（下界）
```

四个 config（`bash run_official.sh CONFIG [LIMIT] [strat]`）：

| config | 含义 |
|---|---|
| **ours** | 我们的 BM25 在真实 ~115k haystack 上取 top-10 session（`flat-session`） |
| **no-mem** | 官方 `no-retrieval`：prompt 为**裸问题**，无任何历史框架——纯模型下界 |
| **oracle** | evidence-only haystack，完美检索上界 |
| **official-bm25** | 官方 `rank_bm25` 基线（需 `pip install rank_bm25`） |

产物在 `official_out/`：`gen/*`（hypothesis）、`*.eval-results-minimax-m3`（评判），
脚本末尾打印 整体/题型/弃答 准确率。

## 错因归因（trace）

`bm25_to_official.py` 除了产出官方检索日志，还为每题落一份 `*.trace.jsonl`：query、
候选 session 数、**每个证据 session 的排名**、recall_any/all@k，以及 `failure_stage`：

- `retrieval_ok` —— 证据全进 top-10（答错则是**生成**的锅）
- `retrieval_partial` / `retrieval_miss` —— 证据部分/全部没进 top-10（**检索**的锅）
- `no_evidence_in_corpus` / `abstention`

```bash
# 只看检索质量（不花 API）：
python3 bm25_to_official.py --out official_out/retr.jsonl --retriever ours --limit 60 --stratified
```

## 对官方代码的最小适配（`official_patches.diff`）

MiniMax-M3 是 reasoning 模型，需两处适配（**判定逻辑/指标不变**）：
1. `run_generation.py`：未列出的模型 `model2maxlength` 兜底 128k；MiniMax 走 tiktoken 做
   长度预算（避免 HF/torch）；transformers 惰性导入；hypothesis 剥 `<think>`。
2. `evaluate_qa.py`：`model_zoo` 加 `minimax-m3`；judge 对 reasoning 模型放开 token 预算
   （官方默认 10 太小）并剥 `<think>` 后再判 `'yes'`。

## mem0 横向对比（官方 harness + MiniMax）

用 **mem0 自己的官方 LongMemEval 评测**（[`memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks)，
即其 README 0.9+ 同一套），把 LLM（抽取+答题+评判）换成 MiniMax-M3、embedder 用本地
`all-MiniLM-L6-v2`、向量库用 chroma——看在我们模型下 mem0 能到多少。

mem0 官方 `run.py` 走 `Mem0Client(mode=oss)` 打一个 mem0 REST server；官方那个 server 带
postgres+auth，原生跑不现实，所以用薄桥 `mem0_shim.py` 暴露它需要的 3 个 endpoint
（`POST /memories`、`POST /search`、`DELETE /memories`），后端直接用**官方 `mem0.Memory`
库**——只桥接传输层，mem0 的 extract→store→search 逻辑不动。

```bash
# 单开 py3.13 venv 装 mem0 全家桶（torch/sentence-transformers/chromadb）
python3.13 -m venv ../../.venv-mem0
../../.venv-mem0/bin/pip install "mem0ai>=2" sentence-transformers chromadb fastapi uvicorn \
    openai aiohttp aiolimiter python-dotenv tqdm pydantic

bash run_mem0.sh start-shim                # 起本地 REST 桥（前台，保持开着）
bash run_mem0.sh run mm30 5 3              # 另开窗口：官方 benchmark，30 题分层，断点续跑
```

**口径**：mem0 跑它自己的官方 harness（自己的答题/评判 prompt），我们的 BM25 跑
LongMemEval 官方 src——两者同模型(MiniMax)、同题库、同准确率指标，是"各自官方 harness、
同模型"的系统级横向对比（mem0 的价值本在它整套 pipeline，硬塞成"只返回 top-k session"
反而失真）。

**注意**：mem0 要 ingest 每题完整 haystack（含干扰项，单题 225-240 个 pair），每个 pair
一次 MiniMax 抽取调用——30 题数千次 reasoning 调用、**数小时**起步（`run.py` 逐 pair 断点
可续）。OSS 版 `add` 不接受自定义 timestamp（cloud-only），记忆 `created_at`=注入时刻，但
`run.py` 本就按时间顺序注入 session，时序大体保留。

## 结果

完整 500 题结果（official harness，MiniMax-M3 读题+评判）见 `official_out/`；汇总表待跑完补入。
