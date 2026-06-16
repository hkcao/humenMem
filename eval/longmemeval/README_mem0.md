# mem0 横向对比 —— 官方 harness + MiniMax-M3（原生跑，无 Docker）

把 **mem0 的官方 LongMemEval 评测**（`vendor/memory-benchmarks`，即 mem0 README 上 0.9+
数字的同一套 harness）跑起来，但把里面的 LLM 换成 **MiniMax-M3**，看在我们模型下 mem0
能到多少——和我们的 BM25（同样 MiniMax 答题/评判）做系统级对照。

## 架构

mem0 官方 harness 的 `run.py` 通过 `Mem0Client(mode="oss")` 打一个 mem0 OSS REST server
（`POST /memories`、`POST /search`、`DELETE /memories`）。官方那个 server 是带
postgres+auth 的重型 FastAPI，原生跑不现实。所以我们用一个**薄桥** `mem0_shim.py`：
暴露这 3 个 endpoint，后端直接用**官方 `mem0.Memory` 库**（真正被测的引擎），只桥接传输层。

- **LLM（抽取 + 答题 + 评判）** = MiniMax-M3（OpenAI 兼容端点）。
- **embedder** = 本地 `all-MiniLM-L6-v2`（sentence-transformers，~90MB，CPU；MiniMax 无
  OpenAI 兼容 embeddings，且 embedder 用谁不影响"mem0 在 MiniMax 下"的主旨）。
- **vector store** = chroma（嵌入式、落盘，无需 server）。

## 跑

```bash
# 0. 单开一个 py3.13 venv 装 mem0 全家桶（torch/sentence-transformers/chromadb）
python3.13 -m venv ../../.venv-mem0
../../.venv-mem0/bin/pip install "mem0ai>=2" sentence-transformers chromadb fastapi uvicorn \
    openai aiohttp aiolimiter python-dotenv tqdm pydantic

# 1. 起本地 mem0 REST 桥（前台，保持开着）
bash run_mem0.sh start-shim

# 2. 另开窗口：跑官方 benchmark（30 题分层，断点续跑）
bash run_mem0.sh run mm30 5 3      # PROJECT=mm30, per-type=5(=30题), workers=3
```

结果落在 `official_out/mem0/`：每题一个 `predicted_mm30/<qid>.json`（含检索到的记忆、
逐 cutoff 的答案+评判），以及汇总 `longmemeval_results_*.json`（按题型/cutoff 的准确率）。

## 注意（口径与对照公平性）

- **系统级对照**：mem0 这组用的是 **mem0 自己的官方 harness**（它自己的答题/评判 prompt），
  我们的 BM25 那组用的是 **LongMemEval 官方 src** 的 prompt。两者都用 MiniMax-M3 当答题+评判、
  同一份 500 题题库、同一个准确率指标——这是"各自跑各自官方 harness、同模型同题库"的横向对比，
  不是共用同一条生成链的受控对照（mem0 的价值本就在它整套 extract→store→search，硬塞成
  "只返回 top-k session"反而失真）。
- **规模**：mem0 要 ingest 每题**完整 haystack（含干扰项，单题 225-240 个 user/assistant
  pair）**，每个 pair 一次 MiniMax 抽取调用。30 题 ≈ 数千次 reasoning 调用，**数小时**起步。
  `run.py` 每个 pair 落一次断点，可随时中断续跑。
- **时序**：OSS 版 mem0 的 `add` 不接受自定义 `timestamp`（cloud-only），记忆 `created_at`
  = 注入时刻；但 `run.py` 本就按时间顺序注入 session，所以时序大体保留（主要可能轻微影响
  temporal-reasoning）。
- **top-k**：mem0 检索的是原子"记忆"而非 session，所以它的 top-k 和我们"top-10 session"
  不是同尺度；报告里用 cutoff 10/20/30 呈现。
