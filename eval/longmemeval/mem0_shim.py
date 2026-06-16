"""Minimal OSS-mode REST bridge so the OFFICIAL mem0 benchmark
(vendor/memory-benchmarks/benchmarks/longmemeval/run.py, which talks to a mem0 OSS
server via `Mem0Client(mode="oss")`) can run natively — no Docker, no postgres/auth.

It exposes exactly the 3 endpoints that Mem0Client(oss) calls:
  POST /memories   {messages, user_id, timestamp?, metadata?} -> {"results": [...]}
  POST /search     {query, user_id, limit}                    -> {"results": [...]}
  DELETE /memories {user_id}                                  -> {"message": ...}

…backed by the OFFICIAL `mem0.Memory` library (the actual engine under test). Only the
transport is bridged; mem0's extract/store/search logic is untouched. Config:
  LLM      = MiniMax-M3 (OpenAI-compatible endpoint) — fact extraction
  embedder = all-MiniLM-L6-v2 (local sentence-transformers, ~90MB, CPU)
  store    = chroma (embedded, on-disk; no server)

Run (py3.13 venv with mem0ai+torch+chromadb):
  .venv-mem0/bin/uvicorn mem0_shim:app --host 127.0.0.1 --port 8888

mem0 2.0.6 OSS: search() takes top_k + filters={'user_id': ...}. The `timestamp` arg on
add() is cloud-only (OSS rejects it), so memories get created_at=ingest-time — but run.py
ingests sessions in chronological order, so created_at preserves chronological order
anyway (the answerer sorts by created_at). Good enough; noted in RESULTS.
"""
import asyncio
import os

os.environ.setdefault("MEM0_TELEMETRY", "false")     # don't phone home to posthog
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, Request
from mem0 import Memory

HERE = os.path.dirname(os.path.abspath(__file__))
_KEY = os.environ.get("MINIMAX_API_KEY") or open(
    os.path.expanduser("~/.minimax_key"), encoding="utf-8").read().strip()
os.environ["OPENAI_API_KEY"] = _KEY  # mem0's openai LLM reads key+base_url from here/config

CONFIG = {
    "llm": {"provider": "openai", "config": {
        "model": "MiniMax-M3",
        "openai_base_url": "https://api.minimaxi.com/v1",
        "temperature": 0,
        "max_tokens": 2048,  # room for MiniMax-M3 <think> before the JSON
    }},
    "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    "vector_store": {"provider": "chroma", "config": {
        "collection_name": "longmemeval",
        "path": os.path.join(HERE, "official_out", "chroma_mem0"),
    }},
}

mem = Memory.from_config(CONFIG)
app = FastAPI()


# mem.add/search/delete are SYNCHRONOUS and block (LLM call + embedding). Offload to a
# threadpool so one in-flight request never freezes the event loop (and others can run).
@app.post("/memories")
async def add_memories(req: Request):
    b = await req.json()
    res = await asyncio.to_thread(
        mem.add, messages=b["messages"], user_id=b["user_id"],
        metadata=b.get("metadata") or None)  # OSS: timestamp arg unsupported (cloud-only)
    return res if isinstance(res, dict) else {"results": res}


@app.post("/search")
async def search_memories(req: Request):
    b = await req.json()
    res = await asyncio.to_thread(
        mem.search, query=b["query"], top_k=int(b.get("limit", 10)),
        filters={"user_id": b["user_id"]})
    return res if isinstance(res, dict) else {"results": res}


@app.delete("/memories")
async def delete_memories(req: Request):
    b = await req.json()
    await asyncio.to_thread(mem.delete_all, user_id=b["user_id"])
    return {"message": "deleted"}


@app.get("/health")
async def health():
    return {"status": "ok"}
