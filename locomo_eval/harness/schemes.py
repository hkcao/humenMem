"""The three schemes under test, each exposing .answer(q) -> {answer, ctx_tokens}.

  theme-mem    : DESIGN.md theme-partitioned memory + budgeted window
  full-context : stuff the whole conversation (accuracy upper bound, token worst)
  bm25-rag     : flat BM25 turn retrieval under the SAME budget as theme-mem
                 (isolates "theme partition + summaries" vs flat keyword retrieval)
"""
import re

from .llm import LLM, n_tokens
from .memory_store import MemoryStore
from . import dataset as ds

ANSWER_SYS = (
    "You answer a question using ONLY the provided context from a conversation "
    "between two people. Be very concise: a name, date, short phrase, or number — "
    "no full sentences unless required. Dates: match the format implied by the "
    "question. If the context does not contain the answer, reply exactly: "
    "No information available.")


def _answer(llm, question, context, budget_tokens):
    usr = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    out = llm.chat([{"role": "system", "content": ANSWER_SYS},
                    {"role": "user", "content": usr}],
                   phase="query", max_tokens=600)
    return out.strip()


# ---------------- theme-mem ----------------
class ThemeMemScheme:
    name = "theme-mem"

    def __init__(self, store: MemoryStore, total_budget=4000, topk=3, strict=False):
        self.store = store
        self.llm = LLM(scheme=self.name)
        self.total_budget = total_budget
        self.topk = 1 if strict else topk

    def answer(self, q):
        themes = self.store.route(q["question"], topk=self.topk)
        context, ctx_tok = self.store.assemble(
            q["question"], themes, total_budget=self.total_budget)
        ans = _answer(self.llm, q["question"], context, self.total_budget)
        return {"answer": ans, "ctx_tokens": ctx_tok, "themes": themes}


class ThemeMemSummaryFirst:
    """Summary-first retrieval (no blind top-k routing):
    Stage 1: core mem + ALL theme summaries -> try to answer.
    Stage 2: if model says it needs detail, it names the relevant themes
             (having seen all summaries), then we load their detail turns."""
    name = "theme-mem-sf"

    def __init__(self, store: MemoryStore, total_budget=4000):
        self.store = store
        self.llm = LLM(scheme=self.name)
        self.total_budget = total_budget

    def answer(self, q):
        sctx, stok = self.store.summaries_context()
        sys = ("You answer a question about a long conversation between two people. "
               "You are given Core Memory and summaries of ALL topics. If these are "
               "enough to answer precisely, return JSON {\"answer\": <concise answer>}. "
               "If you need the detailed turns to be sure, return JSON "
               "{\"need_details\": [topic names to open]} using exact topic names "
               "shown above. Prefer opening details when the question asks for a "
               "specific date, name, or fact not explicit in the summaries.")
        usr = f"{sctx}\n\nQuestion: {q['question']}"
        try:
            out = self.llm.chat_json([{"role": "system", "content": sys},
                                      {"role": "user", "content": usr}],
                                     phase="query", max_tokens=1000)
        except Exception:
            out = {}
        ans = out.get("answer")
        if isinstance(ans, str) and ans.strip() and ans.strip().lower() != "null":
            return {"answer": ans.strip(), "ctx_tokens": stok, "stage": 1,
                    "themes": []}
        need = out.get("need_details") or []
        need = [n for n in need if n in self.store.themes]
        if not need:
            need = list(self.store.themes.keys())  # fall back to all
        budget_left = max(self.total_budget - stok, 1500)
        dctx, dtok = self.store.retrieve_details(q["question"], need, budget_left)
        full = sctx + "\n\n## Retrieved details\n" + dctx
        a = _answer(self.llm, q["question"], full, self.total_budget)
        return {"answer": a, "ctx_tokens": stok + dtok, "stage": 2, "themes": need}


class ThemeMemStateful:
    """Stateful, SEQUENTIAL summary-first retrieval with theme-switch eviction
    (DESIGN.md §4). The window always holds core + ALL theme summaries. Detail
    turns are loaded per question; which themes are *eligible* for detail depends
    on the mode:

      evict=True  : details only from the CURRENT question's theme(s). When the
                    theme changes vs the previous question we EVICT and reload —
                    the window never holds more than one theme's details, so its
                    size stays bounded no matter how many themes get visited.
      evict=False : accumulate — details from the UNION of all themes visited so
                    far; the window grows toward full-conversation size.

    Comparing the two shows what eviction buys: bounded tokens vs unbounded.
    Must be run sequentially (state carries across questions)."""
    sequential = True

    def __init__(self, store: MemoryStore, total_budget=4000, evict=True,
                 accum_cap=28000):
        self.store = store
        self.llm = LLM(scheme=self.name_for(evict))
        self.total_budget = total_budget
        self.evict = evict
        self.accum_cap = accum_cap
        self.reset()

    @staticmethod
    def name_for(evict):
        return "theme-mem-evict" if evict else "theme-mem-accum"

    @property
    def name(self):
        return self.name_for(self.evict)

    def reset(self):
        self.prev_themes = None       # theme set used by the previous question
        self.visited = set()          # all themes visited so far (accumulate)
        self.n_switch = 0

    def answer(self, q):
        sctx, stok = self.store.summaries_context()
        sys = ("You answer a question about a long conversation between two people. "
               "You are given Core Memory and summaries of ALL topics. If these are "
               "enough to answer precisely, return JSON {\"answer\": <concise>}. "
               "If you need detailed turns, return JSON {\"need_details\": [topic "
               "names]} using exact topic names shown. Prefer opening details for a "
               "specific date, name, or fact not explicit in the summaries.")
        try:
            out = self.llm.chat_json(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": f"{sctx}\n\nQuestion: {q['question']}"}],
                phase="query", max_tokens=1000)
        except Exception:
            out = {}
        ans = out.get("answer")
        if isinstance(ans, str) and ans.strip() and ans.strip().lower() != "null":
            # answered from summaries; no detail theme loaded, no switch
            return {"answer": ans.strip(), "ctx_tokens": stok, "stage": 1,
                    "themes": [], "switched": False, "n_switch": self.n_switch}

        need = [n for n in (out.get("need_details") or []) if n in self.store.themes]
        if not need:
            need = list(self.store.themes.keys())
        T = set(need)
        switched = (self.prev_themes is not None and T != self.prev_themes)
        if switched:
            self.n_switch += 1
        self.prev_themes = T

        if self.evict:
            load_themes = list(T)                       # bounded: current only
            budget = max(self.total_budget - stok, 1500)
        else:
            self.visited |= T
            load_themes = list(self.visited)            # grows with themes seen
            budget = self.accum_cap

        dctx, dtok = self.store.retrieve_details(q["question"], load_themes, budget)
        full = sctx + "\n\n## Retrieved details\n" + dctx
        a = _answer(self.llm, q["question"], full, self.total_budget)
        return {"answer": a, "ctx_tokens": stok + dtok, "stage": 2,
                "themes": load_themes, "switched": switched,
                "n_switch": self.n_switch}


def build_theme_memory(sample, persist=True, resume=True, verbose=False):
    cid = sample["sample_id"]
    store = MemoryStore.load_checkpoint(cid) if resume else None
    if store is None:
        store = MemoryStore(cid, persist=persist)
    for sk, iso, dt, turns in ds.iter_sessions(sample["conversation"]):
        if sk in store.ingested:
            continue
        store.ingest_session(sk, iso, turns)
        if verbose:
            print(f"    ingested {sk} ({iso}) | themes={len(store.themes)}",
                  flush=True)
    return store


# ---------------- full-context ----------------
class FullContextScheme:
    name = "full-context"

    def __init__(self, sample):
        self.llm = LLM(scheme=self.name)
        self.context = ds.flatten(sample["conversation"])
        self.ctx_tok = n_tokens(self.context)

    def answer(self, q):
        ans = _answer(self.llm, q["question"], self.context, self.ctx_tok)
        return {"answer": ans, "ctx_tokens": self.ctx_tok, "themes": []}


# ---------------- bm25 flat RAG ----------------
class BM25Scheme:
    name = "bm25-rag"

    def __init__(self, sample, total_budget=4000):
        from rank_bm25 import BM25Okapi
        self.llm = LLM(scheme=self.name)
        self.total_budget = total_budget
        self.chunks = []  # (dia_id, date, speaker, text)
        for sk, iso, dt, turns in ds.iter_sessions(sample["conversation"]):
            for t in turns:
                self.chunks.append((t["dia_id"], iso, t["speaker"], ds.turn_text(t)))
        corpus = [_tok(f"{c[2]}: {c[3]}") for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def answer(self, q):
        scores = self.bm25.get_scores(_tok(q["question"]))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        lines, used = [], 0
        for i in order:
            c = self.chunks[i]
            line = f"[{c[0]} | {c[1]}] {c[2]}: {c[3]}"
            tk = n_tokens(line)
            if used + tk > self.total_budget:
                break
            lines.append(line)
            used += tk
        context = "\n".join(lines)
        ans = _answer(self.llm, q["question"], context, self.total_budget)
        return {"answer": ans, "ctx_tokens": n_tokens(context), "themes": []}


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


# ---------------- mem0 (external memory framework baseline) ----------------
# mem0 stores LLM-extracted facts in a vector store and retrieves by semantic
# similarity. Note: it uses EMBEDDINGS (here a local MiniLM), unlike the
# theme-mem / bm25 schemes which are deliberately embedding-free per DESIGN.md.
import os as _os
import contextlib as _contextlib


@_contextlib.contextmanager
def _account_openai(scheme, phase):
    """Tally token usage of every OpenAI-compatible chat call made inside the
    block (used to capture mem0's internal LLM calls, which bypass our LLM)."""
    from openai.resources.chat.completions import Completions
    from .llm import ACCT
    orig = Completions.create

    def wrapper(self, *a, **kw):
        r = orig(self, *a, **kw)
        try:
            u = r.usage
            rt = getattr(getattr(u, "completion_tokens_details", None),
                         "reasoning_tokens", 0) or 0
            ACCT.add(scheme, phase, {"prompt": u.prompt_tokens,
                                     "completion": u.completion_tokens,
                                     "reasoning": rt})
        except Exception:
            pass
        return r

    Completions.create = wrapper
    try:
        yield
    finally:
        Completions.create = orig


def _mem0_memory(conv_id):
    from mem0 import Memory
    from .llm import _load_env
    _load_env()
    root = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                         "memory_runtime", "mem0", conv_id)
    key = _os.environ["DEEPSEEK_API_KEY"]
    config = {
        "llm": {"provider": "deepseek", "config": {
            "model": _os.environ["DEEPSEEK_MODEL"], "api_key": key,
            "temperature": 0, "max_tokens": 4000}},
        "embedder": {"provider": "huggingface",
                     "config": {"model": "all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "chroma", "config": {
            "collection_name": "c_" + re.sub(r"[^a-z0-9]", "_", conv_id.lower()),
            "path": root}},
    }
    return Memory.from_config(config)


class Mem0Scheme:
    name = "mem0"

    def __init__(self, sample, top_k=10):
        from .llm import LLM as _L
        self.cid = sample["sample_id"]
        self.m = _mem0_memory(self.cid)
        self.top_k = top_k
        self.llm = _L(scheme=self.name)

    def answer(self, q):
        r = self.m.search(q["question"], filters={"user_id": self.cid},
                          top_k=self.top_k)
        mems = [x["memory"] for x in r.get("results", [])]
        context = "\n".join(f"- {m}" for m in mems)
        ans = _answer(self.llm, q["question"], context, 4000)
        return {"answer": ans, "ctx_tokens": n_tokens(context),
                "themes": [], "n_mem": len(mems)}


def build_mem0(sample, verbose=False):
    """Ingest a conversation into mem0 (LLM fact-extraction per session).
    Resumable: skips sessions already added (tracked in a marker file)."""
    import json as _json
    cid = sample["sample_id"]
    m = _mem0_memory(cid)
    root = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                         "memory_runtime", "mem0", cid)
    marker = _os.path.join(root, "ingested.json")
    done = set(_json.load(open(marker))) if _os.path.exists(marker) else set()
    with _account_openai("mem0", "ingest"):
        for sk, iso, dt, turns in ds.iter_sessions(sample["conversation"]):
            if sk in done:
                continue
            msgs = [{"role": "user",
                     "content": f"[{iso}] {t['speaker']}: {ds.turn_text(t)}"}
                    for t in turns]
            m.add(msgs, user_id=cid)
            done.add(sk)
            _json.dump(sorted(done), open(marker, "w"))
            if verbose:
                print(f"    mem0 added {sk} ({iso})", flush=True)
    return m
