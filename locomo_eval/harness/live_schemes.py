"""Live (turn-by-turn streaming) wrappers around the existing schemes.

DialSim-style protocol: turns of a conversation are fed to the agent one at a
time, and questions are injected mid-stream rather than asked cold against a
pre-built memory. Each adapter exposes:

    reset()
    ingest_turn(turn, session_meta)   # streamed, no LLM unless the scheme needs it
    on_session_end(session_meta)      # optional: batch work that needs a full session
    answer(question_text) -> {answer, ctx_tokens, ...}

These are deliberately separate from `harness.schemes` so the existing
cold-start LOCOMO benchmark stays byte-identical. Pick adapters from
`LIVE_SCHEMES` by name in `eval.live`.
"""
from collections import defaultdict

from .llm import LLM, n_tokens
from .memory_store import MemoryStore
from . import dataset as ds
from .schemes import _answer, ANSWER_SYS  # noqa: F401  (reused prompt)


# ---------------- full-context-live ----------------
class FullContextLive:
    """Accumulate all turns into one transcript; answer against the whole buffer.

    Upper-bound on accuracy, worst on tokens. Useful to see when the window
    overflows during a long live session."""
    name = "full-context-live"

    def __init__(self):
        self.llm = LLM(scheme=self.name)
        self.reset()

    def reset(self):
        self._lines = []
        self._cur_session = None

    def ingest_turn(self, turn, sm):
        if sm["session_key"] != self._cur_session:
            self._lines.append(f"\n=== {sm['session_key']} ({sm['raw_dt']}) ===")
            self._cur_session = sm["session_key"]
        self._lines.append(f"[{turn['dia_id']}] {turn['speaker']}: "
                           f"{ds.turn_text(turn)}")

    def on_session_end(self, sm):
        pass

    def answer(self, q_text):
        ctx = "\n".join(self._lines)
        ans = _answer(self.llm, q_text, ctx, budget_tokens=None)
        return {"answer": ans, "ctx_tokens": n_tokens(ctx)}


# ---------------- bm25-rag-live ----------------
class BM25Live:
    """Streaming BM25: every turn becomes a chunk; query-time we rank against
    the chunks observed so far (no future leakage)."""
    name = "bm25-rag-live"

    def __init__(self, total_budget=4000, k_chunks=24):
        from rank_bm25 import BM25Okapi  # type: ignore
        self._BM25 = BM25Okapi
        self.llm = LLM(scheme=self.name)
        self.total_budget = total_budget
        self.k_chunks = k_chunks
        self.reset()

    def reset(self):
        self.chunks = []  # list of (dia_id, date, speaker, text)
        self._tokenized = []

    def ingest_turn(self, turn, sm):
        text = ds.turn_text(turn)
        self.chunks.append((turn["dia_id"], sm["iso_date"], turn["speaker"], text))
        self._tokenized.append(_simple_tok(text))

    def on_session_end(self, sm):
        pass

    def answer(self, q_text):
        if not self.chunks:
            return {"answer": "No information available.", "ctx_tokens": 0}
        bm = self._BM25(self._tokenized)
        scores = bm.get_scores(_simple_tok(q_text))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:self.k_chunks]
        order.sort()  # restore chronological for the prompt
        picked = [self.chunks[i] for i in order]
        ctx_lines = [f"[{d} | {dt} | {sp}] {t}" for d, dt, sp, t in picked]
        ctx = "\n".join(ctx_lines)
        # trim to budget
        while n_tokens(ctx) > self.total_budget and ctx_lines:
            ctx_lines.pop()
            ctx = "\n".join(ctx_lines)
        ans = _answer(self.llm, q_text, ctx, self.total_budget)
        return {"answer": ans, "ctx_tokens": n_tokens(ctx)}


# ---------------- theme-mem-live ----------------
class ThemeMemLive:
    """Live theme memory. Turns are buffered per session; at session-end we
    call MemoryStore.ingest_session (which does the LLM-based theme
    segmentation + core distillation, same as the offline build). Queries
    arriving inside a session see all PREVIOUS sessions' memory plus a flat
    transcript of the in-progress session — no future leakage.

    Modes mirror the offline schemes:
      mode="sf"     stateless summary-first retrieval (no eviction state)
      mode="evict"  stateful with theme-switch eviction (DESIGN.md §4)
      mode="accum"  stateful, accumulating (control)
    """
    def __init__(self, conv_id, total_budget=4000, mode="evict"):
        self.conv_id = conv_id
        self.total_budget = total_budget
        self.mode = mode
        self.name = f"theme-mem-{mode}-live"
        self.llm = LLM(scheme=self.name)
        self.reset()

    def reset(self):
        # Fresh in-memory store — never load a pre-built checkpoint, since
        # the whole point of live eval is to grow memory turn-by-turn.
        self.store = MemoryStore(self.conv_id, persist=False)
        self._cur_session = None
        self._cur_buf = []         # buffered turns for the in-progress session
        self._cur_meta = None
        # state for evict/accum across questions
        self._prev_themes = None
        self._visited = set()
        self._n_switch = 0

    def ingest_turn(self, turn, sm):
        if sm["session_key"] != self._cur_session:
            self._cur_session = sm["session_key"]
            self._cur_meta = sm
            self._cur_buf = []
        self._cur_buf.append(turn)

    def on_session_end(self, sm):
        if not self._cur_buf:
            return
        # Commit the session into the theme store (this does LLM calls).
        self.store.ingest_session(sm["session_key"], sm["iso_date"],
                                  list(self._cur_buf))
        self._cur_buf = []

    def _in_progress_context(self):
        """Flat transcript of the currently-open session (not yet ingested)."""
        if not self._cur_buf:
            return "", 0
        sm = self._cur_meta
        lines = [f"=== {sm['session_key']} ({sm['raw_dt']}) [in progress] ==="]
        for t in self._cur_buf:
            lines.append(f"[{t['dia_id']}] {t['speaker']}: {ds.turn_text(t)}")
        s = "\n".join(lines)
        return s, n_tokens(s)

    def answer(self, q_text):
        from .schemes import ThemeMemSummaryFirst, ThemeMemStateful

        # Pick the inner retrieval policy by mode.
        if self.mode == "sf":
            inner = ThemeMemSummaryFirst(self.store, total_budget=self.total_budget)
            r = inner.answer({"question": q_text})
        else:
            inner = ThemeMemStateful(self.store, total_budget=self.total_budget,
                                     evict=(self.mode == "evict"))
            # carry our cross-question state into the inner scheme
            inner.prev_themes = self._prev_themes
            inner.visited = set(self._visited)
            inner.n_switch = self._n_switch
            r = inner.answer({"question": q_text})
            self._prev_themes = inner.prev_themes
            self._visited = inner.visited
            self._n_switch = inner.n_switch

        # Augment with in-progress session turns the store hasn't ingested yet.
        ip_text, ip_tok = self._in_progress_context()
        r["ctx_tokens"] = r.get("ctx_tokens", 0) + ip_tok
        r["in_progress_tokens"] = ip_tok
        return r


# ---------------- mem0-live ----------------
class Mem0Live:
    """Live mem0: each turn is sent through Memory.add() incrementally.
    Optional — only available if `mem0` is installed."""
    name = "mem0-live"

    def __init__(self, conv_id, top_k=30):
        try:
            import mem0  # type: ignore  # noqa: F401
        except ImportError as e:
            raise RuntimeError("mem0 not installed; pip install -r "
                               "requirements-mem0.txt") from e
        from .schemes import _mem0_memory
        self._mem0_memory = _mem0_memory
        self.conv_id = conv_id
        self.llm = LLM(scheme=self.name)
        self.top_k = top_k
        self.reset()

    def reset(self):
        # Use a separate, ephemeral chroma path so we don't collide with the
        # cold-start mem0 store at memory_runtime/mem0/<conv>/.
        import shutil, os
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "..", "memory_runtime", "mem0_live", self.conv_id)
        shutil.rmtree(root, ignore_errors=True)
        self.mem = self._mem0_memory(self.conv_id, root_subdir="mem0_live")
        self._user_id = "live_user"

    def ingest_turn(self, turn, sm):
        msg = f"{turn['speaker']}: {ds.turn_text(turn)}"
        try:
            self.mem.add(msg, user_id=self._user_id,
                         metadata={"dia_id": turn["dia_id"],
                                   "session": sm["session_key"]})
        except Exception:
            # mem0 sometimes rejects empty/malformed; skip silently — the eval
            # records this scheme's accuracy honestly under failures.
            pass
        self._n_ingested = getattr(self, "_n_ingested", 0) + 1
        if self._n_ingested % 25 == 0:
            print(f"    mem0-live ingested {self._n_ingested} turns", flush=True)

    def on_session_end(self, sm):
        pass

    def answer(self, q_text):
        try:
            hits = self.mem.search(q_text, user_id=self._user_id, limit=self.top_k)
            facts = "\n".join(f"- {h['memory']}" for h in hits.get("results", []))
        except Exception:
            facts = ""
        ans = _answer(self.llm, q_text, facts, budget_tokens=None)
        return {"answer": ans, "ctx_tokens": n_tokens(facts)}


# ---------------- registry ----------------
def _simple_tok(s):
    import re
    return re.findall(r"[a-z0-9]+", (s or "").lower())


LIVE_SCHEMES = {
    "full-context-live": lambda cfg: FullContextLive(),
    "bm25-rag-live":     lambda cfg: BM25Live(total_budget=cfg["budget"]),
    "theme-mem-sf-live":     lambda cfg: ThemeMemLive(cfg["conv_id"],
                                                      total_budget=cfg["budget"],
                                                      mode="sf"),
    "theme-mem-evict-live":  lambda cfg: ThemeMemLive(cfg["conv_id"],
                                                      total_budget=cfg["budget"],
                                                      mode="evict"),
    "theme-mem-accum-live":  lambda cfg: ThemeMemLive(cfg["conv_id"],
                                                      total_budget=cfg["budget"],
                                                      mode="accum"),
    "mem0-live":             lambda cfg: Mem0Live(cfg["conv_id"]),
}
