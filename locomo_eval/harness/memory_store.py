"""Theme-partitioned external memory implementing DESIGN.md.

Layout on disk (faithful to spec; also kept in-memory for fast retrieval):
  <root>/<conv_id>/
    index.md                # Core Memory + Themes index
    <theme>/<YYYY-MM-DD>.md  # daily details, header = ~day summary
    <theme>/summary.md       # overall rollup

Adaptation note for LOCOMO QA: "core mem" here holds stable cross-theme FACTS
about the speakers (names, relationships, persistent life situations), since the
benchmark is fact-recall about two people. DESIGN.md frames core as generalizable
experience; specific facts otherwise live in per-theme summaries.
"""
import os
import re
import json
from collections import defaultdict

from .llm import LLM, n_tokens

CORE_BUDGET = 800          # tokens, per DESIGN.md ~800
STOPWORDS = set("a an the of to in on at for and or but is are was were be been being "
                "i you he she it we they me my your his her our their this that these those "
                "do does did have has had will would can could should what when where who "
                "why how which with about into from as by than then so if not no yes "
                "go went going get got make made say said".split())


def _slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:50] or "misc"


def _keywords(text):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Theme:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        # date -> list of turn dicts {dia_id, speaker, text}
        self.days = defaultdict(list)
        self.day_summaries = {}   # date -> str
        self.related = set()

    def overall_summary(self, max_tokens=400):
        """Roll up day summaries; overall acts as an index (DESIGN §1)."""
        parts = [f"- [{d}] {s}" for d, s in sorted(self.day_summaries.items())]
        text = "\n".join(parts)
        # if too long we'd roll up via LLM; for these sizes joining stays small
        return text


class MemoryStore:
    def __init__(self, conv_id, root=None, persist=True):
        if root is None:
            root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "memory_runtime")
        self.conv_id = conv_id
        self.root = os.path.join(root, conv_id)
        self.persist = persist
        self.core_mem = ""
        self.themes = {}          # name -> Theme
        self.ingested = []        # session_keys already ingested (for resume)
        self.llm = LLM(scheme="theme-mem")
        if persist:
            os.makedirs(self.root, exist_ok=True)

    # pickle: drop the unpicklable LLM client, recreate on load
    def __getstate__(self):
        d = self.__dict__.copy()
        d["llm"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.llm = LLM(scheme="theme-mem")

    def checkpoint(self):
        import pickle
        if self.persist:
            with open(os.path.join(self.root, "store.pkl"), "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, conv_id, root=None):
        import pickle
        if root is None:
            root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "memory_runtime")
        p = os.path.join(root, conv_id, "store.pkl")
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
        return None

    # ---------- ingestion ----------
    def ingest_session(self, session_key, iso_date, turns):
        segments = self._segment(session_key, turns)
        idx = {t["dia_id"]: t for t in turns}
        for seg in segments:
            tname = seg["theme"]
            if tname not in self.themes:
                self.themes[tname] = Theme(tname, seg.get("desc", tname))
            th = self.themes[tname]
            seg_turns = []
            for i in range(seg["start"], seg["end"] + 1):
                if 1 <= i <= len(turns):
                    t = turns[i - 1]
                    rec = {"dia_id": t["dia_id"], "speaker": t["speaker"],
                           "text": _turn_text(t)}
                    th.days[iso_date].append(rec)
                    seg_turns.append(rec)
            # day summary header: append segment summary for this date
            prev = th.day_summaries.get(iso_date, "")
            th.day_summaries[iso_date] = (prev + " " + seg.get("summary", "")).strip()
        self._update_core(session_key, turns)
        self.ingested.append(session_key)
        if self.persist:
            self._flush()
            self.checkpoint()

    def _segment(self, session_key, turns):
        """One LLM call: split session into contiguous topical segments and
        assign each to an existing theme (reuse) or a new theme."""
        numbered = "\n".join(
            f"{i+1}. {t['speaker']}: {_turn_text(t)}" for i, t in enumerate(turns))
        theme_list = "\n".join(
            f"- {n}: {t.desc}" for n, t in self.themes.items()) or "(none yet)"
        sys = ("You segment a chat session into contiguous topical segments and "
               "assign each to a theme. The themes are NOT predefined: reuse an "
               "existing theme name from the list when a segment clearly fits it, "
               "otherwise invent a new concise theme name (2-4 words, lowercase) "
               "that names the topic/event the segment is about. Let the content "
               "decide the themes; do not force segments into ill-fitting themes.")
        usr = (f"Existing themes (may be empty):\n{theme_list}\n\n"
               f"Session turns (numbered):\n{numbered}\n\n"
               "Return JSON: {\"segments\":[{\"theme\":str,\"new\":bool,"
               "\"desc\":str (1 line, only if new),\"start\":int,\"end\":int,"
               "\"summary\":str (1 sentence)}]} covering all turns, "
               "start/end are 1-based inclusive turn numbers.")
        try:
            out = self.llm.chat_json(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": usr}],
                phase="ingest", max_tokens=8000)  # big sessions + reasoning overhead
            segs = out.get("segments", [])
            segs = [s for s in segs if "start" in s and "end" in s and "theme" in s]
            if segs:
                return segs
        except Exception:
            pass
        # fallback (no preset theme): ask the model to name ONE theme for the
        # whole session, so even on failure the theme is model-derived.
        return [self._fallback_theme(turns)]

    def _fallback_theme(self, turns):
        text = "\n".join(f"{t['speaker']}: {_turn_text(t)}" for t in turns)
        try:
            out = self.llm.chat_json(
                [{"role": "system", "content":
                  "Name the single dominant theme of this chat session. Return "
                  "JSON {\"theme\":str (2-4 words, lowercase),\"desc\":str,"
                  "\"summary\":str}."},
                 {"role": "user", "content": text[:6000]}],
                phase="ingest", max_tokens=2000)
            name = (out.get("theme") or "").strip().lower()
            if name:
                return {"theme": name, "new": True,
                        "desc": out.get("desc", name),
                        "start": 1, "end": len(turns),
                        "summary": out.get("summary", "")}
        except Exception:
            pass
        # absolute last resort: a per-session bucket (still not a shared preset)
        sk = f"session-{len(self.ingested)+1}"
        return {"theme": sk, "new": True, "desc": "unclassified session",
                "start": 1, "end": len(turns), "summary": ""}

    def _update_core(self, session_key, turns):
        text = "\n".join(f"{t['speaker']}: {_turn_text(t)}" for t in turns)
        sys = ("You maintain a compact CORE MEMORY of stable, cross-topic facts "
               "about the speakers that stay useful across many topics: identities, "
               "relationships, ongoing life situations, persistent preferences. "
               "Not transient details. Merge new info, dedupe, keep it tight "
               f"(<= {CORE_BUDGET} tokens). Mark superseded facts instead of "
               "silently dropping when something changes.")
        usr = (f"Current core memory:\n{self.core_mem or '(empty)'}\n\n"
               f"New session ({session_key}):\n{text}\n\n"
               "Return the updated core memory as plain markdown bullet points.")
        try:
            new = self.llm.chat([{"role": "system", "content": sys},
                                 {"role": "user", "content": usr}],
                                phase="ingest", max_tokens=2500)
            if new.strip():
                self.core_mem = _truncate_tokens(new.strip(), CORE_BUDGET)
        except Exception:
            pass

    # ---------- retrieval ----------
    def route(self, question, topk=3):
        """Pick theme(s) for a question against the theme index (DESIGN §3)."""
        if not self.themes:
            return []
        theme_list = "\n".join(f"- {n}: {t.desc}" for n, t in self.themes.items())
        sys = ("Route a question to the most relevant theme(s) from the index. "
               "Return JSON {\"themes\":[name,...]} ranked best first, up to "
               f"{topk}. Use exact theme names from the list.")
        usr = f"Theme index:\n{theme_list}\n\nQuestion: {question}"
        try:
            out = self.llm.chat_json([{"role": "system", "content": sys},
                                      {"role": "user", "content": usr}],
                                     phase="query", max_tokens=300)
            names = [n for n in out.get("themes", []) if n in self.themes]
            return names[:topk]
        except Exception:
            return list(self.themes.keys())[:topk]

    def assemble(self, question, theme_names, total_budget=4000, detail_turns=24):
        """Build the working window: core mem + theme summaries + structured
        keyword/date retrieval of detail turns, capped at total_budget tokens."""
        qkw = _keywords(question)
        date_hint = _date_in(question)
        ctx = []
        if self.core_mem:
            ctx.append("## Core Memory\n" + self.core_mem)
        for name in theme_names:
            th = self.themes.get(name)
            if th:
                ctx.append(f"## Theme: {name} — summary\n{th.overall_summary()}")
        # structured retrieval over candidate themes' detail turns
        scored = []
        for name in theme_names:
            th = self.themes.get(name)
            if not th:
                continue
            for date, recs in th.days.items():
                for r in recs:
                    kw = _keywords(r["text"])
                    score = len(qkw & kw)
                    if date_hint and date_hint in date:
                        score += 3
                    if score > 0:
                        scored.append((score, date, name, r))
        scored.sort(key=lambda x: -x[0])
        used = n_tokens("\n\n".join(ctx))
        lines = []
        for score, date, name, r in scored[:detail_turns * 2]:
            line = f"[{r['dia_id']} | {date} | {name}] {r['speaker']}: {r['text']}"
            tk = n_tokens(line)
            if used + tk > total_budget:
                break
            lines.append(line)
            used += tk
            if len(lines) >= detail_turns:
                break
        if lines:
            ctx.append("## Retrieved details\n" + "\n".join(lines))
        text = "\n\n".join(ctx)
        return text, n_tokens(text)

    def summaries_context(self):
        """Stage-1 context for summary-first retrieval: core mem + ALL theme
        summaries (compact, no routing). Returns (text, tokens)."""
        ctx = []
        if self.core_mem:
            ctx.append("## Core Memory\n" + self.core_mem)
        ctx.append("## Topic summaries (all)")
        for n, t in self.themes.items():
            ctx.append(f"### {n}: {t.desc}\n{t.overall_summary()}")
        text = "\n\n".join(ctx)
        return text, n_tokens(text)

    def retrieve_details(self, question, theme_names, max_tokens, detail_turns=40):
        """Stage-2: precise BM25 keyword retrieval of detail turns within the
        chosen themes, plus a date-match boost, capped at max_tokens.

        BM25 (term-frequency weighted) ranks the relevant turn higher than plain
        set-overlap — important for single-fact and temporal questions where the
        evidence turn shares only a few, but discriminative, words."""
        from rank_bm25 import BM25Okapi
        cand = []  # (date, name, rec)
        for name in theme_names:
            th = self.themes.get(name)
            if not th:
                continue
            for date, recs in th.days.items():
                for r in recs:
                    cand.append((date, name, r))
        if not cand:
            return "", 0
        corpus = [_tok(f"{c[2]['speaker']}: {c[2]['text']}") for c in cand]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tok(question))
        date_hint = _date_in(question)
        if date_hint:
            for i, (date, _n, _r) in enumerate(cand):
                if date_hint in date:
                    scores[i] += max(scores) + 1.0  # surface dated turns
        order = sorted(range(len(cand)), key=lambda i: -scores[i])
        lines, used = [], 0
        for i in order:
            if scores[i] <= 0:
                break
            date, name, r = cand[i]
            line = f"[{r['dia_id']} | {date} | {name}] {r['speaker']}: {r['text']}"
            tk = n_tokens(line)
            if used + tk > max_tokens:
                break
            lines.append(line)
            used += tk
            if len(lines) >= detail_turns:
                break
        text = "\n".join(lines)
        return text, n_tokens(text)

    # ---------- persistence ----------
    def _flush(self):
        idx = ["# Core Memory", self.core_mem or "(empty)", "", "# Themes"]
        for n, t in self.themes.items():
            rel = f"  Related: {sorted(t.related)}" if t.related else ""
            idx.append(f"- {n}: {t.desc}{rel}")
        _atomic_write(os.path.join(self.root, "index.md"), "\n".join(idx))
        for n, t in self.themes.items():
            d = os.path.join(self.root, _slug(n))
            os.makedirs(d, exist_ok=True)
            _atomic_write(os.path.join(d, "summary.md"),
                          f"# {n}\n{t.desc}\n\n## Overall\n{t.overall_summary()}")
            for date, recs in t.days.items():
                body = [f"<!-- day summary: {t.day_summaries.get(date,'')} -->"]
                body += [f"[{r['dia_id']}] {r['speaker']}: {r['text']}" for r in recs]
                _atomic_write(os.path.join(d, f"{date}.md"), "\n".join(body))


# ---------- helpers ----------
def _turn_text(t):
    txt = t.get("text", "")
    blip = t.get("blip_caption")
    if blip:
        txt = (txt + f" [shares an image: {blip}]").strip()
    return txt


def _truncate_tokens(text, max_tok):
    from .llm import _ENC
    ids = _ENC.encode(text)
    if len(ids) <= max_tok:
        return text
    return _ENC.decode(ids[:max_tok])


def _date_in(q):
    m = re.search(r"(\d{4})", q)
    return m.group(1) if m else None


def _atomic_write(path, content):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)
