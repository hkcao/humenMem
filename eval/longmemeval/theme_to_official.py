"""Adapter: run the REAL theme-memory scheme (topic routing + extract → store →
topic-aware recall) over LongMemEval haystacks and emit the OFFICIAL retrieval-log
schema that run_generation.py consumes — so the headline theme-memory result is scored
by the same official judge as the BM25 baseline / no-mem / mem0.

Difference from bm25_to_official.py (which is the BM25 *baseline*, topic layer OFF):
here we actually BUILD memory. Per question (each has its own ~115k haystack):

  INGEST  for each session in date order, one MiniMax call extracts durable user facts
          and routes each under a concise TOPIC (reusing existing topics when they fit),
          then store.append()s it into a per-question isolated theme store.
  RECALL  retr.retrieve() (BM25 over content+topic across all topics) pulls the top
          memories for the question; we group them back to their source session so the
          official reader can resolve a real date, and feed our memory TEXT verbatim via
          run_generation's `--merge_key_expansion_into_value replace` path.

The per-question store lives under official_out/theme_stores/<qid>/ (MEMORY_INDEX.md, a
global log.md timeline, and per-topic logs/<day>.md + wiki.md) so the recorded
themes+content are directly inspectable. Output is resumable: completed question_ids in
<out> are skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "theme_memory", "scripts"))

import backoff  # noqa: E402
import openai  # noqa: E402
from openai import OpenAI  # noqa: E402

import store  # noqa: E402  our theme-memory store (root() reads $HANK_MEMORY_DIR)
import retrieve as retr  # noqa: E402  pure-Python BM25 recall

CORE_TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "temporal-reasoning", "knowledge-update"]

# The LLM ROUTES turns to topics + writes a wiki summary. The raw stored under each topic is
# the VERBATIM excerpt of that topic's turns (selected by index, copied unchanged from the
# session) — topic-scoped raw records, not the whole session and not a lossy paraphrase.
ROUTE_PROMPT = """You are organizing a USER's chat history into a topic-based long-term memory.
Below is ONE session whose turns are numbered [1], [2], .... Group its turns into TOPIC(s) —
concise, reusable noun phrases (e.g. "career & education", "pet luna", "home renovation").
Prefer REUSING an existing topic VERBATIM when it fits; create a new one only when nothing
fits. A session may span 1-4 topics. EVERY turn that states something about the user must be
assigned to at least one topic (don't drop user facts); ignore only pure assistant boilerplate.

For each topic give:
- "turns": the list of turn numbers belonging to this topic (the verbatim raw kept for it)
- "summary": exhaustively capture every concrete durable user fact in those turns (education,
  job, numbers, dates, names, preferences, decisions, possessions, plans) — omit no specifics.

Topics that ALREADY EXIST (reuse VERBATIM if any fits):
{topics}

Return ONLY a JSON array (no prose, no markdown fence). Each element:
{{"topic": "<concise topic>", "desc": "<one-line topic description>", "turns": [<turn numbers>], "summary": "<all durable user facts in those turns>"}}
If the session has nothing about the user worth remembering, return [].

Session Date: {date}
Numbered session transcript:
{session}
"""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def _sanitize_topic(t: str) -> str:
    t = re.sub(r"[\\/\n\r\t]", " ", str(t)).strip().strip(".")
    t = re.sub(r"\s+", " ", t)
    return (t or "misc")[:60]


def _safe_ts(date: str) -> str:
    # store's entry regex requires a space-free timestamp token
    return re.sub(r"\s+", "_", str(date).strip()) or "na"


def _turns(session) -> list[str]:
    """Verbatim 'role: content' per turn (full content, kept for storage)."""
    return [f"{t['role']}: {t['content'].strip()}" for t in session if t.get("content")]


def _numbered(turns, per_turn_cap=900) -> str:
    """Numbered transcript for the routing prompt (per-turn capped to bound prompt size;
    storage still uses the full untruncated turn by index)."""
    return "\n".join(f"[{i}] {t[:per_turn_cap]}" for i, t in enumerate(turns, 1))


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
     openai.InternalServerError),
    max_tries=8, max_time=300)
def _chat(client, **kw):
    return client.chat.completions.create(**kw)


def _route(client, model, date, turns, existing):
    """LLM: group this session's turns into topics. Returns list of
    {topic, desc, turns:[idx], summary}. Raw is selected by index, not generated."""
    topics_block = "\n".join(f"- {t} — {d}" for t, d in existing.items()) or "(none yet)"
    prompt = ROUTE_PROMPT.format(topics=topics_block, date=date, session=_numbered(turns))
    comp = _chat(client, model=model, messages=[{"role": "user", "content": prompt}],
                 n=1, temperature=0)
    raw = _THINK.sub("", comp.choices[0].message.content or "").strip()
    m = _ARRAY.search(raw)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and it.get("topic"):
            idx = [int(x) for x in it.get("turns", []) if isinstance(x, (int, float))
                   or (isinstance(x, str) and x.isdigit())]
            out.append({"topic": _sanitize_topic(it["topic"]),
                        "desc": str(it.get("desc", "")).splitlines()[0][:120] if it.get("desc") else "",
                        "turns": idx,
                        "summary": str(it.get("summary", "")).strip()})
    return out


# Wiki updates are LOCAL, not full rewrites: the model judges which NEW FACTS to add, which existing
# numbered lines to revise (superseded values), and which to delete (genuinely obsolete/completed).
# Lines it does not touch are preserved verbatim — so a rewrite can never silently drop a still-true
# fact (the failure mode that lost the boots/sweater and mis-prioritized the root wiki). Deletion is
# explicit and intentional, never a side effect of regenerating the whole document.

TOPIC_WIKI_RULE = (
    "Capture EVERY concrete durable user fact: numbers, quantities, prices, dates, names, brands, "
    "models, measurements, preferences, decisions, possessions, plans, and PENDING actions/tasks "
    "(things to pick up, return, do; deadlines). Keep separate items separate (never collapse a "
    "list into a count). One fact per line.")

ROOT_WIKI_RULE = (
    "Record ONLY cross-cutting, globally-important items: PENDING actions/tasks (pick up, return, "
    "buy, do; deadlines, appointments), key stable personal facts (job, education, location, "
    "relationships), and numbers/dates that matter beyond one topic. Skip fine per-topic minutiae.")

WIKI_UPDATE_PROMPT = """You maintain a {kind} of durable facts about a USER. Fold in the NEW FACTS
from a session dated {date} by a LOCAL UPDATE — do NOT rewrite the whole thing.

{rule}

The CURRENT lines are numbered. Using your judgment, add / revise / delete as needed:
- APPEND facts in NEW FACTS that are genuinely new (not already covered by any current line).
- UPDATE a specific numbered line when a new fact supersedes/changes it — give its number and the
  revised text, keeping the old value noted (e.g. "X is now 25:50 (was 27:12 as of 2023/05/23)").
- DELETE a numbered line only when it is genuinely obsolete, completed, or no longer true (e.g. a
  pending pickup the user has now done) — give its number. Do NOT delete still-true facts to save
  space.
- Lines you do not mention are preserved verbatim — never silently drop a still-true fact.
- If a new fact is already recorded, omit it (no op).
{links}
Return ONLY a JSON object (no prose, no code fence):
{{"append": ["<new fact line>", ...],
  "update": [{{"line": <n>, "text": "<revised line>"}}, ...],
  "delete": [<n>, ...]}}

CURRENT:
{numbered}

NEW FACTS (session {date}):
{facts}
"""

_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _wiki_update(client, model, current_bullets, date, facts, kind, rule, link_topics=None):
    """Model-judged LOCAL update. The model returns add/update/delete ops; store.apply_wiki_ops
    applies them mechanically (untouched lines kept verbatim). On any failure returns
    current_bullets unchanged (no loss). `link_topics` (other existing topics) lets the model
    cross-link a fact to a related topic inline as [[name]]; backlinks are mirrored by the store."""
    numbered = "\n".join(f"[{i}] {b}" for i, b in enumerate(current_bullets, 1)) or "(empty)"
    links = ""
    if link_topics:
        lst = ", ".join(f"[[{t}]]" for t in link_topics)
        links = ("- CROSS-LINK: when a fact clearly relates to another topic, reference it inline "
                 f"with its exact [[name]] so topics interlink for quick jumps. Existing topics: {lst}\n")
    prompt = WIKI_UPDATE_PROMPT.format(kind=kind, rule=rule, date=date, numbered=numbered,
                                       facts=facts, links=links)
    try:
        comp = _chat(client, model=model, messages=[{"role": "user", "content": prompt}],
                     n=1, temperature=0)
    except Exception as e:
        print(f"  [wiki-update-fail keep] {e!r}", flush=True)
        return current_bullets
    raw = _THINK.sub("", comp.choices[0].message.content or "").strip()
    m = _OBJ.search(raw)
    if not m:
        return current_bullets
    try:
        ops = json.loads(m.group(0))
    except Exception:
        return current_bullets
    if not isinstance(ops, dict):
        return current_bullets
    upd = {}
    for u in ops.get("update") or []:
        try:
            t = str(u.get("text", "")).strip()
            if t:
                upd[int(u["line"])] = t
        except Exception:
            continue
    dele = {int(d) for d in ops.get("delete") or [] if str(d).strip().lstrip("-").isdigit()}
    return store.apply_wiki_ops(current_bullets, append=ops.get("append"), update=upd, delete=dele)


def _refresh_wiki(client, model, topic, current, date, facts, link_topics=None):
    """Local, model-judged update of a topic's wiki. Returns the rendered wiki text."""
    bullets = _wiki_update(client, model, store.wiki_bullets(current), date, facts,
                           "topic WIKI", TOPIC_WIKI_RULE, link_topics=link_topics)
    return store.render_wiki(f"# {topic}", bullets)


def _refresh_root_wiki(client, model, current, date, facts):
    """Local, model-judged update of the global (cross-topic) wiki. Returns rendered text."""
    bullets = _wiki_update(client, model, store.wiki_bullets(current), date, facts,
                           "user's GLOBAL cross-topic memory", ROOT_WIKI_RULE)
    return store.render_wiki("# GLOBAL (cross-topic)", bullets)


MERGE_PROMPT = """Here are the TOPICS in one user's memory ("name: description"). Some are
near-duplicates or sub-topics of the same BROADER topic and should be merged; most are distinct
and stay on their own. Only merge topics genuinely about the same thing
(e.g. "farm maintenance & compliance" + "farm operations & expansion" -> "farm").

Topics:
{topics}

Return ONLY a JSON array of merge groups, each: {{"canonical": "<kept name>", "members": ["...","..."]}}
Include ONLY groups with 2+ members; "canonical" should be one of the members or a clean broader
name. Return [] if nothing should merge.
"""


def _cluster_topics(client, model, items):
    block = "\n".join(f"- {t}: {d}" for t, d in items.items())
    try:
        comp = _chat(client, model=model, messages=[{"role": "user",
                     "content": MERGE_PROMPT.format(topics=block)}], n=1, temperature=0)
    except Exception as e:
        print(f"  [merge-cluster-fail] {e!r}", flush=True)
        return []
    raw = _THINK.sub("", comp.choices[0].message.content or "").strip()
    m = _ARRAY.search(raw)
    if not m:
        return []
    try:
        groups = json.loads(m.group(0))
    except Exception:
        return []
    return groups if isinstance(groups, list) else []


def consolidate(client, model):
    """Periodic post-ingest pass: merge near-duplicate topics. Returns #topics merged away."""
    items = store.index_items()
    if len(items) < 3:
        return 0
    merged = 0
    for g in _cluster_topics(client, model, items):
        if not isinstance(g, dict):
            continue
        live = store.index_items()
        members = [m for m in g.get("members", []) if m in live]
        if len(members) < 2:
            continue
        canonical = g.get("canonical") if g.get("canonical") in members else members[0]
        store.merge_topics(canonical, [m for m in members if m != canonical])
        merged += len(members) - 1
    return merged


def ingest(client, model, entry):
    """Build the per-question theme store (HANK_MEMORY_DIR already set). EVERY session is stored
    verbatim in the lossless session floor (store.append_session). The LLM then groups turns into
    topic(s); each topic's verbatim excerpt goes into its day log, and the new facts are folded
    into that topic's living wiki via an incremental LLM refresh (merge/dedup/reconcile).
    Returns (n_raw_entries, n_sessions_routed, n_failed)."""
    order = sorted(range(len(entry["haystack_session_ids"])),
                   key=lambda i: entry["haystack_dates"][i])
    n_raw = n_sess = n_fail = 0
    cur_day = day_date = None
    day_facts = []                        # this day's cross-topic facts, flushed once per day

    def flush_day():
        # end-of-day digest: fold the whole day's facts into the root wiki in ONE refresh
        # (vs once per session) — models "tidy up at the end of the day" and cuts the
        # per-session root-wiki call (~40% of ingest) down to one call per distinct day.
        nonlocal day_facts
        if day_facts:
            rw = _refresh_root_wiki(client, model, store.read_root_wiki(), day_date,
                                    "\n".join(day_facts))
            if rw:
                store.write_root_wiki(rw)
        day_facts = []

    for i in order:
        sid = entry["haystack_session_ids"][i]
        date = entry["haystack_dates"][i]
        turns = _turns(entry["haystack_sessions"][i])
        if not turns:
            continue
        day = store._day_key(date)
        if cur_day is not None and day != cur_day:
            flush_day()                   # day rolled over -> tidy the previous day first
        cur_day, day_date = day, date
        # lossless floor: full session verbatim, regardless of (or before) routing
        store.append_session(sid, "\n".join(turns), timestamp=_safe_ts(date))
        try:
            routes = _route(client, model, date, turns, store.index_items())
        except Exception as e:
            print(f"  [ingest-fail] {entry['question_id']} {sid}: {e!r}", flush=True)
            n_fail += 1
            continue
        if routes:
            n_sess += 1
        for r in routes:
            # topic-scoped raw = the verbatim turns assigned to this topic (no paraphrase)
            excerpt = "\n".join(turns[j - 1] for j in r["turns"] if 1 <= j <= len(turns))
            if not excerpt.strip():
                continue
            store.append(r["topic"], excerpt, source=sid, desc=r["desc"],
                         timestamp=_safe_ts(date))
            n_raw += 1
            if r["summary"]:
                others = [t for t in store.list_topics() if t != r["topic"]]
                merged = _refresh_wiki(client, model, r["topic"], store.read_wiki(r["topic"]),
                                       date, r["summary"], link_topics=others)
                if merged:
                    store.write_wiki(r["topic"], merged)
                    store.sync_backlinks(r["topic"])   # mirror [[links]] as backlinks
        # cross-topic facts accumulate; the root wiki is refreshed once at end-of-day
        day_facts.extend(f"- {r['summary']}" for r in routes if r["summary"])
    flush_day()                           # final day
    return n_raw, n_sess, n_fail


_TOPIC_LIST = re.compile(r"\[.*\]", re.DOTALL)


def _route_topics_for_query(client, model, question, items):
    """MODEL-based topic selection for recall: pick the relevant topics (possibly several)
    for this question from the memory index. Returns a list of topic names from `items`, or
    [] on any failure so the caller falls back to BM25 rank_topics."""
    if not items:
        return []
    topics_block = "\n".join(f"- {t}: {d}" for t, d in items.items())
    prompt = (
        "You are selecting which long-term-memory TOPICS could hold the answer to a "
        "question. Pick EVERY topic that might be relevant (usually 1-4); err toward "
        "including a topic when unsure.\n\n"
        f"Topics:\n{topics_block}\n\n"
        f"Question: {question}\n\n"
        "Return ONLY a JSON array of topic names copied verbatim from the list above "
        "(e.g. [\"topic a\", \"topic b\"]). Return [] if none fit.")
    try:
        comp = _chat(client, model=model, messages=[{"role": "user", "content": prompt}],
                     n=1, temperature=0)
    except Exception as e:
        print(f"  [topic-route-fail -> bm25] {e!r}", flush=True)
        return []
    raw = _THINK.sub("", comp.choices[0].message.content or "").strip()
    m = _TOPIC_LIST.search(raw)
    if not m:
        return []
    try:
        names = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for nm in names if isinstance(names, list) else []:
        nm = str(nm).strip()
        if nm in items:
            out.append(nm)
        else:                                   # tolerant match
            low = nm.lower()
            for t in items:
                if t.lower() == low or low in t.lower():
                    out.append(t)
                    break
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _probe_sufficient(client, model, question, context):
    """YES/NO gate: can `context` alone answer the question confidently? The full context is
    passed un-truncated (don't cut the wiki). max_tokens is left unset so the reasoning model
    can finish its <think> and still emit the YES/NO (an output cap mid-<think> would strip to
    an empty answer and every probe would falsely read NO).
    On API failure we return False so the cascade escalates (safe) instead of crashing."""
    prompt = ("You are answering a user's question using ONLY the memory below.\n\n"
              f"Memory:\n{context}\n\nQuestion: {question}\n\n"
              "Can this question be answered specifically and confidently from the memory "
              "above? Reply with exactly YES or NO.")
    try:
        comp = _chat(client, model=model, messages=[{"role": "user", "content": prompt}],
                     n=1, temperature=0)
    except Exception as e:
        print(f"  [probe-fail -> escalate] {e!r}", flush=True)
        return False
    ans = _THINK.sub("", comp.choices[0].message.content or "").strip().upper()
    return ans.startswith("YES") or ("YES" in ans and "NO" not in ans)


def _wiki_mode(client, model, question):
    """The model decides how to read the wiki layer for THIS question — not a forced rule:
      FULL   — load the whole wiki (counting / listing / aggregation / 'how many·which·all'),
      SEARCH — BM25 a few specific lines (a single-fact lookup; scales as the wiki grows).
    Default 'full' on failure (never under-feed). Returns 'full' or 'bm25'."""
    prompt = ("A question will be answered from a user's memory wiki. Decide how to read it:\n"
              "- FULL: needs the WHOLE wiki — counting, listing, 'how many / which / all / overall', "
              "or aggregating across many facts.\n"
              "- SEARCH: a specific single-fact lookup — a few relevant lines suffice.\n\n"
              f"Question: {question}\n\nReply with exactly FULL or SEARCH.")
    try:
        comp = _chat(client, model=model, messages=[{"role": "user", "content": prompt}],
                     n=1, temperature=0)
    except Exception as e:
        print(f"  [wiki-mode-fail -> full] {e!r}", flush=True)
        return "full"
    ans = _THINK.sub("", comp.choices[0].message.content or "").strip().upper()
    return "bm25" if ("SEARCH" in ans and "FULL" not in ans) else "full"


def _items_from_hits(hits, sid2date, topk, wiki=""):
    """Group raw hits by source session -> official ranked_items (one per corpus_id). BM25
    ranks the sources; the text fed is the topic-scoped excerpts that matched. Wiki, if
    given, prepends the top item."""
    grouped: dict[str, dict] = {}
    for h in hits:
        sid = h["source"]
        if sid not in sid2date:        # only real haystack sessions resolve a date
            continue
        g = grouped.setdefault(sid, {"topics": [], "raws": [], "score": 0.0})
        if h["topic"] not in g["topics"]:
            g["topics"].append(h["topic"])
        if h["content"] not in g["raws"]:
            g["raws"].append(h["content"])
        g["score"] = max(g["score"], h.get("score", 0.0))
    ranked = sorted(grouped.items(), key=lambda kv: kv[1]["score"], reverse=True)[:topk]
    items = []
    for sid, g in ranked:
        label = f"[topics: {', '.join(t for t in g['topics'] if t)}]\n" if any(g["topics"]) else ""
        items.append({"corpus_id": sid, "text": label + "\n".join(g["raws"]),
                      "timestamp": sid2date[sid], "score": round(g["score"], 4)})
    if items and wiki:
        items[0]["text"] = f"[MEMORY WIKI — relevant topics]\n{wiki}\n\n" + items[0]["text"]
    return items


def _wiki_lines(sel):
    """Selected topics' wiki summaries split into BM25-able lines + the full-wiki blob."""
    lines, blobs = [], []
    for t in sel:
        s = store.read_wiki(t)
        if not s:
            continue
        blobs.append(f"## {t}\n{s}")
        for ln in s.splitlines():
            ln = ln.strip()
            if ln.startswith("- "):
                lines.append({"topic": t, "content": ln[2:].strip(), "source": "wiki"})
    return lines, "\n\n".join(blobs)


def _wiki_item(text, entry, sid2date):
    """Wiki context as one official item (carried on a real session id for the date lookup;
    the date itself is irrelevant to a summary)."""
    carrier = entry["haystack_session_ids"][0]
    return [{"corpus_id": carrier, "text": f"[MEMORY WIKI — relevant topics]\n{text}",
             "timestamp": sid2date[carrier], "score": 0.0}]


def recall(client, model, entry, topk, top_topics=5, limit_entries=60):
    """Cascading topic-driven recall (the theme-memory design):
      1. pick relevant topics via the MODEL (BM25 rank_topics is the fallback);
      2a. BM25 within the wiki summaries — if those snippets suffice, stop;
      2b. else feed the FULL wiki — if that suffices, stop;
      3. else escalate to the selected topics' RAW logs (BM25-selected entries);
      4. else fall back to BM25 over the lossless full-session floor (== the bm25 baseline,
         so theme is a strict superset of bm25 and can never recall worse here).
    Each escalation is gated by a sufficiency probe. Returns (ranked_items, sel, tier)."""
    q = entry["question"]
    sid2date = dict(zip(entry["haystack_session_ids"], entry["haystack_dates"]))
    sel = (_route_topics_for_query(client, model, q, store.index_items())
           or [t for t, s in retr.rank_topics(q) if s > 0][:top_topics]
           or store.list_topics())
    wiki_lines, full_wiki = _wiki_lines(sel)
    # cross-topic global wiki: always in view, so facts fragmented across topics still reunite
    gwiki = store.read_root_wiki()
    if gwiki:
        full_wiki = (gwiki + "\n\n" + full_wiki) if full_wiki else gwiki
        for ln in gwiki.splitlines():
            ln = ln.strip()
            if ln.startswith("- "):
                wiki_lines.append({"topic": "GLOBAL", "content": ln[2:].strip(),
                                   "source": "root-wiki"})

    # tier 1 — wiki layer: the MODEL decides BM25-search vs full-load for this question (not a
    # forced rule). If the chosen path doesn't suffice, fall through to the raw/floor tiers.
    if _wiki_mode(client, model, q) == "bm25":
        wh = retr.bm25(q, wiki_lines, limit=limit_entries) if wiki_lines else []
        if wh:
            snippets = "\n".join(f"[{h['topic']}] {h['content']}" for h in wh)
            if _probe_sufficient(client, model, q, snippets):
                return _wiki_item(snippets, entry, sid2date), sel, "wiki_bm25"
    elif full_wiki and _probe_sufficient(client, model, q, full_wiki):
        return _wiki_item(full_wiki, entry, sid2date), sel, "wiki_full"

    # tier 2 — wiki + selected topics' raw logs
    items2 = _items_from_hits(retr.bm25(q, store.all_entries(sel), limit=limit_entries),
                              sid2date, topk, wiki=full_wiki)
    if items2 and _probe_sufficient(client, model, q, "\n\n".join(it["text"] for it in items2)):
        return items2, sel, "topic_raw"

    # tier 3 — bm25 floor: BM25 over the lossless FULL-session corpus (exactly what a flat
    # session-BM25 baseline indexes), so extraction loss can never drop recall below bm25
    floor = store.session_entries()
    hits = retr.bm25(q, floor, limit=limit_entries)
    if not hits:
        hits = sorted(floor, key=lambda e: e["ts"], reverse=True)[:limit_entries]
        for h in hits:
            h.setdefault("score", 0.0)
    items3 = _items_from_hits(hits, sid2date, topk, wiki=full_wiki)
    if not items3:  # empty store — avoid the reader's empty-history assert
        sid0 = entry["haystack_session_ids"][0]
        items3 = [{"corpus_id": sid0, "text": "(no memory extracted)",
                   "timestamp": sid2date[sid0], "score": 0.0}]
    return items3, sel, "full_raw"


def build_out(entry, ranked_items):
    out = {k: entry[k] for k in (
        "question_id", "question_type", "question", "answer", "question_date",
        "haystack_dates", "haystack_sessions", "haystack_session_ids", "answer_session_ids")}
    out["retrieval_results"] = {"query": entry["question"], "ranked_items": ranked_items,
                                "metrics": {"session": {}, "turn": {}}}
    return out


def select(data, limit, stratified, qids=None):
    if qids:
        want = set(qids)
        return [e for e in data if e["question_id"] in want]
    if not limit:
        return data
    if not stratified:
        return data[:limit]
    from collections import defaultdict
    by = defaultdict(list)
    for e in data:
        by[e["question_type"]].append(e)
    picked, i = [], 0
    while len(picked) < limit and any(i < len(v) for v in by.values()):
        for t in CORE_TYPES:
            if i < len(by[t]) and len(picked) < limit:
                picked.append(by[t][i])
        i += 1
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", default=os.path.join(HERE, "data", "longmemeval_s_cleaned.json"))
    p.add_argument("--out", required=True, help="official retrieval-log output path (jsonl)")
    p.add_argument("--stores", default=os.path.join(HERE, "official_out", "theme_stores"))
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--stratified", action="store_true")
    p.add_argument("--qids", help="comma-separated question_ids to run (overrides limit)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--model", default="MiniMax-M3")
    p.add_argument("--base_url", default="https://api.minimaxi.com/v1")
    p.add_argument("--key_file", default=os.path.expanduser("~/.minimax_key"))
    args = p.parse_args()

    client = OpenAI(api_key=open(args.key_file).read().strip(), base_url=args.base_url,
                    timeout=300.0)
    qids = [s.strip() for s in args.qids.split(",")] if args.qids else None
    data = select(json.load(open(args.in_file, encoding="utf-8")), args.limit, args.stratified, qids)

    trace_path = args.out + ".themes.jsonl"
    fo = open(args.out, "w", encoding="utf-8")        # rewritten each launch: recall is cheap and
    ft = open(trace_path, "w", encoding="utf-8")      # ingest is cached via the .ingested marker

    print(f"theme-memory build: {len(data)} questions, topk={args.topk}", flush=True)
    for n, entry in enumerate(data, 1):
        qid = entry["question_id"]
        sdir = os.path.join(args.stores, qid)
        os.environ["HANK_MEMORY_DIR"] = sdir
        marker = os.path.join(sdir, ".ingested")
        try:
            if os.path.exists(marker):        # store already fully ingested -> reuse
                nraw, ns, nfail, cached = 0, 0, 0, True
            else:
                if os.path.isdir(sdir):       # wipe any partial/crashed store
                    shutil.rmtree(sdir)
                nraw, ns, nfail = ingest(client, args.model, entry)
                nmerged = consolidate(client, args.model)   # fold near-duplicate topics
                if nmerged:
                    print(f"  consolidated: -{nmerged} topics", flush=True)
                cached = False
                open(marker, "w").close()     # mark ingest complete
            items, sel, tier = recall(client, args.model, entry, args.topk)
        except Exception as e:                # one bad question must not kill the batch
            print(f"[{n}/{len(data)}] {qid} FAILED, skipping: {e!r}", flush=True)
            continue
        fo.write(json.dumps(build_out(entry, items), ensure_ascii=False) + "\n"); fo.flush()
        ft.write(json.dumps({"question_id": qid, "question_type": entry["question_type"],
                             "n_raw_entries": nraw, "n_topics": len(store.index_items()),
                             "n_sessions_routed": ns, "n_failed_sessions": nfail, "cached": cached,
                             "topics": store.index_items(), "recall_topics": sel,
                             "recall_tier": tier, "n_recalled_sessions": len(items)},
                            ensure_ascii=False) + "\n"); ft.flush()
        print(f"[{n}/{len(data)}] {qid} ({entry['question_type']}): "
              f"{'cached store' if cached else f'{nraw} raw / {ns} sessions routed'} / "
              f"{len(store.index_items())} topics -> tier={tier}, recalled {len(items)} sessions",
              flush=True)

    fo.close(); ft.close()
    print(f"done -> {args.out}\nthemes trace -> {trace_path}\nstores -> {args.stores}", flush=True)


if __name__ == "__main__":
    main()
