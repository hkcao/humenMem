"""External theme-memory store: append-only logs + rebuildable wiki + index.

Layout (root = $HANK_MEMORY_DIR or ~/.claude/hank_memory):
  MEMORY_INDEX.md         topic list + one-line descriptions
  log.md                  GLOBAL timeline: every topic's entries, in write order
  <topic>/logs/<day>.md   append-only entries split by day (YYYY-MM-DD)
  <topic>/wiki.md         rebuildable cache (agent-written or extractive)

The per-topic day logs are the source of truth; log.md is a chronological mirror and
wiki.md is a cache that can be regenerated. Reads fall back to the pre-split single
<topic>/log.md and legacy <topic>/summary.md so older stores keep working.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

INDEX = "MEMORY_INDEX.md"
ROOT_LOG = "log.md"
SESSIONS = "_sessions"          # reserved: lossless full-session floor (bm25-equivalent corpus)
RESERVED = {"session", SESSIONS}  # dirs that are not routing topics
ENTRY_RE = re.compile(r"^## (?P<ts>\S+) \| source=(?P<source>\S+)\s*$")
ROOT_ENTRY_RE = re.compile(r"^## (?P<ts>\S+) \| topic=(?P<topic>.+?) \| source=(?P<source>\S+)\s*$")
INDEX_RE = re.compile(r"- \*\*(?P<topic>.+?)\*\* — (?P<desc>.*)")
_DAY = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})")


def root() -> Path:
    env = os.environ.get("HANK_MEMORY_DIR")
    base = Path(env) if env else Path.home() / ".claude" / "hank_memory"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _topic_dir(topic: str) -> Path:
    d = root() / topic
    d.mkdir(parents=True, exist_ok=True)
    return d


def _day_key(ts: str) -> str:
    """Day bucket for a timestamp: '2023/05/23 (Tue) 02:16' or ISO -> '2023-05-23'."""
    m = _DAY.search(str(ts))
    return f"{m[1]}-{m[2]}-{m[3]}" if m else "undated"


def list_topics() -> list[str]:
    return sorted(p.name for p in root().iterdir() if p.is_dir() and p.name not in RESERVED)


# --- logs (source of truth) -------------------------------------------------

def append(topic, content, source="user", desc=None, timestamp=None) -> None:
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    body = content.strip()
    logs = _topic_dir(topic) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{_day_key(ts)}.md").open("a", encoding="utf-8") as f:
        f.write(f"## {ts} | source={source}\n{body}\n\n")
    # global timeline: same entry, topic-tagged, appended in write order
    with (root() / ROOT_LOG).open("a", encoding="utf-8") as f:
        f.write(f"## {ts} | topic={topic} | source={source}\n{body}\n\n")
    _ensure_index(topic, desc)


def read_log(topic: str) -> str:
    """Concatenate a topic's day logs in date order (fallback: legacy single log.md)."""
    d = root() / topic / "logs"
    if d.is_dir():
        parts = [p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))]
        if parts:
            return "\n".join(parts)
    legacy = root() / topic / "log.md"
    return legacy.read_text(encoding="utf-8") if legacy.exists() else ""


def read_root_log() -> str:
    """The global chronological timeline across all topics."""
    p = root() / ROOT_LOG
    return p.read_text(encoding="utf-8") if p.exists() else ""


def parse_entries(topic: str) -> list[dict]:
    """Parse a topic's logs into entries: {topic, ts, source, content}."""
    entries: list[dict] = []
    cur: dict | None = None
    for line in read_log(topic).splitlines():
        m = ENTRY_RE.match(line)
        if m:
            if cur:
                entries.append(cur)
            cur = {"topic": topic, "ts": m["ts"], "source": m["source"], "content": ""}
        elif cur is not None:
            cur["content"] += line + "\n"
    if cur:
        entries.append(cur)
    for e in entries:
        e["content"] = e["content"].strip()
    return entries


def all_entries(topics=None) -> list[dict]:
    out: list[dict] = []
    for t in (topics or list_topics()):
        out.extend(parse_entries(t))
    return out


# --- lossless session floor (bm25-equivalent corpus) ------------------------
# Every raw session is stored verbatim here, independent of topic routing, so a recall
# fallback over session_entries() covers exactly what a flat session-BM25 baseline would
# — extraction can never drop a session below this floor.

def append_session(sid, text, timestamp=None) -> None:
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    logs = root() / SESSIONS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{_day_key(ts)}.md").open("a", encoding="utf-8") as f:
        f.write(f"## {ts} | source={sid}\n{text.strip()}\n\n")


def session_entries() -> list[dict]:
    """Parse the full-session floor into entries: {topic:'', ts, source=sid, content}."""
    d = root() / SESSIONS / "logs"
    if not d.is_dir():
        return []
    entries: list[dict] = []
    cur: dict | None = None
    for p in sorted(d.glob("*.md")):
        for line in p.read_text(encoding="utf-8").splitlines():
            m = ENTRY_RE.match(line)
            if m:
                if cur:
                    entries.append(cur)
                cur = {"topic": "", "ts": m["ts"], "source": m["source"], "content": ""}
            elif cur is not None:
                cur["content"] += line + "\n"
    if cur:
        entries.append(cur)
    for e in entries:
        e["content"] = e["content"].strip()
    return entries


# --- index ------------------------------------------------------------------

def _read_index() -> dict:
    p = root() / INDEX
    if not p.exists():
        return {}
    items = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = INDEX_RE.match(line)
        if m:
            items[m["topic"]] = m["desc"]
    return items


def _write_index(items: dict) -> None:
    lines = ["# Memory Index", ""]
    lines += [f"- **{t}** — {items[t]}" for t in sorted(items)]
    (root() / INDEX).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_index(topic, desc) -> None:
    items = _read_index()
    if topic not in items or (desc and items[topic] != desc):
        items[topic] = desc or items.get(topic, "(no description)")
        _write_index(items)


def read_index_text() -> str:
    p = root() / INDEX
    return p.read_text(encoding="utf-8") if p.exists() else "# Memory Index\n\n(empty)\n"


def index_items() -> dict:
    """Topic -> one-line description, as recorded in MEMORY_INDEX.md."""
    return _read_index()


# --- wiki (rebuildable cache) -----------------------------------------------

def read_wiki(topic: str) -> str:
    p = root() / topic / "wiki.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    legacy = root() / topic / "summary.md"      # pre-rename stores
    return legacy.read_text(encoding="utf-8") if legacy.exists() else ""


def write_wiki(topic, content) -> None:
    _topic_dir(topic)
    (root() / topic / "wiki.md").write_text(content.strip() + "\n", encoding="utf-8")


def extractive_wiki(topic, max_entries=8) -> str:
    """Deterministic fallback wiki built from the most recent log entries."""
    recent = parse_entries(topic)[-max_entries:]
    lines = [f"# {topic} (extractive wiki of last {len(recent)} entries)", ""]
    for e in recent:
        first = e["content"].splitlines()[0] if e["content"] else ""
        lines.append(f"- [{e['ts']} {e['source']}] {first}")
    return "\n".join(lines)


# --- local wiki edits (add / delete / modify, not full rewrite) -------------
# Shared by the CLI and the eval scheme: the caller (an LLM — the agent, or MiniMax in eval)
# decides the ops; this applies them mechanically so untouched lines are preserved verbatim.

def wiki_bullets(text) -> list[str]:
    """The '- ...' bullet lines of a wiki blob (drops the '# title' header)."""
    return [ln.strip()[2:].strip() for ln in (text or "").splitlines()
            if ln.strip().startswith("- ")]


def render_wiki(title, bullets) -> str:
    return (f"{title}\n\n" + "\n".join(f"- {b}" for b in bullets)) if bullets else ""


def apply_wiki_ops(current_bullets, append=None, update=None, delete=None) -> list[str]:
    """Pure local add/delete/modify over bullet lines, by 1-based line number. `update` is a
    {line: text} map, `delete` a set of line numbers; lines not named are kept verbatim."""
    update = update or {}
    delete = set(delete or ())
    lines = [update.get(i, b) for i, b in enumerate(current_bullets, 1) if i not in delete]
    seen = {b.lower() for b in lines}
    for a in (append or []):
        a = str(a).strip()
        if a and a.lower() not in seen:
            lines.append(a)
            seen.add(a.lower())
    return lines


def update_topic_wiki(topic, append=None, update=None, delete=None, root_wiki=False) -> None:
    """Read → apply local ops → write, for a topic wiki (or the root wiki if root_wiki=True).
    For a topic wiki, any [[other-topic]] links it now contains get mirrored as backlinks."""
    current = read_root_wiki() if root_wiki else read_wiki(topic)
    bullets = apply_wiki_ops(wiki_bullets(current), append, update, delete)
    title = "# GLOBAL (cross-topic)" if root_wiki else f"# {topic}"
    text = render_wiki(title, bullets)
    if root_wiki:
        write_root_wiki(text)
    else:
        write_wiki(topic, text)
        sync_backlinks(topic)


# --- bidirectional wikilinks ([[topic]]) ------------------------------------
# A wiki line may reference a related topic inline as [[topic-name]]. After a topic wiki is
# written, sync_backlinks mirrors each forward link as a backlink on the target, so the two
# wikis interlink and you can jump either way. Additive + idempotent (fail-safe, like recall).

WIKILINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_BACKLINK = "↔ "          # marks an auto-maintained backlink bullet


def link_targets(text) -> set[str]:
    """Topic names referenced as [[name]] in a wiki blob."""
    return {m.strip() for m in WIKILINK.findall(text or "") if m.strip()}


def sync_backlinks(topic) -> list[str]:
    """For every [[X]] in `topic`'s wiki, ensure X's wiki links back to `topic` (append a
    backlink bullet if X doesn't already mention [[topic]]). Returns the topics it touched."""
    src = read_wiki(topic)
    if not src:
        return []
    existing = set(list_topics())
    touched = []
    for tgt in link_targets(src):
        if tgt == topic or tgt not in existing:   # only link real, other topics
            continue
        tgt_wiki = read_wiki(tgt)
        if f"[[{topic}]]" in tgt_wiki:            # already interlinked
            continue
        bullets = wiki_bullets(tgt_wiki)
        bullets.append(f"{_BACKLINK}[[{topic}]]")
        write_wiki(tgt, render_wiki(f"# {tgt}", bullets))
        touched.append(tgt)
    return touched


# --- root wiki (cross-topic global view) ------------------------------------

ROOT_WIKI = "wiki.md"


def read_root_wiki() -> str:
    p = root() / ROOT_WIKI
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_root_wiki(content) -> None:
    (root() / ROOT_WIKI).write_text(content.strip() + "\n", encoding="utf-8")


# --- topic consolidation ----------------------------------------------------

def merge_topics(canonical: str, members) -> None:
    """Fold member topics into `canonical`: move their day logs, concatenate wikis, drop the
    merged-away topic dirs, and rewrite the index. Mechanical only — a caller may re-synthesize
    the canonical wiki afterwards."""
    import shutil
    clogs = _topic_dir(canonical) / "logs"
    clogs.mkdir(parents=True, exist_ok=True)
    wikis = [w for w in [read_wiki(canonical)] if w]
    items = _read_index()
    for m in members:
        if m == canonical:
            continue
        mdir = root() / m
        if not mdir.is_dir():
            continue
        mlogs = mdir / "logs"
        if mlogs.is_dir():
            for f in sorted(mlogs.glob("*.md")):
                with (clogs / f.name).open("a", encoding="utf-8") as out:
                    out.write(f.read_text(encoding="utf-8"))
        legacy = mdir / "log.md"                      # pre-split stores
        if legacy.exists():
            with (clogs / "legacy.md").open("a", encoding="utf-8") as out:
                out.write(legacy.read_text(encoding="utf-8"))
        w = read_wiki(m)
        if w:
            wikis.append(w)
        items.pop(m, None)
        shutil.rmtree(mdir)
    if wikis:
        write_wiki(canonical, "\n\n".join(wikis))
    items.setdefault(canonical, "(merged)")
    _write_index(items)
