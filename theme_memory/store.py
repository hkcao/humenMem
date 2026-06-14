"""External theme-memory store: append-only logs + rebuildable summaries + index.

Layout (root = $HANK_MEMORY_DIR or ~/.claude/hank_memory):
  MEMORY_INDEX.md         topic list + one-line descriptions
  <topic>/log.md          append-only entries, each tagged with timestamp + source
  <topic>/summary.md      rebuildable cache (agent-written or extractive summary)

log.md is the source of truth; summary.md is a cache that can be regenerated.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

INDEX = "MEMORY_INDEX.md"
ENTRY_RE = re.compile(r"^## (?P<ts>\S+) \| source=(?P<source>\S+)\s*$")
INDEX_RE = re.compile(r"- \*\*(?P<topic>.+?)\*\* — (?P<desc>.*)")


def root() -> Path:
    env = os.environ.get("HANK_MEMORY_DIR")
    base = Path(env) if env else Path.home() / ".claude" / "hank_memory"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _topic_dir(topic: str) -> Path:
    d = root() / topic
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_topics() -> list[str]:
    return sorted(p.name for p in root().iterdir() if p.is_dir())


# --- logs (source of truth) -------------------------------------------------

def append(topic, content, source="user", desc=None, timestamp=None) -> None:
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    log = _topic_dir(topic) / "log.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"## {ts} | source={source}\n{content.strip()}\n\n")
    _ensure_index(topic, desc)


def read_log(topic: str) -> str:
    log = root() / topic / "log.md"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def parse_entries(topic: str) -> list[dict]:
    """Parse a topic log into entries: {topic, ts, source, content}."""
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


# --- summaries (rebuildable cache) ------------------------------------------

def read_summary(topic: str) -> str:
    p = root() / topic / "summary.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_summary(topic, content) -> None:
    _topic_dir(topic)
    (root() / topic / "summary.md").write_text(content.strip() + "\n", encoding="utf-8")


def extractive_summary(topic, max_entries=8) -> str:
    """Deterministic fallback summary built from the most recent log entries."""
    recent = parse_entries(topic)[-max_entries:]
    lines = [f"# {topic} (extractive summary of last {len(recent)} entries)", ""]
    for e in recent:
        first = e["content"].splitlines()[0] if e["content"] else ""
        lines.append(f"- [{e['ts']} {e['source']}] {first}")
    return "\n".join(lines)
