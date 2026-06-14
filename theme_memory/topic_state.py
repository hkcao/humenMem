"""Step 2: per-turn topic-state structure (additive, fail-safe).

Decides the current topic for an incoming prompt (sticky: stay unless another topic
clearly wins) and renders a small block injected before each user turn:

  - current-topic anchor (a hint, not a gate)
  - EVERY known topic, tagged with its one-line description (nothing is hidden, so a
    routing miss is never catastrophic — the model can self-correct)
  - a "don't misattribute facts across topics" reminder

This treats reference-ambiguity, stale-anchoring, and source-misattribution without any
deletion. Cross-topic entity bleed (co-presence) is left for a later step.
"""
from __future__ import annotations

import json
import re

import store
import retrieve as retr

# how much a rival topic must beat the current one (by BM25 score) to steal focus
SWITCH_MARGIN = 1.3
# max topics listed in the injected block (current is always included)
MAX_LISTED = 6


def _session_file(session_id: str):
    d = store.root() / "session"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) or "default"
    return d / f"{safe}.json"


def load_state(session_id: str):
    p = _session_file(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("current_topic")
        except Exception:
            return None
    return None


def save_state(session_id: str, current_topic) -> None:
    _session_file(session_id).write_text(
        json.dumps({"current_topic": current_topic}, ensure_ascii=False), encoding="utf-8"
    )


def decide_topic(prompt: str, current):
    """Return (current_topic, ranked) applying stickiness."""
    ranked = retr.rank_topics(prompt)
    if not ranked:
        return current, ranked
    best_t, best_s = ranked[0]
    if current is None:
        return best_t, ranked
    cur_s = dict(ranked).get(current, 0.0)
    if best_t != current and best_s > cur_s * SWITCH_MARGIN:
        return best_t, ranked
    return current, ranked


def render_block(current, ranked) -> str:
    items = store.index_items()
    if not items:
        return ""
    # list by relevance, but never drop the current topic
    order = [t for t, _ in ranked if t in items]
    for t in items:
        if t not in order:
            order.append(t)
    order = order[:MAX_LISTED]
    if current and current in items and current not in order:
        order.append(current)

    lines = [
        f"[主题记忆] 当前主题 = {current or '(未确定)'}(若本轮已切换主题,请按下方标签自行纠正)",
        "已知主题(各带描述,事实勿张冠李戴):",
    ]
    for t in order:
        mark = "(当前)" if t == current else ""
        lines.append(f"- [{t}{mark}] {items.get(t, '(无描述)')}")
    lines.append("提示:回答前先确认事实属于哪个主题;需要细节用 theme-memory 的 retrieve 召回。")
    return "\n".join(lines)


def step(session_id: str, prompt: str):
    """Update session state for this prompt; return (current_topic, block)."""
    if not store.list_topics():
        return None, ""
    current = load_state(session_id)
    current, ranked = decide_topic(prompt, current)
    save_state(session_id, current)
    return current, render_block(current, ranked)
