"""Keyword/BM25 recall over theme-memory logs. Pure-Python, no dependencies."""
from __future__ import annotations

import math
import re
from collections import Counter

import store

# ascii word runs OR single CJK chars -> works for English and Chinese logs
_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25(query, entries, k1=1.5, b=0.75, limit=5) -> list[dict]:
    q_terms = tokenize(query)
    docs = [tokenize(e["content"] + " " + e["topic"]) for e in entries]
    n = len(docs)
    if n == 0 or not q_terms:
        return []
    avgdl = sum(len(d) for d in docs) / n
    df = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(q_terms)}

    scored = []
    for e, d in zip(entries, docs):
        tf = Counter(d)
        dl = len(d)
        score = 0.0
        for t in q_terms:
            freq = tf.get(t, 0)
            if not freq:
                continue
            score += idf[t] * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**e, "score": round(s, 4)} for s, e in scored[:limit]]


def retrieve(query, topic=None, limit=5) -> list[dict]:
    topics = [topic] if topic else None
    return bm25(query, store.all_entries(topics), limit=limit)


def rank_topics(query, limit=None) -> list[tuple]:
    """Rank whole topics by relevance to query (topic doc = wiki + log).

    BM25 fallback for topic selection; the model-based router is preferred when a
    client is available (see topic_state and eval/theme_to_official)."""
    topics = store.list_topics()
    if not topics:
        return []
    docs = [{"topic": t, "content": store.read_wiki(t) + "\n" + store.read_log(t)} for t in topics]
    ranked = bm25(query, docs, limit=limit or len(docs))
    return [(d["topic"], d["score"]) for d in ranked]
