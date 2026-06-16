"""Adapter: run OUR theme-memory BM25 over LongMemEval haystacks and emit the
OFFICIAL retrieval-log schema that vendor/LongMemEval/src/generation/run_generation.py
consumes verbatim. Lets us measure our retriever inside the official harness instead
of a hand-rolled one, so numbers compare horizontally with the LongMemEval board and
with mem0.

Two outputs per run:
  1. <out>                  official retrieval log (jsonl): each line is the dataset
                            entry + `retrieval_results.ranked_items=[{corpus_id,text,
                            timestamp,score}]`, ranked by our BM25. Feed this straight
                            to run_generation.py --retriever_type flat-session.
  2. <out>.trace.jsonl      per-question point-wise I/O trace for ERROR ATTRIBUTION:
                            query -> candidate sessions -> BM25 scores -> top-k pick ->
                            where each evidence session ranked. A wrong final answer can
                            then be split into retrieval-miss vs generation fault.

INPUT is the real haystack longmemeval_s_cleaned.json (~115k-token, ~40-60 sessions with
distractors) — the actual retrieval task. Corpus doc = one per session, user turns only
(matching official run_retrieval session granularity). corpus_id = the literal
haystack_session_id (run_generation looks it up unchanged). Evidence sessions are taken
directly from `answer_session_ids` (present in every file); distractor ids are
`sharegpt_*`/`ultrachat_*` and never marked correct, so no has_answer flag is needed for
session-level recall. (longmemeval_oracle.json is the evidence-only upper-bound variant,
NOT the retrieval task — don't use it here.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "theme_memory"))
import retrieve as retr  # noqa: E402  our pure-Python BM25 engine

CORE_TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "temporal-reasoning", "knowledge-update"]
TRACE_KS = [1, 3, 5, 10, 30, 50]


def session_corpus(entry):
    """Official session-granularity index: returns (texts, corpus_ids).

    text = user turns joined (what gets scored); corpus_id = the literal
    haystack_session_id (run_generation looks it up unchanged)."""
    texts, ids = [], []
    for sid, sess, _ts in zip(entry["haystack_session_ids"], entry["haystack_sessions"],
                              entry["haystack_dates"]):
        text = " ".join(t["content"] for t in sess if t["role"] == "user")
        texts.append(text)
        ids.append(sid)
    return texts, ids


def rank_ours(query, texts):
    """Full ranking of session indices by our BM25. Positive-score docs first (desc),
    then the remaining (zero-score) docs in original order — mirrors argsort while
    keeping every session present in ranked_items, like the official flat retriever."""
    entries = [{"content": t, "topic": "", "_idx": i} for i, t in enumerate(texts)]
    scored = retr.bm25(query, entries, limit=len(entries))  # drops score<=0
    order = [(e["_idx"], e["score"]) for e in scored]
    seen = {i for i, _ in order}
    order += [(i, 0.0) for i in range(len(texts)) if i not in seen]
    return order  # list of (session_idx, score)


def rank_official_bm25(query, texts):
    """Optional baseline: the official rank_bm25.BM25Okapi with naive whitespace
    tokenization (only if rank_bm25 is installed)."""
    from rank_bm25 import BM25Okapi
    import numpy as np
    bm25 = BM25Okapi([t.split(" ") for t in texts])
    scores = bm25.get_scores(query.split(" "))
    order = list(np.argsort(scores)[::-1])
    return [(int(i), float(scores[i])) for i in order]


def build(entry, retriever):
    texts, ids = session_corpus(entry)
    dates = entry["haystack_dates"]
    query = entry["question"]
    order = rank_ours(query, texts) if retriever == "ours" else rank_official_bm25(query, texts)

    ranked_items = [
        {"corpus_id": ids[i], "text": texts[i], "timestamp": dates[i], "score": round(s, 4)}
        for i, s in order
    ]
    out = {k: entry[k] for k in (
        "question_id", "question_type", "question", "answer", "question_date",
        "haystack_dates", "haystack_sessions", "haystack_session_ids", "answer_session_ids")}
    out["retrieval_results"] = {"query": query, "ranked_items": ranked_items,
                                "metrics": {"session": {}, "turn": {}}}

    # --- point-wise trace for error attribution ---
    evidence = sorted(set(entry.get("answer_session_ids", [])))
    is_abs = "_abs" in entry["question_id"]
    ranked_ids = [it["corpus_id"] for it in ranked_items]
    rank_of = {cid: r for r, cid in enumerate(ranked_ids, 1)}
    evidence_ranks = {e: rank_of.get(e) for e in evidence}

    recall = {}
    for k in TRACE_KS:
        topk = set(ranked_ids[:k])
        hit = [e for e in evidence if e in topk]
        recall[f"any@{k}"] = int(bool(hit)) if evidence else None
        recall[f"all@{k}"] = int(len(hit) == len(evidence)) if evidence else None

    k0 = 10  # the cutoff we actually generate at
    in_topk = [e for e in evidence if (evidence_ranks[e] or 1e9) <= k0]
    if is_abs:
        stage = "abstention"          # no evidence by design; downstream should abstain
    elif not evidence:
        stage = "no_evidence_in_corpus"
    elif len(in_topk) == len(evidence):
        stage = "retrieval_ok"        # all evidence within top-k -> any wrong answer is generation
    elif in_topk:
        stage = "retrieval_partial"   # some evidence missing from top-k
    else:
        stage = "retrieval_miss"      # no evidence reached top-k

    trace = {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "is_abstention": is_abs,
        "query": query,
        "n_candidate_sessions": len(texts),
        "n_evidence_sessions": len(evidence),
        "evidence_ranks": evidence_ranks,          # corpus_id -> rank (None = absent)
        "topk_at_10": [{"rank": r, "corpus_id": cid, "score": ranked_items[r - 1]["score"],
                        "is_evidence": cid in evidence} for r, cid in enumerate(ranked_ids[:k0], 1)],
        "recall": recall,
        "failure_stage_if_wrong": stage,
    }
    return out, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", default=os.path.join(HERE, "data", "longmemeval_s_cleaned.json"))
    p.add_argument("--out", required=True, help="official retrieval-log output path")
    p.add_argument("--retriever", default="ours", choices=["ours", "official-bm25"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stratified", action="store_true")
    args = p.parse_args()

    data = json.load(open(args.in_file, encoding="utf-8"))
    if args.limit:
        if args.stratified:
            from collections import defaultdict
            by = defaultdict(list)
            for e in data:
                by[e["question_type"]].append(e)
            picked, i = [], 0
            while len(picked) < args.limit and any(i < len(v) for v in by.values()):
                for t in CORE_TYPES:
                    if i < len(by[t]) and len(picked) < args.limit:
                        picked.append(by[t][i])
                i += 1
            data = picked
        else:
            data = data[:args.limit]

    trace_path = args.out + ".trace.jsonl"
    stage_counts, recall_sum, recall_n = {}, {f"any@10": 0, f"all@10": 0}, 0
    with open(args.out, "w", encoding="utf-8") as fo, open(trace_path, "w", encoding="utf-8") as ft:
        for e in data:
            out, trace = build(e, args.retriever)
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            ft.write(json.dumps(trace, ensure_ascii=False) + "\n")
            stage_counts[trace["failure_stage_if_wrong"]] = stage_counts.get(
                trace["failure_stage_if_wrong"], 0) + 1
            if trace["recall"]["any@10"] is not None:
                recall_sum["any@10"] += trace["recall"]["any@10"]
                recall_sum["all@10"] += trace["recall"]["all@10"]
                recall_n += 1

    print(f"wrote {args.out} ({len(data)} questions, retriever={args.retriever})")
    print(f"wrote {trace_path}")
    print(f"retrieval recall_any@10={recall_sum['any@10']/recall_n:.3f}  "
          f"recall_all@10={recall_sum['all@10']/recall_n:.3f}  (n={recall_n}, non-abstention)")
    print("failure-stage distribution:", json.dumps(stage_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
