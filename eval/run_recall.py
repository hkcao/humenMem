#!/usr/bin/env python3
"""Step 1 verification: store multi-topic content, then check on-demand recall.

Runs against a throwaway memory dir (no API keys needed). Metrics:
  - hit@k:         the gold snippet appears in the top-k retrieved entries
  - topic-correct: the top-1 entry is from the gold topic (no cross-topic bleed)
Exits nonzero if either metric is below 100% on the probe set.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "theme_memory" / "scripts"))


def main():
    probes = json.loads((HERE / "recall_probes" / "probes.json").read_text(encoding="utf-8"))
    os.environ["HANK_MEMORY_DIR"] = tempfile.mkdtemp(prefix="hankmem_eval_")

    import store
    import retrieve as retr

    for e in probes["entries"]:
        store.append(e["topic"], e["content"], source=e.get("source", "user"), desc=e.get("desc"))

    k = 3
    hit = topic_ok = total = 0
    for q in probes["queries"]:
        total += 1
        hits = retr.retrieve(q["q"], limit=k)
        joined = " ".join(h["content"] for h in hits)
        h_ok = q["must_include"] in joined
        t_ok = bool(hits) and hits[0]["topic"] == q["gold_topic"]
        hit += h_ok
        topic_ok += t_ok
        top1 = hits[0]["topic"] if hits else None
        status = "OK  " if h_ok and t_ok else "FAIL"
        print(f"[{status}] q={q['q']!r} top1_topic={top1} hit@{k}={h_ok} topic_correct={t_ok}")

    print(f"\nhit@{k}: {hit}/{total}   topic-correct@1: {topic_ok}/{total}")
    ok = hit == total and topic_ok == total
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
