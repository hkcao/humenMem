#!/usr/bin/env python3
"""Step 2 verification: sticky topic tracking + fail-safe injection.

Feeds an interleaved sequence of prompts and checks:
  - routing: the chosen current topic matches the expected topic each turn
  - fail-safe: every injected block lists ALL known topics (nothing hidden)
Runs against a throwaway memory dir (no API keys). Exits nonzero on any miss.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "theme_memory" / "scripts"))

# (prompt, expected current topic after this turn)
SEQUENCE = [
    ("what's the staging database host?", "deploy-staging"),
    ("and the prod api key?", "deploy-prod"),
    ("how often do I feed the sourdough starter?", "recipe-notes"),
    ("remind me of the staging api key", "deploy-staging"),
    ("what about its database host again?", "deploy-staging"),  # sticky: vague follow-up stays
]


def main():
    probes = json.loads((HERE / "recall_probes" / "probes.json").read_text(encoding="utf-8"))
    os.environ["HANK_MEMORY_DIR"] = tempfile.mkdtemp(prefix="hankmem_routing_")

    import store
    import topic_state

    for e in probes["entries"]:
        store.append(e["topic"], e["content"], source=e.get("source", "user"), desc=e.get("desc"))
    all_topics = set(store.list_topics())

    session = "routing-test"
    route_ok = failsafe_ok = total = 0
    for prompt, expected in SEQUENCE:
        total += 1
        current, block = topic_state.step(session, prompt)
        r_ok = current == expected
        listed = {t for t in all_topics if f"[{t}" in block}
        f_ok = listed == all_topics
        route_ok += r_ok
        failsafe_ok += f_ok
        status = "OK  " if r_ok and f_ok else "FAIL"
        print(f"[{status}] prompt={prompt!r}\n        current={current} (expected {expected}) "
              f"route={r_ok} all_topics_listed={f_ok}")

    print(f"\nrouting: {route_ok}/{total}   fail-safe(all topics listed): {failsafe_ok}/{total}")
    ok = route_ok == total and failsafe_ok == total
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
