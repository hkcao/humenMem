#!/usr/bin/env python3
"""UserPromptSubmit hook: inject the theme-memory topic-state block.

Reads the hook JSON on stdin, classifies the prompt's topic, and emits an
`additionalContext` block. Always exits 0 and stays silent on any error, so it can
never block or break the user's prompt.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "theme_memory", "scripts"))


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    prompt = (data.get("prompt") or "").strip()
    session_id = data.get("session_id") or "default"
    if not prompt:
        return
    try:
        import topic_state
        _, block = topic_state.step(session_id, prompt)
    except Exception:
        return
    if not block:
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": block,
        }
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
