#!/usr/bin/env python3
"""theme-memory tools CLI. Run from the repo root:

    python3 theme_memory/scripts/cli.py <command> [args]

Commands:
  overview                              MEMORY_INDEX + all topic wikis (wiki-first)
  retrieve --query Q [--topic T] [--limit N]   BM25 recall of log entries
  append --topic T --content C [--source S] [--desc D] [--timestamp TS]
  summarize --topic T [--content C]     write wiki.md (agent text, else extractive)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store  # noqa: E402
import retrieve as retr  # noqa: E402


def cmd_overview(a):
    print(store.read_index_text())
    for t in store.list_topics():
        s = store.read_wiki(t)
        if s:
            print(f"\n--- wiki: {t} ---\n{s}")


def cmd_retrieve(a):
    hits = retr.retrieve(a.query, topic=a.topic, limit=a.limit)
    if not hits:
        print("(no matches)")
        return
    for h in hits:
        print(f"[{h['topic']} | {h['ts']} | {h['source']} | score={h['score']}]")
        print(h["content"])
        print()


def cmd_append(a):
    store.append(a.topic, a.content, source=a.source, desc=a.desc, timestamp=a.timestamp)
    print(f"appended to {a.topic}")


def cmd_summarize(a):
    content = a.content or store.extractive_wiki(a.topic)
    store.write_wiki(a.topic, content)
    print(f"wiki written for {a.topic}")


def main():
    p = argparse.ArgumentParser(prog="theme-memory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("overview")
    sp.set_defaults(fn=cmd_overview)

    sp = sub.add_parser("retrieve")
    sp.add_argument("--query", required=True)
    sp.add_argument("--topic")
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(fn=cmd_retrieve)

    sp = sub.add_parser("append")
    sp.add_argument("--topic", required=True)
    sp.add_argument("--content", required=True)
    sp.add_argument("--source", default="user")
    sp.add_argument("--desc")
    sp.add_argument("--timestamp")
    sp.set_defaults(fn=cmd_append)

    sp = sub.add_parser("summarize")
    sp.add_argument("--topic", required=True)
    sp.add_argument("--content")
    sp.set_defaults(fn=cmd_summarize)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
