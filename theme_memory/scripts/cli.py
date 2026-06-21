#!/usr/bin/env python3
"""theme-memory tools CLI. Run from the repo root:

    python3 theme_memory/scripts/cli.py <command> [args]

Commands:
  overview                              MEMORY_INDEX + root wiki + all topic wikis (wiki-first)
  retrieve --query Q [--topic T] [--limit N]   BM25 recall of log entries
  append --topic T --content C [--source S] [--desc D] [--timestamp TS]
  summarize --topic T [--content C]     overwrite a topic wiki (agent text, else extractive)
  wiki (--topic T | --root)             show the wiki with numbered lines; or LOCALLY edit it:
        [--append "fact"]... [--update N "new text"]... [--delete N]...
  merge --into CANONICAL --from a,b,c   fold near-duplicate topics into CANONICAL
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store  # noqa: E402
import retrieve as retr  # noqa: E402


def cmd_overview(a):
    print(store.read_index_text())
    rw = store.read_root_wiki()
    if rw:
        print(f"\n--- root wiki (cross-topic) ---\n{rw}")
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


def cmd_wiki(a):
    if not a.root and not a.topic:
        print("need --topic T or --root")
        return
    # no ops -> show the wiki with numbered lines (so you can pick lines to --update/--delete)
    if not (a.append or a.update or a.delete):
        text = store.read_root_wiki() if a.root else store.read_wiki(a.topic)
        bullets = store.wiki_bullets(text)
        if not bullets:
            print("(empty)")
            return
        for i, b in enumerate(bullets, 1):
            print(f"[{i}] {b}")
        return
    update = {}
    for n, t in (a.update or []):
        if str(n).lstrip("-").isdigit():
            update[int(n)] = t
    delete = {int(n) for n in (a.delete or [])}
    store.update_topic_wiki(a.topic, append=a.append, update=update, delete=delete,
                            root_wiki=a.root)
    print("wiki updated " + ("(root)" if a.root else f"({a.topic})"))


def cmd_merge(a):
    members = [m.strip() for m in a.from_.split(",") if m.strip()]
    store.merge_topics(a.into, [m for m in members if m != a.into])
    print(f"merged {members} -> {a.into}")


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

    sp = sub.add_parser("wiki")
    sp.add_argument("--topic")
    sp.add_argument("--root", action="store_true", help="target the cross-topic root wiki")
    sp.add_argument("--append", action="append", default=[], help="add a new bullet (repeatable)")
    sp.add_argument("--update", action="append", nargs=2, metavar=("LINE", "TEXT"), default=[],
                    help="revise an existing numbered line (repeatable)")
    sp.add_argument("--delete", action="append", type=int, default=[], metavar="LINE",
                    help="delete a numbered line (repeatable)")
    sp.set_defaults(fn=cmd_wiki)

    sp = sub.add_parser("merge")
    sp.add_argument("--into", required=True, help="canonical topic to keep")
    sp.add_argument("--from", dest="from_", required=True, help="comma-separated topics to fold in")
    sp.set_defaults(fn=cmd_merge)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
