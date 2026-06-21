---
name: theme-memory
description: Topic-partitioned external memory for long conversations. Use to recall facts from earlier topics that may have scrolled out of the window, to persist notable facts/decisions under a topic, and to get a wiki-first overview of what is stored. Trigger when the user refers to something discussed earlier, switches topics, or asks about prior context not currently in view.
---

# Theme Memory

A topic-partitioned external store under `~/.claude/hank_memory/` (override with the
`HANK_MEMORY_DIR` env var). Per topic, `logs/<day>.md` (split by day) is the
**append-only source of truth**; `wiki.md` is a **rebuildable cache**. At the root,
`log.md` is a **global timeline**, `wiki.md` is a **cross-topic** view, and
`MEMORY_INDEX.md` lists every topic.

Run all commands from the repo root:

    python3 theme_memory/scripts/cli.py <command> [args]

`retrieve` / `append` are **additive** (recall is always safe — it never deletes or hides).
`wiki` (local edit) and `merge` are **explicit maintenance** ops you invoke deliberately.

## When to use

- **Recall** — the user refers to something from earlier ("that host we set up", "the
  approach from the auth thread") and it is not in the current window → `retrieve`.
- **Overview** — starting out, or unsure what topics exist → `overview`.
- **Persist** — a durable fact or decision worth keeping (a config value, a decision, a
  definition) → `append` under the right topic.
- **Maintain a wiki** — keep a topic's (or the cross-topic root) wiki current with a *local*
  edit: add new facts, revise a superseded line, delete a completed/obsolete one → `wiki`.
- **De-fragment** — when several topics turn out to be the same thing (e.g. `farm-ops` +
  `farm-maint`) → `merge` them into one.
- **Refresh wiki (whole)** — regenerate a topic wiki from scratch → `summarize`.

## Commands

Overview (index + all topic wikis):

    python3 theme_memory/scripts/cli.py overview

Recall via BM25 over logs (omit `--topic` to search all topics):

    python3 theme_memory/scripts/cli.py retrieve --query "staging database host" --limit 5
    python3 theme_memory/scripts/cli.py retrieve --query "api key" --topic deploy-prod

Append a fact (creates the topic + an index entry if new):

    python3 theme_memory/scripts/cli.py append --topic deploy-prod --source user \
      --content "Prod DB host is db-prod.internal:5432" --desc "Production deploy config"

Locally edit a wiki (preferred over a full rewrite — untouched lines are preserved). Show it
with numbered lines first, then add / revise / delete by line number (`--root` targets the
cross-topic root wiki instead of a topic):

    python3 theme_memory/scripts/cli.py wiki --topic deploy-prod                 # show numbered
    python3 theme_memory/scripts/cli.py wiki --topic deploy-prod \
      --append "Staging DB host is db-stg.internal:5432" \
      --update 2 "Prod region is now eu-west-1 (was us-east-1)" \
      --delete 5
    python3 theme_memory/scripts/cli.py wiki --root --append "PENDING: rotate prod API key"

Merge near-duplicate topics into one canonical topic (moves their logs, combines their wikis):

    python3 theme_memory/scripts/cli.py merge --into farm --from "farm-ops,farm-maint"

Whole-wiki rewrite (you supply the text; omit `--content` for an extractive fallback):

    python3 theme_memory/scripts/cli.py summarize --topic deploy-prod --content "Prod env: ..."

## Guidance

- Tag content to the **correct topic** on `append`. When retrieving across topics, read
  the `[topic | ts | source]` header before trusting a snippet — this is what prevents
  cross-topic mix-ups (e.g. staging vs prod config).
- Prefer `retrieve` over guessing when answering "what did we say about X".
- Prefer `wiki` (local add/revise/delete) over `summarize` (full rewrite) to keep a wiki
  current — local edits never silently drop a fact you didn't touch.
