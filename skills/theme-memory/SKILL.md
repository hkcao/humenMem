---
name: theme-memory
description: Topic-partitioned external memory for long conversations. Use to recall facts from earlier topics that may have scrolled out of the window, to persist notable facts/decisions under a topic, and to get a summary-first overview of what is stored. Trigger when the user refers to something discussed earlier, switches topics, or asks about prior context not currently in view.
---

# Theme Memory (Step 1: retrieval & recall)

A topic-partitioned external store under `~/.claude/hank_memory/` (override with the
`HANK_MEMORY_DIR` env var). Per topic, `log.md` is the **append-only source of truth**;
`summary.md` is a **rebuildable cache**; `MEMORY_INDEX.md` lists every topic.

Run all commands from the repo root:

    python3 theme_memory/cli.py <command> [args]

This step only **adds** context (recall); it never deletes or hides anything, so
retrieval is always safe to call.

## When to use

- **Recall** — the user refers to something from earlier ("that host we set up", "the
  approach from the auth thread") and it is not in the current window → `retrieve`.
- **Overview** — starting out, or unsure what topics exist → `overview`.
- **Persist** — a durable fact or decision worth keeping (a config value, a decision, a
  definition) → `append` under the right topic.
- **Refresh summary** — after appending several entries to a topic → `summarize`.

## Commands

Overview (index + all summaries):

    python3 theme_memory/cli.py overview

Recall via BM25 over logs (omit `--topic` to search all topics):

    python3 theme_memory/cli.py retrieve --query "staging database host" --limit 5
    python3 theme_memory/cli.py retrieve --query "api key" --topic deploy-prod

Append a fact (creates the topic + an index entry if new):

    python3 theme_memory/cli.py append --topic deploy-prod --source user \
      --content "Prod DB host is db-prod.internal:5432" --desc "Production deploy config"

Write/refresh a topic summary. You generate the summary text; omit `--content` for a
deterministic extractive fallback built from recent entries:

    python3 theme_memory/cli.py summarize --topic deploy-prod --content "Prod env: ..."

## Guidance

- Tag content to the **correct topic** on `append`. When retrieving across topics, read
  the `[topic | ts | source]` header before trusting a snippet — this is what prevents
  cross-topic mix-ups (e.g. staging vs prod config).
- Prefer `retrieve` over guessing when answering "what did we say about X".
