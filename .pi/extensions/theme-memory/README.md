# theme-memory

A [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) extension that gives an agent
**topic-based long-term memory with theme eviction** — so a single window can hold a long, multi-topic
conversation without lossy auto-compaction. Full design rationale is in [`DESIGN.md`](../../../DESIGN.md).

The core idea: only one theme's detail lives in the model's context at a time. Other themes are evicted
to disk and recalled on demand. A small **core memory** (cross-theme experience/judgment) stays resident.

## What it does

| DESIGN.md | Mechanism (pi hook) |
|---|---|
| §2 startup load | `session_start` + `before_agent_start` inject `index.md` (core mem + theme index) and the active theme's summary into the system prompt |
| §3 theme routing | **keyword prefilter + LLM as the primary judge**; asks the user to confirm when the model is unsure |
| §4 eviction | `context` hook rewrites the message array: theme switch (event-driven) + memory pressure (threshold) |
| §5 append | `agent_end` appends each turn to `<theme>/<date>.md` (atomic write, per-theme lock) |
| §5/§6 maintenance | `/maintain`: daily summary head, `day→week→month→overall` roll-up, core-mem distillation |
| §8 relationships | `/relate` `/merge` `/split` + an auto link pass (keyword prefilter, **LLM decides**) |
| supersede | `memory_supersede` tool marks a fact superseded without deleting history |

## Install

```bash
# project-local (auto-discovered when pi runs in this repo):
#   .pi/extensions/theme-memory/        ← already here
# or global:
cp -r theme-memory ~/.pi/agent/extensions/

# or load explicitly for a one-off:
pi -e ./.pi/extensions/theme-memory/index.ts
```

The extension needs a model with an API key configured in pi (used for routing and maintenance).

## Memory layout

```
$HANK_MEMORY_DIR/                 # default ~/hank_memory
├── index.md                      # resident: # Core Memory  +  # Themes (with [related: …])
└── <theme>/
    ├── YYYY-MM-DD.md             # raw turns; head holds the ~200-token daily summary
    └── summary.md                # # Overall / # Month / # Week / # Day  (hierarchical)
```

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `HANK_MEMORY_DIR` | `~/hank_memory` | memory root (filesystem root `/hank_memory` needs sudo) |
| `THEME_KEEP_RECENT` | `6` | messages always kept regardless of theme (continuity window) |
| `THEME_MAX_KEPT` | off | cap on kept messages; over it, oldest non-recent are pressure-evicted |
| `THEME_LLM_ROUTING` | on | set `0` to fall back to keyword-only routing (no per-turn LLM call) |
| `THEME_STAY_THRESHOLD` | `0.34` | keyword score above which a turn is judged "same topic" and skips the LLM call |
| `THEME_RELATE_PREFILTER` | `0.1` | keyword overlap a theme pair must clear before the LLM is asked to relate them |
| `THEME_MAINTAIN_ON_EXIT` | off | set `1` to run maintenance on `session_shutdown` (heavy — uses the LLM) |

## Commands

| Command | What |
|---|---|
| `/mem` | show root, active theme, theme list, eviction stats |
| `/theme [name]` | show or switch the active theme |
| `/maintain [theme]` | run maintenance for one theme (or all): daily summary + roll-up + core-mem distill + related links |
| `/relate <a> <b>` | add a bidirectional related link |
| `/merge <a> <b> <new>` | merge two themes into a new one (files moved, summaries re-rolled, old dirs left a redirect) |
| `/split <theme> <new> <keyword>` | move daily files containing `<keyword>` into a new theme |

## Tools (LLM-callable)

- `set_theme(theme)` — switch active theme (eviction follows on the next call)
- `memory_recall(theme?, date?, keyword?)` — load detail from a theme's files (structured, no embeddings)
- `memory_supersede(date, note, theme?)` — record that a past fact has been replaced

## Background maintenance via cron

The agent has no daemon, so schedule maintenance externally (DESIGN.md §5):

```cron
0 0 * * *  cd /path/to/repo && HANK_MEMORY_DIR=~/hank_memory \
  pi -ne -e ./.pi/extensions/theme-memory/index.ts --no-session -p "/maintain"
```

## Testing

```bash
HANK_MEMORY_DIR=/tmp/tm-test node test-eviction.mjs   # offline, deterministic, no API cost
```
Covers theme + memory-pressure eviction, index/summary round-trips, isoWeek, atomic write, daily head, bidirectional links.

## Module map

- `index.ts` — wiring: events, tools, commands, the `maintain()` job
- `memory.mjs` — filesystem layer (index/summary parse, daily files, atomic write + lock, supersede, relatedness/merge/split)
- `summarize.mjs` — LLM prompt builders (daily, roll-up, core-mem, theme classify, relate)
- `evict.mjs` — pure eviction logic
- `test-eviction.mjs` — offline test suite

## Known limitations

- Theme routing adds one LLM call on turns that aren't an obvious continuation (gated by `THEME_STAY_THRESHOLD`).
- Maintenance roll-up regenerates week/month/overall from day summaries each run; cost scales with theme count.
- Merge/split are mechanical (file moves + re-roll on next maintain); review the result for large themes.
