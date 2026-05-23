/**
 * theme-memory — full implementation of DESIGN.md (topic-based external memory
 * + theme eviction) as a pi extension. No harness source changes required.
 *
 *   §2 startup load        — session_start + before_agent_start inject index.md
 *                            (core mem + theme index) and the active theme summary
 *   §3 theme routing       — keyword heuristic; confirm with user when uncertain
 *   §4 eviction            — context hook: theme switch (event) + memory pressure
 *   §5 real-time append    — agent_end appends each turn (atomic, locked)
 *   §5/§6 maintenance      — /maintain: daily summary head, hierarchical roll-up,
 *                            core-mem distillation (run via cron: pi -p "/maintain")
 *   §8 relationships       — /relate, /merge, /split + auto related-link pass
 *   supersede, /mem, /theme, tools: set_theme, memory_recall, memory_supersede
 *
 * Memory root: $HANK_MEMORY_DIR else ~/hank_memory. Tunables:
 *   THEME_KEEP_RECENT (default 6), THEME_MAX_KEPT (default off),
 *   THEME_RELATE_THRESHOLD (default 0.25), THEME_MAINTAIN_ON_EXIT (default off).
 */
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { complete } from "@earendil-works/pi-ai";
import * as mem from "./memory.mjs";
import * as sum from "./summarize.mjs";
import { tagMessagesByTurn, evict, messageText } from "./evict.mjs";

const KEEP_RECENT = Number(process.env.THEME_KEEP_RECENT) || 6;
const MAX_KEPT = Number(process.env.THEME_MAX_KEPT) || Infinity;
const STAY_THRESHOLD = Number(process.env.THEME_STAY_THRESHOLD) || 0.34; // keyword gate: skip LLM when clearly same topic
const RELATE_PREFILTER = Number(process.env.THEME_RELATE_PREFILTER) || 0.1; // keyword gate before asking LLM to relate
const LLM_ROUTING = process.env.THEME_LLM_ROUTING !== "0"; // model-primary routing (default on)
const log = (...a: unknown[]) => console.error("[theme-mem]", ...a);
const ts = () => new Date().toISOString();
const parseJson = (text: string | null): any | null => {
  const m = text?.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try {
    return JSON.parse(m[0]);
  } catch {
    return null;
  }
};

export default function (pi: ExtensionAPI) {
  mem.ensure(mem.ROOT);
  let currentTheme = "general";
  const turnThemes: string[] = [];
  let lastEviction = { kept: 0, evicted: 0, pressure: 0, themes: [] as string[] };

  // ---- LLM helper for maintenance (uses the active model + its key) ----
  async function llm(ctx: any, prompt: string, maxTokens = 1024): Promise<string | null> {
    const model = ctx.model;
    if (!model) return null;
    const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
    if (!auth.ok || !auth.apiKey) return null;
    const res = await complete(
      model,
      { messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }] },
      { apiKey: auth.apiKey, headers: auth.headers, maxTokens, signal: ctx.signal },
    );
    return res.content.filter((c: any) => c.type === "text").map((c: any) => c.text).join("\n").trim() || null;
  }

  // ===== §2 startup: rebuild theme history (survives restart / -c) =====
  pi.on("session_start", (_event, ctx) => {
    turnThemes.length = 0;
    for (const entry of ctx.sessionManager.getEntries() as any[]) {
      if (entry.type === "custom" && entry.customType === "theme-turn") turnThemes.push(entry.data.theme);
    }
    if (turnThemes.length) currentTheme = turnThemes[turnThemes.length - 1];
    log(`session_start root=${mem.ROOT} themes=[${mem.listThemes().join(", ")}] current=${currentTheme} turns=${turnThemes.length}`);
  });

  // §3 theme judgment: keyword PREFILTER + LLM PRIMARY decision (+ user confirm if model unsure).
  async function classifyTheme(ctx: any, prompt: string): Promise<{ theme: string; isNew: boolean; source: string }> {
    const idx = mem.readIndex();
    const known = idx.themes.map((t) => t.name);
    if (!known.length) return { theme: currentTheme, isNew: false, source: "empty" };

    // Cheap keyword gate: if clearly continuing the current topic, stay without an LLM call.
    const scores = known.map((t) => ({ t, s: mem.scorePrompt(prompt, t) })).sort((a, b) => b.s - a.s);
    const curScore = mem.scorePrompt(prompt, currentTheme);
    if (!LLM_ROUTING || (curScore >= STAY_THRESHOLD && curScore >= scores[0].s)) {
      const top = scores[0];
      const pick = top && top.s >= STAY_THRESHOLD && top.s > curScore ? top.t : currentTheme;
      return { theme: pick, isNew: false, source: LLM_ROUTING ? "keyword-stay" : "keyword" };
    }

    // Model is the primary judge; keyword scores go in only as a hint.
    const hints = scores.slice(0, 5).map((x) => `${x.t}(${x.s.toFixed(2)})`).join(", ");
    const j = parseJson(await llm(ctx, sum.themeClassifyPrompt(prompt, currentTheme, idx.themes, hints), 200));
    if (!j || !j.theme) return { theme: currentTheme, isNew: false, source: "fallback" };
    const isNew = !!j.new && !known.includes(j.theme);

    // Model unsure -> confirm with the user when a UI is available (§3/§4).
    if (typeof j.confidence === "number" && j.confidence < 0.5 && ctx.hasUI && j.theme !== currentTheme) {
      const choice = await ctx.ui.select(
        `你现在是在讨论哪个主题/事件?(模型猜:${j.theme}${isNew ? " 新" : ""})`,
        [`继续当前(${currentTheme})`, ...known.filter((t) => t !== currentTheme), `+ 新主题: ${j.theme}`],
      );
      if (choice?.startsWith("+ 新主题")) return { theme: j.theme, isNew: true, source: "confirm" };
      if (choice && !choice.startsWith("继续当前")) return { theme: choice, isNew: false, source: "confirm" };
      return { theme: currentTheme, isNew: false, source: "confirm-stay" };
    }
    return { theme: j.theme, isNew, source: "llm" };
  }

  // ===== §3 routing + §2 load: pick theme, inject memory =====
  pi.on("before_agent_start", async (event, ctx) => {
    const prompt = event.prompt || "";

    const decision = await classifyTheme(ctx, prompt);
    if (decision.theme !== currentTheme) {
      log(`route "${currentTheme}" -> "${decision.theme}" (${decision.source}${decision.isNew ? ", new" : ""})`);
      currentTheme = decision.theme;
      mem.ensure(mem.themeDir(currentTheme));
      mem.upsertTheme(currentTheme, "");
    }
    turnThemes.push(currentTheme);

    // Inject resident core mem + theme index + active theme overall summary.
    const idx = mem.readIndex();
    const overall = mem.readSummary(currentTheme).overall;
    const themeList = idx.themes
      .map((t) => `- ${t.name}: ${t.desc}${t.related.length ? ` [related: ${t.related.join(", ")}]` : ""}`)
      .join("\n");
    const block =
      `\n\n## Persistent Memory (always resident)\n\n# Core Memory\n${idx.coreMem || "(empty)"}\n\n` +
      `# Themes\n${themeList || "(none)"}\n\n# Active theme: ${currentTheme}\n${overall || "(no summary yet)"}\n\n` +
      `Call memory_recall to load detail on demand; call set_theme when the topic clearly changes.`;
    return { systemPrompt: event.systemPrompt + block };
  });

  // ===== §4 eviction: theme switch + memory pressure =====
  pi.on("context", (event) => {
    const tags = tagMessagesByTurn(event.messages, turnThemes);
    const { kept, evictedCount, evictedThemes, pressureEvicted } = evict(
      event.messages,
      tags,
      currentTheme,
      KEEP_RECENT,
      MAX_KEPT,
    );
    lastEviction = { kept: kept.length, evicted: evictedCount, pressure: pressureEvicted, themes: evictedThemes.filter(Boolean) as string[] };
    if (evictedCount || pressureEvicted) {
      log(`context: theme-evicted ${evictedCount} [${lastEviction.themes.join(", ")}], pressure-evicted ${pressureEvicted}, kept ${kept.length}/${event.messages.length} for "${currentTheme}"`);
    }
    return { messages: kept };
  });

  // ===== §5 real-time append =====
  pi.on("agent_end", async (event) => {
    const blocks = ((event.messages as any[]) || [])
      .map((m) => {
        const t = messageText(m).trim();
        return t ? `**${m.role}** (${ts()}):\n${t}\n` : "";
      })
      .filter(Boolean);
    if (blocks.length) await mem.appendTurn(currentTheme, blocks);
    pi.appendEntry("theme-turn", { theme: currentTheme });
    log(`agent_end: appended ${blocks.length} msg -> ${currentTheme}/${mem.today()}.md`);
  });

  // ===== §5/§6 background maintenance (run via cron: pi -e ... -p "/maintain") =====
  async function maintain(ctx: any, scope?: string) {
    const themes = scope ? [scope] : mem.listThemes();
    const date = mem.today();
    let dailyN = 0;
    for (const theme of themes) {
      await mem.withLock(theme, async () => {
        const raw = mem.readFile(mem.dailyFile(theme, date));
        if (!raw.trim()) return;
        const s = mem.readSummary(theme);
        // 1) today's daily summary -> summary.days + daily file head
        const ds = await llm(ctx, sum.dailyPrompt(date, raw));
        if (ds) {
          s.days[date] = ds;
          mem.setDailyHead(theme, date, ds);
          dailyN++;
        }
        // 2) roll up day -> week -> month -> overall
        const wk = mem.isoWeek(new Date(date));
        const daysInWeek = Object.entries(s.days).filter(([d]) => mem.isoWeek(new Date(d)) === wk);
        if (daysInWeek.length) {
          const w = await llm(ctx, sum.rollupPrompt("week", daysInWeek.map(([d, v]) => `${d}: ${v}`).join("\n"), s.weeks[wk]));
          if (w) s.weeks[wk] = w;
        }
        const mk = mem.monthKey(date);
        const daysInMonth = Object.entries(s.days).filter(([d]) => d.startsWith(mk));
        if (daysInMonth.length) {
          const m = await llm(ctx, sum.rollupPrompt("month", daysInMonth.map(([d, v]) => `${d}: ${v}`).join("\n"), s.months[mk]));
          if (m) s.months[mk] = m;
        }
        const months = Object.entries(s.months);
        if (months.length) {
          const o = await llm(ctx, sum.rollupPrompt("overall", months.map(([k, v]) => `${k}: ${v}`).join("\n"), s.overall));
          if (o) s.overall = o;
        }
        mem.writeSummary(theme, s);
      });
    }
    // 3) §6 core-mem distillation across themes (today's day summaries)
    const cross = themes
      .map((t) => {
        const d = mem.readSummary(t).days[date];
        return d ? `【${t}】${d}` : "";
      })
      .filter(Boolean)
      .join("\n");
    let coreUpdated = false;
    if (cross) {
      const core = await llm(ctx, sum.coreMemPrompt(cross, mem.readIndex().coreMem));
      if (core) {
        mem.setCoreMem(core);
        coreUpdated = true;
      }
    }
    // 4) §8 auto related-link pass: keyword PREFILTER, then the MODEL decides.
    const snippet = (t: string) =>
      (mem.readSummary(t).overall || mem.listDaily(t).map((f) => mem.readFile(mem.dailyFile(t, f.replace(/\.md$/, "")))).join("\n")).slice(0, 600);
    const ths = mem.listThemes();
    let links = 0;
    for (let i = 0; i < ths.length; i++)
      for (let j = i + 1; j < ths.length; j++) {
        if (mem.relatedness(ths[i], ths[j]) < RELATE_PREFILTER) continue; // not even lexically close
        const r = parseJson(await llm(ctx, sum.relatePrompt(ths[i], snippet(ths[i]), ths[j], snippet(ths[j])), 50));
        if (r?.related) {
          mem.addRelated(ths[i], ths[j]);
          links++;
        }
      }
    return { themes: themes.length, dailyN, coreUpdated, links };
  }

  // ===== tools (LLM-callable) =====
  pi.registerTool({
    name: "set_theme",
    label: "Set Theme",
    description: "Switch the active memory theme. Other themes' history is evicted from context next call.",
    promptGuidelines: ["Call set_theme when the conversation clearly moves to a different topic/theme."],
    parameters: Type.Object({ theme: Type.String({ description: "theme name, kebab-case" }) }),
    async execute(_id, params) {
      currentTheme = params.theme.trim() || currentTheme;
      mem.ensure(mem.themeDir(currentTheme));
      mem.upsertTheme(currentTheme, "");
      log(`set_theme -> ${currentTheme}`);
      return { content: [{ type: "text", text: `Active theme is now "${currentTheme}".` }], details: {} };
    },
  });
  pi.registerTool({
    name: "memory_recall",
    label: "Memory Recall",
    description: "Load detail from a theme's memory files by date or keyword (structured retrieval, no embeddings).",
    promptGuidelines: ["Call memory_recall when you need past detail not present in the resident summary."],
    parameters: Type.Object({
      theme: Type.Optional(Type.String()),
      date: Type.Optional(Type.String({ description: "YYYY-MM-DD" })),
      keyword: Type.Optional(Type.String()),
    }),
    async execute(_id, params) {
      const t = params.theme || currentTheme;
      if (!mem.listThemes().includes(t)) return { content: [{ type: "text", text: `No memory for theme "${t}".` }], details: {} };
      const overall = mem.readSummary(t).overall;
      let files = mem.listDaily(t);
      if (params.date) files = files.filter((f) => f.includes(params.date!));
      let detail = files.map((f) => `--- ${f} ---\n${mem.readFile(mem.dailyFile(t, f.replace(/\.md$/, "")))}`).join("\n\n");
      if (params.keyword) {
        const kw = params.keyword.toLowerCase();
        detail = detail.split("\n").filter((l) => l.toLowerCase().includes(kw)).join("\n") || `(no lines matching "${params.keyword}")`;
      }
      const text = `# overall\n${overall || "(none)"}\n\n# detail\n${detail || "(none)"}`.slice(0, 4000);
      return { content: [{ type: "text", text }], details: {} };
    },
  });
  pi.registerTool({
    name: "memory_supersede",
    label: "Memory Supersede",
    description: "Mark an earlier fact as superseded when the user updates/contradicts it (no physical delete).",
    promptGuidelines: ["Call memory_supersede when new info replaces something recorded earlier."],
    parameters: Type.Object({
      date: Type.String({ description: "date of the entry being superseded, YYYY-MM-DD" }),
      note: Type.String({ description: "what replaces it" }),
      theme: Type.Optional(Type.String()),
    }),
    async execute(_id, params) {
      await mem.supersede(params.theme || currentTheme, params.date, params.note);
      return { content: [{ type: "text", text: `Recorded supersede of ${params.date}.` }], details: {} };
    },
  });

  // ===== commands (user) =====
  const out = (ctx: any, msg: string) => (ctx.hasUI ? ctx.ui.notify(msg, "info") : console.error(msg));

  pi.registerCommand("mem", {
    description: "Show theme-memory status",
    handler: async (_args, ctx) => {
      const idx = mem.readIndex();
      out(
        ctx,
        `theme-memory @ ${mem.ROOT}\n  active: ${currentTheme}\n  themes: ${mem.listThemes().join(", ") || "(none)"}\n` +
          `  turns: ${turnThemes.length}\n  core mem: ${idx.coreMem.split("\n").filter(Boolean).length} lines\n` +
          `  last eviction: kept ${lastEviction.kept}, theme-evicted ${lastEviction.evicted} [${lastEviction.themes.join(", ")}], pressure-evicted ${lastEviction.pressure}`,
      );
    },
  });
  pi.registerCommand("theme", {
    description: "Switch active memory theme",
    handler: async (args, ctx) => {
      const t = args.trim();
      if (t) {
        currentTheme = t;
        mem.ensure(mem.themeDir(currentTheme));
        mem.upsertTheme(currentTheme, "");
      }
      out(ctx, `active theme: ${currentTheme}`);
    },
  });
  pi.registerCommand("maintain", {
    description: "Run background memory maintenance (daily summary, roll-up, core mem). Optional: theme name.",
    handler: async (args, ctx) => {
      out(ctx, `maintaining ${args.trim() || "all themes"}…`);
      const r = await maintain(ctx, args.trim() || undefined);
      out(ctx, `maintain done: ${r.themes} theme(s), ${r.dailyN} daily summary, core ${r.coreUpdated ? "updated" : "unchanged"}, ${r.links} related link(s).`);
    },
  });
  pi.registerCommand("merge", {
    description: "Merge two themes into a new one: /merge <a> <b> <new-name>",
    handler: async (args, ctx) => {
      const [a, b, n] = args.trim().split(/\s+/);
      if (!a || !b || !n) return out(ctx, "usage: /merge <a> <b> <new-name>");
      await mem.mergeThemes(a, b, n);
      out(ctx, `merged ${a} + ${b} -> ${n}`);
    },
  });
  pi.registerCommand("relate", {
    description: "Add a bidirectional related link: /relate <a> <b>",
    handler: async (args, ctx) => {
      const [a, b] = args.trim().split(/\s+/);
      if (!a || !b) return out(ctx, "usage: /relate <a> <b>");
      mem.addRelated(a, b);
      out(ctx, `linked ${a} <-> ${b}`);
    },
  });
  pi.registerCommand("split", {
    description: "Split daily files containing a keyword into a new theme: /split <theme> <new-name> <keyword>",
    handler: async (args, ctx) => {
      const [t, n, ...kw] = args.trim().split(/\s+/);
      if (!t || !n || !kw.length) return out(ctx, "usage: /split <theme> <new-name> <keyword>");
      const moved = await mem.splitTheme(t, n, kw.join(" "));
      out(ctx, `split ${moved} file(s) from ${t} -> ${n}`);
    },
  });

  // optional: run maintenance on exit (idle proxy). Off by default — heavy (LLM).
  if (process.env.THEME_MAINTAIN_ON_EXIT === "1") {
    pi.on("session_shutdown", async (_event, ctx) => {
      try {
        await maintain(ctx);
      } catch (e) {
        log("maintain-on-exit failed:", (e as Error).message);
      }
    });
  }
}
