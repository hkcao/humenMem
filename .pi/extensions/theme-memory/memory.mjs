// Filesystem layer for the theme-memory extension (DESIGN.md §1, §5, §8).
// Pure Node — no pi, no LLM — so it is unit-testable offline.
// Implements: index.md (core mem + theme index w/ related links), per-theme
// daily files with summary heads, hierarchical summary.md, supersede,
// relatedness/merge/split, atomic writes, per-theme locks.

import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

export const ROOT = process.env.HANK_MEMORY_DIR || path.join(os.homedir(), "hank_memory");
export const ensure = (d) => fs.mkdirSync(d, { recursive: true });
export const themeDir = (t) => path.join(ROOT, t);
export const dailyFile = (t, date = today()) => path.join(themeDir(t), `${date}.md`);
export const summaryFile = (t) => path.join(themeDir(t), "summary.md");
export const indexFile = () => path.join(ROOT, "index.md");
export const readFile = (f) => (fs.existsSync(f) ? fs.readFileSync(f, "utf8") : "");

export const today = () => new Date().toISOString().slice(0, 10);
export const monthKey = (date = today()) => date.slice(0, 7);
export function isoWeek(d = new Date()) {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const day = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((date - yearStart) / 86400000 + 1) / 7);
  return `${date.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

// ---------- atomic write + per-theme lock ----------
export function atomicWrite(file, content) {
  ensure(path.dirname(file));
  const tmp = `${file}.tmp.${process.pid}.${Date.now()}`;
  fs.writeFileSync(tmp, content);
  fs.renameSync(tmp, file);
}
export function atomicAppend(file, text) {
  atomicWrite(file, readFile(file) + text);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** Mutual exclusion across processes (append vs. maintenance). Non-reentrant. */
export async function withLock(theme, fn, { retries = 200, delayMs = 20, staleMs = 60000 } = {}) {
  const dir = themeDir(theme);
  ensure(dir);
  const lock = path.join(dir, ".lock");
  let fd;
  for (let i = 0; i < retries; i++) {
    try {
      fd = fs.openSync(lock, "wx");
      break;
    } catch {
      try {
        if (Date.now() - fs.statSync(lock).mtimeMs > staleMs) fs.unlinkSync(lock); // steal stale lock
      } catch {}
      await sleep(delayMs);
    }
  }
  if (fd === undefined) throw new Error(`theme-memory: could not lock "${theme}"`);
  try {
    return await fn();
  } finally {
    fs.closeSync(fd);
    try {
      fs.unlinkSync(lock);
    } catch {}
  }
}

// ---------- index.md (core mem + theme index) ----------
export function parseIndex(raw) {
  const core = [];
  const themes = [];
  let section = "";
  for (const line of raw.split("\n")) {
    const h = line.match(/^#\s+(.+)/);
    if (h) {
      section = h[1].toLowerCase();
      continue;
    }
    if (section.startsWith("core")) {
      if (line.trim()) core.push(line);
    } else if (section.startsWith("theme")) {
      const m = line.match(/^-\s+(\S+?):\s*(.*)$/);
      if (!m) continue;
      let desc = m[2];
      const related = [];
      const r = desc.match(/\[related:\s*([^\]]*)\]\s*$/i);
      if (r) {
        desc = desc.slice(0, r.index).trim();
        related.push(...r[1].split(",").map((s) => s.trim()).filter(Boolean));
      }
      themes.push({ name: m[1], desc, related });
    }
  }
  return { coreMem: core.join("\n"), themes };
}
export function serializeIndex({ coreMem, themes }) {
  const t = themes
    .map((x) => `- ${x.name}: ${x.desc}${x.related && x.related.length ? ` [related: ${x.related.join(", ")}]` : ""}`)
    .join("\n");
  return `# Core Memory\n\n${coreMem || ""}\n\n# Themes\n\n${t}\n`;
}
export const readIndex = () => parseIndex(readFile(indexFile()));
export const writeIndex = (idx) => atomicWrite(indexFile(), serializeIndex(idx));
export const listThemes = () =>
  fs.existsSync(ROOT)
    ? fs.readdirSync(ROOT, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name)
    : [];
export function upsertTheme(name, desc) {
  const idx = readIndex();
  const ex = idx.themes.find((t) => t.name === name);
  if (ex) {
    if (desc) ex.desc = desc;
  } else idx.themes.push({ name, desc: desc || name, related: [] });
  writeIndex(idx);
}
export function setCoreMem(text) {
  const idx = readIndex();
  idx.coreMem = text;
  writeIndex(idx);
}
export function addRelated(a, b) {
  if (a === b) return;
  const idx = readIndex();
  for (const [x, y] of [[a, b], [b, a]]) {
    const t = idx.themes.find((th) => th.name === x);
    if (t && !t.related.includes(y)) t.related.push(y);
  }
  writeIndex(idx);
}

// ---------- daily files ----------
export const listDaily = (theme) => {
  const dir = themeDir(theme);
  return fs.existsSync(dir)
    ? fs.readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.md$/.test(f)).sort()
    : [];
};
/** Append a turn's blocks to today's daily file (locked). */
export const appendTurn = (theme, blocks) =>
  withLock(theme, async () => atomicAppend(dailyFile(theme), `\n${blocks.join("\n")}`));

const HEAD = "<!-- daily-summary -->";
/** Write/replace the ~200 token summary at the head of a daily file. Caller must hold the lock. */
export function setDailyHead(theme, date, summary) {
  const f = dailyFile(theme, date);
  let body = readFile(f);
  const re = new RegExp(`^# ${date} 小结\\n${HEAD}\\n[\\s\\S]*?${HEAD}\\n\\n`);
  body = body.replace(re, "");
  atomicWrite(f, `# ${date} 小结\n${HEAD}\n${summary.trim()}\n${HEAD}\n\n${body.replace(/^\n+/, "")}`);
}

// ---------- hierarchical summary.md (day -> week -> month -> overall) ----------
function parseSub(body, map) {
  for (const blk of body.split(/^## /m).map((s) => s.trim()).filter(Boolean)) {
    const [k, ...r] = blk.split("\n");
    map[k.trim()] = r.join("\n").trim();
  }
}
export function parseSummary(raw) {
  const out = { overall: "", months: {}, weeks: {}, days: {} };
  for (const sec of raw.split(/^# /m).map((s) => s.trim()).filter(Boolean)) {
    const [head, ...rest] = sec.split("\n");
    const body = rest.join("\n").trim();
    const h = head.toLowerCase();
    if (h.startsWith("overall")) out.overall = body;
    else if (h.startsWith("month")) parseSub(body, out.months);
    else if (h.startsWith("week")) parseSub(body, out.weeks);
    else if (h.startsWith("day")) parseSub(body, out.days);
  }
  return out;
}
export function serializeSummary(s) {
  const sub = (m) => Object.entries(m).sort().map(([k, v]) => `## ${k}\n${v}`).join("\n\n");
  return `# Overall\n\n${s.overall || ""}\n\n# Month\n\n${sub(s.months)}\n\n# Week\n\n${sub(s.weeks)}\n\n# Day\n\n${sub(s.days)}\n`;
}
export const readSummary = (theme) => parseSummary(readFile(summaryFile(theme)));
export const writeSummary = (theme, s) => atomicWrite(summaryFile(theme), serializeSummary(s));

// ---------- supersede (contradictions; no physical delete) ----------
export const supersede = (theme, date, note) =>
  withLock(theme, async () => atomicAppend(dailyFile(theme), `\n> supersede ${date}: ${note}\n`));

// ---------- relatedness / merge / split (§8) ----------
const STOP = new Set("的 了 和 与 是 在 我 你 它 把 个 也 就 a an the to of and or for is on in".split(/\s+/));
export const tokens = (text) =>
  (text.toLowerCase().match(/[a-z0-9]+|[一-龥]/g) || []).filter((w) => !STOP.has(w));
function themeText(theme) {
  return readSummary(theme).overall + " " + listDaily(theme).map((f) => readFile(path.join(themeDir(theme), f))).join(" ");
}
/** Overlap coefficient of keyword sets in [0,1]. */
export function relatedness(a, b) {
  const ta = new Set(tokens(themeText(a)));
  const tb = new Set(tokens(themeText(b)));
  if (!ta.size || !tb.size) return 0;
  let inter = 0;
  for (const w of ta) if (tb.has(w)) inter++;
  return inter / Math.min(ta.size, tb.size);
}
/** Score a free-text prompt against a theme's keyword set in [0,1]. */
export function scorePrompt(prompt, theme) {
  const pt = new Set(tokens(prompt));
  const tt = new Set(tokens(themeText(theme)));
  if (!pt.size || !tt.size) return 0;
  let inter = 0;
  for (const w of pt) if (tt.has(w)) inter++;
  return inter / pt.size;
}
export async function mergeThemes(a, b, newName) {
  ensure(themeDir(newName));
  for (const src of [a, b]) {
    for (const f of listDaily(src)) {
      const target = path.join(themeDir(newName), f);
      const prev = readFile(target);
      const add = readFile(path.join(themeDir(src), f));
      atomicWrite(target, prev ? `${prev}\n\n${add}` : add);
    }
  }
  const idx = readIndex();
  const ta = idx.themes.find((t) => t.name === a);
  const tb = idx.themes.find((t) => t.name === b);
  const related = [...new Set([...(ta?.related || []), ...(tb?.related || [])])].filter(
    (x) => x !== a && x !== b && x !== newName,
  );
  idx.themes = idx.themes.filter((t) => t.name !== a && t.name !== b);
  idx.themes.push({ name: newName, desc: `${ta?.desc || a} + ${tb?.desc || b}`, related });
  writeIndex(idx);
  for (const src of [a, b]) atomicWrite(path.join(themeDir(src), "MERGED.md"), `> supersede: merged into ${newName} on ${today()}\n`);
}
export async function splitTheme(theme, newName, keyword) {
  ensure(themeDir(newName));
  let moved = 0;
  for (const f of listDaily(theme)) {
    const p = path.join(themeDir(theme), f);
    const content = readFile(p);
    if (keyword && content.toLowerCase().includes(keyword.toLowerCase())) {
      atomicWrite(path.join(themeDir(newName), f), content);
      fs.unlinkSync(p);
      moved++;
    }
  }
  upsertTheme(newName, `split from ${theme}`);
  addRelated(theme, newName);
  return moved;
}
