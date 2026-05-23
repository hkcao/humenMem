// Offline deterministic tests — no pi, no network, no API cost.
//   HANK_MEMORY_DIR=/tmp/tm-test node test-eviction.mjs
import { tagMessagesByTurn, evict, messageText } from "./evict.mjs";
import * as mem from "./memory.mjs";
import * as fs from "node:fs";

let pass = 0, fail = 0;
const ok = (name, cond) => { (cond ? pass++ : fail++); console.log(`${cond ? "PASS" : "FAIL"}  ${name}`); };

// ---------- 1. theme eviction + memory pressure ----------
const turnThemes = ["cooking", "cooking", "pi-design", "pi-design", "taxes", "pi-design"];
const messages = [{ role: "system", content: [{ type: "text", text: "(core)" }] }];
turnThemes.forEach((t, i) => {
  messages.push({ role: "user", content: [{ type: "text", text: `[${t}] q${i}` }] });
  messages.push({ role: "assistant", content: [{ type: "text", text: `[${t}] a${i}` }] });
});
const tags = tagMessagesByTurn(messages, turnThemes);

const e1 = evict(messages, tags, "pi-design", 4);
ok("theme eviction drops cooking", e1.evictedThemes.includes("cooking") && e1.evictedCount === 4);
ok("theme eviction keeps system+recent", e1.kept[0].role === "system" && e1.kept.length === 9);

const e2 = evict(messages, tags, "pi-design", 4, 6); // maxKept=6 -> pressure
ok("memory pressure trims to cap", e2.kept.length === 6 && e2.pressureEvicted === 3);

// ---------- 2. filesystem layer (index / summary parse round-trips) ----------
mem.ensure(mem.ROOT);
const idx = { coreMem: "- a\n- b", themes: [{ name: "x", desc: "X", related: ["y"] }, { name: "y", desc: "Y", related: ["x"] }] };
ok("index round-trip", JSON.stringify(mem.parseIndex(mem.serializeIndex(idx))) === JSON.stringify(idx));

const s = { overall: "ov", months: { "2026-05": "m" }, weeks: { "2026-W21": "w" }, days: { "2026-05-23": "d" } };
ok("summary round-trip", JSON.stringify(mem.parseSummary(mem.serializeSummary(s))) === JSON.stringify(s));

ok("isoWeek", mem.isoWeek(new Date("2026-05-23")) === "2026-W21");
ok("tokens drops stopwords", !mem.tokens("我 的 pi design").includes("的"));

// ---------- 3. atomic write + daily head ----------
mem.writeIndex(idx);
ok("writeIndex+readIndex", mem.readIndex().themes.length === 2);
mem.addRelated("x", "z");
ok("addRelated bidirectional", mem.readIndex().themes.find((t) => t.name === "x").related.includes("z"));

mem.ensure(mem.themeDir("x"));
mem.atomicAppend(mem.dailyFile("x"), "body line\n");
mem.setDailyHead("x", mem.today(), "daily summary here");
const daily = mem.readFile(mem.dailyFile("x"));
ok("daily head present", daily.startsWith(`# ${mem.today()} 小结`) && daily.includes("body line"));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
