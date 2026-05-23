// Eviction logic (DESIGN.md §4) — two independent triggers:
//   1. theme switch  — drop messages tagged to other themes
//   2. memory pressure — if same-theme history still exceeds maxKept, drop the
//      oldest evictable (non-core, non-recent) messages too
// Pure (no pi imports) so it runs under plain `node` for offline testing.

/** Extract concatenated text from a message's content parts. */
export function messageText(m) {
  if (!m || !Array.isArray(m.content)) return "";
  return m.content
    .filter((c) => c && c.type === "text" && typeof c.text === "string")
    .map((c) => c.text)
    .join("\n");
}

/**
 * Tag each message with the theme of the turn it belongs to. Turn boundaries
 * are user messages, walked in order; system/custom messages get null (never
 * evicted — they carry resident core memory).
 */
export function tagMessagesByTurn(messages, turnThemes) {
  let turn = -1;
  return messages.map((m) => {
    if (m.role === "user") turn += 1;
    const conversational = m.role === "user" || m.role === "assistant" || m.role === "toolResult";
    return { theme: conversational ? (turnThemes[turn] ?? null) : null };
  });
}

/**
 * Keep: core (theme null) + active-theme turns + last `keepRecent` messages.
 * Then, under memory pressure (kept > maxKept), drop the oldest evictable
 * messages (not core, not within keepRecent) until at/below the cap.
 */
export function evict(messages, tags, currentTheme, keepRecent, maxKept = Infinity) {
  const n = messages.length;
  let kept = [];
  const keptTags = [];
  const evictedThemes = new Set();
  let evictedCount = 0;
  for (let i = 0; i < n; i++) {
    const t = tags[i].theme;
    const isRecent = i >= n - keepRecent;
    if (t === null || t === currentTheme || isRecent) {
      kept.push(messages[i]);
      keptTags.push({ theme: t, isRecent });
    } else {
      evictedCount++;
      evictedThemes.add(t);
    }
  }

  let pressureEvicted = 0;
  if (kept.length > maxKept) {
    const need = kept.length - maxKept;
    const drop = new Set();
    for (let j = 0; j < kept.length && drop.size < need; j++) {
      if (keptTags[j].theme !== null && !keptTags[j].isRecent) drop.add(j); // oldest first
    }
    kept = kept.filter((_, j) => !drop.has(j));
    pressureEvicted = drop.size;
  }

  return { kept, evictedCount, evictedThemes: [...evictedThemes], pressureEvicted };
}
