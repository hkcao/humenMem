// LLM prompt builders for background maintenance (DESIGN.md §5, §6).
// Pure strings — the actual model call lives in index.ts via ctx.modelRegistry.

export const dailyPrompt = (date, conversationText) =>
  `把下面 ${date} 的对话压缩成不超过 200 token 的中文小结,只保留关键结论/决定/事实,去掉寒暄与过程:\n\n${conversationText}`;

export function rollupPrompt(level, itemsText, prev) {
  const name = { week: "周", month: "月", overall: "总体" }[level] || level;
  const budget = level === "overall" ? 200 : 300;
  const prevBlock = prev ? `\n已有的${name}摘要(可在此基础上修订):\n${prev}\n` : "";
  return `把下面这些下一级摘要合并成一个不超过 ${budget} token 的中文「${name}摘要」,去重、保留关键结论,不要逐条罗列:${prevBlock}\n\n${itemsText}`;
}

// Theme routing (§3): the MODEL is the primary judge; keyword scores are only a hint.
export function themeClassifyPrompt(prompt, currentTheme, themes, hints) {
  const list = themes.map((t) => `- ${t.name}: ${t.desc}`).join("\n") || "(none)";
  return (
    `你是对话主题分类器。已知主题:\n${list}\n\n当前主题:${currentTheme}\n` +
    `关键词预筛(仅供参考,分数高=用词更接近,不代表语义相关):${hints || "(无)"}\n\n` +
    `用户最新消息:\n"""${prompt}"""\n\n` +
    `判断这条消息应归到哪个主题。优先复用已有主题;只有明显是新话题时才新建,新建时给出 kebab-case 名称。` +
    `主要依据消息的语义,关键词分数仅作参考。\n` +
    `仅输出 JSON:{"theme":"主题名","new":true/false,"confidence":0到1之间}。confidence 是你对该判断的把握。`
  );
}

// Relatedness (§8): MODEL decides if two themes are substantively related; keyword overlap only prefilters.
export function relatePrompt(a, textA, b, textB) {
  return (
    `判断以下两个对话主题是否「内容上有实质联系、值得建立关联链接」(不是只因碰巧用词重叠)。\n\n` +
    `主题A = ${a}:\n${textA || "(空)"}\n\n主题B = ${b}:\n${textB || "(空)"}\n\n` +
    `仅输出 JSON:{"related":true/false}。`
  );
}

export const coreMemPrompt = (crossThemeText, prevCore, budget = 800) =>
  `你在维护一段跨主题的「核心记忆」——是经验、原则与判断力,不是具体事实(具体事实归各主题摘要)。` +
  `结合已有核心记忆与今天的跨主题内容,产出更新后的核心记忆:markdown 无序列表,总量不超过 ${budget} token,` +
  `合并相近条目、去重,只保留可泛化的高价值结论。直接输出列表,不要解释。\n\n` +
  `已有核心记忆:\n${prevCore || "(空)"}\n\n今天的跨主题内容:\n${crossThemeText}`;
