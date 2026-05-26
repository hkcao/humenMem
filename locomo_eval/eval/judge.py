"""Answer scoring.

Aligns with snap-research/locomo `task_eval/evaluation.py`:
  - cat5 (adversarial): correct iff prediction contains "no information
    available" or "not mentioned" (exact-phrase match, official rule).
  - cat1 (multi-hop): token-F1 with max-F1 averaging over comma-split
    multi-answer pairs.
  - cat2/3/4: token-F1 (Porter-style normalization).

Also computes an LLM-judge 0/1 verdict (paraphrase-tolerant). Both
metrics are returned per question so reports can show F1 (official) and
LLM-judge side by side.
"""
import re
import string
from collections import Counter

from harness.llm import LLM

_judge_llm = None


def _llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(scheme="judge")
    return _judge_llm


# --- official LOCOMO cat5 rule (task_eval/evaluation.py) ---
_ABSTAIN_OFFICIAL = ("no information available", "not mentioned")


def is_abstain(pred):
    p = (pred or "").lower()
    return any(s in p for s in _ABSTAIN_OFFICIAL)


# --- token-F1 (mirrors SQuAD / LOCOMO normalization) ---
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.U)


def _normalize(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p_toks)
    rec = n / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _multi_f1(pred, gold):
    """LOCOMO multi-hop: gold may be comma-separated; average max-F1 over golds."""
    golds = [g.strip() for g in str(gold).split(",") if g.strip()]
    if len(golds) <= 1:
        return _f1(pred, gold)
    # For each gold answer, take the max F1 across pred chunks (also split).
    preds = [p.strip() for p in str(pred).split(",") if p.strip()] or [pred]
    scores = [max(_f1(p, g) for p in preds) for g in golds]
    return sum(scores) / len(scores)


def judge(q, pred):
    cat = q.get("category")
    if cat == 5:
        ok = is_abstain(pred)
        return {"correct": ok, "f1": float(ok), "mode": "adversarial"}

    gold = str(q.get("answer"))
    f1 = _multi_f1(pred, gold) if cat == 1 else _f1(pred, gold)

    # LLM-judge for paraphrase tolerance (kept alongside official F1).
    sys = ("You grade a predicted answer against the gold answer for a question "
           "about a conversation. Mark correct if the prediction conveys the same "
           "core information as gold (paraphrase, equivalent date formats, or a "
           "prediction that contains the gold fact are all correct). Mark incorrect "
           "if it misses, contradicts, or abstains. Return JSON {\"correct\":bool}.")
    usr = (f"Question: {q['question']}\nGold answer: {gold}\n"
           f"Predicted answer: {pred}\n")
    try:
        out = _llm().chat_json([{"role": "system", "content": sys},
                                {"role": "user", "content": usr}],
                               phase="judge", max_tokens=200)
        llm_ok = bool(out.get("correct"))
        mode = "judge"
    except Exception:
        llm_ok = f1 >= 0.5
        mode = "f1-fallback"
    return {"correct": llm_ok, "f1": round(f1, 4), "mode": mode}
