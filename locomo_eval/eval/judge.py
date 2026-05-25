"""LLM-judge of answer correctness (LOCOMO J-score style).

Normal categories: predicted is correct if it conveys the gold answer (allowing
paraphrase, date-format, superset). Category 5 (adversarial): correct iff the
model ABSTAINS (says no info / refuses), since the question is unanswerable and
the gold `adversarial_answer` is the tempting-but-wrong lure.
"""
import re
from harness.llm import LLM

_judge_llm = None


def _llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(scheme="judge")
    return _judge_llm


_ABSTAIN = re.compile(r"no information available|not (mentioned|available|provided|"
                      r"specified|enough)|cannot (be )?(answer|determin)|don'?t know|"
                      r"unknown|unclear|no (info|mention|record)", re.I)


def is_abstain(pred):
    return bool(_ABSTAIN.search(pred or ""))


def judge(q, pred):
    cat = q.get("category")
    if cat == 5:
        # correct iff model abstained
        return {"correct": is_abstain(pred), "mode": "adversarial"}
    gold = q.get("answer")
    gold = str(gold)
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
        return {"correct": bool(out.get("correct")), "mode": "judge"}
    except Exception:
        # fallback: token overlap
        g = set(re.findall(r"[a-z0-9]+", gold.lower()))
        p = set(re.findall(r"[a-z0-9]+", (pred or "").lower()))
        return {"correct": bool(g) and len(g & p) / len(g) >= 0.5, "mode": "overlap"}
