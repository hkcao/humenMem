"""LongMemEval harness: retrieval (our BM25) + generation/judge prompts (verbatim).

Faithfully replicates the official LongMemEval prompts and judging logic
(src/generation/run_generation.py and src/evaluation/evaluate_qa.py) so results are
comparable. Retrieval uses OUR theme-memory BM25 engine at session granularity.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "theme_memory"))
import retrieve as retr  # noqa: E402  (our BM25 engine)

# --- generation prompt (verbatim, non-CoT "direct" reading) ----------------
GEN_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer the "
    "question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\n"
    "Current Date: {}\nQuestion: {}\nAnswer:"
)

# --- judge prompts (verbatim from get_anscheck_prompt) ----------------------
J_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If "
    "the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
J_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If "
    "the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In "
    "addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one "
    "errors (e.g., predicting 19 days when the answer is 18), the model's response is "
    "still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
J_KNOWLEDGE = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If "
    "the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the "
    "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
J_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the points "
    "in the rubric. The response is correct as long as it recalls and utilizes the "
    "user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)
J_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some other "
    "information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)


def judge_prompt(question_id, question_type, question, answer, response) -> str:
    if "_abs" in question_id:
        tmpl = J_ABSTENTION
    elif question_type == "temporal-reasoning":
        tmpl = J_TEMPORAL
    elif question_type == "knowledge-update":
        tmpl = J_KNOWLEDGE
    elif question_type == "single-session-preference":
        tmpl = J_PREFERENCE
    else:
        tmpl = J_STANDARD
    return tmpl.format(question, answer, response)


def parse_label(judge_response: str) -> bool:
    """Official rule: 'yes' substring (case-insensitive) -> correct."""
    return "yes" in (judge_response or "").lower()


# --- data ------------------------------------------------------------------

def load_instances(path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- retrieval (our BM25 over sessions) ------------------------------------

def _session_docs(inst) -> list:
    """One doc per session, content = user turns only (matches LongMemEval's index)."""
    docs = []
    for sid, sess, date in zip(inst["haystack_session_ids"], inst["haystack_sessions"],
                               inst["haystack_dates"]):
        user_text = " ".join(t["content"] for t in sess if t.get("role") == "user")
        # topic="" so the session id (which encodes answer/noans) never leaks into scoring
        docs.append({"topic": "", "content": user_text, "sid": sid, "date": date, "session": sess})
    return docs


def retrieve_sessions(inst, topk, config="bm25") -> list:
    """Return selected sessions (list of docs) for the question."""
    if config == "oracle":
        evid = set(inst.get("answer_session_ids", []))
        docs = _session_docs(inst)
        return [d for d in docs if d["sid"] in evid]
    if config == "no-mem":
        return []
    # default: our BM25 ranking, top-k sessions
    ranked = retr.bm25(inst["question"], _session_docs(inst), limit=topk)
    return ranked


# --- generation prompt assembly --------------------------------------------

def format_history(sessions, char_budget=150_000) -> str:
    """Verbatim LongMemEval session block format, NL turn rendering, sorted by date."""
    sessions = sorted(sessions, key=lambda s: s.get("date", ""))
    blocks = []
    for i, s in enumerate(sessions, 1):
        turns = "".join(f"\n\n{t['role']}: {t['content']}" for t in s["session"])
        blocks.append(f"\n### Session {i}:\nSession Date: {s['date']}\nSession Content:\n{turns}\n")
    hist = "".join(blocks)
    return hist[:char_budget]


def gen_prompt(inst, sessions) -> str:
    history = format_history(sessions) if sessions else "(no relevant history found)"
    return GEN_TEMPLATE.format(history, inst.get("question_date", ""), inst["question"])
