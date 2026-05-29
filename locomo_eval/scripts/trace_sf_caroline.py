"""Reproduce theme-mem-sf-live's answer to:
   "When did Caroline go to the LGBTQ support group?"
and dump every stage's input/output.

Mirrors what eval.live does up to the moment that question is injected
(which is at the end of session_1 for this question — its evidence dia
ids live in session_1, May 7-8 2023).
"""
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from harness import dataset as ds
from harness.memory_store import MemoryStore
from harness.schemes import ThemeMemSummaryFirst, ANSWER_SYS
from harness.llm import LLM

DUMP = []


def log(label, value):
    DUMP.append({"step": label, "value": value})
    print(f"\n===== {label} =====")
    if isinstance(value, str):
        print(value if len(value) < 2000 else value[:2000] + f"...[+{len(value)-2000} chars]")
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2)[:2000])


def main():
    data = ds.load()
    sample = data[0]  # conv-26
    qa = sample["qa"]
    # find Caroline support-group question
    target_q = next(q for q in qa
                    if q["question"] == "When did Caroline go to the LGBTQ support group?")
    log("target_question", target_q)

    # ingest session_1 only — mirrors live state at the moment the
    # question is injected (its evidence anchor is in session_1).
    store = MemoryStore(sample["sample_id"], persist=False)
    sessions = list(ds.iter_sessions(sample["conversation"]))
    sk, iso, dt, turns = sessions[0]
    log("session_1_meta", {"session_key": sk, "iso_date": iso,
                            "turns": len(turns)})
    store.ingest_session(sk, iso, turns)
    log("themes_after_session_1", list(store.themes.keys()))
    log("core_mem_after_session_1", store.core_mem)
    for tname, t in store.themes.items():
        log(f"theme[{tname}].desc", t.desc)
        log(f"theme[{tname}].day_summaries", dict(t.day_summaries))
        log(f"theme[{tname}].turns_by_day",
            {d: [f"[{x['dia_id']}] {x['speaker']}: {x['text']}" for x in xs]
             for d, xs in t.days.items()})

    # Stage 1: instrument answer() — re-implement inline so we can capture
    sctx, stok = store.summaries_context()
    log("stage1_summaries_context", sctx)
    log("stage1_summaries_tokens", stok)

    llm = LLM(scheme="trace-sf")
    sys_msg = ("You answer a question about a long conversation between two people. "
               "You are given Core Memory and summaries of ALL topics. If these are "
               "enough to answer precisely, return JSON {\"answer\": <concise answer>}. "
               "If you need the detailed turns to be sure, return JSON "
               "{\"need_details\": [topic names to open]} using exact topic names "
               "shown above. Prefer opening details when the question asks for a "
               "specific date, name, or fact not explicit in the summaries.")
    usr_msg = f"{sctx}\n\nQuestion: {target_q['question']}"
    log("stage1_user_prompt", usr_msg)

    out = llm.chat_json([{"role": "system", "content": sys_msg},
                         {"role": "user", "content": usr_msg}],
                        phase="query", max_tokens=1000)
    log("stage1_model_output_json", out)

    # Decide stage 2
    ans = out.get("answer")
    if isinstance(ans, str) and ans.strip() and ans.strip().lower() != "null":
        log("decision", f"STAGE 1 answered: {ans!r}")
    else:
        need = out.get("need_details") or []
        log("stage1_needed_themes_raw", need)
        valid = [n for n in need if n in store.themes]
        log("stage1_needed_themes_valid", valid)
        if not valid:
            valid = list(store.themes.keys())
            log("fallback", "no valid themes named -> opening ALL themes")

        dctx, dtok = store.retrieve_details(target_q["question"], valid,
                                            max_tokens=max(4000 - stok, 1500))
        log("stage2_retrieved_details", dctx)
        log("stage2_retrieved_tokens", dtok)
        full = sctx + "\n\n## Retrieved details\n" + dctx
        log("stage2_full_context", full)

        a = llm.chat([{"role": "system", "content": ANSWER_SYS},
                      {"role": "user",
                       "content": f"Context:\n{full}\n\nQuestion: {target_q['question']}\nAnswer:"}],
                     phase="query", max_tokens=600).strip()
        log("stage2_model_answer", a)

    json.dump(DUMP, open("scripts/trace_sf_caroline.json", "w"),
              ensure_ascii=False, indent=2)
    print("\nFull trace -> scripts/trace_sf_caroline.json")


if __name__ == "__main__":
    main()
