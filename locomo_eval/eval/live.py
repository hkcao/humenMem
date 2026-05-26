"""DialSim-style LIVE evaluation: replay a LOCOMO conversation turn-by-turn,
inject each QA question at the earliest turn where all its `evidence` dia_ids
have been observed. No future leakage; same agent instance answers all
questions in order, so cross-question state (windows, eviction, mem0 facts)
is exercised.

This runner is intentionally separate from `eval.run` (cold-start LOCOMO) so
that benchmark stays byte-identical. The judge is reused from `eval.judge`.

Usage (from locomo_eval/):
  python -m eval.live --samples 0 --max-questions 50 --budget 4000 \
      --model-ctx 128000 \
      --schemes full-context-live,bm25-rag-live,theme-mem-evict-live
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

from harness import dataset as ds
from harness.llm import ACCT
from harness.live_schemes import LIVE_SCHEMES
from eval import judge as J


# --------------- timeline construction ---------------
def build_timeline(sample, questions):
    """Interleave turns and question-injections.

    For each question with evidence, attach it to the LAST evidence dia_id —
    i.e. ask the moment the agent could in principle know the answer.
    Adversarial (cat 5) and evidence-less questions are appended to the very
    end of the conversation so the agent sees the whole story first.

    Returns a list of events:
      {"kind": "session_start", "meta": {...}}
      {"kind": "turn", "turn": {...}, "meta": {...}}
      {"kind": "session_end", "meta": {...}}
      {"kind": "question", "q": <qa dict>}
    """
    conv = sample["conversation"]

    # last dia_id (chronological) for each session
    sessions = list(ds.iter_sessions(conv))
    chrono_index = {}  # dia_id -> (session_idx, turn_idx)
    for si, (sk, iso, dt, turns) in enumerate(sessions):
        for ti, t in enumerate(turns):
            chrono_index[t["dia_id"]] = (si, ti)

    def _latest(dia_ids):
        present = [d for d in dia_ids if d in chrono_index]
        if not present:
            return None
        return max(present, key=lambda d: chrono_index[d])

    inject_at = defaultdict(list)  # dia_id -> [questions]
    tail = []
    for q in questions:
        ev = q.get("evidence") or []
        anchor = _latest(ev) if ev else None
        if anchor is None:
            tail.append(q)
        else:
            inject_at[anchor].append(q)

    events = []
    for si, (sk, iso, dt, turns) in enumerate(sessions):
        meta = {"session_key": sk, "iso_date": iso, "raw_dt": dt,
                "session_idx": si}
        events.append({"kind": "session_start", "meta": meta})
        for t in turns:
            events.append({"kind": "turn", "turn": t, "meta": meta})
            for q in inject_at.get(t["dia_id"], []):
                events.append({"kind": "question", "q": q})
        events.append({"kind": "session_end", "meta": meta})
    for q in tail:
        events.append({"kind": "question", "q": q})
    return events


# --------------- driver ---------------
def run_one(agent, events, model_ctx):
    """Drive a single agent through the timeline. Returns per-question records."""
    agent.reset() if hasattr(agent, "reset") else None
    results = []
    for ev in events:
        k = ev["kind"]
        if k == "session_start":
            pass
        elif k == "turn":
            agent.ingest_turn(ev["turn"], ev["meta"])
        elif k == "session_end":
            if hasattr(agent, "on_session_end"):
                agent.on_session_end(ev["meta"])
        elif k == "question":
            q = ev["q"]
            t0 = time.time()
            try:
                r = agent.answer(q["question"])
            except Exception as e:  # noqa: BLE001
                r = {"answer": f"[ERROR: {e}]", "ctx_tokens": 0}
            verdict = J.judge(q, r["answer"])
            ctx = r.get("ctx_tokens", 0)
            results.append({
                "question": q["question"], "category": q.get("category"),
                "gold": ds.gold_answer(q), "pred": r["answer"],
                "ctx_tokens": ctx,
                "ctx_overflow": bool(model_ctx and ctx > model_ctx),
                "in_progress_tokens": r.get("in_progress_tokens", 0),
                "stage": r.get("stage"),
                "correct": verdict["correct"],
                "f1": verdict.get("f1", 0.0),
                "judge_mode": verdict["mode"],
                "latency": round(time.time() - t0, 1),
            })
    return results


def aggregate(results):
    by_cat = defaultdict(lambda: [0, 0, 0.0])
    ctx = []
    for r in results:
        by_cat[r["category"]][0] += int(r["correct"])
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][2] += float(r.get("f1", 0.0))
        ctx.append(r["ctx_tokens"])
    n = sum(v[1] for v in by_cat.values())
    return {
        "overall_acc": round(sum(v[0] for v in by_cat.values()) / n, 4) if n else 0,
        "overall_f1": round(sum(v[2] for v in by_cat.values()) / n, 4) if n else 0,
        "n": n,
        "by_category": {ds.CAT_NAME.get(c, c): {
            "acc": round(v[0] / v[1], 4) if v[1] else 0,
            "f1": round(v[2] / v[1], 4) if v[1] else 0,
            "n": v[1]} for c, v in sorted(by_cat.items())},
        "avg_ctx_tokens": round(sum(ctx) / len(ctx)) if ctx else 0,
        "max_ctx_tokens": max(ctx) if ctx else 0,
        "n_overflow": sum(1 for r in results if r.get("ctx_overflow")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="0")
    ap.add_argument("--max-questions", type=int, default=50)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--model-ctx", type=int, default=0)
    ap.add_argument("--schemes",
                    default="full-context-live,bm25-rag-live,theme-mem-evict-live")
    ap.add_argument("--out", default="results_live")
    args = ap.parse_args()

    data = ds.load()
    os.makedirs(args.out, exist_ok=True)
    sample_idx = [int(x) for x in args.samples.split(",")]
    want = args.schemes.split(",")

    all_report = {}
    for si in sample_idx:
        sample = data[si]
        cid = sample["sample_id"]
        qa = sample["qa"][:args.max_questions] if args.max_questions else sample["qa"]
        events = build_timeline(sample, qa)
        n_q = sum(1 for e in events if e["kind"] == "question")
        print(f"\n### sample {si} ({cid}): {n_q} questions injected into "
              f"{sum(1 for e in events if e['kind'] == 'turn')} turns | schemes={want}",
              flush=True)

        cfg = {"budget": args.budget, "conv_id": cid}
        scheme_reports, per_scheme = {}, {}
        for name in want:
            if name not in LIVE_SCHEMES:
                print(f"  unknown scheme: {name}", flush=True)
                continue
            try:
                agent = LIVE_SCHEMES[name](cfg)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {name}: init failed ({e})", flush=True)
                continue
            print(f"  running {name}...", flush=True)
            t0 = time.time()
            res = run_one(agent, events, args.model_ctx)
            per_scheme[name] = res
            rep = aggregate(res)
            rep["wall_s"] = round(time.time() - t0)
            rep["model_ctx"] = args.model_ctx
            scheme_reports[name] = rep
            print(f"    {name}: acc={rep['overall_acc']} f1={rep['overall_f1']} "
                  f"avg_ctx={rep['avg_ctx_tokens']} peak={rep['max_ctx_tokens']} "
                  f"oom={rep['n_overflow']} ({rep['wall_s']}s)", flush=True)

        all_report[cid] = {"schemes": scheme_reports, "tokens": ACCT.snapshot()}
        json.dump({"report": all_report[cid], "results": per_scheme},
                  open(os.path.join(args.out, f"{cid}.json"), "w"),
                  ensure_ascii=False, indent=2)

    json.dump(all_report, open(os.path.join(args.out, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n==== LIVE SUMMARY ====")
    print(json.dumps(all_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
