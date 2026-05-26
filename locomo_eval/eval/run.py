"""Pilot runner: build memory, answer questions across schemes, judge, report.

Usage:
  python -m eval.run --samples 0 --max-questions 50 --workers 8 \
                     --schemes theme-mem,full-context,bm25-rag --budget 4000
Run from locomo_eval/ dir.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from harness import dataset as ds
from harness.llm import ACCT
from harness import schemes as S
from eval import judge as J


def stratified(qa, max_q, seed=0):
    if not max_q or max_q >= len(qa):
        return qa
    by = defaultdict(list)
    for q in qa:
        by[q.get("category")].append(q)
    import random
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out, cats = [], sorted(by)
    i = 0
    while len(out) < max_q:
        c = cats[i % len(cats)]
        if by[c]:
            out.append(by[c].pop())
        i += 1
        if all(not by[c] for c in cats):
            break
    return out


def _one(scheme, q, model_ctx):
    t0 = time.time()
    try:
        r = scheme.answer(q)
    except Exception as e:  # noqa: BLE001
        r = {"answer": f"[ERROR: {e}]", "ctx_tokens": 0, "themes": []}
    verdict = J.judge(q, r["answer"])
    ctx = r["ctx_tokens"]
    overflow = bool(model_ctx and ctx > model_ctx)
    return {
        "question": q["question"], "category": q.get("category"),
        "gold": ds.gold_answer(q), "pred": r["answer"],
        "ctx_tokens": ctx, "ctx_overflow": overflow,
        "themes": r.get("themes", []),
        "switched": r.get("switched"), "stage": r.get("stage"),
        "correct": verdict["correct"], "f1": verdict.get("f1", 0.0),
        "judge_mode": verdict["mode"],
        "latency": round(time.time() - t0, 1),
    }


def run_scheme(scheme, questions, model_ctx):
    # All schemes run sequentially: stateful ones need it, and stateless ones
    # must use the same order so cross-scheme comparison is apples-to-apples
    # (e.g. continuous-conversation evaluation requires fixed question order).
    if hasattr(scheme, "reset"):
        scheme.reset()
    return [_one(scheme, q, model_ctx) for q in questions]


def aggregate(results):
    by_cat = defaultdict(lambda: [0, 0, 0.0])  # cat -> [correct, total, f1_sum]
    ctx = []
    for r in results:
        by_cat[r["category"]][0] += int(r["correct"])
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][2] += float(r.get("f1", 0.0))
        ctx.append(r["ctx_tokens"])
    tot_c = sum(v[0] for v in by_cat.values())
    tot_n = sum(v[1] for v in by_cat.values())
    tot_f1 = sum(v[2] for v in by_cat.values())
    return {
        "overall_acc": round(tot_c / tot_n, 4) if tot_n else 0,
        "overall_f1": round(tot_f1 / tot_n, 4) if tot_n else 0,
        "n": tot_n,
        "by_category": {ds.CAT_NAME.get(c, c): {
            "acc": round(v[0] / v[1], 4) if v[1] else 0,
            "f1": round(v[2] / v[1], 4) if v[1] else 0,
            "n": v[1]}
            for c, v in sorted(by_cat.items())},
        "avg_ctx_tokens": round(sum(ctx) / len(ctx)) if ctx else 0,
        "max_ctx_tokens": max(ctx) if ctx else 0,
        "n_overflow": sum(1 for r in results if r.get("ctx_overflow")),
        "n_switched": sum(1 for r in results if r.get("switched")),
        "n_stage1": sum(1 for r in results if r.get("stage") == 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="0", help="comma sample indices")
    ap.add_argument("--max-questions", type=int, default=50)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--model-ctx", type=int, default=0,
                    help="declared model window (tokens); questions whose "
                         "ctx_tokens exceed this are recorded as overflow. "
                         "0 disables the check.")
    ap.add_argument("--schemes", default="theme-mem,full-context,bm25-rag")
    ap.add_argument("--strict", action="store_true", help="theme-mem top-1 routing")
    ap.add_argument("--stratified", action="store_true",
                    help="stratified sampling across categories (default: keep "
                         "original conversation order so topic-switching effects "
                         "are visible and stateful schemes are evaluated fairly)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data = ds.load()
    sample_idx = [int(x) for x in args.samples.split(",")]
    want = args.schemes.split(",")
    os.makedirs(args.out, exist_ok=True)

    all_report = {}
    for si in sample_idx:
        sample = data[si]
        cid = sample["sample_id"]
        if args.stratified:
            questions = stratified(sample["qa"], args.max_questions)
        else:
            questions = sample["qa"][:args.max_questions] if args.max_questions \
                else sample["qa"]
        print(f"\n### sample {si} ({cid}): {len(questions)} questions "
              f"(of {len(sample['qa'])}) | schemes={want}", flush=True)

        store = None
        if any(w.startswith("theme-mem") for w in want):
            from harness.memory_store import MemoryStore
            store = MemoryStore.load_checkpoint(cid)
            n_sess = len(ds.session_keys(sample["conversation"]))
            if store is None or len(store.ingested) < n_sess:
                have = 0 if store is None else len(store.ingested)
                print(f"  ERROR: theme memory not fully built "
                      f"({have}/{n_sess}). Run: python -m eval.build "
                      f"--samples {si}", flush=True)
                want = [w for w in want if w != "theme-mem"]
            else:
                print(f"  loaded theme memory: themes={len(store.themes)} "
                      f"sessions={len(store.ingested)}", flush=True)

        scheme_reports = {}
        per_scheme_results = {}
        for name in want:
            if name == "theme-mem":
                sc = S.ThemeMemScheme(store, total_budget=args.budget,
                                      strict=args.strict)
            elif name == "theme-mem-sf":
                if store is None:
                    print("  skip theme-mem-sf: no store", flush=True)
                    continue
                sc = S.ThemeMemSummaryFirst(store, total_budget=args.budget)
            elif name in ("theme-mem-evict", "theme-mem-accum"):
                if store is None:
                    print(f"  skip {name}: no store", flush=True)
                    continue
                sc = S.ThemeMemStateful(store, total_budget=args.budget,
                                        evict=(name == "theme-mem-evict"))
            elif name == "full-context":
                sc = S.FullContextScheme(sample)
            elif name == "bm25-rag":
                sc = S.BM25Scheme(sample, total_budget=args.budget)
            elif name == "mem0":
                sc = S.Mem0Scheme(sample, top_k=30)
            else:
                continue
            print(f"  running {name}...", flush=True)
            t0 = time.time()
            res = run_scheme(sc, questions, args.model_ctx)
            per_scheme_results[name] = res
            rep = aggregate(res)
            rep["wall_s"] = round(time.time() - t0)
            rep["model_ctx"] = args.model_ctx
            scheme_reports[name] = rep
            print(f"    {name}: acc={rep['overall_acc']} "
                  f"f1={rep['overall_f1']} "
                  f"avg_ctx={rep['avg_ctx_tokens']} "
                  f"peak={rep['max_ctx_tokens']} "
                  f"oom={rep['n_overflow']} ({rep['wall_s']}s)", flush=True)

        all_report[cid] = {
            "schemes": scheme_reports,
            "tokens": ACCT.snapshot(),
        }
        json.dump({"report": all_report[cid],
                   "results": per_scheme_results},
                  open(os.path.join(args.out, f"{cid}.json"), "w"),
                  ensure_ascii=False, indent=2)

    json.dump(all_report, open(os.path.join(args.out, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n==== SUMMARY ====")
    print(json.dumps(all_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
