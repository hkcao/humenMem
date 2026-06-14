#!/usr/bin/env python3
"""LongMemEval Phase B: end-to-end QA with our retrieval + MiniMax-M3, judged the
official LongMemEval way.

Stages (run all with `all`, or individually):
  gen      retrieve (our BM25) -> build verbatim prompt -> MiniMax-M3 answer
  eval     judge each hypothesis with the official per-type / abstention prompts
  metrics  task-averaged + overall + abstention accuracy, plus per-type

Usage:
  python3 run_phaseb.py all  --config bm25 --topk 10 --limit 30 --stratified
  python3 run_phaseb.py all  --config bm25 --topk 10            # full 500
  python3 run_phaseb.py metrics --tag bm25_k10_n30
"""
import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H
import mmclient as MM

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

ORACLE = os.path.join(DATA, "longmemeval_oracle.json")
HAYSTACK = os.path.join(DATA, "longmemeval_s_cleaned.json")
CORE_TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "temporal-reasoning", "knowledge-update"]


def select(refs, limit, stratified):
    if not limit:
        return refs
    if not stratified:
        return refs[:limit]
    by_type = defaultdict(list)
    for r in refs:
        by_type[r["question_type"]].append(r)
    picked, i = [], 0
    while len(picked) < limit and any(i < len(v) for v in by_type.values()):
        for t in CORE_TYPES:
            if i < len(by_type[t]) and len(picked) < limit:
                picked.append(by_type[t][i])
        i += 1
    return picked


# --- stages -----------------------------------------------------------------

def stage_gen(args, refs):
    print(f"[gen] loading haystack {HAYSTACK} ...", flush=True)
    hay = {x["question_id"]: x for x in H.load_instances(HAYSTACK)}
    todo = [r for r in refs if r["question_id"] in hay]
    print(f"[gen] {len(todo)} questions, config={args.config} topk={args.topk}", flush=True)

    def work(ref):
        inst = hay[ref["question_id"]]
        sessions = H.retrieve_sessions(inst, args.topk, config=args.config)
        try:
            hyp = MM.chat(H.gen_prompt(inst, sessions), max_tokens=2048)
        except Exception as e:
            print(f"  ! gen failed {ref['question_id']}: {e}", flush=True)
            return None  # not written -> retried on rerun
        return {"question_id": ref["question_id"], "question_type": ref["question_type"],
                "hypothesis": hyp}

    path = os.path.join(OUT, f"hyp_{args.tag}.jsonl")
    incremental_pool(work, todo, path, args.workers, "gen")
    return path


def stage_eval(args, refs):
    hyps = read_jsonl(os.path.join(OUT, f"hyp_{args.tag}.jsonl"))
    ref_by_id = {r["question_id"]: r for r in refs}
    print(f"[eval] judging {len(hyps)} hypotheses with MiniMax-M3 ...", flush=True)

    def work(h):
        r = ref_by_id[h["question_id"]]
        prompt = H.judge_prompt(r["question_id"], r["question_type"], r["question"],
                                r["answer"], h["hypothesis"])
        try:
            resp = MM.chat(prompt, max_tokens=1024)
        except Exception as e:
            print(f"  ! judge failed {h['question_id']}: {e}", flush=True)
            return None  # not written -> retried on rerun
        return {**h, "autoeval_label": {"model": MM.MODEL, "label": H.parse_label(resp)},
                "judge_raw": resp}

    path = os.path.join(OUT, f"eval_{args.tag}.jsonl")
    incremental_pool(work, hyps, path, args.workers, "eval")
    return path


def stage_metrics(args):
    path = os.path.join(OUT, f"eval_{args.tag}.jsonl")
    rows = read_jsonl(path)
    per_type, abst = defaultdict(list), []
    for r in rows:
        label = bool(r["autoeval_label"]["label"])
        if "_abs" in r["question_id"]:
            abst.append(label)
        else:
            per_type[r["question_type"]].append(label)

    print(f"\n===== LongMemEval Phase B metrics  (tag={args.tag}) =====")
    print(f"judge/answer model: {MM.MODEL}")
    type_accs = []
    for t in CORE_TYPES:
        labs = per_type.get(t, [])
        if labs:
            acc = sum(labs) / len(labs)
            type_accs.append(acc)
            print(f"  {t:28s} {acc:.3f}  (n={len(labs)})")
    all_non_abs = [x for labs in per_type.values() for x in labs]
    task_avg = sum(type_accs) / len(type_accs) if type_accs else 0.0
    overall = sum(all_non_abs) / len(all_non_abs) if all_non_abs else 0.0
    print(f"  {'-'*40}")
    print(f"  Task-averaged accuracy   {task_avg:.3f}  (mean of {len(type_accs)} types)")
    print(f"  Overall accuracy         {overall:.3f}  (micro, n={len(all_non_abs)})")
    if abst:
        print(f"  Abstention accuracy      {sum(abst)/len(abst):.3f}  (n={len(abst)})")
    print("=" * 56)


# --- helpers ----------------------------------------------------------------

def incremental_pool(fn, items, out_path, workers, tag):
    """Append each result as it completes; skip ids already in out_path; results that
    return None (failed) are not written, so a rerun retries exactly those."""
    done_ids = set()
    if os.path.exists(out_path):
        done_ids = {r["question_id"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["question_id"] not in done_ids]
    print(f"  [{tag}] {len(done_ids)} cached, {len(todo)} to run", flush=True)

    lock = threading.Lock()
    done, total = len(done_ids), len(items)
    failed = 0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                else:
                    failed += 1
                if (done + failed) % 10 == 0 or done == total:
                    print(f"  [{tag}] {done}/{total} done, {failed} failed", flush=True)
    print(f"  [{tag}] wrote {out_path} ({done}/{total}, {failed} failed this pass)", flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["gen", "eval", "metrics", "all"])
    p.add_argument("--config", default="bm25", choices=["bm25", "oracle", "no-mem"])
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stratified", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    if not args.tag:
        n = args.limit or "all"
        args.tag = f"{args.config}_k{args.topk}_n{n}{'_strat' if args.stratified else ''}"

    refs = H.load_instances(ORACLE)
    refs = select(refs, args.limit, args.stratified)
    print(f"selected {len(refs)} questions; type dist: "
          f"{dict(Counter(r['question_type'] for r in refs))}; "
          f"abs={sum('_abs' in r['question_id'] for r in refs)}", flush=True)

    if args.stage in ("gen", "all"):
        stage_gen(args, refs)
    if args.stage in ("eval", "all"):
        stage_eval(args, refs)
    if args.stage in ("metrics", "all"):
        stage_metrics(args)


if __name__ == "__main__":
    main()
