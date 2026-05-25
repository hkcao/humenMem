"""Diagnose theme-mem failures: did the gold evidence turns reach the context?

Splits failure into RETRIEVAL (evidence not in assembled context) vs REASONING
(evidence present but answer wrong). Run from locomo_eval/.
"""
import re
from collections import defaultdict

from harness import dataset as ds
from harness.memory_store import MemoryStore
from eval.run import stratified

store = MemoryStore.load_checkpoint("conv-26")
sample = ds.load()[0]
questions = stratified(sample["qa"], 50)  # same seed -> same 50 as the run

cat_stat = defaultdict(lambda: {"n": 0, "ev_in_ctx": 0, "routed_has_ev": 0})
for q in questions:
    ev = q.get("evidence", []) or []
    if not ev:
        continue
    themes = store.route(q["question"], topk=3)
    ctx, _ = store.assemble(q["question"], themes)
    # which dia_ids exist anywhere in the routed themes (routing recall ceiling)
    routed_ids = set()
    for n in themes:
        th = store.themes.get(n)
        if th:
            for recs in th.days.values():
                routed_ids.update(r["dia_id"] for r in recs)
    in_ctx = all(e in ctx for e in ev)
    in_routed = all(e in routed_ids for e in ev)
    c = q["category"]
    cat_stat[c]["n"] += 1
    cat_stat[c]["ev_in_ctx"] += int(in_ctx)
    cat_stat[c]["routed_has_ev"] += int(in_routed)

print(f"{'category':12} {'n':>3} {'routing_recall':>14} {'ctx_recall':>11}")
for c in sorted(cat_stat):
    s = cat_stat[c]
    rr = s["routed_has_ev"] / s["n"]
    cr = s["ev_in_ctx"] / s["n"]
    print(f"{ds.CAT_NAME.get(c,c):12} {s['n']:>3} {rr:>14.2f} {cr:>11.2f}")
tot_n = sum(s["n"] for s in cat_stat.values())
print(f"\nrouting_recall = all evidence turns live in the top-3 routed themes")
print(f"ctx_recall     = all evidence turns survived into the 4K assembled window")
print(f"(evidence-bearing questions: {tot_n})")
