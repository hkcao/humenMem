"""Render results/summary.json into a comparison table (accuracy vs token cost).

Usage (from locomo_eval/): python -m eval.report results/summary.json
"""
import json
import sys

from harness import dataset as ds

CATS = ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]


def main(path="results/summary.json"):
    report = json.load(open(path))
    for cid, blob in report.items():
        schemes = blob["schemes"]
        tokens = blob["tokens"]  # "scheme/phase" -> {prompt,completion,reasoning,calls}
        print(f"\n## {cid}\n")
        # accuracy table: LLM-judge (paraphrase-tolerant)
        head = ["scheme", "overall"] + CATS + ["avg_ctx", "peak_ctx", "oom"]
        rows = [head]
        for name, rep in schemes.items():
            bc = rep["by_category"]
            row = [name, f"{rep['overall_acc']:.2f}"]
            for c in CATS:
                row.append(f"{bc[c]['acc']:.2f}" if c in bc else "-")
            row.append(str(rep["avg_ctx_tokens"]))
            row.append(str(rep.get("max_ctx_tokens", 0)))
            row.append(str(rep.get("n_overflow", 0)))
            rows.append(row)
        print("### LLM-judge accuracy\n")
        _print_table(rows)

        # F1 table (LOCOMO-official scoring)
        print("\n### token-F1 (LOCOMO official; cat5 = abstain 0/1)\n")
        frows = [["scheme", "overall"] + CATS]
        for name, rep in schemes.items():
            bc = rep["by_category"]
            row = [name, f"{rep.get('overall_f1', 0):.2f}"]
            for c in CATS:
                row.append(f"{bc[c].get('f1', 0):.2f}" if c in bc else "-")
            frows.append(row)
        _print_table(frows)

        # token cost table (query phase = inference; ingest = one-time build)
        print("\n### token cost\n")
        trows = [["scheme", "query_prompt", "query_completion(+reason)",
                  "ingest_total", "query_calls"]]
        for name in schemes:
            q = tokens.get(f"{name}/query", {})
            ing = tokens.get(f"{name}/ingest", {})
            ing_tot = ing.get("prompt", 0) + ing.get("completion", 0)
            trows.append([
                name, str(q.get("prompt", 0)),
                f"{q.get('completion',0)} ({q.get('reasoning',0)})",
                str(ing_tot), str(q.get("calls", 0))])
        _print_table(trows)
        j = tokens.get("judge/judge", {})
        print(f"\n(judge tokens, not charged to schemes: "
              f"prompt={j.get('prompt',0)} completion={j.get('completion',0)})")


def _print_table(rows):
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for ri, r in enumerate(rows):
        print("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
        if ri == 0:
            print("|" + "|".join("-" * (w[i] + 2) for i in range(len(r))) + "|")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/summary.json")
