"""Build (resumable) theme memory for one or more samples, then exit.

Sequential reasoning-model ingest is slow and may be killed; just re-run this —
it resumes from the per-session checkpoint (store.pkl). Exits 0 only when the
sample is fully ingested.

Usage (from locomo_eval/): python -m eval.build --samples 0
"""
import argparse
import sys
import time

from harness import dataset as ds
from harness import schemes as S
from harness.memory_store import MemoryStore
from harness.llm import ACCT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="0")
    args = ap.parse_args()
    data = ds.load()
    done_all = True
    for si in (int(x) for x in args.samples.split(",")):
        sample = data[si]
        cid = sample["sample_id"]
        n = len(ds.session_keys(sample["conversation"]))
        existing = MemoryStore.load_checkpoint(cid)
        have = len(existing.ingested) if existing else 0
        print(f"[{cid}] sessions={n} already_ingested={have}", flush=True)
        t0 = time.time()
        store = S.build_theme_memory(sample, resume=True, verbose=True)
        ok = len(store.ingested) >= n
        print(f"[{cid}] now ingested={len(store.ingested)}/{n} "
              f"themes={len(store.themes)} (+{time.time()-t0:.0f}s) "
              f"{'COMPLETE' if ok else 'INCOMPLETE -- rerun to continue'}",
              flush=True)
        done_all = done_all and ok
    print("INGEST TOKENS:", ACCT.snapshot(), flush=True)
    return 0 if done_all else 2


if __name__ == "__main__":
    sys.exit(main())
