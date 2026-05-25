"""Poll DeepSeek balance; once a tiny call succeeds (account topped up),
run the query pilot (memory already cached, no rebuild) and print SUMMARY.

Usage (from locomo_eval/): nohup ../.venv/bin/python -m eval.wait_and_run &
"""
import os
import sys
import time
import subprocess

from openai import OpenAI
from harness.llm import _load_env


def balance_ok():
    _load_env()
    c = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
               base_url=os.environ["DEEPSEEK_BASE_URL"])
    try:
        c.chat.completions.create(model=os.environ["DEEPSEEK_MODEL"],
                                  messages=[{"role": "user", "content": "hi"}],
                                  max_tokens=2)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]


def main():
    print("WAITING_FOR_BALANCE", flush=True)
    while True:
        ok, err = balance_ok()
        if ok:
            print("BALANCE_OK -> launching query pilot", flush=True)
            break
        print(f"poll: still insufficient ({err})", flush=True)
        time.sleep(90)
    cmd = ["../.venv/bin/python", "-m", "eval.run", "--samples", "0",
           "--max-questions", "50", "--workers", "10", "--budget", "4000",
           "--schemes", "theme-mem,full-context,bm25-rag"]
    with open("results_run.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"QUERY_RUN_DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
