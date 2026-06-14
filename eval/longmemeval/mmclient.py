"""Minimal MiniMax-M3 client (CN endpoint, OpenAI-compatible chat/completions).

MiniMax-M3 is a reasoning model: it emits a <think>...</think> block before the answer,
so we request enough tokens and strip the think block before returning.

Key is read from $MINIMAX_API_KEY or ~/.minimax_key (kept outside the repo, never
committed).
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("MINIMAX_BASE", "https://api.minimaxi.com/v1")
MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M3")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Global pacing: enforce a minimum gap between request starts across all threads, to stay
# under MiniMax's per-minute quota (set via MINIMAX_MIN_INTERVAL seconds; 0 = off).
_MIN_INTERVAL = float(os.environ.get("MINIMAX_MIN_INTERVAL", "0"))
_rate_lock = threading.Lock()
_last_start = [0.0]


def _throttle():
    if _MIN_INTERVAL <= 0:
        return
    with _rate_lock:
        wait = _last_start[0] + _MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_start[0] = time.monotonic()


def _key() -> str:
    k = os.environ.get("MINIMAX_API_KEY")
    if not k:
        p = os.path.expanduser("~/.minimax_key")
        if os.path.exists(p):
            k = open(p, encoding="utf-8").read().strip()
    if not k:
        raise RuntimeError("set MINIMAX_API_KEY or create ~/.minimax_key")
    return k


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def chat(prompt, max_tokens=800, temperature=0.0, retries=8) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    last = None
    for attempt in range(retries):
        try:
            _throttle()
            req = urllib.request.Request(
                BASE + "/chat/completions", data=body,
                headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read().decode("utf-8"))
            choices = d.get("choices")
            if not choices:
                raise RuntimeError(f"no choices: {str(d)[:300]}")
            content = choices[0].get("message", {}).get("content", "")
            out = strip_think(content)
            return out if out else content.strip()
        except urllib.error.HTTPError as e:  # rate limit gets a much longer backoff
            last = e
            time.sleep(min(15 * (attempt + 1), 90) if e.code == 429 else min(2 ** attempt, 30))
        except Exception as e:  # network / transient -> short backoff
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"minimax call failed after {retries} tries: {last}")
