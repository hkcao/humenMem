"""DeepSeek (OpenAI-compatible) client + global token accounting.

deepseek-v4-pro is a reasoning model: completion_tokens includes reasoning_tokens.
We track prompt/completion/reasoning separately, tagged by (scheme, phase), so we
can report token cost per scheme for ingest vs query phases.
"""
import os
import json
import time
import threading
from collections import defaultdict

from openai import OpenAI
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


def _load_env():
    """Find .env in cwd or any ancestor dir and load it."""
    d = os.getcwd()
    while True:
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)
            return
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


class Accountant:
    """Thread-safe token tally, keyed by (scheme, phase)."""

    def __init__(self):
        self._lock = threading.Lock()
        # key -> dict(prompt, completion, reasoning, calls)
        self.tally = defaultdict(lambda: defaultdict(int))

    def add(self, scheme, phase, usage):
        with self._lock:
            t = self.tally[(scheme, phase)]
            t["prompt"] += usage.get("prompt", 0)
            t["completion"] += usage.get("completion", 0)
            t["reasoning"] += usage.get("reasoning", 0)
            t["calls"] += 1

    def snapshot(self):
        with self._lock:
            return {f"{s}/{p}": dict(v) for (s, p), v in self.tally.items()}

    def reset(self):
        with self._lock:
            self.tally.clear()


ACCT = Accountant()


class LLM:
    def __init__(self, scheme="global"):
        _load_env()
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ["DEEPSEEK_BASE_URL"],
        )
        self.model = os.environ["DEEPSEEK_MODEL"]
        self.scheme = scheme

    def chat(self, messages, phase="query", max_tokens=2048, temperature=0.0,
             json_mode=False, retries=4):
        kwargs = dict(model=self.model, messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(**kwargs)
                u = r.usage
                rt = 0
                ctd = getattr(u, "completion_tokens_details", None)
                if ctd is not None:
                    rt = getattr(ctd, "reasoning_tokens", 0) or 0
                ACCT.add(self.scheme, phase, {
                    "prompt": u.prompt_tokens,
                    "completion": u.completion_tokens,
                    "reasoning": rt,
                })
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"LLM call failed after {retries} tries: {last}")

    def chat_json(self, messages, phase="query", max_tokens=2048, retries=4):
        """Chat expecting JSON; tolerant parse (strips code fences)."""
        raw = self.chat(messages, phase=phase, max_tokens=max_tokens,
                        json_mode=True, retries=retries)
        return _parse_json(raw)


def _parse_json(raw):
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    try:
        return json.loads(raw)
    except Exception:
        # last-ditch: find first { .. last }
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j != -1:
            return json.loads(raw[i:j + 1])
        raise
