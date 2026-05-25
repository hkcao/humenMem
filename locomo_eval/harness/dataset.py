"""LOCOMO dataset loader + helpers.

Categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial.
For cat 5 the gold field is `adversarial_answer` and correct behavior is to abstain.
"""
import json
import os
from datetime import datetime

CAT_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain",
            4: "single-hop", 5: "adversarial"}

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "locomo10.json")


def load(path=None):
    return json.load(open(path or _DEFAULT))


def session_keys(conv):
    ks = [k for k in conv if k.startswith("session") and "date" not in k]
    return sorted(ks, key=lambda x: int(x.split("_")[1]))


def parse_date(dt_str):
    """'1:56 pm on 8 May, 2023' -> ISO date '2023-05-08' (date part only)."""
    try:
        part = dt_str.split(" on ", 1)[1]
        return datetime.strptime(part.strip(), "%d %B, %Y").strftime("%Y-%m-%d")
    except Exception:
        return "0000-00-00"


def turn_text(turn):
    txt = turn.get("text", "")
    blip = turn.get("blip_caption")
    if blip:
        txt = (txt + f" [shares an image: {blip}]").strip()
    return txt


def iter_sessions(conv):
    """Yield (session_key, iso_date, raw_dt, turns[])."""
    for sk in session_keys(conv):
        dt = conv.get(sk + "_date_time", "")
        yield sk, parse_date(dt), dt, conv[sk]


def flatten(conv):
    """Whole conversation as text, for full-context baseline."""
    lines = []
    for sk, iso, dt, turns in iter_sessions(conv):
        lines.append(f"\n=== {sk} ({dt}) ===")
        for t in turns:
            lines.append(f"[{t['dia_id']}] {t['speaker']}: {turn_text(t)}")
    return "\n".join(lines)


def gold_answer(q):
    if q.get("category") == 5:
        return q.get("adversarial_answer", "")
    return str(q.get("answer", ""))
