"""走 OpenAI 端点的精度回归(量化/引擎变更后跑,防数字类回退)。

  .venv-deploy/bin/python -m tn.serve.endpoint_eval --url http://127.0.0.1:8104 --n 1200
"""

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict

import httpx

from tn.constants import SEP, TN_PREFIX
from tn.parser import ParseError, parse_and_apply, render_output

DIGIT_CLASSES = {"NUMBER", "MONEY", "PHONE"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", default="data/blind_v0.jsonl")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.Random(0).shuffle(recs)
    recs = recs[: args.n]

    stats = Counter()
    cls_acc = defaultdict(lambda: [0, 0])
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=30.0) as client:
        model = (await client.get(f"{args.url}/v1/models")).json()["data"][0]["id"]

        async def one(rec):
            async with sem:
                r = await client.post(f"{args.url}/v1/completions", json={
                    "model": model, "prompt": TN_PREFIX + rec["src"] + SEP,
                    "max_tokens": 128, "temperature": 0.0})
            text = r.json()["choices"][0]["text"]
            gold = parse_and_apply(rec["src"], render_output(
                [tuple(e) for e in rec["edits"]]))
            try:
                pred = parse_and_apply(rec["src"], text)
            except ParseError:
                stats["parse_fail"] += 1
                pred = None
            ok = pred == gold
            stats["correct"] += ok
            stats["total"] += 1
            for c in set(rec["meta"]["classes"] or ["NEG"]):
                cls_acc[c][1] += 1
                cls_acc[c][0] += ok
            if not ok and any(c in DIGIT_CLASSES for c in rec["meta"]["classes"]):
                stats["digit_sent_wrong"] += 1

        await asyncio.gather(*(one(r) for r in recs))

    print(json.dumps({
        "url": args.url, "n": stats["total"],
        "sentence_acc": round(stats["correct"] / stats["total"], 4),
        "parse_fail": stats["parse_fail"],
        "digit_sent_wrong": stats["digit_sent_wrong"],
        "per_class": {c: round(a / t, 4) for c, (a, t) in sorted(cls_acc.items())},
    }, ensure_ascii=False), flush=True)
    print("EP_EVAL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
