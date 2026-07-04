"""HTTP 层压测(调用方口径,含网络/序列化开销)。

  .venv-deploy/bin/python -m tn.serve.http_bench --url http://127.0.0.1:8100/tn --n 300
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100/tn")
    ap.add_argument("--data", default="data/blind_v0.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    srcs = [json.loads(l)["src"] for l in open(args.data, encoding="utf-8")]
    random.Random(1).shuffle(srcs)
    srcs = srcs[: args.n]

    async with httpx.AsyncClient(timeout=5.0) as client:
        for s in srcs[:10]:  # 预热
            await client.post(args.url, json={"text": s})

        lat, fallbacks = [], 0
        sem = asyncio.Semaphore(args.concurrency)

        async def one(s):
            nonlocal fallbacks
            async with sem:
                t0 = time.perf_counter()
                r = (await client.post(args.url, json={"text": s})).json()
                lat.append((time.perf_counter() - t0) * 1000)
                fallbacks += r.get("fallback", False)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(s) for s in srcs))
        wall = time.perf_counter() - t0

    print(json.dumps({
        "concurrency": args.concurrency, "n": len(lat),
        "p50_ms": round(pctl(lat, 50), 2), "p90_ms": round(pctl(lat, 90), 2),
        "p99_ms": round(pctl(lat, 99), 2), "mean_ms": round(statistics.mean(lat), 2),
        "fallback": fallbacks, "qps": round(len(lat) / wall, 1),
    }))


if __name__ == "__main__":
    asyncio.run(main())
