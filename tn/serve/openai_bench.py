"""对 OpenAI 兼容 completions 端点压测(lmdeploy api_server / vLLM 通用)。

  .venv-deploy/bin/python -m tn.serve.openai_bench --url http://127.0.0.1:8101 --n 500
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

from tn.constants import SEP, TN_PREFIX
from tn.parser import ParseError, apply_edits, parse_edits
from tn.serve.postcheck import filter_edits


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def bench(client, url, model, srcs, concurrency):
    lat, toks, fails = [], [], 0
    sem = asyncio.Semaphore(concurrency)

    async def one(src):
        nonlocal fails
        async with sem:
            t0 = time.perf_counter()
            r = await client.post(f"{url}/v1/completions", json={
                "model": model, "prompt": TN_PREFIX + src + SEP,
                "max_tokens": 128, "temperature": 0.0})
            body = r.json()
            text = body["choices"][0]["text"]
            try:
                edits, _ = filter_edits(parse_edits(text))
                apply_edits(src, edits)
            except ParseError:
                fails += 1
            lat.append((time.perf_counter() - t0) * 1000)
            toks.append(body["usage"]["completion_tokens"])

    t0 = time.perf_counter()
    await asyncio.gather(*(one(s) for s in srcs))
    wall = time.perf_counter() - t0
    return {
        "concurrency": concurrency, "n": len(lat),
        "p50_ms": round(pctl(lat, 50), 2), "p90_ms": round(pctl(lat, 90), 2),
        "p99_ms": round(pctl(lat, 99), 2), "mean_ms": round(statistics.mean(lat), 2),
        "mean_new_tokens": round(statistics.mean(toks), 1),
        "p99_new_tokens": pctl(toks, 99),
        "parse_fail": fails, "qps": round(len(lat) / wall, 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8101")
    ap.add_argument("--data", default="data/blind_v0.jsonl")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    srcs = [json.loads(l)["src"] for l in open(args.data, encoding="utf-8")]
    random.Random(0).shuffle(srcs)
    srcs = srcs[: args.n]

    async with httpx.AsyncClient(timeout=30.0) as client:
        model = (await client.get(f"{args.url}/v1/models")).json()["data"][0]["id"]
        for s in srcs[:20]:
            await client.post(f"{args.url}/v1/completions", json={
                "model": model, "prompt": TN_PREFIX + s + SEP,
                "max_tokens": 128, "temperature": 0.0})
        for c in (1, 4, 8):
            print(json.dumps(await bench(client, args.url, model, srcs, c)),
                  flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
