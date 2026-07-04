"""端到端延迟压测(引擎级,batch=1 顺序 + 并发两种模式)。

计时口径 = tokenize(引擎内)+ generate + detokenize + parse_and_apply + 生成后校验,
即业务视角的单请求全链路(不含网络)。

用法:
  .venv-deploy/bin/python -m tn.serve.bench --model runs/sft_v1/final --n 500
"""

import argparse
import json
import random
import statistics
import time

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

from tn.constants import EOS_ID, SEP, TN_PREFIX
from tn.parser import ParseError, apply_edits, parse_edits
from tn.serve.postcheck import filter_edits


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/sft_v1/final")
    ap.add_argument("--data", default="data/blind_v0.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    srcs = []
    for line in open(args.data, encoding="utf-8"):
        srcs.append(json.loads(line)["src"])
    random.Random(0).shuffle(srcs)
    srcs = srcs[: args.n]

    pipe = pipeline(
        args.model,
        backend_config=TurbomindEngineConfig(
            dtype="bfloat16",
            cache_max_entry_count=0.05,   # 0.6B 短句,KV 需求极小
            enable_prefix_caching=True,
            max_batch_size=4,
        ),
    )
    gen = GenerationConfig(
        max_new_tokens=128,
        top_k=1,                          # greedy
        temperature=1.0,
        stop_token_ids=[EOS_ID],
    )

    def run_one(src: str):
        t0 = time.perf_counter()
        out = pipe([TN_PREFIX + src + SEP], gen_config=gen, do_preprocess=False)[0]
        text = out.text
        try:
            edits = parse_edits(text)
            edits, blocked = filter_edits(edits)
            normalized = apply_edits(src, edits)
            ok = True
        except ParseError:
            normalized, blocked, ok = src, 0, False
        ms = (time.perf_counter() - t0) * 1000
        return ms, out.generate_token_len, ok, blocked, normalized

    # 预热(CUDA graph 捕获 / 缓存)
    for s in srcs[:20]:
        run_one(s)

    # 顺序 batch=1(线上主路径口径)
    lat, toks, fails, blocked_total = [], [], 0, 0
    for s in srcs:
        ms, tk, ok, blocked, _ = run_one(s)
        lat.append(ms)
        toks.append(tk)
        fails += (not ok)
        blocked_total += blocked
    print(json.dumps({
        "mode": "sequential_bs1",
        "n": len(lat),
        "p50_ms": round(pctl(lat, 50), 2),
        "p90_ms": round(pctl(lat, 90), 2),
        "p99_ms": round(pctl(lat, 99), 2),
        "max_ms": round(max(lat), 2),
        "mean_ms": round(statistics.mean(lat), 2),
        "mean_new_tokens": round(statistics.mean(toks), 1),
        "p99_new_tokens": pctl(toks, 99),
        "ms_per_token_est": round(
            statistics.mean(x / max(1, t) for x, t in zip(lat, toks)), 2),
        "parse_fail": fails,
        "postcheck_blocked": blocked_total,
    }, ensure_ascii=False), flush=True)

    # 并发(continuous batching 下的排队影响)
    import threading
    lat_c = []
    lock = threading.Lock()
    idx = {"i": 0}

    def worker():
        while True:
            with lock:
                i = idx["i"]
                if i >= len(srcs):
                    return
                idx["i"] += 1
            ms, *_ = run_one(srcs[i])
            with lock:
                lat_c.append(ms)

    ts = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    print(json.dumps({
        "mode": f"concurrent_x{args.concurrency}",
        "n": len(lat_c),
        "p50_ms": round(pctl(lat_c, 50), 2),
        "p99_ms": round(pctl(lat_c, 99), 2),
        "qps": round(len(lat_c) / wall, 1),
    }, ensure_ascii=False), flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
