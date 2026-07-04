"""TRT-LLM 进程内 LLM API 延迟探针(无 HTTP、无 orchestrator IPC)。

  CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/usr/lib64/openmpi/lib \
    .venv-trt/bin/python -m tn.serve.trt_inproc_probe
"""

import json
import random
import statistics
import time

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from tn.constants import SEP, TN_PREFIX


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    llm = LLM(
        model="runs/sft_v1/final-trt",
        cuda_graph_config=CudaGraphConfig(enable_padding=True,
                                          batch_sizes=[1, 2, 4, 8]),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.2),
        max_batch_size=8,
        max_seq_len=1024,
    )
    greedy = SamplingParams(max_tokens=128, temperature=0.0)

    # 纯 decode 速率
    long_sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True)
    best = 9e9
    for _ in range(4):
        t0 = time.perf_counter()
        out = llm.generate([TN_PREFIX + "今天天气不错。" + SEP], long_sp)
        dt = (time.perf_counter() - t0) * 1000
        n = len(out[0].outputs[0].token_ids)
        best = min(best, dt / n)
    print(f"inproc decode: {best:.3f} ms/token", flush=True)

    # 真实分布单请求延迟
    srcs = [json.loads(l)["src"] for l in open("data/blind_v0.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(srcs)
    srcs = srcs[:400]
    for s in srcs[:20]:
        llm.generate([TN_PREFIX + s + SEP], greedy)
    lat, toks = [], []
    for s in srcs:
        t0 = time.perf_counter()
        out = llm.generate([TN_PREFIX + s + SEP], greedy)
        lat.append((time.perf_counter() - t0) * 1000)
        toks.append(len(out[0].outputs[0].token_ids))
    print(json.dumps({
        "mode": "trt_inproc_bs1", "n": len(lat),
        "p50_ms": round(pctl(lat, 50), 2), "p90_ms": round(pctl(lat, 90), 2),
        "p99_ms": round(pctl(lat, 99), 2), "mean_ms": round(statistics.mean(lat), 2),
        "mean_new_tokens": round(statistics.mean(toks), 1),
    }), flush=True)
    print("INPROC_DONE", flush=True)


if __name__ == "__main__":
    main()
