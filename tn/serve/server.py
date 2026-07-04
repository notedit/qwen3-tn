"""TN 推理服务(生产形态参考实现)。

  .venv-deploy/bin/python -m tn.serve.server --model runs/sft_v1/final --port 8100

POST /tn {"text": "..."}  →
  {"normalized": "...", "edits": [[a,r],...], "fallback": false,
   "postcheck_blocked": 0, "latency_ms": 12.3}

设计对应 docs/PLAN.md §5:
- greedy + eos 早停;无编辑句 = prefill + 1 step
- 生成后校验:非法编辑逐 span 拦截(回退原文),不整句作废
- 超时熔断(默认 35ms):返回 fallback=true,由上游走 WFST 全链路
- /healthz 探活;/metrics 简易计数(解析失败率/拦截数/超时数)
"""

import argparse
import asyncio
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from tn.constants import EOS_ID, SEP, TN_PREFIX
from tn.parser import ParseError, apply_edits, parse_edits
from tn.serve.postcheck import filter_edits

app = FastAPI()
STATE: dict = {}
METRICS: Counter = Counter()


class TNRequest(BaseModel):
    text: str
    timeout_ms: float = 35.0


def _infer(src: str):
    out = STATE["pipe"]([TN_PREFIX + src + SEP],
                        gen_config=STATE["gen"], do_preprocess=False)[0]
    return out.text


@app.post("/tn")
async def tn(req: TNRequest):
    t0 = time.perf_counter()
    METRICS["requests"] += 1
    loop = asyncio.get_running_loop()
    try:
        raw = await asyncio.wait_for(
            loop.run_in_executor(STATE["pool"], _infer, req.text),
            timeout=req.timeout_ms / 1000)
    except asyncio.TimeoutError:
        METRICS["timeout_fallback"] += 1
        return {"normalized": req.text, "edits": [], "fallback": True,
                "reason": "timeout",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    try:
        edits = parse_edits(raw)
    except ParseError as e:
        METRICS["parse_fail"] += 1
        return {"normalized": req.text, "edits": [], "fallback": True,
                "reason": f"parse:{e.kind.value}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    edits, blocked = filter_edits(edits)
    METRICS["postcheck_blocked"] += blocked
    try:
        normalized = apply_edits(req.text, edits)
    except ParseError as e:
        METRICS["parse_fail"] += 1
        return {"normalized": req.text, "edits": [], "fallback": True,
                "reason": f"apply:{e.kind.value}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    return {"normalized": normalized, "edits": edits, "fallback": False,
            "postcheck_blocked": blocked,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return dict(METRICS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/sft_v1/final")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-batch", type=int, default=4)
    args = ap.parse_args()

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
    STATE["pipe"] = pipeline(
        args.model,
        backend_config=TurbomindEngineConfig(
            dtype="bfloat16",
            cache_max_entry_count=0.05,
            enable_prefix_caching=True,
            max_batch_size=args.max_batch,
        ),
    )
    STATE["gen"] = GenerationConfig(
        max_new_tokens=128, top_k=1, temperature=1.0, stop_token_ids=[EOS_ID])
    # 线程数 = max_batch:超出即在此排队,continuous batching 合批
    STATE["pool"] = ThreadPoolExecutor(max_workers=args.max_batch)
    # 预热
    for _ in range(8):
        _infer("预热句子,今天是3月8日。")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
