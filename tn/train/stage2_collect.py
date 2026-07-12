"""Stage-2 rejection sampling 收集器(PLAN §3.2)。

对给定池子,用 Stage-1 模型 temperature 采样 n 次 + greedy 一次;任一采样的归一化
结果与 gold 不一致即判为难例。输出 = 难例(原 gold 标签)+ 等量随机正确样本,
供低 lr 精调。错误只用于「选样本」,标签仍是程序化 gold —— 不引入模型输出为标签。

用法:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m tn.train.stage2_collect \
      --model runs/sft_v6/final --pool data/pool_s2.jsonl --out data/stage2_v6.jsonl
"""

import argparse
import json
import random
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tn.constants import EOS_ID, SEP, TN_PREFIX
from tn.parser import ParseError, parse_and_apply, render_output
from tn.train.build_dataset import load_jsonl


@torch.inference_mode()
def gen(model, tok, srcs, do_sample, n, temperature=0.8, max_new_tokens=160):
    enc = tok([TN_PREFIX + s + SEP for s in srcs], return_tensors="pt", padding=True,
              padding_side="left", add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature if do_sample else None,
        num_return_sequences=n, eos_token_id=EOS_ID, pad_token_id=EOS_ID)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [texts[i * n:(i + 1) * n] for i in range(len(srcs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    recs = load_jsonl(args.pool)
    print(f"pool={len(recs)}", flush=True)

    hard, easy = [], []
    stats = Counter()
    t0 = time.time()
    for i in range(0, len(recs), args.batch):
        chunk = recs[i:i + args.batch]
        srcs = [r["src"] for r in chunk]
        sampled = gen(model, tok, srcs, True, args.n_samples, args.temperature)
        greedy = gen(model, tok, srcs, False, 1)
        for rec, outs, g in zip(chunk, sampled, greedy):
            gold = parse_and_apply(rec["src"], render_output([tuple(e) for e in rec["edits"]]))
            bad = 0
            for o in outs + g:
                try:
                    if parse_and_apply(rec["src"], o) != gold:
                        bad += 1
                except ParseError:
                    bad += 1
            greedy_ok = True
            try:
                greedy_ok = parse_and_apply(rec["src"], g[0]) == gold
            except ParseError:
                greedy_ok = False
            if bad:
                rec["meta"]["s2"] = {"bad": bad, "greedy_ok": greedy_ok}
                hard.append(rec)
                stats["hard"] += 1
                stats["greedy_wrong"] += int(not greedy_ok)
            else:
                easy.append(rec)
        if (i // args.batch) % 10 == 0:
            print(f"[{time.time() - t0:6.0f}s] {i + len(chunk)}/{len(recs)} "
                  f"hard={stats['hard']} greedy_wrong={stats['greedy_wrong']}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(easy)
    keep = hard + easy[: len(hard)]  # 难例 + 等量正确样本
    rng.shuffle(keep)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE hard={len(hard)} (greedy_wrong={stats['greedy_wrong']}) "
          f"out={len(keep)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
