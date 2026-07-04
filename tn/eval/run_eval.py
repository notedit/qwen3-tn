"""评测:greedy 生成 → parse_and_apply → 五项指标 + 分桶 + badcase 导出。

用法:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m tn.eval.run_eval \
      --model runs/sft_v0/final --data data/blind_v0.jsonl --out runs/sft_v0/eval_blind

指标:
- sentence_acc   整句归一化结果与 gold 完全一致
- parse_fail     ParseError 占比(核心监控)
- span_p / span_r  编辑级精确/召回(span 位置 + 读法完全一致)
- digit_misread  NUMBER/MONEY/PHONE gold span 上模型给出错误读法的比例(P0)
- overtrigger    误报编辑数 / 句子数
- 分桶:class × domain 句准率
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tn.constants import EOS_ID, SEP, TN_PREFIX
from tn.parser import Failure, ParseError, parse_and_apply, parse_edits, render_output
from tn.train.build_dataset import load_jsonl

DIGIT_CLASSES = {"NUMBER", "MONEY", "PHONE"}

from tn.verbalizer import split_gold_edit  # noqa: E402


def acceptable_replacements(anchor: str, replacement: str, cls: str, ctx: str) -> set[str]:
    """gold 编辑的全部可接受替换文本(读法风格变体,如 幺/一、两/二)。"""
    got = split_gold_edit(anchor, replacement, cls, ctx)
    if got is None:
        return {replacement}
    p, s, _, _, cands = got
    pre = anchor[:p]
    suf = anchor[len(anchor) - s:] if s else ""
    return {pre + r + suf for r in cands}


def edits_to_spans(src: str, edits: list[tuple[str, str]]):
    spans = []
    cursor = 0
    for a, r in edits:
        pos = src.find(a, cursor)
        if pos == -1:
            raise ParseError(Failure.ANCHOR_NOT_FOUND, a)
        spans.append((pos, pos + len(a), r))
        cursor = pos + len(a)
    return spans


@torch.inference_mode()
def generate_batch(model, tok, srcs: list[str], max_new_tokens=200) -> list[str]:
    prompts = [TN_PREFIX + s + SEP for s in srcs]
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
              add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=EOS_ID,
        pad_token_id=EOS_ID,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    recs = load_jsonl(args.data)
    if args.limit:
        recs = recs[: args.limit]

    n = len(recs)
    stats = Counter()
    bucket = defaultdict(lambda: [0, 0])  # key -> [correct, total]
    gen_tokens = []
    badcases = []

    t0 = time.time()
    for i in range(0, n, args.batch):
        chunk = recs[i:i + args.batch]
        outs = generate_batch(model, tok, [r["src"] for r in chunk])
        for rec, out in zip(chunk, outs):
            src = rec["src"]
            gold_edits = [tuple(e) for e in rec["edits"]]
            gold_norm = parse_and_apply(src, render_output(gold_edits))
            gen_tokens.append(len(tok(out, add_special_tokens=False)["input_ids"]) + 1)

            err_type = None
            pred_norm = None
            pred_edits = []
            try:
                pred_edits = parse_edits(out)
                pred_norm = parse_and_apply(src, out)
            except ParseError as e:
                stats["parse_fail"] += 1
                err_type = f"parse:{e.kind.value}"

            correct = pred_norm == gold_norm
            if correct:
                stats["sent_correct"] += 1
            elif err_type is None:
                err_type = "wrong_output"

            # span 级(仅在可解析时)
            if pred_norm is not None:
                try:
                    ps = set(edits_to_spans(src, pred_edits))
                    gs_list = edits_to_spans(src, gold_edits)
                    gs = set(gs_list)
                except ParseError:
                    ps, gs, gs_list = set(), set(), []
                tp = len(ps & gs)
                stats["span_tp"] += tp
                stats["span_pred"] += len(ps)
                stats["span_gold"] += len(gs)
                stats["overtrigger_edits"] += len(ps - gs)

                # 可接受变体口径:同位置 span、读法属于 verbalizer 合法集合即算对
                pred_pos = {(s, e): r for s, e, r in ps}
                classes = rec["meta"]["classes"]
                ctxs = rec["meta"].get("ctxs", [""] * len(classes))
                sent_acceptable = len(ps) == len(gs_list)
                for (cls, ctx, (s_, e_, r_), (a_, _)) in zip(
                        classes, ctxs, gs_list, gold_edits):
                    acc_set = acceptable_replacements(a_, r_, cls, ctx)
                    pr = pred_pos.get((s_, e_))
                    ok = pr is not None and pr in acc_set
                    stats["span_tp_acc"] += int(ok)
                    if not ok:
                        sent_acceptable = False
                    if cls in DIGIT_CLASSES:
                        stats["digit_total"] += 1
                        if pr is not None and pr != r_:
                            stats["digit_misread"] += 1
                        if pr is not None and pr not in acc_set:
                            stats["digit_misread_acc"] += 1
                if sent_acceptable and not correct:
                    stats["sent_acceptable_only"] += 1
                if sent_acceptable or correct:
                    stats["sent_acceptable"] += 1

            # 分桶
            classes = rec["meta"]["classes"] or ["NEG"]
            for c in set(classes):
                bucket[f"class:{c}"][1] += 1
                bucket[f"class:{c}"][0] += int(correct)
            dom = rec["meta"].get("domain", "?")
            bucket[f"domain:{dom}"][1] += 1
            bucket[f"domain:{dom}"][0] += int(correct)
            kind = rec["meta"].get("kind", "?")
            bucket[f"kind:{kind}"][1] += 1
            bucket[f"kind:{kind}"][0] += int(correct)

            if not correct and len(badcases) < 300:
                badcases.append({"src": src, "gold": gold_edits,
                                 "pred_raw": out, "err": err_type})
        if i // args.batch % 10 == 0:
            print(f"eval {i + len(chunk)}/{n} acc so far "
                  f"{stats['sent_correct'] / max(1, i + len(chunk)):.4f}", flush=True)

    report = {
        "model": args.model,
        "data": args.data,
        "n": n,
        "sentence_acc": stats["sent_correct"] / n,
        "sentence_acc_acceptable": stats["sent_acceptable"] / n,
        "parse_fail": stats["parse_fail"] / n,
        "span_p": stats["span_tp"] / max(1, stats["span_pred"]),
        "span_r": stats["span_tp"] / max(1, stats["span_gold"]),
        "span_r_acceptable": stats["span_tp_acc"] / max(1, stats["span_gold"]),
        "digit_misread": stats["digit_misread"] / max(1, stats["digit_total"]),
        "digit_misread_acceptable": stats["digit_misread_acc"] / max(1, stats["digit_total"]),
        "digit_total": stats["digit_total"],
        "overtrigger_per_sent": stats["overtrigger_edits"] / n,
        "gen_tokens_mean": sum(gen_tokens) / n,
        "gen_tokens_p99": sorted(gen_tokens)[int(0.99 * n)],
        "eval_seconds": round(time.time() - t0, 1),
        "buckets": {k: {"acc": c / t, "n": t}
                    for k, (c, t) in sorted(bucket.items())},
    }
    with open(f"{args.out}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(f"{args.out}/badcases.jsonl", "w", encoding="utf-8") as f:
        for b in badcases:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "buckets"},
                     ensure_ascii=False, indent=2), flush=True)
    print("\n== 分桶(句准率) ==", flush=True)
    for k, v in report["buckets"].items():
        flag = "  ⚠" if v["acc"] < 0.95 else ""
        print(f"  {k:28s} acc={v['acc']:.4f} n={v['n']}{flag}", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
