"""造数 runner:异步批量调 DeepSeek,校验入库,断点续跑。

用法(经 launch.sh 后台跑):
  .venv/bin/python -m tn.datagen.run_gen --out data/train_v0.jsonl --n 5000 --seed 1
  .venv/bin/python -m tn.datagen.run_gen --out data/blind_v0.jsonl --n 3000 --seed 900001 --variant 1

断点续跑:输出文件里已有的 spec id 自动跳过,直接重跑同一命令即可。
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter

from tn.anchor import AnchorError, apply_spans, build_edit_script
from tn.constants import has_pua
from tn.datagen.client import DeepSeekClient
from tn.datagen.specs import Spec, build_specs, judge_prompt, negative_prompt, positive_prompt
from tn.parser import ParseError, parse_and_apply, render_output
from tn.verbalizer import is_valid, valid_readings

DIGIT_RE = re.compile(r"[0-90-9]")
SYMBOL_RE = re.compile(r"[¥$%℃~～£€±½⅓⅔¼¾⅕⅛Ω×÷√²³π≈°′″·]")
HANZI_NUM_RE = re.compile(r"[一二三四五六七八九十百千万亿两零]")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
BOUNDARY_BAD = re.compile(r"[0-9A-Za-z.0-9]")

# 语境-读法冲突守卫:LLM 造句可能把 quantity 语境写成编号语境(或反之),
# 导致 gold 与句子语义相悖(标签噪声)。高精度词法规则,命中即弃。
_CODE_CUE_BEFORE = re.compile(r"(编号|代码|号码|工号|单号|卡号|证号|房号|账号|编码|代号)[是为:: ]?$")
_QTY_CUE_BEFORE = re.compile(r"(共|多达|达到|达|约|近|超过|增至|增加到|减至|高达)$")
_QTY_CUE_AFTER = re.compile(r"^(个|人|件|元|次|条|名|张|吨|台|只|辆|栋|间|份|岁)")


def _find_free(text: str, frag: str, occupied: list[tuple[int, int]]):
    """找 frag 首个不与 occupied 重叠且边界不粘连的出现位置,无则 None。"""
    start = 0
    while True:
        pos = text.find(frag, start)
        if pos == -1:
            return None
        end = pos + len(frag)
        before = text[pos - 1] if pos > 0 else ""
        after = text[end] if end < len(text) else ""
        if (all(e <= pos or s >= end for s, e in occupied)
                and not BOUNDARY_BAD.match(before) and not BOUNDARY_BAD.match(after)):
            return pos, end
        start = pos + 1


def ctx_conflict(text: str, s: int, e: int, cls: str, ctx: str) -> bool:
    if cls != "NUMBER":
        return False
    before = text[max(0, s - 4):s]
    after = text[e:e + 2]
    if ctx == "quantity":
        return bool(_CODE_CUE_BEFORE.search(before)) or after.startswith("号")
    if ctx == "code":
        return bool(_QTY_CUE_BEFORE.search(before)) or bool(_QTY_CUE_AFTER.match(after))
    return False


def _validate_positive(spec: Spec, text: str):
    """返回 (record, None) 或 (None, 拒绝原因)。"""
    text = text.strip()
    if not text or "\n" in text or has_pua(text):
        return None, "bad_text"
    if not 5 <= len(text) <= 120:
        return None, "bad_length"
    spans = []
    cursor = 0
    for it in spec.items:
        pos = text.find(it.written, cursor)
        if pos == -1:
            return None, "fragment_missing"
        before = text[pos - 1] if pos > 0 else ""
        after = text[pos + len(it.written)] if pos + len(it.written) < len(text) else ""
        if BOUNDARY_BAD.match(before) or BOUNDARY_BAD.match(after):
            return None, "fragment_glued"
        if ctx_conflict(text, pos, pos + len(it.written), it.cls, it.ctx):
            return None, "ctx_conflict"
        spans.append((pos, pos + len(it.written), it.reading))
        cursor = pos + len(it.written)
    # 负槽位:必须原样出现且不与编辑 span 重叠;其内部数字不算越界
    protected = [(s, e) for s, e, _ in spans]
    for d in spec.distractors:
        got = _find_free(text, d, protected)
        if got is None:
            return None, "distractor_missing"
        protected.append(got)
    # 保护区之外不得有任何数字/NSW 符号(否则编辑脚本对该句不完备 → 标签噪声)
    rest = "".join(ch for i, ch in enumerate(text)
                   if not any(s <= i < e for s, e in protected))
    if DIGIT_RE.search(rest) or SYMBOL_RE.search(rest):
        return None, "extra_digits"
    # 三道校验:gold 反向校验 / 锚点可编译 / round-trip
    for it in spec.items:
        if not is_valid(it.cls, it.written, it.ctx, it.reading):
            return None, "gold_invalid"
    try:
        edits = build_edit_script(text, spans)
    except AnchorError:
        return None, "anchor_fail"
    try:
        if parse_and_apply(text, render_output(edits)) != apply_spans(text, spans):
            return None, "roundtrip_mismatch"
    except ParseError:
        return None, "roundtrip_parse_error"
    rec = {
        "id": spec.id,
        "src": text,
        "edits": [list(e) for e in edits],
        "meta": {
            "classes": [it.cls for it in spec.items],
            "ctxs": [it.ctx for it in spec.items],
            "nsw": [[it.written, it.reading] for it in spec.items],
            "distractors": spec.distractors,
            "domain": spec.domain,
            "kind": "positive",
            "source": "synth-dsv4",
        },
    }
    return rec, None


_MULTI_LATIN_RE = re.compile(r"[A-Za-z]+(?: [A-Za-z]+)+")


def _validate_negative(spec: Spec, text: str):
    text = text.strip()
    if not text or "\n" in text or has_pua(text):
        return None, "bad_text"
    if not 5 <= len(text) <= 120:
        return None, "bad_length"
    if spec.neg_type == "models":
        # 数字只允许出现在给定不展开片段内
        protected = []
        for d in spec.distractors:
            got = _find_free(text, d, protected)
            if got is None:
                return None, "distractor_missing"
            protected.append(got)
        rest = "".join(ch for i, ch in enumerate(text)
                       if not any(s <= i < e for s, e in protected))
        if DIGIT_RE.search(rest) or SYMBOL_RE.search(rest):
            return None, "neg_has_digits"
    elif DIGIT_RE.search(text) or SYMBOL_RE.search(text):
        return None, "neg_has_digits"
    if spec.neg_type == "hanzi_num" and not HANZI_NUM_RE.search(text):
        return None, "neg_missing_hanzi_num"
    if spec.neg_type == "latin" and not LATIN_RE.search(text):
        return None, "neg_missing_latin"
    if spec.neg_type == "latin_long" and not _MULTI_LATIN_RE.search(text):
        return None, "neg_missing_latin"
    if spec.neg_type in ("plain", "entity", "classical") and LATIN_RE.search(text):
        return None, "neg_has_latin"
    rec = {
        "id": spec.id,
        "src": text,
        "edits": [],
        "meta": {"classes": [], "ctxs": [], "domain": spec.domain,
                 "kind": f"negative_{spec.neg_type}", "source": "synth-dsv4"},
    }
    return rec, None


# ---- judge 质检:对歧义类样本做选择题校验,判错即弃 ----

_CONFUSABLE = {("NUMBER", "code"), ("NUMBER", "quantity"), ("PHONE", "phone"),
               ("SCORE", "score"), ("SERIAL", "serial")}


def _confusion_candidates(it) -> list[str]:
    cands = {it.reading}
    if it.cls in ("NUMBER", "PHONE", "SERIAL"):
        for cls2, ctx2 in (("NUMBER", "quantity"), ("NUMBER", "code"), ("PHONE", "phone")):
            if (cls2, ctx2) != (it.cls, it.ctx):
                for r in list(valid_readings(cls2, it.written, ctx2))[:1]:
                    cands.add(r)
    elif it.cls == "SCORE":
        cands |= set(list(valid_readings("TIME", it.written, "time"))[:1])
    if len(cands) < 2:
        return []
    return sorted(cands)


async def _qa_check(client: DeepSeekClient, rec: dict, spec: Spec, text: str) -> bool:
    """True=通过或无需质检;False=判定标签可疑,弃样本。"""
    for it in spec.items:
        if (it.cls, it.ctx) not in _CONFUSABLE:
            continue
        cands = _confusion_candidates(it)
        if not cands:
            continue
        obj = await client.chat_json(
            judge_prompt(text, it.written, cands), temperature=0.0, max_tokens=50)
        if not obj or "choice" not in obj:
            return True  # judge 不可用不拦截,只拦明确判错
        idx = ord(str(obj["choice"]).strip()[:1].upper()) - 65
        if not (0 <= idx < len(cands)) or cands[idx] != it.reading:
            return False
    return True


# ---- 主流程 ----

def _load_done(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def _worker(client, batch, variant, out_q, stats, qa_ratio, rng):
    kind = batch[0].kind
    prompt = positive_prompt(batch, variant) if kind == "positive" else negative_prompt(batch)
    try:
        obj = await client.chat_json(prompt, temperature=0.9, max_tokens=2400)
    except Exception as e:  # 重试穷尽,留给下次续跑
        stats[f"api_fail:{type(e).__name__}"] += len(batch)
        return
    if not isinstance(obj, dict) or not isinstance(obj.get("sentences"), list):
        stats["bad_json"] += len(batch)
        return
    by_id = {}
    for s in obj["sentences"]:
        if isinstance(s, dict) and "id" in s and "text" in s:
            by_id[str(s["id"])] = str(s["text"])
    for spec in batch:
        text = by_id.get(spec.id)
        if text is None:
            stats["missing_in_response"] += 1
            continue
        rec, err = (_validate_positive(spec, text) if kind == "positive"
                    else _validate_negative(spec, text))
        if err:
            stats[f"reject:{err}"] += 1
            continue
        if kind == "positive" and rng.random() < qa_ratio:
            if not await _qa_check(client, rec, spec, text):
                stats["qa_dropped"] += 1
                with open(os.environ.get("REVIEW_POOL", "data/review_pool.jsonl"),
                          "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            stats["qa_passed"] += 1
        stats["accepted"] += 1
        await out_q.put(rec)


async def _writer(path: str, out_q: asyncio.Queue, seen_src: set):
    with open(path, "a", encoding="utf-8") as f:
        while True:
            rec = await out_q.get()
            if rec is None:
                break
            if rec["src"] in seen_src:  # 精确去重(n-gram 去重离线做)
                continue
            seen_src.add(rec["src"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--neg-ratio", type=float, default=0.25)
    ap.add_argument("--pair-ratio", type=float, default=0.0,
                    help="code/quantity 最小对占比(逐位 vs 数值消歧监督)")
    ap.add_argument("--boost", default="",
                    help='类别权重倍率,如 "MONEY:3,FRACTION:3"')
    ap.add_argument("--variant", type=int, default=0, help="prompt 措辞版本,盲测集用 1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--qa-ratio", type=float, default=0.15,
                    help="正样本抽此比例做 judge 质检")
    ap.add_argument("--batch", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.boost:
        from tn.datagen.specs import set_class_boost
        set_class_boost({k: float(v) for k, v in
                         (kv.split(":") for kv in args.boost.split(","))})
    specs = build_specs(args.n, args.seed, args.neg_ratio, pair_ratio=args.pair_ratio)
    done = _load_done(args.out)
    todo = [s for s in specs if s.id not in done]
    seen_src = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                seen_src.add(json.loads(line)["src"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"total={len(specs)} done={len(done)} todo={len(todo)}", flush=True)

    client = DeepSeekClient(concurrency=args.concurrency)
    stats = Counter()
    rng = random.Random(args.seed + 777)
    out_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    writer = asyncio.create_task(_writer(args.out, out_q, seen_src))

    # 同 kind 组 batch(正/负样本 prompt 不同)
    batches = []
    for kind in ("positive", "negative"):
        group = [s for s in todo if s.kind == kind]
        bs = args.batch if kind == "positive" else args.batch + 3
        batches += [group[i:i + bs] for i in range(0, len(group), bs)]
    rng.shuffle(batches)

    t0 = time.time()
    pending = set()
    max_inflight = args.concurrency * 2
    for i, b in enumerate(batches):
        pending.add(asyncio.create_task(
            _worker(client, b, args.variant, out_q, stats, args.qa_ratio, rng)))
        if len(pending) >= max_inflight:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"[{el:7.0f}s] batch {i}/{len(batches)} "
                  f"accepted={stats['accepted']} stats={dict(stats)}", flush=True)
    if pending:
        await asyncio.wait(pending)
    await out_q.put(None)
    await writer
    print(f"DONE in {time.time() - t0:.0f}s stats={dict(stats)}", flush=True)
    print(f"requests={client.n_requests}", flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
