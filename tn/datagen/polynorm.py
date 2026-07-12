"""Apple PolyNorm-Bench zh-CN → 本项目 jsonl 评测集。

来源:github.com/apple/ml-speech-polynorm-bench(整句 written→spoken 对,27 类 × 20 句)。
编辑 span 用字符级 diff 抽取(插入/删除吸收相邻原文字符以满足锚点契约),
gold 读法保留 Apple 原标注(不经我方 verbalizer),round-trip 校验后入库。
category 映射到本项目 class 供 acceptable 口径;范围外类别标 UNK(仅 strict 口径)。

用法:
  .venv/bin/python -m tn.datagen.polynorm --src <groundtruth.jsonl> --out data/polynorm_eval.jsonl
"""

import argparse
import difflib
import json
from collections import Counter

from tn.anchor import AnchorError, apply_spans, build_edit_script
from tn.parser import ParseError, parse_and_apply, render_output

# Apple category → (本项目 class, ctx);未列出的 → ("UNK", "")
CATEGORY_MAP = {
    "Date": ("DATE", "date"),
    "Time": ("TIME", "time"),
    "Cardinal": ("NUMBER", "quantity"),
    "Decimal": ("NUMBER", "quantity"),
    "Phone Number": ("PHONE", "phone"),
    "Currency": ("MONEY", "money"),
    "Fractions": ("FRACTION", "fraction"),
    "Sports score": ("SCORE", "score"),
    "Unit": ("MEASURE", "measure"),
    "Version Number": ("VERSION", "version"),
    "License Plate or Serial Number": ("SERIAL", "serial"),
    "Vehicle or Product Code": ("SERIAL", "serial"),
    "ISBN": ("NUMBER", "code"),
}


def extract_edits(src: str, tgt: str):
    """字符 diff → [(start, end, replacement)];插入/删除吸收相邻原文字符。"""
    sm = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        spans.append([i1, i2, tgt[j1:j2]])
    fixed = []
    for s, e, r in spans:
        if s == e or not r:  # 插入或删除:锚点必须非空、读法必须非空 → 吸收邻字
            if s > 0:
                s -= 1
                r = src[s] + r
            elif e < len(src):
                e += 1
                r = r + src[e - 1]
            else:
                return None
        if fixed and s <= fixed[-1][1]:  # 与前一 span 相接/重叠 → 合并
            ps, pe, pr = fixed.pop()
            gap = src[pe:s]
            s, r = ps, pr + gap + r
        fixed.append([s, e, r])
    return [(s, e, r) for s, e, r in fixed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = Counter()
    with open(args.out, "w", encoding="utf-8") as fout:
        for line in open(args.src, encoding="utf-8"):
            rec = json.loads(line)
            src, tgt = rec["original_text"].strip(), rec["normalized_text"].strip()
            cat = rec["category"]
            if not src or "\n" in src:
                stats["reject:bad_text"] += 1
                continue
            spans = extract_edits(src, tgt)
            if spans is None:
                stats["reject:diff_fail"] += 1
                continue
            try:
                edits = build_edit_script(src, spans)
                if parse_and_apply(src, render_output(edits)) != tgt \
                        or apply_spans(src, spans) != tgt:
                    stats["reject:roundtrip"] += 1
                    continue
            except (AnchorError, ParseError):
                stats["reject:anchor"] += 1
                continue
            cls, ctx = CATEGORY_MAP.get(cat, ("UNK", ""))
            fout.write(json.dumps({
                "id": f"polynorm-{rec['index']}",
                "src": src,
                "edits": [list(e) for e in edits],
                "meta": {"classes": [cls] * len(edits), "ctxs": [ctx] * len(edits),
                         "domain": cat, "kind": "positive" if edits else "negative",
                         "source": "polynorm-bench"},
            }, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats[f"kept:{cat}"] += 1
    total = stats["kept"] + sum(v for k, v in stats.items() if k.startswith("reject"))
    print(f"kept {stats['kept']}/{total}  rejects:",
          {k[7:]: v for k, v in stats.items() if k.startswith("reject")})


if __name__ == "__main__":
    main()
