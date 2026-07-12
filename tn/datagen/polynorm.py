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

# 人工复核后移除的样本(2026-07-12 逐条 review):index → 原因
# 判定原则:合理的覆盖缺口(数学式/罗马数字/度分秒)保留;以下四类移除 ——
#   symbol   符号逐字读(点/斜杠/井号/下标),属 Stage-A 规则层职责,非 TN 模型任务
#   reorder  读法需要跨 span 重排(N→北纬 前置等),span-edit 契约结构性不可能
#   policy   与本项目已定读法政策冲突(型号不展开 / v 前缀读法未定)
#   gold     Apple 标注自身错误或 diff 碎片伪影
BLOCKLIST = {
    # URL/Email、Hashtag、化学下标:symbol
    **{i: "symbol" for i in (246, 248, 256, 259, 261, 263, 265, 271, 273, 275,
                             277, 279, 341, 345, 351, 352, 354, 360)},
    # 生物编号:E. 删点 / 罗马 II.4:symbol
    503: "symbol", 512: "symbol",
    # 乐谱:拍号领域读法(4/4→四四)/音符符号/Op.No.展开/罗马和弦:symbol
    **{i: "symbol" for i in (461, 462, 463, 465, 468, 471, 472, 473, 474, 477, 480)},
    # 法律编号:杠/斜杠/括号读法约定:symbol
    **{i: "symbol" for i in (369, 371, 374, 375, 376, 377, 379)},
    # 坐标:方位字母重排(N→北纬前置):reorder
    **{i: "reorder" for i in (401, 402, 405, 406, 409, 410, 413, 414, 417, 418, 419)},
    # 数学式 gold 碎片伪影(括号/竖线/空格锚点):gold
    222: "gold", 227: "gold", 238: "gold",
    # 罗马数字:LVI 重排 / XI五 gold 丢字
    124: "reorder", 137: "gold",
    # 版本号:v 读「版本」约定未定 / Windows·Android 与「型号不展开」政策冲突:policy
    421: "policy", 423: "policy", 425: "policy", 426: "policy",
    430: "policy", 438: "policy",
    # 产品型号:iPhone13 展开与政策冲突 / P/N 斜杠符号读
    381: "policy", 389: "symbol",
    # gold 伪影补充:normalized 吞句号(Mbps)/删空格(Chrome 114、React 18.2.0)
    196: "gold", 433: "gold", 440: "gold",
    # gold 伪影:$1百万 金额拆分错乱 / 锚点吞逗号(坐标)/ 删前导空格(Release)
    168: "gold", 407: "gold", 428: "gold",
}

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
            if int(rec["index"]) in BLOCKLIST:
                stats[f"blocked:{BLOCKLIST[int(rec['index'])]}"] += 1
                continue
            # nbsp → 空格(输入归一化,生产前端同样预处理)
            src = rec["original_text"].replace(" ", " ").strip()
            tgt = rec["normalized_text"].replace(" ", " ").strip()
            # 伪影修正:normalized 带句末标点而 original 没有 → 剥离(否则读法混入标点)
            while tgt and tgt[-1] in "。.!?!?" and (not src or src[-1] != tgt[-1]):
                tgt = tgt[:-1]
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
