"""对已产出的 jsonl 补打新校验规则(目前:NUMBER 语境-读法冲突守卫)。

用法:
  .venv/bin/python -m tn.datagen.postfilter data/train_v1.jsonl

原地重写(先写 .tmp 再替换),打印删除统计。被删 id 重跑同一条 run_gen 命令
会按新规则重新生成(断点续跑机制)。
"""

import json
import os
import sys

from tn.datagen.run_gen import ctx_conflict
from tn.verbalizer import split_gold_edit


def check_rec(rec: dict) -> str | None:
    """返回删除原因,None=保留。"""
    edits = rec.get("edits") or []
    if not edits:
        return None
    meta = rec["meta"]
    classes = meta.get("classes", [])
    ctxs = meta.get("ctxs", [])
    if len(classes) != len(edits):
        return None  # 老格式缺信息,不动
    src = rec["src"]
    cursor = 0
    for (anchor, repl), cls, ctx in zip(edits, classes, ctxs):
        pos = src.find(anchor, cursor)
        if pos == -1:
            return "broken_anchor"
        got = split_gold_edit(anchor, repl, cls, ctx)
        if got is not None:
            p, s, _, _, _ = got
            ws, we = pos + p, pos + len(anchor) - s
        else:
            ws, we = pos, pos + len(anchor)
        if ctx_conflict(src, ws, we, cls, ctx):
            return "ctx_conflict"
        cursor = pos + len(anchor)
    return None


def main():
    path = sys.argv[1]
    kept, dropped = [], {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        reason = check_rec(rec)
        if reason is None:
            kept.append(line)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, path)
    print(f"{path}: kept={len(kept)} dropped={dropped}")


if __name__ == "__main__":
    main()
