"""标贝开源 TN 数据集(char.bmes)→ 本项目 jsonl 评测集。

标贝标注是字级读法决策(DIGIT=逐位 / CARDINAL=数值 / HYPHEN_RATIO=读比 / …),
本转换器只取其 NSW 定位与消歧信息,gold 读法一律由自家 verbalizer render 生成
(方案 v1.1 原则:gold 不经第三方)。无法映射到 12 类或 render 失败的句子整句丢弃,
丢弃原因即 verbalizer 覆盖率基线,输出到 coverage 报告。

用法:
  .venv/bin/python -m tn.datagen.databaker \
      --bmes dev.char.bmes test.char.bmes \
      --out data/databaker_eval.jsonl --report reports/databaker_coverage.md

注意:标贝数据为非商用许可,产物仅用于内部评测,不得混入生产训练集。
"""

import argparse
import json
import re
from collections import Counter, defaultdict

from tn.anchor import AnchorError, apply_spans, build_edit_script
from tn.constants import has_pua
from tn.parser import ParseError, parse_and_apply, render_output
from tn.verbalizer import is_valid, render

# 数字体 tag(可互相拼接成一个数字串;区分逐位/数值两种模式)
NUM_TAGS = {"DIGIT", "CARDINAL", "NUM_TWO_LIANG", "MINUTE_CARDINAL",
            "DAY_CARDINAL", "MONTH_CARDINAL"}
DIGIT_MODE = {"DIGIT"}
PASS_TAGS = {"SELF", "PUNC"}

_DIGIT_RE = re.compile(r"[0-90-9]")


def load_bmes(path: str):
    """→ [(text, [(chars, tag), ...]), ...],tag 已去 BMES 前缀并按组合并。"""
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split()
        ch, tag = (" ", parts[0]) if len(parts) == 1 else (parts[0], parts[-1])
        pre, name = tag.split("-", 1) if "-" in tag else ("S", tag)
        if cur and pre in ("M", "E") and cur[-1][1] == name:
            cur[-1][0] += ch
        else:
            cur.append([ch, name])
    if cur:
        sents.append(cur)
    out = []
    for groups in sents:
        text = "".join(g[0] for g in groups)
        toks, pos = [], 0
        for g_text, g_tag in groups:
            toks.append((pos, pos + len(g_text), g_text, g_tag))
            pos += len(g_text)
        out.append((text, toks))
    return out


class Uncovered(Exception):
    def __init__(self, reason: str, sample: str = ""):
        self.reason, self.sample = reason, sample


def _num_mode(tags: list[str]) -> str:
    """小数点后的 DIGIT 是「小数部分逐位读」(read_number 本就如此),不算逐位模式。"""
    head = tags[: tags.index("POINT")] if "POINT" in tags else tags
    return "digit" if any(t in DIGIT_MODE for t in head) else "cardinal"


def assemble_units(text: str, toks: list) -> list[tuple[int, int, str, str, str]]:
    """NSW 连续段 + 上下文 → [(start, end, written, cls, ctx)];无法映射抛 Uncovered。

    返回的 span 边界尽量对齐训练约定(DATE 含年月日、PERCENT 含 %、MONEY 含 万亿元)。
    """
    # 1) 非 SELF/PUNC 的连续 token 归为一个 run
    runs = []
    for tk in toks:
        if tk[3] in PASS_TAGS:
            continue
        if runs and runs[-1][-1][1] == tk[0]:
            runs[-1].append(tk)
        else:
            runs.append([tk])

    units = []
    consumed_runs = set()
    for ri, run in enumerate(runs):
        if ri in consumed_runs:
            continue
        s, e = run[0][0], run[-1][1]
        w = text[s:e]
        tags = [t for _, _, _, t in run]
        tagset = set(tags)

        # 纯英文字母段:保持原样不改写(与训练负样本约定一致)
        if tagset == {"ENG_LETTER"}:
            continue

        bad = tagset - NUM_TAGS - {"POINT", "ENG_LETTER", "MEASURE_UNIT", "VERBATIM",
                                   "COLON_HOUR", "COLON_MINUTE", "HYPHEN_RANGE",
                                   "HYPHEN_RATIO", "SLASH_FRACTION"}
        if bad:
            raise Uncovered(f"tag:{sorted(bad)[0]}", w)

        after = text[e:e + 3]

        # 字母+数字(NGC4349 / A380):SERIAL,字母保留数字逐位
        if "ENG_LETTER" in tagset:
            if re.fullmatch(r"[A-Za-z]+[0-9]+", w) and tagset <= {"ENG_LETTER"} | NUM_TAGS:
                units.append((s, e, w, "SERIAL", "serial"))
                continue
            raise Uncovered("eng_digit_mix", w)

        if "MEASURE_UNIT" in tagset:
            units.append((s, e, w, "MEASURE", "measure"))  # render 失败会计入 unit 缺口
            continue
        if "VERBATIM" in tagset:
            if w.endswith("%") and tags.count("VERBATIM") == 1 and "HYPHEN_RANGE" not in tagset:
                units.append((s, e, w, "PERCENT", "percent"))
                continue
            if "HYPHEN_RANGE" in tagset and re.fullmatch(r"\d+%[-~—–~]\d+%", w):
                units.append((s, e, w, "RANGE", "range"))
                continue
            raise Uncovered("verbatim_symbol", w)
        if "COLON_HOUR" in tagset or "COLON_MINUTE" in tagset:
            units.append((s, e, w, "TIME", "time"))
            continue
        if "HYPHEN_RATIO" in tagset:
            units.append((s, e, w, "SCORE", "score"))
            continue
        if "HYPHEN_RANGE" in tagset:
            if _num_mode(tags) == "digit":
                raise Uncovered("range_digit_mode", w)  # 年份区间逐位读,RANGE 类未覆盖
            units.append((s, e, w, "RANGE", "range"))
            continue
        if "SLASH_FRACTION" in tagset:
            units.append((s, e, w, "FRACTION", "fraction"))
            continue

        # 纯数字串(可含小数点/千分位逗号)
        mode = _num_mode(tags)
        if mode == "digit":
            if after.startswith("年"):
                units.append((s, e + 1, w + "年", "DATE", "date"))
            else:
                units.append((s, e, w, "NUMBER", "code"))
            continue
        # cardinal:先尝试拼日期 M月D日/号(月/日为 SELF,需跨 run 合并)
        if after.startswith("月") and ri + 1 < len(runs):
            nxt = runs[ri + 1]
            n_s, n_e = nxt[0][0], nxt[-1][1]
            n_tags = [t for _, _, _, t in nxt]
            if (n_s == e + 1 and set(n_tags) <= NUM_TAGS and _num_mode(n_tags) == "cardinal"
                    and text[n_e:n_e + 1] in ("日", "号")):
                units.append((s, n_e + 1, text[s:n_e + 1], "DATE", "date"))
                consumed_runs.add(ri + 1)
                continue
        # 金额 / 万亿 / 单位后缀吸收(对齐训练 span 约定)
        m = re.match(r"(万|亿)?(美元|元)", after)
        if m and m.group(2):
            suf = m.group(0)
            units.append((s, e + len(suf), w + suf, "MONEY", "money"))
            continue
        if after.startswith(("万", "亿")):
            units.append((s, e + 1, w + after[0], "NUMBER", "quantity"))
            continue
        m = re.match(r"(公里|米|斤|度)", after)
        if m:
            units.append((s, e + len(m.group(1)), w + m.group(1), "MEASURE", "measure"))
            continue
        units.append((s, e, w, "NUMBER", "quantity"))
    return units


def convert_sentence(text: str, toks: list, liang_positions: set):
    """→ (record_fields, None) 或 (None, 丢弃原因)。"""
    if has_pua(text) or "\n" in text:
        return None, "bad_text"
    if not 5 <= len(text) <= 150:
        return None, "bad_length"
    try:
        units = assemble_units(text, toks)
    except Uncovered as u:
        return None, u.reason

    spans, classes, ctxs, nsw = [], [], [], []
    for s, e, w, cls, ctx in units:
        reading = render(cls, w, ctx)
        if cls == "NUMBER" and ctx == "quantity" and s in liang_positions and w == "2":
            reading = "两"  # 标贝 NUM_TWO_LIANG:该 2 读「两」
        if reading is None:
            return None, f"render:{cls}:{w[:12]}"
        if not is_valid(cls, w, ctx, reading):
            return None, f"invalid:{cls}:{w[:12]}"
        spans.append((s, e, reading))
        classes.append(cls)
        ctxs.append(ctx)
        nsw.append([w, reading])

    # span 外不得有残留数字(标注噪声或组装遗漏 → 弃)
    covered = [False] * len(text)
    for s, e, _ in spans:
        for i in range(s, e):
            covered[i] = True
    for i, ch in enumerate(text):
        if _DIGIT_RE.match(ch) and not covered[i]:
            return None, "stray_digit"

    try:
        edits = build_edit_script(text, spans)
    except AnchorError:
        return None, "anchor_fail"
    try:
        if parse_and_apply(text, render_output(edits)) != apply_spans(text, spans):
            return None, "roundtrip_mismatch"
    except ParseError:
        return None, "roundtrip_parse_error"
    return {"edits": [list(e) for e in edits], "classes": classes,
            "ctxs": ctxs, "nsw": nsw}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="reports/databaker_coverage.md")
    args = ap.parse_args()

    stats = Counter()
    reject_examples = defaultdict(list)
    recs, seen = [], set()
    for path in args.bmes:
        split = path.split("/")[-1].split(".")[0]
        for idx, (text, toks) in enumerate(load_bmes(path)):
            liang = {tk[0] for tk in toks if tk[3] == "NUM_TWO_LIANG"}
            got, err = convert_sentence(text, toks, liang)
            if err:
                stats["reject:" + ":".join(err.split(":")[:2])] += 1
                if len(reject_examples[err]) < 3:
                    reject_examples[err].append(text[:60])
                continue
            if text in seen:
                stats["dup"] += 1
                continue
            seen.add(text)
            kind = "positive" if got["edits"] else "negative_real"
            recs.append({
                "id": f"databaker-{split}-{idx:05d}",
                "src": text,
                "edits": got["edits"],
                "meta": {"classes": got["classes"], "ctxs": got["ctxs"],
                         "nsw": got["nsw"], "domain": "百科",
                         "kind": kind, "source": "databaker"},
            })
            stats["kept"] += 1
            stats[f"kept:{kind}"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(v for k, v in stats.items() if k.startswith("reject:")) \
        + stats["kept"] + stats["dup"]
    lines = [f"# 标贝 BMES → jsonl 覆盖率报告\n",
             f"输入 {total} 句 → 保留 {stats['kept']}"
             f"(positive {stats['kept:positive']} / negative {stats['kept:negative_real']}),"
             f"重复 {stats['dup']},保留率 {stats['kept'] / max(1, total):.1%}\n",
             "## 丢弃原因(= verbalizer 覆盖缺口,按频次排序)\n"]
    for k, v in stats.most_common():
        if k.startswith("reject:"):
            lines.append(f"- {k[7:]}: {v}")
    lines.append("\n## 各原因样例\n")
    for reason, exs in sorted(reject_examples.items(),
                              key=lambda kv: -len(kv[1])):
        lines.append(f"- `{reason}`")
        for e in exs:
            lines.append(f"  - {e}")
    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nwrote {stats['kept']} records -> {args.out}")


if __name__ == "__main__":
    main()
