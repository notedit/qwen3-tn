"""parser/anchor 契约测试:手写用例 + 大规模随机 fuzz。

fuzz 两个性质:
1. round-trip:随机原文 + 随机合法 span 编辑,build_edit_script → render_output →
   parse_and_apply 的结果必须与按区间直接替换完全一致。
2. 鲁棒性:随机损坏的模型输出,parse_and_apply 要么返回 str,要么抛 ParseError,
   绝不抛其他异常。
"""

import random

import pytest

from tn.anchor import AnchorError, apply_spans, build_edit_script
from tn.parser import Failure, ParseError, parse_and_apply, parse_edits, render_output

# ---------- 手写用例 ----------

def test_basic():
    src = "会议定在18号下午3:30,预算¥1200万。"
    out = "18号->十八号\n3:30->三点三十分\n¥1200万->一千二百万元"
    assert parse_and_apply(src, out) == "会议定在十八号下午三点三十分,预算一千二百万元。"


def test_no_edit():
    assert parse_and_apply("今天天气不错。", "") == "今天天气不错。"
    assert parse_and_apply("今天天气不错。", "  \n ") == "今天天气不错。"


def test_repeated_nsw_sequential_match():
    src = "他18岁,住18号楼。"
    out = "18->十八\n18号->幺八号"
    assert parse_and_apply(src, out) == "他十八岁,住幺八号楼。"


def test_anchor_with_context_chars():
    # 扩窗锚点:读法里带上下文字符(非汉字)需放行
    src = "编号A18和A19。"
    out = "A18->A幺八\nA19->A幺九"
    assert parse_and_apply(src, out) == "编号A幺八和A幺九。"


def test_separator_in_anchor_uses_last():
    src = "温度-5->-3度。"
    out = "-5->-3->零下五到零下三"
    assert parse_and_apply(src, out) == "温度零下五到零下三度。"


@pytest.mark.parametrize("out,kind", [
    ("18号", Failure.NO_SEPARATOR),
    ("->十八", Failure.EMPTY_ANCHOR),
    ("18号->", Failure.EMPTY_READING),
    ("99号->九十九", Failure.ANCHOR_NOT_FOUND),
    ("18号->十八k号", Failure.ILLEGAL_CHARS),
    ("18号->十八", Failure.PUA_IN_OUTPUT),
])
def test_failures(out, kind):
    with pytest.raises(ParseError) as e:
        parse_and_apply("会议定在18号。", out)
    assert e.value.kind == kind


def test_out_of_order():
    src = "上午9点和下午6点。"
    with pytest.raises(ParseError) as e:
        parse_and_apply(src, "6点->六点\n9点->九点")
    assert e.value.kind == Failure.ANCHOR_OUT_OF_ORDER


# ---------- fuzz ----------

HANZI = "的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动"
ASCII = "abcdefgHIJKLMN0123456789 .-:/%¥$"
PUNCT = ",。、!?;:()《》"
READ = "零一二三四五六七八九十百千万亿点分秒号年月日元块幺两负"


def rand_src(rng: random.Random) -> str:
    n = rng.randint(1, 80)
    pools = [HANZI, HANZI, HANZI, ASCII, PUNCT]
    return "".join(rng.choice(rng.choice(pools)) for _ in range(n))


def rand_spans(rng: random.Random, src: str) -> list[tuple[int, int, str]]:
    spans = []
    pos = 0
    while pos < len(src) and len(spans) < 6:
        if rng.random() < 0.4:
            start = rng.randint(pos, len(src) - 1)
            end = min(len(src), start + rng.randint(1, 5))
            reading = "".join(rng.choice(READ) for _ in range(rng.randint(1, 8)))
            spans.append((start, end, reading))
            pos = end
        else:
            pos += rng.randint(1, 10)
    return spans


def test_fuzz_round_trip():
    rng = random.Random(20260704)
    total, skipped = 0, 0
    for _ in range(100_000):
        src = rand_src(rng)
        spans = rand_spans(rng, src)
        try:
            edits = build_edit_script(src, spans)
        except AnchorError:
            skipped += 1  # 随机文本可无法唯一化(如重复片段),数据侧本就弃用
            continue
        total += 1
        got = parse_and_apply(src, render_output(edits))
        expect = apply_spans(src, spans)
        assert got == expect, f"src={src!r} spans={spans} edits={edits}"
    # 唯一化失败率不应离谱(随机文本比真实文本更难唯一化,阈值放宽)
    assert total > 0 and skipped / (total + skipped) < 0.35, (total, skipped)


def corrupt(rng: random.Random, out: str) -> str:
    ops = rng.randint(1, 3)
    s = out
    for _ in range(ops):
        c = rng.random()
        if c < 0.25 and s:
            i = rng.randrange(len(s))
            s = s[:i] + s[i + 1:]  # 删字符
        elif c < 0.5:
            i = rng.randint(0, len(s))
            junk = rng.choice(["->", "\n", "", "x", "十", "?"])
            s = s[:i] + junk + s[i:]  # 插入
        elif c < 0.75 and "\n" in s:
            lines = s.split("\n")
            rng.shuffle(lines)
            s = "\n".join(lines)  # 打乱行序
        else:
            s = s + rng.choice(["\n18->", "->", "垃圾行", ""])
    return s


def test_fuzz_robustness():
    rng = random.Random(42)
    for _ in range(50_000):
        src = rand_src(rng)
        spans = rand_spans(rng, src)
        try:
            out = render_output(build_edit_script(src, spans))
        except AnchorError:
            out = "18号->十八号\n3:30->三点"
        bad = corrupt(rng, out)
        try:
            r = parse_and_apply(src, bad)
            assert isinstance(r, str)
        except ParseError:
            pass  # 唯一允许的异常


def test_parse_edits_type_safety():
    with pytest.raises(ParseError):
        parse_edits(None)  # type: ignore[arg-type]
