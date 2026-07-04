"""数据侧锚点生成:把 (span 区间, 读法) 列表编译成与 parser 顺序解析语义一致的编辑脚本。

规则(与 docs/PLAN.md §1.1 一致):
- 锚点默认取 NSW 本体;若从当前游标 find 的第一个命中不在正确位置,向两侧扩窗
  (优先小窗,±2 字封顶)直到第一命中即正确位置。
- 扩窗带入的上下文字符须同时拼进读法两侧(锚点被整体替换)。
- 锚点不得含换行 / `->` / PUA;越界或无法唯一化则抛 AnchorError,该样本弃用。
"""

from tn.constants import SEPARATOR, has_pua

MAX_EXPAND = 2

# 小窗优先的扩窗尝试序列 (left, right)
_EXPANSIONS = sorted(
    ((l, r) for l in range(MAX_EXPAND + 1) for r in range(MAX_EXPAND + 1)),
    key=lambda p: (p[0] + p[1], p[0]),
)


class AnchorError(Exception):
    pass


def build_edit_script(src: str, spans: list[tuple[int, int, str]]) -> list[tuple[str, str]]:
    """spans: [(start, end, reading)],须已按 start 排序且互不重叠。

    返回 [(anchor, replacement)],保证 parse_and_apply(src, render_output(...))
    与直接按区间替换的结果一致。
    """
    for (s1, e1, _), (s2, _, _) in zip(spans, spans[1:]):
        if s2 < e1:
            raise AnchorError(f"overlapping spans: ({s1},{e1}) vs start {s2}")

    edits = []
    cursor = 0
    for i, (start, end, reading) in enumerate(spans):
        if not (0 <= start < end <= len(src)):
            raise AnchorError(f"span out of range: ({start},{end})")
        if start < cursor:
            raise AnchorError(f"span ({start},{end}) overlaps previous anchor window")
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(src)

        chosen = None
        for lx, rx in _EXPANSIONS:
            l = max(start - lx, cursor)
            r = min(end + rx, next_start)
            anchor = src[l:r]
            if "\n" in anchor or SEPARATOR in anchor or has_pua(anchor):
                continue
            if src.find(anchor, cursor) == l:
                chosen = (l, r, anchor)
                break
        if chosen is None:
            raise AnchorError(f"cannot uniquify span ({start},{end}) {src[start:end]!r}")

        l, r, anchor = chosen
        replacement = src[l:start] + reading + src[end:r]
        edits.append((anchor, replacement))
        cursor = r
    return edits


def apply_spans(src: str, spans: list[tuple[int, int, str]]) -> str:
    """按区间直接替换(gold 归一化结果,用于校验 round-trip)。"""
    parts = []
    prev = 0
    for start, end, reading in spans:
        parts.append(src[prev:start])
        parts.append(reading)
        prev = end
    parts.append(src[prev:])
    return "".join(parts)
