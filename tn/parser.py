"""输出解析与应用:parse_and_apply(src, model_output) -> normalized_text。

契约(冻结):
- 模型输出为若干行 `锚点->读法`,按原文出现顺序;无编辑时输出为空串。
- 行内以最后一个 `->` 分隔(锚点可含 `->`,读法不可能含:合法读法字符集不含 - >)。
- 解析器宽松第一匹配:从上一条编辑结束位置向后 find 锚点,取第一个命中。
- 任何异常情况抛 ParseError(带枚举类型),调用方走 fallback;绝不抛其他异常。
"""

from enum import Enum

from tn.constants import SEPARATOR, has_pua, is_reading_char


class Failure(Enum):
    NO_SEPARATOR = "no_separator"          # 行内无 ->
    EMPTY_ANCHOR = "empty_anchor"
    EMPTY_READING = "empty_reading"
    ANCHOR_NOT_FOUND = "anchor_not_found"  # 剩余文本中找不到锚点
    ANCHOR_OUT_OF_ORDER = "anchor_out_of_order"  # 锚点只出现在已消费区域(顺序倒置)
    ILLEGAL_CHARS = "illegal_chars"        # 读法含非法字符(且不来自锚点上下文)
    PUA_IN_OUTPUT = "pua_in_output"


class ParseError(Exception):
    def __init__(self, kind: Failure, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind.value}: {detail}")


def parse_edits(output: str) -> list[tuple[str, str]]:
    """模型原始输出(不含 eos)→ [(anchor, reading), ...]。空输出 → []。"""
    if not isinstance(output, str):
        raise ParseError(Failure.NO_SEPARATOR, f"non-str output: {type(output)}")
    if has_pua(output):
        raise ParseError(Failure.PUA_IN_OUTPUT, repr(output[:50]))
    edits = []
    for line in output.split("\n"):
        # 只跳过空白行;内容行不 strip——锚点/读法的边缘空白是有效字符(如扩窗带入的空格)
        line = line.rstrip("\r")
        if not line.strip():
            continue
        idx = line.rfind(SEPARATOR)
        if idx == -1:
            raise ParseError(Failure.NO_SEPARATOR, line[:80])
        anchor, reading = line[:idx], line[idx + len(SEPARATOR):]
        if not anchor:
            raise ParseError(Failure.EMPTY_ANCHOR, line[:80])
        if not reading:
            raise ParseError(Failure.EMPTY_READING, line[:80])
        edits.append((anchor, reading))
    return edits


def apply_edits(src: str, edits: list[tuple[str, str]]) -> str:
    parts = []
    cursor = 0
    for anchor, reading in edits:
        for ch in reading:
            # 锚点扩窗带入的上下文字符会原样出现在读法里,予以放行
            if not is_reading_char(ch) and ch not in anchor:
                raise ParseError(Failure.ILLEGAL_CHARS, f"{ch!r} in {reading[:40]!r}")
        pos = src.find(anchor, cursor)
        if pos == -1:
            if src.find(anchor) != -1:
                raise ParseError(Failure.ANCHOR_OUT_OF_ORDER, anchor[:80])
            raise ParseError(Failure.ANCHOR_NOT_FOUND, anchor[:80])
        parts.append(src[cursor:pos])
        parts.append(reading)
        cursor = pos + len(anchor)
    parts.append(src[cursor:])
    return "".join(parts)


def parse_and_apply(src: str, output: str) -> str:
    return apply_edits(src, parse_edits(output))


def render_output(edits: list[tuple[str, str]]) -> str:
    """编辑脚本 → 模型目标输出串(不含 eos;无编辑 → 空串)。datagen/train 共用。"""
    return "\n".join(f"{a}{SEPARATOR}{r}" for a, r in edits)
