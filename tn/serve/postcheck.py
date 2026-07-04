"""在线生成后校验:模型输出的每条编辑,验证「书面形式→读法」是合法路径。

模型不输出 class/ctx,故对全部 (class, ctx) 组合尝试反推(split_gold_edit),
任一组合接受即放行;全部拒绝 → 该编辑判为幻觉,丢弃(span 回退原文)。
单条编辑校验成本 <1ms(12 类 × 常数个 ctx 的集合运算)。
"""

from tn.verbalizer import split_gold_edit
from tn.verbalizer.classes import CLASSES

# 每类可能的 ctx(与 datagen 一致;"" 表示类自身默认)
_CTXS = {
    "NUMBER": ["quantity", "code"],
    "DATE": ["date"], "TIME": ["time"], "MONEY": ["money"], "PHONE": ["phone"],
    "MEASURE": [""], "PERCENT": [""], "FRACTION": [""], "SCORE": [""],
    "VERSION": [""], "RANGE": [""], "SERIAL": [""],
}


def edit_is_legal(anchor: str, replacement: str) -> bool:
    for cls in CLASSES:
        for ctx in _CTXS[cls]:
            if split_gold_edit(anchor, replacement, cls, ctx) is not None:
                return True
    return False


def filter_edits(edits: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """返回 (通过的编辑, 被拦截数)。被拦截的 span 保持原文(等价回退默认读法)。"""
    kept = []
    blocked = 0
    for a, r in edits:
        if edit_is_legal(a, r):
            kept.append((a, r))
        else:
            blocked += 1
    return kept, blocked
