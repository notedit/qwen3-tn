"""verbalizer:gold 读法的唯一来源(不经 LLM)。

三件套:
- sample_nsw(cls, rng)  -> NSWSample(written, reading, ctx)   书面形式采样 + gold 渲染
- valid_readings(cls, written, ctx) -> set[str]               该书面形式的全部合法读法
- is_valid(cls, written, ctx, reading) -> bool                反向校验(入库/在线共用)
"""

from tn.verbalizer.classes import (
    CLASSES,
    NSWSample,
    is_valid,
    sample_nsw,
    split_gold_edit,
    valid_readings,
)

__all__ = ["CLASSES", "NSWSample", "sample_nsw", "valid_readings", "is_valid",
           "split_gold_edit"]
