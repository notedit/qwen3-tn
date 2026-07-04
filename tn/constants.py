"""全局常量:prompt 格式、token id、合法读法字符集。

spec 冻结后本文件不得随意改动(parser/datagen/train/eval 共同依赖)。
"""

MODEL_ID = "Qwen/Qwen3-0.6B-Base"

# 任务标记复用 Qwen3 词表 FIM 保留 token(单 token、中文语料不出现,实测于 2026-07-04)
TN_PREFIX = "<|fim_prefix|>"   # 充当 <|tn|>,id 151659
SEP = "<|fim_suffix|>"         # 充当 <|sep|>,id 151661
EOS = "<|endoftext|>"          # id 151643
TN_PREFIX_ID = 151659
SEP_ID = 151661
EOS_ID = 151643

# 编辑行分隔符:行内最后一个 SEPARATOR 之前为锚点,之后为读法
SEPARATOR = "->"

# 合法读法字符:汉字 + 少量标点(读法主体必须是汉字;锚点扩窗带入的上下文字符
# 由 parser 单独放行,不在此集合内)
READING_PUNCT = set(",。、!?;:·")

def is_hanzi(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"

def is_reading_char(ch: str) -> bool:
    return is_hanzi(ch) or ch in READING_PUNCT

def has_pua(s: str) -> bool:
    return any("\ue000" <= c <= "\uf8ff" for c in s)
