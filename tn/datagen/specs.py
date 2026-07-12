"""造数规格采样与 prompt 构造。

正样本:verbalizer 采样 NSW(gold 程序化)→ LLM 只负责把书面形式嵌进自然句子。
    v2 起可混入「负槽位」干扰片段(不展开的型号/英文名称,可含数字),教模型选择性编辑。
负样本(v2 扩到 7 类):无数字普通句 / 汉字数字句 / 英文缩写句 / 专名密集句 /
    长英文名称句 / 文言古诗句 / 型号句(数字仅在不展开片段内)。
"""

import random
from dataclasses import dataclass, field

from tn.verbalizer import NSWSample, render, sample_nsw

# class 采样权重:线上分布 × 难度加权(TIME/SCORE/RANGE 过采样)
CLASS_WEIGHTS = {
    "NUMBER": 0.22, "DATE": 0.15, "TIME": 0.12, "MONEY": 0.12, "PHONE": 0.08,
    "MEASURE": 0.08, "PERCENT": 0.06, "FRACTION": 0.03, "SCORE": 0.05,
    "VERSION": 0.03, "RANGE": 0.04, "SERIAL": 0.02,
}

DOMAINS = ["新闻报道", "日常对话", "客服对话", "小说叙述", "社交媒体帖子",
           "百科条目", "体育赛事报道", "科技评测"]
LEN_BUCKETS = [(10, 20), (20, 40), (40, 60), (60, 90)]

# 负样本类型及权重(v2:过度触发是 OOD 主要失效模式,负监督多样化)
NEG_TYPES = ["plain", "hanzi_num", "latin", "entity", "latin_long", "classical", "models"]
NEG_WEIGHTS = [0.15, 0.20, 0.10, 0.20, 0.10, 0.10, 0.15]

# 负槽位:约定「不展开」的型号/名称(词形特征:英文词 + 空格 + 数字,
# 与 SERIAL 类的「大写字母紧贴数字」词形区分开,避免标签冲突)
MODEL_FRAGS = [
    "iPhone 15", "iPhone 12", "iPad Pro 11", "Windows 11", "Windows 10",
    "iOS 17", "Android 14", "Java 8", "Python 3", "USB 3.0", "Office 365",
    "Boeing 737", "Surface Pro 9", "PlayStation 5", "Xbox 360", "FIFA 23",
]
# 纯英文名称(无数字):抗「翻译幻觉」型干扰
LATIN_FRAGS = [
    "Deep Purple", "Pink Floyd", "The Beatles", "Manchester United",
    "Real Madrid", "National Geographic", "Discovery Channel", "Whole Foods",
    "Starbucks Reserve", "Bauhaus", "Notre Dame", "Silicon Valley",
]
DISTRACTOR_FRAGS = MODEL_FRAGS + LATIN_FRAGS

CTX_DESC = {
    "quantity": "数量(按数值读,如 18 读作十八)",
    "code": "编号/代码(逐位读,如 368 读作三六八;适用于公交路号、房间号、门牌号、编号、代号、产品代码等)",
    "phone": "电话号码",
    "date": "日期",
    "time": "钟点时间",
    "money": "金额",
    "measure": "计量(数字+单位)",
    "percent": "百分比",
    "fraction": "分数(几分之几)",
    "score": "比赛比分",
    "version": "版本号",
    "range": "数值范围(从A到B)",
    "serial": "型号/编号(字母保留,数字逐位读)",
}


@dataclass
class Spec:
    id: str
    kind: str                 # positive | negative
    domain: str
    length: tuple[int, int]
    items: list[NSWSample] = field(default_factory=list)
    neg_type: str = ""
    distractors: list[str] = field(default_factory=list)  # 不展开片段(正样本负槽位 / models 负样本)


def _weighted_cls(rng: random.Random) -> str:
    return rng.choices(list(CLASS_WEIGHTS), list(CLASS_WEIGHTS.values()))[0]


def _sample_items(rng: random.Random) -> list[NSWSample]:
    n = rng.choices([1, 2, 3], [0.7, 0.25, 0.05])[0]
    for _ in range(10):  # 重采样避免书面形式互为子串(定位会歧义)
        items = [sample_nsw(_weighted_cls(rng), rng) for _ in range(n)]
        ws = [it.written for it in items]
        if all(ws[i] not in ws[j] for i in range(n) for j in range(n) if i != j):
            return items
    return items[:1]


def build_specs(n: int, seed: int, neg_ratio: float = 0.25,
                distractor_ratio: float = 0.35,
                pair_ratio: float = 0.0) -> list[Spec]:
    """确定性生成 n 条规格(id 稳定,断点续跑依赖此性质)。

    pair_ratio:最小对占比 —— 同一数字书面形式以 code/quantity 两种语境成对出现
    (两条独立句子),是逐位 vs 数值消歧的最高密度监督。
    """
    rng = random.Random(seed)
    specs = []
    while len(specs) < n:
        sid = f"s{seed}-{len(specs):07d}"
        domain = rng.choice(DOMAINS)
        length = rng.choice(LEN_BUCKETS)
        r = rng.random()
        if r < neg_ratio:
            neg_type = rng.choices(NEG_TYPES, NEG_WEIGHTS)[0]
            dis = rng.sample(MODEL_FRAGS, rng.choice([1, 2])) \
                if neg_type == "models" else []
            specs.append(Spec(sid, "negative", domain, length,
                              neg_type=neg_type, distractors=dis))
        elif r < neg_ratio + pair_ratio and n - len(specs) >= 2:
            w = str(rng.randint(10, 99999))
            for ctx in ("code", "quantity"):
                sid = f"s{seed}-{len(specs):07d}"
                item = NSWSample("NUMBER", ctx, w, render("NUMBER", w, ctx))
                specs.append(Spec(sid, "positive", rng.choice(DOMAINS),
                                  rng.choice(LEN_BUCKETS), items=[item]))
        else:
            dis = []
            if rng.random() < distractor_ratio:
                dis = rng.sample(DISTRACTOR_FRAGS, 1 if rng.random() < 0.8 else 2)
            specs.append(Spec(sid, "positive", domain, length,
                              items=_sample_items(rng), distractors=dis))
    return specs


# ---------------- prompts ----------------

_POS_SYSTEM = "你是中文语料造句助手,严格按要求输出 JSON,不输出任何其他内容。"

# 盲测集用第二套措辞(variant=1),降低与训练集的 prompt 相关性
_POS_HEADER = [
    "为下列每个条目写一句自然、口语化程度适中的中文句子。",
    "请给下面每个条目分别创作一个符合语境的中文句子。",
]

_POS_RULES = """要求:
1. 句子必须原样包含条目给出的全部「片段」,一字不差,且按给出顺序出现;
2. 片段紧邻的前后字符不能是数字、字母或小数点(避免粘连);
3. 除给定片段外,句子中不得出现任何其他阿拉伯数字,也不得出现 ¥ $ % ℃ ~ 等符号;
   但允许并鼓励出现汉字数字表达(如"三十多岁""两点半"),它们不算违反本条;
4. 语境必须符合片段说明,风格:{domain},长度约 {lo}-{hi} 字;
5. 输出 JSON:{{"sentences":[{{"id":"条目id","text":"句子"}},...]}}"""

_NEG_PROMPTS = {
    "plain": "写一句{domain}风格的中文句子,长度{lo}-{hi}字,不得包含任何阿拉伯数字、英文字母或 ¥ $ % ℃ 等符号。",
    "hanzi_num": "写一句{domain}风格的中文句子,长度{lo}-{hi}字,其中的数字表达全部用汉字(如“两点半”“三十多岁”“五百多人”“十八号”),不得出现阿拉伯数字或英文字母。",
    "latin": "写一句{domain}风格的中文句子,长度{lo}-{hi}字,自然地包含英文缩写或单词(如 GDP、AI、USB、VIP、Pro),但不得包含任何阿拉伯数字。",
    "entity": "写一句{domain}风格的中文句子,长度{lo}-{hi}字,信息密集地包含多个专有名词(人名、机构名、地名、作品名,可用音译汉字名),不得出现阿拉伯数字、英文字母或符号。",
    "latin_long": "写一句{domain}风格的中文句子,长度{lo}-{hi}字,自然包含一个较长的英文名称或词组(两个以上英文单词,如 Deep Purple、National Geographic),不得包含任何阿拉伯数字。",
    "classical": "写一句文言文或古诗词风格的中文句子(可以是引用古文的叙述),长度{lo}-{hi}字,不得出现阿拉伯数字、英文字母或符号。",
    "models": "写一句{domain}风格的中文句子,长度{lo}-{hi}字,自然地原样包含以下片段:{frags}。除这些片段内部的数字外,句中不得出现其他阿拉伯数字,也不得出现 ¥ $ % ℃ 等符号。",
}


def positive_prompt(specs: list[Spec], variant: int = 0) -> list[dict]:
    lines = []
    for sp in specs:
        frag = ";".join(
            f"片段{k + 1}:「{it.written}」({CTX_DESC.get(it.ctx, it.ctx)})"
            for k, it in enumerate(sp.items))
        for d in sp.distractors:
            frag += f";片段(型号/名称,原样嵌入即可):「{d}」"
        lines.append(f"- id={sp.id} {frag}")
    sp0 = specs[0]
    body = (_POS_HEADER[variant % len(_POS_HEADER)] + "\n"
            + _POS_RULES.format(domain=sp0.domain, lo=sp0.length[0], hi=sp0.length[1])
            + "\n条目:\n" + "\n".join(lines))
    return [{"role": "system", "content": _POS_SYSTEM},
            {"role": "user", "content": body}]


def negative_prompt(specs: list[Spec]) -> list[dict]:
    lines = []
    for sp in specs:
        req = _NEG_PROMPTS[sp.neg_type].format(
            domain=sp.domain, lo=sp.length[0], hi=sp.length[1],
            frags="、".join(f"「{d}」" for d in sp.distractors))
        lines.append(f"- id={sp.id} {req}")
    body = ("为下列每个条目各写一句中文句子。\n"
            "输出 JSON:{\"sentences\":[{\"id\":\"条目id\",\"text\":\"句子\"},...]}\n"
            "条目:\n" + "\n".join(lines))
    return [{"role": "system", "content": _POS_SYSTEM},
            {"role": "user", "content": body}]


def judge_prompt(text: str, written: str, candidates: list[str]) -> list[dict]:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(candidates))
    body = (f"句子:{text}\n"
            f"问题:句中「{written}」在朗读时的正确读法是哪一个?\n"
            f"{opts}\n"
            '只输出 JSON:{"choice":"字母"}')
    return [{"role": "system", "content": "你是中文文本正则化(TTS 读法)专家,只输出 JSON。"},
            {"role": "user", "content": body}]
