"""12 类 semiotic class:书面形式采样、gold 渲染、合法读法集合、反向校验。

每类实现三个函数:
- _sample_X(rng) -> (written, reading, ctx)
- _valid_X(written, ctx) -> set[str]   该书面形式全部合法读法(canonical 必在其中)
- _render_X(written, ctx) -> str|None  任意书面形式 → canonical 读法(与 sampler 产出一致);
  无法解析返回 None。供真实语料转数据 / 覆盖率扫描使用。

约定:written 是句子中将被替换的完整 NSW 片段;reading 全部为汉字。
ctx 是语境标签,喂给 LLM 造句时约束语境(phone/quantity/code/...)。
"""

import random
from dataclasses import dataclass
from itertools import product

from tn.verbalizer.num_core import (
    digit_string_readings,
    integer_readings,
    number_readings,
    read_digits,
    read_integer,
    read_number,
)


@dataclass
class NSWSample:
    cls: str
    ctx: str
    written: str
    reading: str


# ---------------- NUMBER ----------------

def _rand_int(rng: random.Random, max_digits: int = 8) -> int:
    return rng.randint(0, 10 ** rng.randint(1, max_digits) - 1)


def _sample_number(rng):
    # code 权重上调(标贝真实分布 DIGIT 逐位占 NSW 41%,v1 的 0.25 严重偏低)
    ctx = rng.choices(["quantity", "code"], [0.65, 0.35])[0]
    if ctx == "code":
        r = rng.random()
        if r < 0.12:  # 点分纪念日/代号(3.15 → 三一五,点不读)
            s = f"{rng.randint(1, 12)}.{rng.randint(1, 30)}"
            return s, read_digits(s), ctx
        if r < 0.22:  # 连字符编号(订单号 2023-5678,连字符不读)
            s = f"{rng.randint(100, 2099)}-{rng.randint(100, 9999)}"
            return s, read_digits(s), ctx
        if r < 0.32:  # 长证件号(15-18 位,考验长串读法风格一致性)
            s = "".join(rng.choice("0123456789") for _ in range(rng.choice([15, 17, 18])))
            return s, read_digits(s), ctx
        s = "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 8)))
        return s, read_digits(s), ctx
    style = rng.random()
    if style < 0.15:  # 万/亿 后缀
        unit = rng.choice(["万", "亿"])
        x = rng.choice([str(rng.randint(1, 9999)),
                        f"{rng.randint(1, 99)}.{rng.randint(1, 9)}"])
        return f"{x}{unit}", read_number(x) + unit, "quantity"
    if style < 0.3:  # 小数
        a, b = rng.randint(0, 999), rng.randint(0, 99)
        s = f"{a}.{b:02d}" if rng.random() < 0.3 else f"{a}.{rng.randint(0, 9)}"
        return s, read_number(s), "quantity"
    if style < 0.36:  # 负数 / ± 公差
        n = rng.randint(1, 200)
        if rng.random() < 0.25:
            x = f"{rng.randint(0, 9)}.{rng.randint(1, 99):02d}" if rng.random() < 0.5 else str(n)
            return f"±{x}", "正负" + read_number(x), "quantity"
        return f"-{n}", "负" + read_integer(n), "quantity"
    if style < 0.44:  # 千分位
        n = rng.randint(1000, 99_999_999)
        return f"{n:,}", read_integer(n), "quantity"
    n = _rand_int(rng)
    return str(n), read_integer(n), "quantity"


def _valid_number(written, ctx):
    if ctx == "code":
        return digit_string_readings(written)
    if written.startswith("±"):
        return {"正负" + r for r in number_readings(written[1:])}
    for unit in ("万", "亿"):
        if written.endswith(unit):
            return {r + unit for r in number_readings(written[:-1])}
    return number_readings(written, liang_alone=True)


# ---------------- DATE ----------------

def _read_year(y) -> str:
    return read_digits(str(y))


def _sample_date(rng):
    # 30% 历史年份(含 3 位,如 976年),压真实语料中的古代纪年错读
    y = rng.randint(100, 1948) if rng.random() < 0.3 else rng.randint(1949, 2032)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    tpl = rng.choices(
        ["ymd", "ymd_dash", "ymd_dot", "md", "y", "d_hao", "md_hao", "md_slash"],
        [0.24, 0.09, 0.04, 0.16, 0.14, 0.11, 0.07, 0.15])[0]
    if y < 1000 and tpl in ("ymd_dash", "ymd_dot"):
        tpl = "ymd"  # 3 位年份不出现连字符/点分式(书面上不自然)
    if tpl == "md_slash":  # 12/31 式月日
        return f"{m}/{d}", f"{read_integer(m)}月{read_integer(d)}日", "date"
    if tpl == "y" and y >= 2000 and rng.random() < 0.15:  # 两位简写年(08年)
        return f"{y % 100:02d}年", f"{read_digits(f'{y % 100:02d}')}年", "date"
    if tpl == "ymd":
        return f"{y}年{m}月{d}日", f"{_read_year(y)}年{read_integer(m)}月{read_integer(d)}日", "date"
    if tpl == "ymd_dash":
        return f"{y}-{m:02d}-{d:02d}", f"{_read_year(y)}年{read_integer(m)}月{read_integer(d)}日", "date"
    if tpl == "ymd_dot":
        return f"{y}.{m}.{d}", f"{_read_year(y)}年{read_integer(m)}月{read_integer(d)}日", "date"
    if tpl == "md":
        return f"{m}月{d}日", f"{read_integer(m)}月{read_integer(d)}日", "date"
    if tpl == "md_hao":
        return f"{m}月{d}号", f"{read_integer(m)}月{read_integer(d)}号", "date"
    if tpl == "y":
        return f"{y}年", f"{_read_year(y)}年", "date"
    return f"{d}号", f"{read_integer(d)}号", "date"


def _valid_date(written, ctx):
    w = written
    out = set()
    import re
    m = re.fullmatch(r"(\d{3,4})[年\-./](?:(\d{1,2})[月\-./](\d{1,2})[日号]?)?", w) \
        or re.fullmatch(r"(\d{3,4})年", w)
    if m and m.group(1) and len(m.groups()) >= 3 and m.group(2):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        for suf in ("日", "号"):
            out.add(f"{_read_year(y)}年{read_integer(mo)}月{read_integer(d)}{suf}")
        return out
    m = re.fullmatch(r"(\d{3,4})年", w)  # 3 位:公元早期年份(如 976年)
    if m:
        return {f"{_read_year(m.group(1))}年", f"{read_integer(int(m.group(1)))}年"}
    m = re.fullmatch(r"(\d{2})年", w)  # 两位简写年(08年 → 零八年;也可作时长读)
    if m:
        return {f"{_read_year(m.group(1))}年", f"{read_integer(int(m.group(1)))}年"}
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})([日号])", w)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        return {f"{read_integer(mo)}月{read_integer(d)}{suf}" for suf in ("日", "号")}
    m = re.fullmatch(r"(\d{1,2})([日号])", w)
    if m:
        d, suf = int(m.group(1)), m.group(2)
        return {f"{read_integer(d)}{suf}"}
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", w)
    if m and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 31:
        mo, d = int(m.group(1)), int(m.group(2))
        return {f"{read_integer(mo)}月{read_integer(d)}{suf}" for suf in ("日", "号")}
    return set()


# ---------------- TIME ----------------

def _read_hour(h: int) -> str:
    return "两" if h == 2 else read_integer(h)


def _sample_time(rng):
    h = rng.randint(0, 23)
    mi = rng.randint(0, 59)
    if rng.random() < 0.15:
        se = rng.randint(0, 59)
        w = f"{h}:{mi:02d}:{se:02d}"
        r = f"{_read_hour(h)}点{_read_min(mi)}{_read_sec(se)}"
        return w, r, "time"
    if mi and rng.random() < 0.12:  # 原文自带「分」:23:59分(span 含分字)
        w = f"{h}:{mi:02d}分"
        return w, f"{_read_hour(h)}点{_read_min(mi)}", "time"
    w = f"{h}:{mi:02d}"
    return w, f"{_read_hour(h)}点{_read_min(mi)}", "time"


def _read_min(mi: int) -> str:
    if mi == 0:
        return ""
    if mi < 10:
        return f"零{read_integer(mi)}分"
    return f"{read_integer(mi)}分"


def _read_sec(se: int) -> str:
    if se == 0:
        return ""
    if se < 10:
        return f"零{read_integer(se)}秒"
    return f"{read_integer(se)}秒"


def _valid_time(written, ctx):
    if written.endswith("分") and ":" in written:
        return _valid_time(written[:-1], ctx)
    parts = written.split(":")
    if not all(p.isdigit() for p in parts) or not 2 <= len(parts) <= 3:
        return set()
    h, mi = int(parts[0]), int(parts[1])
    se = int(parts[2]) if len(parts) == 3 else None
    hours = {"两点"} if h == 2 else set()
    hours.add(read_integer(h) + "点")
    out = set()
    for hr in hours:
        if se is not None:
            out.add(f"{hr}{_read_min(mi)}{_read_sec(se)}")
            if se == 0:
                out.add(f"{hr}{_read_min(mi)}零秒")
            continue
        out.add(f"{hr}{_read_min(mi)}")
        if mi:  # 无「分」变体(原文时间后紧跟"分"字的场景)
            out.add(f"{hr}{'零' if mi < 10 else ''}{read_integer(mi)}")
        if mi == 30:
            out.add(f"{hr}半")
        if mi == 0:
            out.add(f"{hr}整")
    return out


# ---------------- MONEY ----------------

# 币种词表(rules-as-data):首个读法为 canonical
_CUR_PREFIX = {  # 前缀符号(长优先匹配:US$ 先于 $,JP¥ 先于 ¥)
    "US$": ["美元"], "HK$": ["港币", "港元"], "CAD$": ["加元"], "NT$": ["新台币"],
    "JP¥": ["日元"], "£": ["英镑"], "€": ["欧元"], "$": ["美元"], "¥": ["元", "人民币"],
}
_CUR_CODE = {  # 字母代码前缀(与数字间可有空格/nbsp)
    "USD": ["美元"], "EUR": ["欧元"], "GBP": ["英镑"], "JPY": ["日元"],
    "CNY": ["人民币", "元"], "HKD": ["港币", "港元"], "AUD": ["澳元"], "CAD": ["加元"],
    "KRW": ["韩元"], "THB": ["泰铢"], "INR": ["印度卢比", "卢比"],
    "SGD": ["新加坡元", "新元"], "MYR": ["林吉特", "马来西亚林吉特"],
    "IDR": ["印尼盾"], "RUB": ["卢布"],
}
_CUR_HANZI = ["美元", "欧元", "日元", "英镑", "港币", "港元", "韩元", "泰铢",
              "卢布", "加元", "澳元", "新台币", "人民币", "元"]  # 汉字后缀(长优先)


def _parse_money(w: str):
    """written → (数字串, 万/亿, 读法列表);非金额返回 None。"""
    readings = None
    for p in sorted(_CUR_PREFIX, key=len, reverse=True):
        if w.startswith(p):
            readings, w = _CUR_PREFIX[p], w[len(p):]
            break
    if readings is None:
        for c in sorted(_CUR_CODE, key=len, reverse=True):
            if w.startswith(c):
                readings, w = _CUR_CODE[c], w[len(c):].lstrip("  ")
                break
    if readings is None:
        for s in sorted(_CUR_HANZI, key=len, reverse=True):
            if w.endswith(s):
                readings, w = [s], w[: -len(s)]
                break
    if readings is None:
        return None
    big = ""
    for unit in ("万", "亿"):
        if w.endswith(unit):
            big, w = unit, w[:-1]
    if not w or not all(c.isdigit() or c in ".,," for c in w):
        return None
    return w, big, readings


def _sample_money(rng):
    r = rng.random()
    if r < 0.6:  # 主流:人民币/美元(与 v1 分布一致)
        sym, suffix = rng.choice([("¥", ""), ("", "元"), ("$", ""), ("", "美元")])
    elif r < 0.72:  # 符号型外币
        sym, suffix = rng.choice(["£", "€", "HK$", "JP¥"]), ""
    elif r < 0.93:  # 代码型外币(半数带空格)
        code = rng.choice(list(_CUR_CODE))
        sep = " " if rng.random() < 0.5 else ""
        sym, suffix = code + sep, ""
    else:  # 汉字后缀外币
        sym, suffix = "", rng.choice(["欧元", "日元", "英镑", "港币"])
    cur = _parse_money(f"{sym}1{suffix}")[2][0]
    if rng.random() < 0.25:  # 大额 + 万/亿
        unit = rng.choice(["万", "亿"])
        x = str(rng.randint(1, 9999)) if rng.random() < 0.7 else \
            f"{rng.randint(1, 99)}.{rng.randint(1, 9)}"
        return f"{sym}{x}{unit}{suffix}", f"{read_number(x)}{unit}{cur}", "money"
    if rng.random() < 0.35:  # 小数金额
        a, b = rng.randint(0, 9999), rng.randint(1, 99)
        x = f"{a}.{b:02d}" if rng.random() < 0.5 else f"{a}.{rng.randint(1, 9)}"
        return f"{sym}{x}{suffix}", f"{read_number(x)}{cur}", "money"
    n = rng.randint(1, 999_999)
    return f"{sym}{n}{suffix}", f"{read_integer(n)}{cur}", "money"


def _valid_money(written, ctx):
    got = _parse_money(written)
    if got is None:
        return set()
    w, big, readings = got
    out = {r + big + cur for r in number_readings(w, liang_alone=True)
           for cur in readings}
    if not big and "." in w and readings[0] == "元":
        a, b = w.split(".", 1)
        if len(b) == 1 and b != "0":
            out.add(f"{read_integer(int(a))}块{read_digits(b)}")
            out.add(f"{read_integer(int(a))}元{read_digits(b)}角")
    return out


def _render_money(written, ctx):
    got = _parse_money(written)
    if got is None:
        return None
    w, big, readings = got
    return read_number(w) + big + readings[0]


# ---------------- PHONE ----------------

def _sample_phone(rng):
    kind = rng.choices(["mobile", "landline", "short", "service"], [0.5, 0.2, 0.2, 0.1])[0]
    if kind == "mobile":
        w = "1" + rng.choice("3456789") + "".join(rng.choice("0123456789") for _ in range(9))
    elif kind == "landline":
        area = rng.choice(["010", "021", "0755", "0571", "025"])
        w = area + "-" + "".join(rng.choice("0123456789") for _ in range(8))
    elif kind == "short":
        w = rng.choice(["110", "119", "120", "122", "114", "12306", "12315", "12345"])
    else:
        w = "400-" + "".join(rng.choice("0123456789") for _ in range(3)) + "-" + \
            "".join(rng.choice("0123456789") for _ in range(4))
    return w, read_digits(w, yao=True), "phone"


def _valid_phone(written, ctx):
    return digit_string_readings(written.replace("-", ""))


# ---------------- MEASURE ----------------

_UNIT_READINGS = {
    "km": ["公里", "千米"], "m": ["米"], "cm": ["厘米"], "mm": ["毫米"],
    "kg": ["公斤", "千克"], "g": ["克"], "t": ["吨"],
    "L": ["升"], "ml": ["毫升"], "℃": ["摄氏度", "度"], "°C": ["摄氏度", "度"],
    "米": ["米"], "公里": ["公里"], "斤": ["斤"], "度": ["度"],
    "V": ["伏特", "伏"], "A": ["安培", "安"], "Ω": ["欧姆", "欧"],
    "W": ["瓦特", "瓦"], "kW": ["千瓦"], "mA": ["毫安"], "mAh": ["毫安时"],
    "Hz": ["赫兹", "赫"], "kHz": ["千赫兹", "千赫"], "MHz": ["兆赫兹", "兆赫"],
    "lux": ["勒克斯"], "kcal": ["千卡", "大卡"], "kPa": ["千帕", "千帕斯卡"],
    "°": ["度"],
}


def _parse_degree(written: str):
    """度分秒(31°14′ / 22°15′0″)→ (度, 分, 秒|None);非该形态返回 None。"""
    import re
    m = re.fullmatch(r"(\d{1,3})°(\d{1,2})′(?:(\d{1,2})″)?", written)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def _degree_reading(d: int, mi: int, se) -> str:
    r = f"{read_integer(d)}度{read_integer(mi)}分"
    if se is not None:
        r += f"{read_integer(se)}秒"
    return r
# 前置模板单位:读法为「模板 format 数字」而非「数字+单位」
_UNIT_TPL = {"mph": "每小时{}英里", "km/h": "每小时{}公里",
             "m/s": "每秒{}米", "Mbps": "每秒{}兆比特"}


def _read_signed(x: str) -> str:
    """数字读法,支持 ± 前缀(正负)。"""
    if x.startswith("±"):
        return "正负" + read_number(x[1:])
    return read_number(x)


def _signed_readings(num: str, liang_alone=False) -> set[str]:
    if num.startswith("±"):
        return {"正负" + r for r in number_readings(num[1:], liang_alone)}
    return number_readings(num, liang_alone)


def _measure_num_ok(num: str) -> bool:
    body = num[1:] if num[:1] in "±-" else num
    return bool(body) and all(c.isdigit() or c == "." for c in body)


def _sample_measure(rng):
    if rng.random() < 0.06:  # 度分秒(坐标/角度)
        d, mi = rng.randint(0, 179), rng.randint(0, 59)
        se = rng.randint(0, 59) if rng.random() < 0.3 else None
        w = f"{d}°{mi}′" + (f"{se}″" if se is not None else "")
        return w, _degree_reading(d, mi, se), "measure"
    if rng.random() < 0.1:  # 前置模板单位(速度/网速)
        unit = rng.choice(list(_UNIT_TPL))
        x = str(rng.randint(1, 999))
        return f"{x}{unit}", _UNIT_TPL[unit].format(read_number(x)), "measure"
    unit = rng.choice(list(_UNIT_READINGS))
    if rng.random() < 0.3:
        x = f"{rng.randint(0, 99)}.{rng.randint(1, 9)}"
    elif unit in ("℃", "°C") and rng.random() < 0.3:
        x = f"-{rng.randint(1, 30)}" if rng.random() < 0.7 else f"±{rng.randint(1, 9)}"
    else:
        x = str(rng.randint(1, 9999))
    r = _read_signed(x)
    if x == "2":
        r = "两"
    return f"{x}{unit}", r + _UNIT_READINGS[unit][0], "measure"


def _valid_measure(written, ctx):
    deg = _parse_degree(written)
    if deg:
        return {_degree_reading(*deg)}
    for unit in sorted(_UNIT_TPL, key=len, reverse=True):
        if written.endswith(unit):
            num = written[: -len(unit)]
            if _measure_num_ok(num):
                return {_UNIT_TPL[unit].format(n) for n in _signed_readings(num)}
    for unit in sorted(_UNIT_READINGS, key=len, reverse=True):
        if written.endswith(unit):
            num = written[: -len(unit)]
            if _measure_num_ok(num):
                nums = _signed_readings(num, liang_alone=True)
                return {n + u for n in nums for u in _UNIT_READINGS[unit]}
    return set()


# ---------------- PERCENT ----------------

def _sample_percent(rng):
    if rng.random() < 0.25:
        x = f"{rng.randint(0, 99)}.{rng.randint(1, 9)}"
    elif rng.random() < 0.1:
        x = f"-{rng.randint(1, 30)}"
    else:
        x = str(rng.randint(0, 200))
    r = read_number(x.lstrip("-"))
    pre = "负" if x.startswith("-") else ""
    return f"{x}%", f"{pre}百分之{r}", "percent"


def _valid_percent(written, ctx):
    if not written.endswith("%"):
        return set()
    x = written[:-1]
    neg = x.startswith("-")
    body = x.lstrip("-")
    if not body or not all(c.isdigit() or c == "." for c in body):
        return set()
    out = set()
    for r in number_readings(body):
        out.add(("负" if neg else "") + "百分之" + r)
        if neg:
            out.add("百分之负" + r)
    return out


# ---------------- FRACTION ----------------

_VULGAR = {"½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4),
           "⅕": (1, 5), "⅛": (1, 8)}


def _frac_canonical(a: int, b: int) -> str:
    return f"{read_integer(b)}分之{read_integer(a)}"


def _sample_fraction(rng):
    if rng.random() < 0.35:  # unicode 分数字符,60% 带整数前缀(带分数)
        ch = rng.choice(list(_VULGAR))
        a, b = _VULGAR[ch]
        if rng.random() < 0.6:
            n = rng.randint(1, 9)
            return f"{n}{ch}", f"{read_integer(n)}又{_frac_canonical(a, b)}", "fraction"
        return ch, _frac_canonical(a, b), "fraction"
    b = rng.choice([100, 1000]) if rng.random() < 0.2 else rng.randint(2, 999)
    a = rng.randint(1, min(b - 1, 99))
    return f"{a}/{b}", _frac_canonical(a, b), "fraction"


def _valid_fraction(written, ctx):
    import re
    m = re.fullmatch(r"(\d*)([½⅓⅔¼¾⅕⅛])", written)
    if m:
        a, b = _VULGAR[m.group(2)]
        base = {_frac_canonical(a, b)}
        if not m.group(1):
            return base
        return {f"{rn}又{f}" for rn in integer_readings(int(m.group(1))) for f in base}
    if "/" not in written:
        return set()
    a, b = written.split("/", 1)
    if not (a.isdigit() and b.isdigit()):
        return set()
    out = {f"{rb}分之{ra}"
           for ra in integer_readings(int(a)) for rb in integer_readings(int(b))}
    if int(b) in (100, 1000):  # 百分之/千分之 变体(canonical 仍为 X分之Y,防标签漂移)
        pre = "百分之" if int(b) == 100 else "千分之"
        out |= {pre + ra for ra in integer_readings(int(a))}
    return out


# ---------------- SCORE ----------------

def _sample_score(rng):
    hi = 150 if rng.random() < 0.3 else 30  # 覆盖篮球等大比分
    a, b = rng.randint(0, hi), rng.randint(0, hi)
    sep = rng.choices([":", "-", "–"], [0.6, 0.3, 0.1])[0]
    if rng.random() < 0.08:  # 小数比率(如 1:1.5)
        w = f"1{sep}{a}.{rng.randint(1, 9)}"
        aa, bb = w.split(sep, 1)
        return w, f"{read_number(aa)}比{read_number(bb)}", "score"
    return f"{a}{sep}{b}", f"{read_integer(a)}比{read_integer(b)}", "score"


_SCORE_SEPS = ":-–—"


def _split_score(written):
    import re
    for sep in _SCORE_SEPS:
        if sep in written:
            a, b = written.split(sep, 1)
            if re.fullmatch(r"\d+(\.\d+)?", a) and re.fullmatch(r"\d+(\.\d+)?", b):
                return a, b
    return None


def _valid_score(written, ctx):
    ab = _split_score(written)
    if ab is None:
        return set()
    a, b = ab
    return {f"{ra}比{rb}"
            for ra in number_readings(a) for rb in number_readings(b)}


# ---------------- VERSION ----------------

def _ver_seg(s: str) -> str:
    """版本段 canonical:≥3 位逐位(2023 → 二零二三),短段按数值。"""
    return read_digits(s) if len(s) >= 3 else read_integer(int(s))


def _sample_version(rng):
    if rng.random() < 0.15:  # 年份式版本(2023.5.1 → 二零二三点五点一)
        segs = [str(rng.randint(2015, 2026)), str(rng.randint(1, 12))]
        if rng.random() < 0.6:
            segs.append(str(rng.randint(0, 28)))
    else:
        segs = [str(rng.randint(0, 20)) for _ in range(rng.randint(2, 3))]
    w = ".".join(segs)
    return w, "点".join(_ver_seg(s) for s in segs), "version"


def _valid_version(written, ctx):
    segs = written.split(".")
    if not all(s.isdigit() for s in segs) or len(segs) < 2:
        return set()
    per_seg = [list({read_integer(int(s))} | ({read_digits(s)} if len(s) > 1 else set()))
               for s in segs]
    return {"点".join(combo) for combo in product(*per_seg)}


# ---------------- RANGE ----------------

def _sample_range(rng):
    a = rng.randint(1, 500)
    b = rng.randint(a + 1, a + 500)
    sep = rng.choice(["-", "~", "—", "–"])
    if rng.random() < 0.2:
        return f"{a}%{sep}{b}%", f"百分之{read_integer(a)}到百分之{read_integer(b)}", "range"
    return f"{a}{sep}{b}", f"{read_integer(a)}到{read_integer(b)}", "range"


def _valid_range(written, ctx):
    import re
    m = re.fullmatch(r"(\d+)(%?)[-~—–~](\d+)(%?)", written)
    if not m:
        return set()
    a, pa, b, pb = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    out = set()
    for ra in integer_readings(a):
        for rb in integer_readings(b):
            if pa and pb:
                out.add(f"百分之{ra}到百分之{rb}")
                out.add(f"百分之{ra}到{rb}")
            elif not pa and not pb:
                out.add(f"{ra}到{rb}")
            elif pb and not pa:
                out.add(f"百分之{ra}到百分之{rb}")
                out.add(f"{ra}到百分之{rb}")
    return out


# ---------------- SERIAL ----------------

_PLATE_PROVINCES = "京沪粤川浙苏鲁豫冀晋辽吉黑皖闽赣湘鄂桂琼渝黔滇陕甘青宁新"


def _sample_serial(rng):
    r = rng.random()
    if r < 0.15:  # 车牌:京A·12345(· 不读)
        prov = rng.choice(_PLATE_PROVINCES)
        letter = rng.choice("ABCDEFGHKMNPQRSTXYZ")
        digits = "".join(rng.choice("0123456789") for _ in range(5))
        sep = "·" if rng.random() < 0.7 else ""
        w = f"{prov}{letter}{sep}{digits}"
        return w, _serial_read(w, False), "serial"
    if r < 0.4:  # 数字前置(5G)/ 交错(H1N1)
        if rng.random() < 0.5:
            w = rng.choice("123456789") + rng.choice(["G", "K", "D", "S", "P"])
        else:
            w = (rng.choice("ABCDEFGHKMN") + rng.choice("0123456789")
                 + rng.choice("KLMNPRSTV") + rng.choice("0123456789"))
        return w, _serial_read(w, False), "serial"
    letters = "".join(rng.choice("ABCDEFGHKMNPQRSTXYZ") for _ in range(rng.randint(1, 2)))
    digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 5)))
    w = letters + digits
    return w, _serial_read(w, False), "serial"


def _serial_read(written: str, yao: bool) -> str | None:
    """位置保持式:数字逐位、字母/汉字原样、·- 分隔符静音(5G→五G,H1N1→H一N一)。"""
    out = []
    for c in written:
        if c.isdigit():
            out.append(read_digits(c, yao=yao))
        elif c in "·-#":
            continue
        elif c.isalpha():
            out.append(c)
        else:
            return None
    return "".join(out)


def _valid_serial(written, ctx):
    if not any(c.isdigit() for c in written):
        return set()
    return {r for r in (_serial_read(written, False), _serial_read(written, True))
            if r is not None}


# ---------------- render(canonical,与 sampler 产出一致) ----------------

def _render_number(written, ctx):
    if ctx == "code":
        ok = written and all(c.isdigit() or c in ".-#" for c in written) \
            and any(c.isdigit() for c in written)
        return read_digits(written) if ok else None
    if written.startswith("±"):
        return "正负" + read_number(written[1:])
    for unit in ("万", "亿"):
        if written.endswith(unit):
            return read_number(written[:-1]) + unit
    return read_number(written)


def _render_date(written, ctx):
    import re
    m = re.fullmatch(r"(\d{3,4})年(?:(\d{1,2})月(\d{1,2})([日号]))?", written)
    if m:
        if m.group(2) is None:
            return f"{_read_year(m.group(1))}年"
        return (f"{_read_year(m.group(1))}年{read_integer(int(m.group(2)))}月"
                f"{read_integer(int(m.group(3)))}{m.group(4)}")
    m = re.fullmatch(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", written)
    if m:
        return (f"{_read_year(m.group(1))}年{read_integer(int(m.group(2)))}月"
                f"{read_integer(int(m.group(3)))}日")
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})([日号])", written)
    if m:
        return f"{read_integer(int(m.group(1)))}月{read_integer(int(m.group(2)))}{m.group(3)}"
    m = re.fullmatch(r"(\d{1,2})([日号])", written)
    if m:
        return f"{read_integer(int(m.group(1)))}{m.group(2)}"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", written)
    if m and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 31:
        return f"{read_integer(int(m.group(1)))}月{read_integer(int(m.group(2)))}日"
    m = re.fullmatch(r"(\d{2})年", written)
    if m:
        return f"{read_digits(m.group(1))}年"
    return None


def _render_time(written, ctx):
    if written.endswith("分") and ":" in written:
        return _render_time(written[:-1], ctx)
    parts = written.split(":")
    if not all(p.isdigit() for p in parts) or not 2 <= len(parts) <= 3:
        return None
    h, mi = int(parts[0]), int(parts[1])
    se = int(parts[2]) if len(parts) == 3 else None
    return f"{_read_hour(h)}点{_read_min(mi)}" + (_read_sec(se) if se is not None else "")


def _render_phone(written, ctx):
    digits = written.replace("-", "")
    return read_digits(digits, yao=True) if digits.isdigit() else None


def _render_measure(written, ctx):
    deg = _parse_degree(written)
    if deg:
        return _degree_reading(*deg)
    for unit in sorted(_UNIT_TPL, key=len, reverse=True):
        if written.endswith(unit):
            num = written[: -len(unit)]
            if _measure_num_ok(num):
                return _UNIT_TPL[unit].format(_read_signed(num))
    for unit in sorted(_UNIT_READINGS, key=len, reverse=True):
        if written.endswith(unit):
            num = written[: -len(unit)]
            if _measure_num_ok(num):
                r = "两" if num == "2" else _read_signed(num)
                return r + _UNIT_READINGS[unit][0]
    return None


def _render_percent(written, ctx):
    if not written.endswith("%"):
        return None
    x = written[:-1]
    body = x.lstrip("-")
    if not body or not all(c.isdigit() or c == "." for c in body):
        return None
    return ("负" if x.startswith("-") else "") + "百分之" + read_number(body)


def _render_fraction(written, ctx):
    import re
    m = re.fullmatch(r"(\d*)([½⅓⅔¼¾⅕⅛])", written)
    if m:
        a, b = _VULGAR[m.group(2)]
        core = _frac_canonical(a, b)
        return f"{read_integer(int(m.group(1)))}又{core}" if m.group(1) else core
    if "/" not in written:
        return None
    a, b = written.split("/", 1)
    if not (a.isdigit() and b.isdigit()):
        return None
    return _frac_canonical(int(a), int(b))


def _render_score(written, ctx):
    ab = _split_score(written)
    if ab is None:
        return None
    return f"{read_number(ab[0])}比{read_number(ab[1])}"


def _render_version(written, ctx):
    segs = written.split(".")
    if not all(s.isdigit() for s in segs) or len(segs) < 2:
        return None
    return "点".join(_ver_seg(s) for s in segs)


def _render_range(written, ctx):
    import re
    m = re.fullmatch(r"(\d+)(%?)[-~—–~](\d+)(%?)", written)
    if not m:
        return None
    a, b = read_integer(int(m.group(1))), read_integer(int(m.group(3)))
    if m.group(2) and m.group(4):
        return f"百分之{a}到百分之{b}"
    if not m.group(2) and not m.group(4):
        return f"{a}到{b}"
    if m.group(4):
        return f"百分之{a}到百分之{b}"
    return None


def _render_serial(written, ctx):
    if not any(c.isdigit() for c in written) or not any(c.isalpha() for c in written):
        return None
    return _serial_read(written, yao=False)


# ---------------- ROMAN(罗马数字) ----------------

_ROMAN_VALS = [("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100),
               ("XC", 90), ("L", 50), ("XL", 40), ("X", 10), ("IX", 9),
               ("V", 5), ("IV", 4), ("I", 1)]


def _int_to_roman(n: int) -> str:
    out = []
    for sym, val in _ROMAN_VALS:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def _roman_to_int(s: str) -> int | None:
    if not s or any(c not in "MDCLXVI" for c in s):
        return None
    i, n = 0, 0
    for sym, val in _ROMAN_VALS:
        while s[i:i + len(sym)] == sym:
            n += val
            i += len(sym)
    return n if i == len(s) and s == _int_to_roman(n) else None


def _sample_roman(rng):
    if rng.random() < 0.15:  # 年份:罗马 → 逐位读(MMXXIII年)
        y = rng.randint(1900, 2026)
        return _int_to_roman(y), read_digits(str(y)), "roman_year"
    n = rng.randint(1, 60)
    return _int_to_roman(n), read_integer(n), "roman"


def _valid_roman(written, ctx):
    n = _roman_to_int(written)
    if n is None:
        return set()
    out = set(integer_readings(n))
    out |= {"第" + r for r in integer_readings(n)}  # 届/世纪等序数语境(XXIII届→第二十三届)
    if n >= 1000:  # 年份逐位变体
        out.add(read_digits(str(n)))
    return out


def _render_roman(written, ctx):
    n = _roman_to_int(written)
    if n is None:
        return None
    if ctx == "roman_year" and n >= 1000:
        return read_digits(str(n))
    return read_integer(n)


# ---------------- MATH(简单算式) ----------------

_MATH_OPS = {"+": "加", "-": "减", "×": "乘", "*": "乘", "÷": "除以",
             "=": "等于", "≈": "约等于"}


def _sample_math(rng):
    kind = rng.choices(["arith", "chain", "pow", "sqrt", "pi"],
                       [0.4, 0.2, 0.15, 0.15, 0.1])[0]
    if kind == "arith":
        a, b = rng.randint(0, 99), rng.randint(1, 99)
        op = rng.choice(["+", "-", "×", "÷"])
        if op == "-" and a < b:
            a, b = b, a  # 结果非负(负号会与减号 token 混淆)
        c = {"+": a + b, "-": a - b, "×": a * b, "÷": a}[op]
        if op == "÷":
            a = b * c  # 整除
        w = f"{a}{op}{b}={c}"
        r = f"{read_integer(a)}{_MATH_OPS[op]}{read_integer(b)}等于{read_integer(c)}"
        return w, r, "math"
    if kind == "chain":
        a, b, c = rng.randint(1, 20), rng.randint(1, 9), rng.randint(1, 20)
        op1, op2 = rng.choice([("+", "+"), ("×", "+"), ("+", "×")])
        v = eval(f"{a}{op1.replace('×', '*')}{b}{op2.replace('×', '*')}{c}")  # noqa: S307
        w = f"{a}{op1}{b}{op2}{c}={v}"
        r = (f"{read_integer(a)}{_MATH_OPS[op1]}{read_integer(b)}"
             f"{_MATH_OPS[op2]}{read_integer(c)}等于{read_integer(v)}")
        return w, r, "math"
    if kind == "pow":
        a, p = rng.randint(2, 20), rng.choice([2, 3])
        sup = "²" if p == 2 else "³"
        w = f"{a}{sup}={a ** p}"
        pr = "的平方" if p == 2 else "的三次方"
        return w, f"{read_integer(a)}{pr}等于{read_integer(a ** p)}", "math"
    if kind == "sqrt":
        b = rng.randint(2, 30)
        paren = rng.random() < 0.3
        x = f"({b * b})" if paren else str(b * b)
        return f"√{x}={b}", f"根号{read_integer(b * b)}等于{read_integer(b)}", "math"
    x = f"3.{rng.choice(['14', '1416'])}"
    return f"π≈{x}", f"圆周率约等于{read_number(x)}", "math"


_FULLWIDTH_OPS = str.maketrans("＝＋－＊（）", "=+-*()")


def _math_readings(written: str) -> set[str] | None:
    """算式逐 token 解析 → 读法集合(变体:乘/乘以、的平方/的二次方)。"""
    import re
    written = written.translate(_FULLWIDTH_OPS)
    toks = re.findall(r"\d+(?:\.\d+)?|√|π|²|³|[+\-×*÷=≈()]", written)
    if not toks or "".join(toks) != written.replace(" ", ""):
        return None
    parts: list[list[str]] = []
    for t in toks:
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            parts.append(sorted(number_readings(t)))
        elif t == "√":
            parts.append(["根号"])
        elif t == "π":
            parts.append(["圆周率"])
        elif t == "²":
            parts.append(["的平方", "的二次方"])
        elif t == "³":
            parts.append(["的三次方", "的立方"])
        elif t in "()":
            parts.append([""])
        elif t == "×" or t == "*":
            parts.append(["乘", "乘以"])
        else:
            parts.append([_MATH_OPS[t]])
    out = {""}
    for p in parts:
        out = {a + b for a in out for b in p}
        if len(out) > 4000:
            return None
    return out


def _valid_math(written, ctx):
    return _math_readings(written) or set()


def _render_math(written, ctx):
    import re
    written = written.translate(_FULLWIDTH_OPS)
    toks = re.findall(r"\d+(?:\.\d+)?|√|π|²|³|[+\-×*÷=≈()]", written)
    if not toks or "".join(toks) != written.replace(" ", ""):
        return None
    canon = {"√": "根号", "π": "圆周率", "²": "的平方", "³": "的三次方",
             "(": "", ")": "", "*": "乘", "×": "乘"}
    out = []
    for t in toks:
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            out.append(read_number(t))
        else:
            out.append(canon.get(t, _MATH_OPS.get(t)))
    return "".join(out) if all(x is not None for x in out) else None


# ---------------- registry ----------------

_REGISTRY = {
    "NUMBER": (_sample_number, _valid_number, _render_number),
    "DATE": (_sample_date, _valid_date, _render_date),
    "TIME": (_sample_time, _valid_time, _render_time),
    "MONEY": (_sample_money, _valid_money, _render_money),
    "PHONE": (_sample_phone, _valid_phone, _render_phone),
    "MEASURE": (_sample_measure, _valid_measure, _render_measure),
    "PERCENT": (_sample_percent, _valid_percent, _render_percent),
    "FRACTION": (_sample_fraction, _valid_fraction, _render_fraction),
    "SCORE": (_sample_score, _valid_score, _render_score),
    "VERSION": (_sample_version, _valid_version, _render_version),
    "RANGE": (_sample_range, _valid_range, _render_range),
    "SERIAL": (_sample_serial, _valid_serial, _render_serial),
    "ROMAN": (_sample_roman, _valid_roman, _render_roman),
    "MATH": (_sample_math, _valid_math, _render_math),
}

CLASSES = list(_REGISTRY)


def sample_nsw(cls: str, rng: random.Random) -> NSWSample:
    written, reading, ctx = _REGISTRY[cls][0](rng)
    return NSWSample(cls=cls, ctx=ctx, written=written, reading=reading)


def valid_readings(cls: str, written: str, ctx: str = "") -> set[str]:
    try:
        return _REGISTRY[cls][1](written, ctx)
    except (ValueError, KeyError):
        return set()


def is_valid(cls: str, written: str, ctx: str, reading: str) -> bool:
    return reading in valid_readings(cls, written, ctx)


def render(cls: str, written: str, ctx: str = "") -> str | None:
    """任意书面形式 → canonical 读法;无法解析或不在合法集合内返回 None。"""
    try:
        r = _REGISTRY[cls][2](written, ctx)
    except (ValueError, KeyError, IndexError):
        return None
    if r is None or r not in valid_readings(cls, written, ctx):
        return None
    return r


def split_gold_edit(anchor: str, replacement: str, cls: str, ctx: str):
    """反推带扩窗上下文的 gold 编辑:(prefix_len, suffix_len, written, reading, 合法读法集)。

    anchor = prefix + written + suffix,replacement = prefix + reading + suffix,
    prefix/suffix ≤ 2 字(tn.anchor.MAX_EXPAND)。无法反推返回 None。
    """
    for p in range(0, 3):
        for s in range(0, 3):
            if p + s >= len(anchor) + 1 or p + s > len(replacement):
                continue
            pre = anchor[:p]
            suf = anchor[len(anchor) - s:] if s else ""
            if not replacement.startswith(pre) or (suf and not replacement.endswith(suf)):
                continue
            reading = replacement[p: len(replacement) - s if s else len(replacement)]
            written = anchor[p: len(anchor) - s if s else len(anchor)]
            cands = valid_readings(cls, written, ctx)
            if reading in cands:
                return p, s, written, reading, cands
    return None
