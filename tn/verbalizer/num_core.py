"""中文数字读法核心原语:整数/小数/逐位,及合法变体集合。"""

DIG = "零一二三四五六七八九"
DIG_YAO = "零幺二三四五六七八九"
_UNITS = ("", "十", "百", "千")
_GUNITS = ("", "万", "亿", "万亿")


def read_digits(s: str, yao: bool = False) -> str:
    """逐位读数字串(电话/编号/年份)。yao=True 时 1 读幺。"""
    table = DIG_YAO if yao else DIG
    return "".join(table[int(c)] for c in s if c.isdigit())


def digit_string_readings(s: str) -> set[str]:
    """逐位读法的合法集合:全串一式 / 全串幺式。"""
    return {read_digits(s, False), read_digits(s, True)}


def read_integer(n: int) -> str:
    """数量读法。0→零,15→十五,110→一百一十,100000001→一亿零一。"""
    if n < 0:
        return "负" + read_integer(-n)
    if n == 0:
        return "零"
    s = str(n)
    if len(s) > 16:
        raise ValueError(f"integer too large: {n}")
    out: list[str] = []
    zero_pending = False
    group_emitted = False
    L = len(s)
    for i, ch in enumerate(s):
        pos = L - 1 - i
        d = int(ch)
        u = pos % 4
        g = pos // 4
        if d == 0:
            zero_pending = True
        else:
            if out and zero_pending:
                out.append("零")
            zero_pending = False
            out.append(DIG[d])
            if u:
                out.append(_UNITS[u])
            group_emitted = True
        if u == 0:
            if group_emitted and g > 0:
                out.append(_GUNITS[g])
            group_emitted = False
    r = "".join(out)
    if 10 <= n < 20:
        r = r[1:]  # 一十五 → 十五
    return r


def read_number(s: str) -> str:
    """字符串数字(可含负号/小数点/千分位逗号)→ 数量读法。"""
    s = s.replace(",", "").replace(",", "")
    neg = s.startswith("-") or s.startswith("−")
    if neg:
        s = s[1:]
    if "." in s:
        a, b = s.split(".", 1)
        r = read_integer(int(a or "0")) + "点" + read_digits(b)
    else:
        r = read_integer(int(s))
    return ("负" + r) if neg else r


def _liang(reading: str) -> str:
    for a, b in (("二百", "两百"), ("二千", "两千"), ("二万", "两万"), ("二亿", "两亿")):
        reading = reading.replace(a, b)
    return reading


def integer_readings(n: int, liang_alone: bool = False) -> set[str]:
    """整数的合法读法集合:canonical + 两 变体 + 一十/十 变体。

    liang_alone=True 时把单独的 2 也接受为「两」(量词语境)。
    """
    c = read_integer(n)
    s = {c, _liang(c)}
    if 10 <= abs(n) < 20:
        prefix = "负" if n < 0 else ""
        s.add(prefix + "一" + c.removeprefix("负"))
    if abs(n) == 2 and liang_alone:
        s.add(("负" if n < 0 else "") + "两")
    return s


def number_readings(s: str, liang_alone: bool = False) -> set[str]:
    """字符串数字的合法读法集合(整数含变体;小数变体作用于整数部分)。"""
    s2 = s.replace(",", "").replace(",", "")
    if "." not in s2:
        return integer_readings(int(s2), liang_alone)
    c = read_number(s2)
    return {c, _liang(c)}
