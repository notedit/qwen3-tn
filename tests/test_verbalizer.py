"""verbalizer 单测:数字核心 + 12 类手写断言 + 采样/校验自洽性。"""

import random

import pytest

from tn.verbalizer import CLASSES, is_valid, sample_nsw, valid_readings
from tn.verbalizer.num_core import (
    integer_readings,
    read_digits,
    read_integer,
    read_number,
)

# ---------- 数字核心 ----------

@pytest.mark.parametrize("n,expect", [
    (0, "零"), (1, "一"), (2, "二"), (10, "十"), (11, "十一"), (15, "十五"),
    (20, "二十"), (105, "一百零五"), (110, "一百一十"), (200, "二百"),
    (1000, "一千"), (1001, "一千零一"), (1010, "一千零一十"), (1100, "一千一百"),
    (2005, "二千零五"), (9999, "九千九百九十九"), (10000, "一万"),
    (10001, "一万零一"), (10500000, "一千零五十万"), (100000001, "一亿零一"),
    (200000, "二十万"), (123456789, "一亿二千三百四十五万六千七百八十九"),
    (1000000000000, "一万亿"), (-3, "负三"), (-15, "负十五"),
])
def test_read_integer(n, expect):
    assert read_integer(n) == expect


@pytest.mark.parametrize("s,expect", [
    ("12.5", "十二点五"), ("0.5", "零点五"), ("3.14", "三点一四"),
    ("1,234", "一千二百三十四"), ("-2.5", "负二点五"), ("100.01", "一百点零一"),
])
def test_read_number(s, expect):
    assert read_number(s) == expect


def test_read_digits():
    assert read_digits("110") == "一一零"
    assert read_digits("110", yao=True) == "幺幺零"
    assert read_digits("2024") == "二零二四"


def test_integer_variants():
    assert "两百" in integer_readings(200)
    assert "一十五" in integer_readings(15)
    assert "两" in integer_readings(2, liang_alone=True)
    assert "两" not in integer_readings(2)


# ---------- NUMBER ----------

@pytest.mark.parametrize("w,ctx,r,ok", [
    ("18", "quantity", "十八", True),
    ("18", "quantity", "一八", False),
    ("18", "code", "一八", True),
    ("18", "code", "幺八", True),
    ("18", "code", "十八", False),
    ("1200万", "quantity", "一千二百万", True),
    ("1.5亿", "quantity", "一点五亿", True),
    ("2", "quantity", "两", True),
    ("-3", "quantity", "负三", True),
    ("1,234", "quantity", "一千二百三十四", True),
    ("12.5", "quantity", "十二点五", True),
    ("110", "quantity", "一百一十", True),
    ("110", "code", "幺幺零", True),
])
def test_number(w, ctx, r, ok):
    assert is_valid("NUMBER", w, ctx, r) is ok


# ---------- DATE ----------

@pytest.mark.parametrize("w,r,ok", [
    ("2024年3月8日", "二零二四年三月八日", True),
    ("2024年3月8日", "二千零二十四年三月八日", False),
    ("2024-03-08", "二零二四年三月八日", True),
    ("2024-03-08", "二零二四年三月八号", True),
    ("3月18号", "三月十八号", True),
    ("18号", "十八号", True),
    ("18号", "幺八号", False),
    ("1998年", "一九九八年", True),
    ("12月31日", "十二月三十一日", True),
])
def test_date(w, r, ok):
    assert is_valid("DATE", w, "date", r) is ok


# ---------- TIME ----------

@pytest.mark.parametrize("w,r,ok", [
    ("3:30", "三点三十分", True),
    ("3:30", "三点半", True),
    ("3:05", "三点零五分", True),
    ("3:05", "三点五分", False),
    ("15:00", "十五点", True),
    ("15:00", "十五点整", True),
    ("2:00", "两点", True),
    ("2:00", "二点", True),
    ("12:30:15", "十二点三十分十五秒", True),
])
def test_time(w, r, ok):
    assert is_valid("TIME", w, "time", r) is ok


# ---------- MONEY ----------

@pytest.mark.parametrize("w,r,ok", [
    ("¥1200万", "一千二百万元", True),
    ("¥99", "九十九元", True),
    ("12.5元", "十二点五元", True),
    ("12.5元", "十二块五", True),
    ("12.5元", "十二元五角", True),
    ("$30", "三十美元", True),
    ("2元", "两元", True),
    ("2元", "二元", True),
    ("3.5亿元", "三点五亿元", True),
])
def test_money(w, r, ok):
    assert is_valid("MONEY", w, "money", r) is ok


# ---------- PHONE ----------

@pytest.mark.parametrize("w,r,ok", [
    ("110", "幺幺零", True),
    ("110", "一一零", True),
    ("110", "一百一十", False),
    ("13800138000", "幺三八零零幺三八零零零", True),
    ("010-12345678", "零幺零幺二三四五六七八", True),
    ("400-123-4567", "四零零幺二三四五六七", True),
])
def test_phone(w, r, ok):
    assert is_valid("PHONE", w, "phone", r) is ok


# ---------- MEASURE / PERCENT / FRACTION / SCORE / VERSION / RANGE / SERIAL ----------

@pytest.mark.parametrize("cls,w,r,ok", [
    ("MEASURE", "3km", "三公里", True),
    ("MEASURE", "3km", "三千米", True),
    ("MEASURE", "2kg", "两公斤", True),
    ("MEASURE", "-5℃", "负五摄氏度", True),
    ("MEASURE", "-5℃", "负五度", True),
    ("MEASURE", "1.5L", "一点五升", True),
    ("PERCENT", "25%", "百分之二十五", True),
    ("PERCENT", "0.5%", "百分之零点五", True),
    ("PERCENT", "-3%", "负百分之三", True),
    ("PERCENT", "25%", "二十五", False),
    ("FRACTION", "3/4", "四分之三", True),
    ("FRACTION", "3/4", "三分之四", False),
    ("SCORE", "2:1", "二比一", True),
    ("SCORE", "2:1", "两比一", False),
    ("VERSION", "2.0", "二点零", True),
    ("VERSION", "15.1", "十五点一", True),
    ("VERSION", "2.0.1", "二点零点一", True),
    ("VERSION", "2.10", "二点一零", True),
    ("RANGE", "3-5", "三到五", True),
    ("RANGE", "10%-20%", "百分之十到百分之二十", True),
    ("RANGE", "10~20", "十到二十", True),
    ("SERIAL", "A380", "A三八零", True),
    ("SERIAL", "A380", "A幺八零", False),
    ("SERIAL", "G102", "G幺零二", True),
])
def test_misc_classes(cls, w, r, ok):
    assert is_valid(cls, w, "", r) is ok


# ---------- 自洽性:采样出的 gold 必须通过自家反向校验 ----------

def test_sample_self_consistency():
    rng = random.Random(7)
    for cls in CLASSES:
        for _ in range(500):
            s = sample_nsw(cls, rng)
            assert s.reading, (cls, s)
            assert is_valid(cls, s.written, s.ctx, s.reading), \
                f"{cls}: {s.written!r} -> {s.reading!r} 未通过反向校验"


def test_render_reproduces_sampler_canonical():
    # render 契约:对 sampler 产出的任意书面形式,渲染结果 == sampler 的 gold
    from tn.verbalizer import render
    rng = random.Random(13)
    for cls in CLASSES:
        for _ in range(500):
            s = sample_nsw(cls, rng)
            assert render(cls, s.written, s.ctx) == s.reading, \
                f"{cls}: render({s.written!r}) != {s.reading!r}"


def test_valid_readings_bounded():
    # 校验集合规模必须有界(防组合爆炸)
    rng = random.Random(11)
    for cls in CLASSES:
        for _ in range(200):
            s = sample_nsw(cls, rng)
            assert len(valid_readings(cls, s.written, s.ctx)) < 5000, (cls, s.written)
