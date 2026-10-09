# -*- coding: utf-8 -*-
"""LongBench-v2 四选一解析——统一权威实现（E117b，GPT TL-LBV2-PARSER-037 修复）。

历史 bug：pred.py / eval.py 两份手写解析共享的第三级兜底
    re.search(r"\\b([ABCD])\\b", t, flags=re.IGNORECASE)
会把英文冠词 "a"（及正文中任意首个独立字母）判成选项 A；第二条
"The correct answer is ([A-D])" 启用 IGNORECASE 同样会把
"The correct answer is a difficult choice..." 中的冠词 a 判成 A。
该行为既不等于回答语义，也偏离官方评分器（THUDM/LongBench pred.py
仅接受两条大小写敏感格式），可同时制造假阳性与假阴性——503 题 ×3 臂
重放实测：aavg 4 个假阳性，三臂排序反转（aavg 32.60 → 31.81）。

修复口径：正式评分只使用官方两条大小写敏感模式，无匹配记 None / 0 分，
不再接受任何兜底。不做星号剥离（官方无此行为；503×3 重放实测剥离与否
对解析结果逐位等价）。本模块无重依赖（仅 re），生成端（GPU 推理机）与
评分端共同 import，消除两份手写正则的漂移。
"""
import re

# scorer 版本标识——receipt / 结果 JSON 应引用本字段做解析器身份绑定
LBV2_PARSER_VERSION = "official-strict-v1"

# 官方两条格式（THUDM/LongBench pred.py，大小写敏感，无其他兜底）
_P_OFFICIAL_1 = re.compile(r"The correct answer is \(([A-D])\)")
_P_OFFICIAL_2 = re.compile(r"The correct answer is ([A-D])")


def extract_choice_official(text):
    """官方口径解析：'The correct answer is (X)' → 'The correct answer is X'，
    无匹配返回 None。空/None 输入返回 None。"""
    if not text:
        return None
    m = _P_OFFICIAL_1.search(text)
    if m:
        return m.group(1)
    m = _P_OFFICIAL_2.search(text)
    if m:
        return m.group(1)
    return None
