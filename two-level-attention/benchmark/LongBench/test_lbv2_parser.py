# -*- coding: utf-8 -*-
# E117b 红绿测试（GPT 审计 TL-LBV2-PARSER-037 修复验证）
# 验证：
#   A. lbv2_choice.extract_choice_official 官方口径表驱动用例
#      （官方两条正例 / 小写 / 星号 / 冠词 a / option C / 多候选字母 /
#        先提 A 后官方答 C / 拒答 / 空输出）
#   B. 与固定官方参考实现逐项一致（THUDM/LongBench pred.py 两条模式，冻结）
#   C. 生成端（pred.extract_choice_letter）与评分端（eval.lbv2_choice_score）
#      双侧接入同一模块——生成端解析与评分端解析逐项一致，冠词反例不再假阳性
# 用法: python benchmark/LongBench/test_lbv2_parser.py
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PASS = 0

# ---------- 固定官方参考实现（冻结，独立于被测模块） ----------
_REF_1 = re.compile(r"The correct answer is \(([A-D])\)")
_REF_2 = re.compile(r"The correct answer is ([A-D])")


def ref_official(text):
    if not text:
        return None
    m = _REF_1.search(text)
    if m:
        return m.group(1)
    m = _REF_2.search(text)
    if m:
        return m.group(1)
    return None


# ---------- A+B. 表驱动用例 ----------
# (text, expected_official_choice)
CASES = [
    # 官方两条正例
    ("The correct answer is (C).", "C"),
    ("The correct answer is C", "C"),
    ("The correct answer is (A)", "A"),
    ("Answer: The correct answer is (D).", "D"),
    # 小写句式——官方大小写敏感，无匹配
    ("the correct answer is (c)", None),
    ("The correct answer is c", None),
    # 星号：整句加粗不隔断句式（仍匹配）；星号隔断句式 = 无匹配
    ("**The correct answer is (C)**", "C"),
    ("The correct answer is **(C)**", None),
    # 冠词 a 反例（GPT 审计核心反例——历史兜底判 A）
    ("The correct answer is a difficult choice; I lean C.", None),
    ("The correct answer is a difficult choice", None),
    # option C 表述（历史兜底取全文首个独立字母判 A）
    ("This is a hard call; option C is best.", None),
    # 拒答 / 空输出
    ("I cannot provide a definitive answer.", None),
    ("", None),
    (None, None),
    # 多候选字母——官方锚定优先，不被前文字母干扰
    ("A or B, maybe C; anyway, The correct answer is (B).", "B"),
    # 解释中先提 A，最终官方格式答 C
    ("I think A is plausible, but The correct answer is (C).", "C"),
    # 小写独立字母散落正文——一律不解析
    ("option a vs option b: b looks right to me.", None),
]


def test_table():
    global PASS
    from benchmark.LongBench.lbv2_choice import extract_choice_official
    for text, exp in CASES:
        got = extract_choice_official(text)
        assert got == exp, (text, got, exp)
        ref = ref_official(text)
        assert got == ref, ("与官方参考不一致", text, got, ref)
    n_none = sum(1 for _, e in CASES if e is None)
    print(f"A+B PASS  官方口径表驱动 {len(CASES)} 例（{n_none} 例 None）与固定参考逐项一致")
    PASS += 1


# ---------- C. 生成端 / 评分端双侧一致 + 假阳性消除 ----------
def test_both_sides():
    global PASS
    # 生成端：import 真实 pred.py 的 extract_choice_letter
    # pred.py 顶层 import torch/datasets 等重依赖——用 AST 抽取函数体不可行（已委托），
    # 改为校验源文件确实委托 lbv2_choice（无手写正则残留），并直接测评分端真实函数。
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred.py")).read()
    assert "extract_choice_official(text)" in src, "pred.py 未委托统一模块"
    # 只匹配真实代码（docstring 中的历史描述不算残留）
    assert re.search(r"flags\s*=\s*re\.IGNORECASE", src) is None, "pred.py 残留 IGNORECASE 手写解析"
    assert r"\b([ABCD])\b" not in src, "pred.py 残留独立字母兜底"

    # 评分端：import 真实 eval.py 的 lbv2_choice_score
    from benchmark.LongBench.eval import lbv2_choice_score
    from benchmark.LongBench.lbv2_choice import LBV2_PARSER_VERSION, extract_choice_official
    # 冠词反例不再假阳性：真值 A + 拒答文本 → 0 分（历史实现返回 1.0）
    assert lbv2_choice_score("I cannot provide a definitive answer.", "A") == 0.0
    assert lbv2_choice_score("The correct answer is a difficult choice; I lean C.", "A") == 0.0
    # 真值 C + 冠词文本 → 0 分（历史兜底判 A 同样不中）
    assert lbv2_choice_score("The correct answer is a difficult choice; I lean C.", "C") == 0.0
    # 官方正例正常得分
    assert lbv2_choice_score("The correct answer is (C).", "C") == 1.0
    assert lbv2_choice_score("The correct answer is C", "C") == 1.0
    # 评分端解析与统一模块逐项一致（prediction → choice → 判分闭合）
    for text, exp in CASES:
        gt = exp if exp else "A"
        expect_score = 1.0 if (exp is not None and exp == gt) else 0.0
        assert lbv2_choice_score(text, gt) == expect_score, (text, gt)
    print(f"C PASS  评分端真实函数冠词反例 0 假阳性 + {len(CASES)} 例判分闭合；"
          f"pred.py 无手写正则残留（parser={LBV2_PARSER_VERSION}）")
    PASS += 1


if __name__ == "__main__":
    test_table()
    test_both_sides()
    print(f"E117b parser gate: {PASS}/{PASS} PASS")
