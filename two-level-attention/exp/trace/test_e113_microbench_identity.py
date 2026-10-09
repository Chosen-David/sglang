#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 microbench 050 修复的红绿单测（CPU-only，无需 GPU/无需跑性能）。

对应 GPT 审计 TL-E113-BENCH-PROVENANCE-050 的修复验收（方案 1-3）：
  T1 check_pair fail-closed 门：一致状态 → 空错误列表；
     assign 注错 / k_live 注错 / cnt/sq/簇心 sums 超容差 → 各自报错（红）
  T2 输出内容 SHA 自洽：canonical_bytes/output_content_sha256/verify_output_sha256
     往返一致；篡改 cases 后旧 SHA 校验必须失败（红）
  T3 原子发布：atomic_write_bytes 落盘内容一致 + 临时文件清理干净
  T4 身份工具：git_identity 对真实仓库返回 SHA；file_sha256 与 hashlib 一致；
     build_identity 双侧字段齐备（050 manifest 契约）

用法：python3 test_e113_microbench_identity.py   （exp/trace/ 目录下，纯 CPU）
"""
import glob
import hashlib
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import e113_microbench as MB   # noqa: E402  （模块级只 import torch，实现加载在 main 内）

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def make_state_pair(T=64, H=2, K=16, dd=8, seed=123):
    """构造一对（参考态, 被测态）贪心终态张量（初始完全一致，CPU 张量）。"""
    g = torch.Generator().manual_seed(seed)
    sums = torch.randn(H, K, dd, generator=g)
    cnt = torch.rand(H, K, generator=g) * 8 + 1
    sq = (sums * sums).sum(-1)
    kl = torch.tensor([3, 5], dtype=torch.long)
    assign = torch.stack([torch.randint(0, 3, (T,), generator=g),
                          torch.randint(0, 5, (T,), generator=g)]).long()
    r = (sums.clone(), cnt.clone(), sq.clone(), kl.clone(), assign.clone())
    t = (sums.clone(), cnt.clone(), sq.clone(), kl.clone(), assign.clone())
    return r, t


# ================================================================ T1 fail-closed 门
def t1_gate_red_green():
    name = "T1 check_pair fail-closed（一致过 / 各注错必红）"
    try:
        # 绿：完全一致 → 无错误
        r, t = make_state_pair()
        assert MB.check_pair(r, t, "green") == [], "一致状态不应报错"
        # 红 1：assign 单点注错
        r2, t2 = make_state_pair()
        t2[4][0, 7] = (t2[4][0, 7] + 1) % 3
        errs = MB.check_pair(r2, t2, "assign")
        assert any("assign mismatch" in e for e in errs), f"assign 注错未检出: {errs}"
        # 红 2：k_live 注错
        r3, t3 = make_state_pair()
        t3[3][1] += 1
        errs = MB.check_pair(r3, t3, "klive")
        assert any("k_live 不一致" in e for e in errs), f"k_live 注错未检出: {errs}"
        # 红 3-5：cnt / sq / 簇心 sums 各自超容差
        for idx, key, perturb in ((0, "簇心 sums", lambda a: a + 1.0),
                                  (1, "cnt", lambda a: a + 1.0),
                                  (2, "sq", lambda a: a + 1.0)):
            r4, t4 = make_state_pair()
            live = int(r4[3].max().item())
            t4l = list(t4)
            t4l[idx][:, :live] = perturb(t4l[idx][:, :live])
            errs = MB.check_pair(r4, tuple(t4l), key)
            assert any(key in e for e in errs), f"{key} 超容差未检出: {errs}"
        report(name, True, "1 绿 + 5 红（assign/k_live/sums/cnt/sq）全检出")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T2 内容 SHA 自洽
def t2_output_sha():
    name = "T2 输出内容 SHA 自洽（往返一致 / 篡改必红）"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "fake.json")
            doc = {"meta": {"probe": "x", "started": "t"}, "cases": [{"T": 8, "speedup": 1.0}]}
            doc["output_content_sha256"] = MB.output_content_sha256(doc)
            MB.atomic_write_bytes(out, json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8"))
            assert MB.verify_output_sha256(out), "发布文件内容 SHA 校验失败（不应发生）"
            # 红：篡改 cases 但保留旧 SHA → 校验必须失败
            bad = json.loads(open(out, encoding="utf-8").read())
            bad["cases"][0]["speedup"] = 999.0
            MB.atomic_write_bytes(out, json.dumps(bad, indent=1, ensure_ascii=False).encode("utf-8"))
            assert not MB.verify_output_sha256(out), "篡改后旧 SHA 仍通过——防篡改失效"
            # 无 SHA 键的文件也必须拒绝（fail-closed：claimed is None → False）
            MB.atomic_write_bytes(out, json.dumps({"meta": {}}, indent=1).encode())
            assert not MB.verify_output_sha256(out), "缺 SHA 键的文件不应通过校验"
        report(name, True, "往返校验过 + 篡改红 + 缺键红")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T3 原子发布
def t3_atomic_publish():
    name = "T3 atomic_write_bytes（落盘一致 + 临时文件清理）"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "pub.json")
            data = json.dumps({"a": 1}, indent=1).encode("utf-8")
            MB.atomic_write_bytes(out, data)
            assert open(out, "rb").read() == data, "落盘内容与写入不一致"
            leftovers = glob.glob(os.path.join(td, "*.tmp-*"))
            assert not leftovers, f"临时文件残留: {leftovers}"
            # 覆盖发布（replace 语义）同样成立
            MB.atomic_write_bytes(out, b'{"a": 2}')
            assert open(out, "rb").read() == b'{"a": 2}', "二次发布覆盖失败"
        report(name, True, "内容一致 + tmp 清零 + 覆盖发布过")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ T4 身份工具
def t4_identity_tools():
    name = "T4 git_identity / file_sha256 / build_identity（050 manifest 契约）"
    try:
        root = MB.DEFAULT_ROOT
        gi = MB.git_identity(root)
        assert gi["git_sha"] and len(gi["git_sha"]) == 40, f"git SHA 缺失: {gi}"
        assert isinstance(gi["git_dirty"], bool), f"dirty 应为 bool: {gi}"
        # 非 git 目录 → 显式 None（不静默冒充）
        with tempfile.TemporaryDirectory() as td:
            gi2 = MB.git_identity(td)
            assert gi2["git_sha"] is None and gi2["git_dirty"] is None, \
                f"非 git 目录应显式 None: {gi2}"
        # file_sha256 与 hashlib 口径一致
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"print(1)\n")
            tmpf = f.name
        try:
            assert MB.file_sha256(tmpf) == hashlib.sha256(b"print(1)\n").hexdigest(), \
                "file_sha256 与 hashlib 不一致"
        finally:
            os.unlink(tmpf)
        # build_identity：双侧字段齐备（050 manifest 契约）
        ident = MB.build_identity(root, root)
        for side in ("ref", "tri"):
            assert ident[side]["impl_sha256"] and len(ident[side]["impl_sha256"]) == 64, \
                f"{side} 实现 SHA256 缺失"
            assert ident[side]["git"]["git_sha"], f"{side} git 身份缺失"
        assert ident["same_root"] is True
        assert os.path.isfile(ident["ref"]["impl_file"]) and os.path.isfile(ident["tri"]["impl_file"])
        assert ident["microbench_script_sha256"], "脚本自身 SHA 缺失"
        report(name, True, f"git={gi['git_sha'][:12]}… dirty={gi['git_dirty']}，"
                           f"双侧 SHA 齐，非 git 目录显式 None")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    t1_gate_red_green()
    t2_output_sha()
    t3_atomic_publish()
    t4_identity_tools()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 60)
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - n_fail}，失败 {n_fail}")
    sys.exit(1 if n_fail else 0)
