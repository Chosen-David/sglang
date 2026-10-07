#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113 kernel 的 exp/trace 入口（E113b e2e 集成后的转发层）。

kernel 本体已收编进生产路径 sparse_attn/indexer/greedy_triton.py
（tli_indexer.py `_greedy_cluster_pass` 调度器直接调用；远程部署 rsync
sparse_attn 即携带）。本文件按文件路径加载本体并转发导出，保证
exp/trace 下单测/bench 脚本的 `from e113_greedy_triton import ...`
继续可用，且测的就是生产本体（单一事实源，无副本漂移）。

原型与实证历史（对拍单测 T1-T5、microbench、SEG-GREEDY 仿真）见
research/docs/e113_method_kernel_design.md。
"""
import importlib.util
import os

_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "sparse_attn", "indexer", "greedy_triton.py",
)

_spec = importlib.util.spec_from_file_location("tli_greedy_triton_kernel", _KERNEL_PATH)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

greedy_pass_triton = _m.greedy_pass_triton
greedy_build_triton = _m.greedy_build_triton
_greedy_pass_kernel = _m._greedy_pass_kernel

__all__ = ["greedy_pass_triton", "greedy_build_triton", "_greedy_pass_kernel"]
