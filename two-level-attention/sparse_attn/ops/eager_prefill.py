# -*- coding: utf-8 -*-
"""稀疏 prefill（质量路径，TLI_SPARSE_PREFILL=1 门控）。

用户 2026-10-08 指令：e2e 的 prefill 和 decode 都应上 indexer，对齐
MoBA/NSA/DSA 论文的 chunked sparse prefill 口径。

chunk 共享选择语义（MoBA 口径，MoBA 论文 §3.2 TopK Gating）：
  prefill 的 L 行 query 按 CHUNK=2048 行切块；对每个 chunk 取末行 q 调
  indexer.prepare_mask 做一次选择（末行因果可见全部前缀 → 选择对块内
  所有行合法），chunk 内所有行共享该选择集，再与各自行因果下三角做
  交集（行 r 只可见 [0, r]）；sink 区 [0,128) 由 indexer 的 mask 强制
  保留（compute_mask 的 topk_mask[..., :sink_tok]=True）。

首 chunk 与短行 dense：chunk 起始位置 < K2（tia_level2_topk，默认 1024）
的行（含整个首 chunk）走 dense——等价 identity 选择，对齐 B02/S1 的
短行语义（短序列 dense 等价）。

swa 语义【10-09 TL-PREFILL-SWA-002 修复（GPT 审计）】：共享选择只强制
chunk 末行的 sink/SWA，块内早行自己的 128-token 局部窗口不保证在选集内
（例：chunk [2048,3000) 首行的自身 SWA [1921,2048] 与末行 SWA [2872,2999]
零重叠）→ 每行显式 OR 自身保护带（sink [0,128) + 自身窗口 [p-127,p]，
∩ 因果），保护 token 不占 mid top-k 预算（对齐 decode 侧 sink/swa 强制
区在 K2_mid 之外的契约）。

已知成本（E107a 诊断）：每 chunk 的 prepare_mask 走 prepare_index 全量
重建 O(S²/chunk)——质量路径可接受；sglang serving 路径已有 F1-F3 增量化。
"""
import os

import torch
import torch.nn.functional as F
from einops import rearrange, einsum

# MoBA 口径的 chunk 行数（gate 粒度）
SPARSE_PREFILL_CHUNK = 2048
# 行子块 fp32 score 峰值元素数预算（128M 元素 ≈ 512MB fp32）
_ROW_BLOCK_MAX_ELEM = 1 << 27


def _chunk_rows_attn(q_c, k_c, v_c, sel, pos_lo, G, softmax_scale,
                     sink_tok=0, swa_tok=0):
    """单个 chunk 的行级 attention（dense 与稀疏共用，dense 时 sel=None）。

    q_c: [1, len, H, D]；k_c/v_c: [1, T, Hkv, D]；sel: [Hkv, T]（kv-head
    共享）或 [H, T]（E103 per_q_head）的 token 级 bool 选择，None=dense。
    行 r 的绝对位置 = pos_lo + r，可见 keys [0, pos_lo + r] ∩ sel。

    TL-PREFILL-SWA-002：sink_tok/swa_tok > 0 时（稀疏 chunk），每行额外
    OR 自身保护带：sink 列 [0, min(sink_tok, p+1)) ∪ 自身窗口
    [max(0, p-swa_tok+1), p]（再 ∩ 因果）。dense（sel=None）保护带 ⊆ 因果
    恒为 no-op，路径不变。

    数值口径对齐 decode 侧 eager_decoding_attn：score 在 fp32 计算
    （q*scale → fp32、k → fp32），softmax fp32 后转回模型 dtype 与 v 相乘。
    """
    len_r, H, D = q_c.shape[1], q_c.shape[2], q_c.shape[3]
    T, Hkv = k_c.shape[1], k_c.shape[2]
    device = q_c.device

    # 选择集展开到 [Hkv, G, T] 形状（与 b_s 的 head 布局对齐）
    if sel is None:
        sel_e = None
    elif sel.shape[0] == Hkv:
        sel_e = sel[:, None, :]          # [Hkv, 1, T] 广播到组内 G 个 q-head
    else:
        # E103 per_q_head：[H, T] → [Hkv, G, T]（q-head 布局 (h g) 与 b_s 一致）
        sel_e = sel.reshape(Hkv, G, T)

    # 行子块：控制 fp32 score 张量 [R, Hkv, G, T] 峰值内存
    R = max(1, _ROW_BLOCK_MAX_ELEM // (H * max(T, 1)))
    o_c = torch.empty_like(q_c)
    kcol = torch.arange(T, device=device)
    for r0 in range(0, len_r, R):
        r1 = min(r0 + R, len_r)
        b_q = rearrange(
            q_c[0, r0:r1] * softmax_scale, "r (h g) d -> r h g d", g=G
        ).to(torch.float32)
        b_k = k_c[0].to(torch.float32)
        b_s = einsum(b_q, b_k, "r h g d, t h d -> r h g t")
        # 因果下三角：行 r（绝对位置 pos_lo+r0+i）只可见 t <= 该位置
        causal = (
            torch.arange(pos_lo + r0, pos_lo + r1, device=device)[:, None]
            >= kcol[None, :]
        )  # [r, T]
        if sel_e is None:
            m = causal[:, None, None, :]
        else:
            m = sel_e[None, :, :, :] & causal[:, None, None, :]
            # TL-PREFILL-SWA-002：行级保护带（自身 sink + 自身 swa 窗口），
            # ∩ 因果后 OR 进共享选择——块内早行的局部窗口不再依赖末行选集
            if sink_tok > 0 or swa_tok > 0:
                pos_r = torch.arange(pos_lo + r0, pos_lo + r1, device=device)
                prot = (kcol[None, :] < sink_tok) | (
                    kcol[None, :] >= (pos_r - swa_tok + 1)[:, None]
                )
                m = m | (prot & causal)[:, None, None, :]
        # 防御：选择 ∩ 因果全空的行兜底对角线（sink 强制 + pos>=2048 下
        # 不应触发；防御性保 softmax 无 NaN）
        if not bool(m.any(dim=-1).all()):
            diag = torch.arange(pos_lo + r0, pos_lo + r1, device=device)[
                :, None
            ] == kcol[None, :]
            m = m | diag[:, None, None, :]
        b_s = torch.where(m, b_s, float("-inf"))
        b_p = F.softmax(b_s, dim=-1).to(q_c.dtype)
        b_o = einsum(b_p, v_c[0], "r h g t, t h d -> r h g d")
        o_c[0, r0:r1] = rearrange(b_o, "r h g d -> r (h g) d").to(q_c.dtype)
    return o_c


def sparse_prefill_attn(
    indexer,
    q,
    k,
    v,
    softmax_scale=None,
    chunk_size=None,
    dense_below=None,
):
    """稀疏 chunked prefill attention 主入口（patch 层调用）。

    q: [1, L, H, D]；k/v: [1, S, Hkv, D]（S == L，B=1 无 padding）。
    返回 o: [1, L, H, D]。

    注意：不改 tli_indexer.py 的单行 query 断言（保护在跑语义）——本函数
    每 chunk 只喂末行单行 q（reshape 成 [1,1,H,D]）。
    """
    L, H, D = q.shape[1], q.shape[2], q.shape[3]
    S = k.shape[1]
    assert q.shape[0] == 1 and k.shape[0] == 1, "质量路径假设 B=1"
    assert L == S, "质量路径假设 prefill 时 q/k 等长（无 prefix 续写）"
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    if chunk_size is None:
        chunk_size = int(
            os.environ.get("TLI_SPARSE_PREFILL_CHUNK", SPARSE_PREFILL_CHUNK)
        )
    if dense_below is None:
        dense_below = getattr(indexer.args, "tia_level2_topk", 1024)

    G = H // k.shape[2]
    # 行级保护带尺寸（与 indexer mask 强制区同源；getattr 兜底对齐默认配置）
    swa_tok = int(getattr(indexer, "sliding_window_size", 128))
    _bs = int(getattr(indexer.args, "tia_block_size", 64))
    sink_tok = int(getattr(indexer, "sink_blocks", 2)) * _bs
    o = torch.empty_like(q)
    for c0 in range(0, L, chunk_size):
        c1 = min(c0 + chunk_size, L)
        # 因果可见前缀切片：中间 chunk 的末行不可见 [c1, L) 的未来 token，
        # 切片保证选择严格因果（末行视角 mask 的 swa 强制区同样是块末
        # 视角最近 128）
        k_c = k[:, :c1]
        v_c = v[:, :c1]
        if c0 < dense_below:
            # 首 chunk / 短行：dense（identity 选择，对齐 B02/S1 短行语义；
            # 保护带 ⊆ 因果恒 no-op，不传保持 dense 路径逐位不变）
            sel = None
            prot_args = dict(sink_tok=0, swa_tok=0)
        else:
            # 末行 q 单行调用（indexer 单行 query 设计，规避多行 assert）
            last_q = q[:, c1 - 1 : c1]  # [1, 1, H, D]
            cu = torch.tensor([0, c1], dtype=torch.int32, device=k.device)
            q_ids = cu[1:] - 1  # 末行位置（= prepare_seqlens(cu) - 1 口径）
            mask, _bs = indexer.prepare_mask(
                last_q, q_ids, k_c, cu, softmax_scale
            )
            m0 = mask[0, 0]        # [h, T_pad]（h=Hkv 共享 或 H per_q_head）
            sel = m0[..., :c1]     # 截到真实 token 数（pad 块尾巴去除）
            # 稀疏 chunk：行级保护带生效（TL-PREFILL-SWA-002）
            prot_args = dict(sink_tok=sink_tok, swa_tok=swa_tok)
        o[:, c0:c1] = _chunk_rows_attn(
            q[:, c0:c1], k_c, v_c, sel, c0, G, softmax_scale, **prot_args
        )
    return o
