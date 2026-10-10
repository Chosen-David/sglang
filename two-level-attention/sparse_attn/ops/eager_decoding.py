import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

def eager_decoding_attn(
    q: torch.Tensor, # [1, tq, hq, d]
    k: torch.Tensor, # [1, tk, h, d]
    v: torch.Tensor, # [1, tk, h, d]
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    if softmax_scale is None:
        softmax_scale = k.shape[-1] ** -0.5
    q, k, v, mask = (x.squeeze(0) for x in (q, k, v, mask))
    o = q.new_zeros(q.shape[:-1] + (v.shape[-1],))
    G = q.shape[-2] // k.shape[-2]
    bos_k, eos_k = cu_seqlens_k[0], cu_seqlens_k[1]
    # ---- E103：kv-head 共享消融消费端 ----
    # 共享口径：mask [Hkv, tk]（kv-head 级），repeat 后 [Hkv, tk*bs] 直接广播到组内
    # 全部 G 个 q-head（b_mask[:, None, :]）——「同组共享同一份选择」的落点；
    # per_q_head 口径：mask [H, tk]（q-head 级），按 G 拆回 [Hkv, G, tk*bs]
    # 逐元素对应——每个 q-head 用自己的选择，不做组内广播。
    m0 = mask[0]
    per_qh = m0.shape[0] != k.shape[-2]
    # B05 修复（GPT 审查 2026-10-08）：原版 [:, :eos_k-bos_k] 切的是 dim1
    # （per_qh 的 G 轴 / shared 的 Hkv 轴）而非最后一维 token 轴——无 padding
    # 时恰好是 no-op 故长期未暴露；投影路径 Tpad>T 时会广播失败或掩码不裁。
    # 改为裁 token 轴；无 padding（tk*bs == eos_k-bos_k）时与原版逐位等价。
    if per_qh:
        g_m = m0.shape[0] // k.shape[-2]
        b_mask = repeat(
            m0, '(h g) tk -> h g (tk bs)', g=g_m, bs=block_size
        )[..., :eos_k - bos_k]
    else:
        b_mask = repeat(mask[0], 'h tk -> h (tk bs)', bs=block_size)[..., :eos_k - bos_k]

    b_q = rearrange(q[0] * softmax_scale, '(h g) d -> h g d', g=G).to(torch.float32)
    b_k = k.to(torch.float32)
    b_v = v
    b_s = einsum(b_q, b_k, 'h g d, t h d -> h g t')
    if per_qh:
        # E103：b_mask 已是 [Hkv, G, T] 与 b_s 同形，逐元素对应（每 q-head 自己的选择）
        b_s = torch.where(b_mask, b_s, float('-inf'))
    else:
        b_s = torch.where(b_mask[:, None, :], b_s, float('-inf'))
    # 【B7 修复（kimi3 清单 F12，2026-10-08）】全 False 行防线（消费端）：
    # 任一 head 行全 False 时 softmax 对全 -inf 行产生 NaN 输出。indexer
    # 契约上由 sink/swa 正交强制保证不可达，此为防御性 fail-closed（契约
    # 破损时显式报错优于 NaN 静默扩散）。代价 = 每次调用一次 host 同步
    # （any().all() → bool）；本函数是 HF 参考路径，非 sglang 生产热路径。
    if not bool(b_mask.any(dim=-1).all()):
        raise RuntimeError(
            "eager_decoding_attn: 选择 mask 存在全 False 行——softmax 将"
            "产生 NaN。indexer 契约应保证 sink/swa 强制区使每行至少一个"
            "有效 token；请检查 indexer 的 mask 输出。"
        )
    b_p = F.softmax(b_s, dim=-1).to(q.dtype)
    b_o = rearrange(einsum(b_p, b_v, 'h g t, t h d -> h g d'), 'h g d -> (h g) d')
    o[0] = b_o
    return o.unsqueeze(0)
