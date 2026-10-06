# KVCache-Factory 方法（SnapKV/H2O/PyramidKV）的 Qwen3 适配层（E81 baseline 统一口径）
# 背景：KVCache-Factory 只 monkeypatch llama 系列且为 transformers 4.4x 旧签名
# （past_key_value 单数——4.56 调用方传复数会静默落 kwargs，正是 E71 踩过的坑）。
# 本适配层在 transformers 4.56.2 的 Qwen3Attention 骨架上插压缩钩子：
#   ① q/k_norm（Qwen3 特有 QK-RMSNorm）按原版位置应用；
#   ② cache 全程 kv-head 粒度（GQA 8 heads）——Cluster 的 _gqa_groups 自动走
#     kv_head 路径（gqa_score_agg=mean 聚合），attention 前标准 repeat；
#   ③ prefill 判据 = past_len==0 且 q_len>1（每样本重建 cluster，天然复位
#     PyramidKV 的 steps 计数，无需 patch prepare_inputs_for_generation）；
#   ④ 压缩后 attention 用 transformers 自带 ALL_ATTENTION_FUNCTIONS，
#     mask 4D 切片对齐压缩后 kv 长度（标准 KV 压缩近似，同 KVCache-Factory llama 版）。
# 预算口径：--max_capacity_prompt 1024 与 TLI/TIA/Quest 主表 token 预算对齐。
import sys

KVCF = "/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory"
if KVCF not in sys.path:
    sys.path.insert(0, KVCF)

import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from pyramidkv.pyramidkv_utils import (
    H2OKVCluster,
    PyramidKVCluster,
    SnapKVCluster,
    StreamingLLMKVCluster,
)

# 全局 rotary 注册表：replace_qwen3 时填入 model.model.rotary_emb，
# attention patch 在 decode 步按「原始（未压缩）长度」重算 cos/sin 用。
_ROTARY = {}


def _rope_cos_sin(position_ids, ref_tensor, config):
    """按给定位置计算 Qwen3 默认 RoPE 的 cos/sin（[b, s, head_dim]）。

    优先用模型真实 rotary_emb（数值与原版逐位一致）；未注册时用
    config.rope_theta 手动计算（Qwen3 默认 rope 无 attention_scaling，
    fp32 计算后 cast 到 x.dtype——与 Qwen3RotaryEmbedding.forward 同款）。
    """
    rot = _ROTARY.get("emb")
    if rot is not None:
        return rot(ref_tensor, position_ids)
    dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (
        torch.arange(0, dim, 2, dtype=torch.float32, device=ref_tensor.device) / dim))
    freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(ref_tensor.dtype), emb.sin().to(ref_tensor.dtype)


def _repeat_kv(hidden_states, n_rep):
    """标准 GQA repeat（transformers 同款）：[b, kv_h, s, d] -> [b, h, s, d]"""
    b, kv_h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None, :, :].expand(b, kv_h, n_rep, s, d).reshape(
        b, kv_h * n_rep, s, d
    )


def _init_cluster(self, method):
    """按 KVCache-Factory run_longbench.py 推荐配置构造 Cluster（每 prefill 重建）"""
    cfg = self.config
    cap = getattr(cfg, "max_capacity_prompt", 1024)
    ws = getattr(cfg, "window_size", 8)
    ks = getattr(cfg, "kernel_size", 7)
    pool = getattr(cfg, "pooling", "maxpool")
    merge = getattr(cfg, "merge", None)
    agg = getattr(cfg, "gqa_score_agg", "mean")
    if method == "snapkv":
        self.kv_cluster = SnapKVCluster(
            window_size=ws, max_capacity_prompt=cap, kernel_size=ks,
            pooling=pool, merge=merge, gqa_score_agg=agg)
    elif method == "h2o":
        self.kv_cluster = H2OKVCluster(
            window_size=ws, max_capacity_prompt=cap, kernel_size=ks,
            pooling=pool, merge=merge, gqa_score_agg=agg)
    elif method == "pyramidkv":
        self.kv_cluster = PyramidKVCluster(
            num_hidden_layers=cfg.num_hidden_layers, layer_idx=self.layer_idx,
            window_size=ws, max_capacity_prompt=cap, kernel_size=ks,
            pooling=pool, merge=merge, gqa_score_agg=agg)
    elif method == "streamingllm":
        self.kv_cluster = StreamingLLMKVCluster(
            window_size=ws, max_capacity_prompt=cap,
            gqa_score_agg=agg)
    else:
        raise ValueError(f"unknown kvcf method: {method}")


def make_qwen3_forward(method):
    def qwen3_attn_forward_kvcf(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3 QK-RMSNorm：view 到 head 形状后 norm 再 transpose（4.56 原版位置）
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        # 位置语义修正（KVCache-Factory llama 版 self.kv_seq_len/_seen_tokens
        # 机制的 4.56 等价物）：压缩只缩 cache 张量长度，不缩 RoPE 位置——
        # decode 步的 q/k 位置必须从「原始（未压缩）prefill 长度」继续，
        # 否则保留 key（RoPE 位置散布在 0..orig_len）与 decode q（位置 =
        # 压缩后 cache 长度）的相对距离全错，needle 检索崩。
        past_len = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        q_len = hidden_states.shape[1]
        is_prefill = past_len == 0 and q_len > 1
        if is_prefill:
            self.kv_orig_len = q_len
        elif past_key_values is not None and getattr(self, "kv_orig_len", None) is not None:
            true_pos = torch.arange(
                self.kv_orig_len, self.kv_orig_len + q_len,
                device=hidden_states.device).unsqueeze(0)
            cos, sin = _rope_cos_sin(true_pos, value_states, self.config)
            self.kv_orig_len += q_len

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # 关键：get_seq_length() 不带参数返回第 0 层长度——L0 压缩写入后
            # 其余层的 prefill 判据会失效（只有第 0 层被压缩的崩坏根源）。
            # 必须查本层长度。
            # 口径：query_head 粒度（KVCache-Factory reference 默认，
            # kv_cache_granularity="query_head"）——repeat 到 query-head 后
            # 走 Cluster 的 groups==1 原版路径（逐 query-head topk 选择，
            # 与 SnapKV 官方 GQA 行为一致；kv_head+mean 聚合是可选低显存
            # 口径，非 runner 默认）。
            nkg = self.num_key_value_groups
            if is_prefill:
                # prefill 首步：压缩（每样本重建 cluster = 参数复位）
                _init_cluster(self, method)
                kr = _repeat_kv(key_states, nkg)
                vr = _repeat_kv(value_states, nkg)
                kc, vc = self.kv_cluster.update_kv(
                    kr, query_states, vr, attention_mask, nkg)
                # M6 sink-guard：压缩后把 prefill 首 sink_n 个 token cat 回 cache
                # 前部（PSI 严格口径的 baseline 等价物——sink 不参与压缩竞争，
                # 永久保留）。总预算 = max_capacity + sink_n（sink 计外），与
                # PSI「sink/swa 强制保留不计入预算」对齐。
                sink_n = getattr(self.config, "kvcf_sink_guard", 0)
                if sink_n > 0 and sink_n < kc.shape[2]:
                    kc = torch.cat([kr[:, :, :sink_n, :], kc], dim=2)
                    vc = torch.cat([vr[:, :, :sink_n, :], vc], dim=2)
                past_key_values.update(kc, vc, self.layer_idx, cache_kwargs)
                key_states, value_states = kc, vc
            else:
                # decode：query-head 粒度追加，与 prefill 压缩后粒度一致
                kr = _repeat_kv(key_states, nkg)
                vr = _repeat_kv(value_states, nkg)
                key_states, value_states = past_key_values.update(
                    kr, vr, self.layer_idx, cache_kwargs)

        # GQA repeat 交给 attention_interface 内部处理（4.56 的 sdpa 用
        # enable_gqa/repeat_kv、eager 用 repeat_kv——外部再 repeat 会双重展开）

        q_len2, k_len2 = query_states.shape[-2], key_states.shape[-2]
        if q_len2 > 1 and k_len2 < q_len2:
            # 仅 prefill 压缩后（q>1 且 k<q）需要显式 mask：
            # ① 若为 None 放行，sdpa 推断 is_causal=True，flash 后端对非方阵
            #    causal 是右下对齐（最后 query 只看最后 1 个 key）——崩坏；
            # ② decode 步（q==1）绝不能进来：triu 对单行会把可见性裁成
            #    「只看 cache 第 0 个 token」，信息全丢（本 bug 已实测踩过）。
            # 显式构造左上对齐 causal（j≤i 截断 k）= KVCache-Factory llama 版
            # 4D mask 切片的同款压缩近似语义。
            if attention_mask is not None and attention_mask.dim() == 4:
                attention_mask = attention_mask[..., :k_len2]
            else:
                min_val = torch.finfo(query_states.dtype).min
                mask = torch.full((q_len2, k_len2), min_val,
                                  dtype=query_states.dtype, device=query_states.device)
                mask = torch.triu(mask, diagonal=1)  # 下三角 j≤i 可见
                attention_mask = mask[None, None, :, :]

        # 4.56 原版语义：eager 不在 ALL_ATTENTION_FUNCTIONS 里（特殊分支直取）
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        if self.config._attn_implementation == "eager":
            attention_interface = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        # k/v 已是 query-head 粒度（32 heads）——sdpa/eager 内部的
        # repeat_kv(module.num_key_value_groups) 会再 ×4 成 128 崩坏，
        # 用 shim（n_rep=1）替代 self 传入，跳过内部 repeat。
        import types
        shim = types.SimpleNamespace(
            num_key_value_groups=1, is_causal=True, training=self.training)
        attn_output, attn_weights = attention_interface(
            shim, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    return qwen3_attn_forward_kvcf


def replace_qwen3(model, method, max_capacity=1024, window_size=8,
                  kernel_size=7, pooling="maxpool", sink_guard=0):
    """patch 所有 Qwen3Attention 层并注入预算配置"""
    from transformers.models.qwen3 import modeling_qwen3 as m

    fwd = make_qwen3_forward(method)
    m.Qwen3Attention.forward = fwd
    # 注册模型真实 rotary_emb（decode 步位置重算用，数值与原版逐位一致）
    rotary = getattr(model, "model", model).rotary_emb
    _ROTARY["emb"] = rotary
    for layer in model.model.layers:
        c = layer.self_attn.config
        c.max_capacity_prompt = max_capacity
        c.window_size = window_size
        c.kernel_size = kernel_size
        c.pooling = pooling
        c.merge = None
        c.gqa_score_agg = "mean"
        c.kvcf_sink_guard = sink_guard
    print(f"[kvcf_qwen3] patched Qwen3Attention -> {method} "
          f"(budget={max_capacity}, window={window_size}, kernel={kernel_size}, "
          f"pooling={pooling}, sink_guard={sink_guard})")
