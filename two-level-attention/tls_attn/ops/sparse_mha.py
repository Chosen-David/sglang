import torch
import tilelang as tl
import tilelang.language as T
from typing import Optional


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_fwd_kernel(
    num_heads: int, 
    num_kv_heads: int,
    group_size: int,
    dim_k: int,
    dim_v: int,
    topk: int,
    block_G: int,
    block_S: int,
    scaling: float,
    threads: int = 256,
    num_stages: int = 0,
):
    batch = T.symbolic("batch")
    max_batch = T.symbolic("max_batch")
    seqlen_kv = T.symbolic("seqlen_kv")

    block_H = group_size
    
    q_shape = [batch, 1, num_heads, dim_k]
    k_shape = [max_batch, seqlen_kv, num_kv_heads, dim_k]
    v_shape = [max_batch, seqlen_kv, num_kv_heads, dim_v]
    indices_shape = [batch, num_kv_heads, topk]
    o_shape = [batch, 1, num_heads, dim_v]
    # lse_shape = [batch, num_heads]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),
        K: T.Tensor(k_shape, T.bfloat16),
        V: T.Tensor(v_shape, T.bfloat16),
        Indices: T.Tensor(indices_shape, T.int32),
        O: T.Tensor(o_shape, T.bfloat16),
        # Lse: T.Tensor(lse_shape, T.float32),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            loop_range = T.ceildiv(topk, block_S)

            Q_shared = T.alloc_shared([block_G, dim_k], T.bfloat16)
            K_shared = T.alloc_shared([block_S, dim_k], T.bfloat16)
            V_shared = T.alloc_shared([block_S, dim_v], T.bfloat16)

            mask = T.alloc_shared([block_S], T.bool)

            acc_s = T.alloc_fragment([block_G, block_S], T.float32)
            acc_s_cast = T.alloc_shared([block_G, block_S], T.bfloat16)
            acc_o = T.alloc_fragment([block_G, dim_v], T.float32)

            max_scores = T.alloc_fragment([block_G], T.float32)
            max_scores_prev = T.alloc_fragment([block_G], T.float32)
            scale_i = T.alloc_fragment([block_G], T.float32)
            sum_scores_i = T.alloc_fragment([block_G], T.float32)
            
            lse = T.alloc_fragment([block_G], T.float32)

            T.fill(max_scores, float('-inf'))
            T.fill(lse, 0)
            T.fill(acc_o, 0)

            T.copy(Q[i_b, 0, i_h * block_H:(i_h + 1) * block_H, :], Q_shared[:block_H, :])
            for i, j in T.Parallel(block_G, dim_k):
                Q_shared[i, j] = Q_shared[i, j] * scaling

            for i_s in T.Pipelined(loop_range, num_stages=num_stages):
                T.fill(K_shared, 0)
                T.fill(V_shared, 0)
                for i in T.Parallel(block_S):
                    mask[i] = T.if_then_else(
                        i_s * block_S + i < topk, 
                        Indices[i_b, i_h, i_s * block_S + i] != -1, False
                    )
                for i, j in T.Parallel(block_S, dim_k):
                    K_shared[i, j] = T.if_then_else( 
                        mask[i],
                        K[i_b, Indices[i_b, i_h, i_s * block_S + i], i_h, j], 0
                    )
                for i, j in T.Parallel(block_S, dim_v):
                    V_shared[i, j] = T.if_then_else(
                        mask[i],
                        V[i_b, Indices[i_b, i_h, i_s * block_S + i], i_h, j], 0
                    )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_G, block_S):
                    acc_s[i, j] = T.if_then_else(mask[j], acc_s[i, j], float('-inf'))

                T.copy(max_scores, max_scores_prev)
                T.reduce_max(acc_s, max_scores, dim=-1, clear=False)
                for i in T.Parallel(block_G):
                    scale_i[i] = T.exp(max_scores_prev[i] - max_scores[i])
                for i, j in T.Parallel(block_G, block_S):
                    acc_s[i, j] = T.exp(acc_s[i, j] - max_scores[i])
                T.reduce_sum(acc_s, sum_scores_i, dim=-1)
                for i in T.Parallel(block_G):
                    lse[i] = lse[i] * scale_i[i] + sum_scores_i[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_G, dim_v):
                    acc_o[i, j] *= scale_i[i]
                T.gemm(acc_s_cast, V_shared, acc_o, clear_accum=False)
            
            for i, j in T.Parallel(block_G, dim_v):
                acc_o[i, j] /= lse[i]
            # for i in T.Parallel(block_G):
            #     lse[i] = T.log(lse[i]) + max_scores[i]

            # T.copy(acc_o[:block_H, :], O[i_b, i_h*block_H:i_h*block_H+block_H, :]) # BUG
            for i, j in T.Parallel(block_G, dim_v):
                if i < block_H:
                    O[i_b, 0, i_h*block_H + i, j] = acc_o[i, j]

            # T.copy(lse[:block_H], Lse[i_b, i_h*block_H:i_h*block_H+block_H])

    return main


class SparseMHAInterface:
    
    def __init__(self,
        num_heads: int,
        num_kv_heads: int,
        dim_k: int,
        dim_v: int,
        topk: int,
        scaling: Optional[float] = None, 
    ) -> None:
        if scaling is None:
            scaling = dim_k ** -0.5
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.topk = topk
        self.block_G = max(16, self.group_size)
        self.block_S = 128
        self.scaling = scaling

        self.fwd_kernel = mha_fwd_kernel(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            group_size=self.group_size,
            dim_k=self.dim_k,
            dim_v=self.dim_v,
            topk=self.topk,
            block_G=self.block_G,
            block_S=self.block_S,
            scaling=self.scaling,
        )

    def forward(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        batch = q.shape[0]
        o = torch.empty((batch, 1, self.num_heads, self.dim_v), dtype=torch.bfloat16, device=q.device)
        # lse = torch.empty((batch, self.num_heads), dtype=torch.float32, device=q.device)

        self.fwd_kernel(q, k, v, token_indices, o)
        return o
    
    def forward_with_buffer(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_indices: torch.Tensor,
        o: torch.Tensor,
    ):
        self.fwd_kernel(q, k, v, token_indices, o)


def get_abs_err(y: torch.Tensor, x: torch.Tensor):
    x = x.to(torch.float32).nan_to_num(posinf=0)
    y = y.to(torch.float32).nan_to_num(posinf=0)
    return (x-y).flatten().abs().max().item()


def get_err_ratio(y, x):
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def ref_attn(
    q: torch.Tensor,             # [B, H, D]
    k: torch.Tensor,             # [B, S, H, D]
    v: torch.Tensor,
    token_indices: torch.Tensor,
    scaling: float = None
):
    from einops import einsum, rearrange, repeat
    B, S, H, D, Dv = *k.shape, v.shape[-1]
    HQ = q.shape[-2]
    G = HQ // H
    if scaling is None:
        scaling = D ** -0.5

    device = q.device
    dtype_q = q.dtype

    o = torch.zeros(
        B, HQ, Dv, 
        device=device, 
        dtype=dtype_q
    )

    # 遍历 batch 和 head
    for b in range(B):
        for h in range(H):
            indices = token_indices[b, h]
            indices = indices[indices >= 0]
            b_q = q[b, h*G:h*G+G, :] * scaling  # [G, D]
            b_k = k[b, indices, h, :]
            b_v = v[b, indices, h, :]
            b_s = einsum(b_q, b_k, 'g d, s d -> g s')
            b_p = torch.softmax(b_s, dim=-1)
            b_o = einsum(b_p, b_v, 'g s, s dv -> g dv')
            o[b, h*G:h*G+G, :] = b_o

    return o


@torch.no_grad()
def test():
    torch.manual_seed(42)
    
    batch = 4
    num_heads = 4
    num_kv_heads = 1
    dim_k = 128
    dim_v = 128
    max_seqlen = 512
    topk = 32

    q = torch.randn(batch, num_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, max_seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, max_seqlen, num_kv_heads, dim_v, dtype=torch.bfloat16, device="cuda")
    token_indices = torch.randn(batch, num_kv_heads, max_seqlen, dtype=torch.bfloat16, device="cuda").sort(dim=-1).indices[..., :topk]
    token_indices = token_indices.to(torch.int32)

    # ref_score, ref_indices
    ref_o = ref_attn(q, k, v, token_indices)
    print("ref_o:", ref_o[..., :8])

    indexer_wrapper = SparseMHAInterface(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        dim_v=dim_v,
        topk=topk,
    )
    # score, indices
    o = indexer_wrapper.forward(q.unsqueeze(1), k, v, token_indices).squeeze(0)

    print("o:", o[..., :8])
    print(f"abs: {get_abs_err(ref_o, o)}, ratio: {get_err_ratio(ref_o, o)}")

if __name__ == '__main__':
    test()
