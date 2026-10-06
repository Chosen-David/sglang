import torch
import tilelang as tl
import tilelang.language as T
from typing import Optional

@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_indexer_topk_kernel(
    num_kv_heads: int, 
    TopK: int,
    block_TK: int,
    sliding_blocks: int,
    threads: int = 256,
):
    batch = T.symbolic("batch")
    max_batch = T.symbolic("max_batch")
    max_length = T.symbolic('max_length')
    block_TK2 = block_TK * 2
    num_stages = block_TK2.bit_length() - 1 # log2

    @T.macro
    def bitonic_sort(
        scores_shared: T.SharedBuffer([2, block_TK2], T.float32),
        indices_shared: T.SharedBuffer([2, block_TK2], T.int32),
        ascending: T.bool,
        sort_by_value: T.bool,
    ):
        flip = T.alloc_shared([block_TK2], dtype=T.bool)
        T.sync_threads()
        for i_0 in T.serial(num_stages):
            for i_1 in T.serial(i_0 + 1):
                for i in T.Parallel(block_TK2):
                    j = i ^ (1 << (i_0 - i_1))
                    order = ((i & (1 << (i_0 + 1))) != 0) ^ ascending
                    if sort_by_value:
                        desc = T.if_then_else(
                            i < j, 
                            scores_shared[0, i] > scores_shared[0, j], 
                            scores_shared[0, j] > scores_shared[0, i]
                        )
                    else:
                        desc = T.if_then_else(
                            i < j, 
                            indices_shared[0, i] > indices_shared[0, j], 
                            indices_shared[0, j] > indices_shared[0, i]
                        )
                    flip[i] = order == desc
                T.sync_threads()
                for i in T.Parallel(block_TK2):
                    j = i ^ (1 << (i_0 - i_1))
                    if flip[i] != 0:
                        scores_shared[1, i] = scores_shared[0, j]
                        indices_shared[1, i] = indices_shared[0, j]
                    else:
                        scores_shared[1, i] = scores_shared[0, i]
                        indices_shared[1, i] = indices_shared[0, i]
                T.sync_threads()
                for i in T.Parallel(block_TK2):
                    scores_shared[0, i] = scores_shared[1, i]
                    indices_shared[0, i] = indices_shared[1, i]
                T.sync_threads()

    @T.prim_func
    def main(
        Scores: T.Tensor([batch, num_kv_heads, max_length], T.float32),
        TopkScores: T.Tensor([batch, num_kv_heads, TopK], T.float32),
        TopkIndices: T.Tensor([batch, num_kv_heads, TopK], T.int32),
        Lengths: T.Tensor([max_batch], T.int32),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by,):
            scores_shared = T.alloc_shared([2, block_TK2], T.float32)
            indices_shared = T.alloc_shared([2, block_TK2], T.int32)

            i_b = bx
            i_h = by
            seqlen = Lengths[i_b]
            loop_range = T.ceildiv(seqlen, block_TK)

            T.fill(scores_shared, float('-inf'))
            T.fill(indices_shared, -1)
            T.sync_threads()
            for i_s in T.serial(loop_range): # loop_range
                T.copy(Scores[i_b, i_h, i_s * block_TK:(i_s + 1) * block_TK], scores_shared[0, block_TK:])
                T.sync_threads()
                
                for i in T.Parallel(block_TK):
                    scores_shared[0, block_TK + i] = T.if_then_else(
                        i_s * block_TK + i < seqlen, 
                        scores_shared[0, block_TK + i], float('-inf')
                    )
                    indices_shared[0, block_TK + i] = T.if_then_else(
                        i_s * block_TK + i < seqlen,
                        i_s * block_TK + i, -1
                    )
                T.sync_threads()

                bitonic_sort(scores_shared, indices_shared, False, True)
                T.sync_threads()

            T.copy(scores_shared[0, :TopK], TopkScores[i_b, i_h, :TopK])
            T.copy(indices_shared[0, :TopK], TopkIndices[i_b, i_h, :TopK])

    return main


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_indexer_fwd_kernel(
    num_heads: int, 
    num_kv_heads: int,
    group_size: int,
    dim_k: int,
    block_G: int,
    block_S: int,
    scaling: float,
    sliding_blocks: int = 3,
    threads: int = 256,
    num_stages: int = 0,
):
    batch = T.symbolic("batch")
    max_batch = T.symbolic("max_batch")
    seqlen_kv = T.symbolic("seqlen_kv")
    max_length = T.symbolic("max_length")
    
    q_shape = [batch, 1, num_heads, dim_k]
    k_shape = [max_batch, seqlen_kv, num_kv_heads, dim_k]
    scores_shape = [batch, num_kv_heads, max_length]
    lengths_shape = [max_batch]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),
        K_min: T.Tensor(k_shape, T.bfloat16),
        K_max: T.Tensor(k_shape, T.bfloat16),
        Scores: T.Tensor(scores_shape, T.float32),
        Lengths: T.Tensor(lengths_shape, T.int32),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_G, dim_k], T.bfloat16)
            K_min_shared = T.alloc_shared([block_S, dim_k], T.bfloat16)
            K_max_shared = T.alloc_shared([block_S, dim_k], T.bfloat16)
            Q_pos_frag = T.alloc_fragment([block_G, dim_k], T.bfloat16)
            Q_neg_frag = T.alloc_fragment([block_G, dim_k], T.bfloat16)

            acc_s = T.alloc_fragment([block_G, block_S], T.float32)
            acc_o = T.alloc_fragment([block_S], T.float32)

            i_b = bx
            i_h = by
            seqlen = Lengths[i_b]
            loop_range = T.ceildiv(seqlen, block_S)

            T.fill(Q_shared, 0)

            T.copy(Q[i_b, 0, i_h * group_size:(i_h + 1) * group_size, :], Q_shared[:group_size, :])
            for i, j in T.Parallel(block_G, dim_k):
                Q_pos_frag[i, j] = T.if_then_else(Q_shared[i, j] >= 0, Q_shared[i, j], 0) * scaling
            for i, j in T.Parallel(block_G, dim_k):
                Q_neg_frag[i, j] = T.if_then_else(Q_shared[i, j] < 0, Q_shared[i, j], 0) * scaling

            for i_s in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K_max[i_b, i_s * block_S:(i_s + 1) * block_S, i_h, :], K_max_shared)
                T.gemm(Q_pos_frag, K_max_shared, acc_s, transpose_B=True, clear_accum=True) # [block_H, k_dim] * [block_T, k_dim].T => [block_H, block_T]
                T.copy(K_min[i_b, i_s * block_S:(i_s + 1) * block_S, i_h, :], K_min_shared)
                T.gemm(Q_neg_frag, K_min_shared, acc_s, transpose_B=True, clear_accum=False)
                T.reduce_sum(acc_s, acc_o, dim=0, clear=True)
                if (i_s == loop_range - 1) or (i_s == loop_range - 2):
                    for i in T.Parallel(block_S):
                        acc_o[i] = T.if_then_else(
                            i_s * block_S + i < seqlen - sliding_blocks, 
                            acc_o[i], 
                            float('inf')
                        )
                        acc_o[i] = T.if_then_else(
                            i_s * block_S + i < seqlen, 
                            acc_o[i], 
                            float('-inf')
                        )
                T.copy(acc_o, Scores[i_b, i_h, i_s * block_S:(i_s + 1) * block_S])

    return main


class KVO_MHAIndexerLevel1Interface:
    
    def __init__(self,
        num_heads: int,
        num_kv_heads: int,
        dim_k: int,
        topk: int,
        sliding_blocks: int,
        scaling: Optional[float] = None, 
    ) -> None:
        if scaling is None:
            scaling = dim_k ** -0.5
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.dim_k = dim_k
        self.block_G = max(16, self.group_size)
        self.block_S = 128
        self.scaling = scaling
        self.sliding_blocks = sliding_blocks
        self.topk = topk
        self.block_TK = tl.next_power_of_2(topk)

        self.fwd_kernel = mha_indexer_fwd_kernel(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            group_size=self.group_size,
            dim_k=self.dim_k,
            block_G=self.block_G,
            block_S=self.block_S,
            scaling=self.scaling,
            sliding_blocks=self.sliding_blocks
        )
        self.topk_kernel = mha_indexer_topk_kernel(
            num_kv_heads=self.num_kv_heads,
            TopK=self.topk,
            block_TK=self.block_TK,
            sliding_blocks=self.sliding_blocks,
        )

    def forward(self,
        q: torch.Tensor,
        k_min: torch.Tensor,
        k_max: torch.Tensor,
        lengths: torch.Tensor,
    ):
        batch = q.shape[0]
        max_length = max(lengths.amax().item(), 1)
        scores = torch.full((batch, self.num_kv_heads, max_length), float('-inf'), dtype=torch.float32, device=q.device)
        topk_scores = torch.zeros((batch, self.num_kv_heads, self.topk), dtype=torch.float32, device=q.device)
        topk_indices = torch.zeros((batch, self.num_kv_heads, self.topk), dtype=torch.int32, device=q.device)
        self.fwd_kernel(q, k_min, k_max, scores, lengths)
        self.topk_kernel(scores, topk_scores, topk_indices, lengths)
        return scores, topk_scores, topk_indices

    def forward_with_buffer(self,
        q: torch.Tensor,
        k_min: torch.Tensor,
        k_max: torch.Tensor,
        lengths: torch.Tensor,
        scores: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        self.fwd_kernel(q, k_min, k_max, scores, lengths)
        self.topk_kernel(scores, topk_scores, topk_indices, lengths)


def ref_indexer_topk(
    q: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    sliding_blocks: int,
):
    from einops import einsum, rearrange, repeat
    group_size = q.shape[-2] // k_max.shape[-2]
    scale = k_max.shape[-1] ** -0.5
    q_pos = torch.where(q >= 0, q * scale, 0)
    q_neg = torch.where(q <= 0, q * scale, 0)
    k_max = repeat(k_max, 'b s h d -> b s (h g) d', g=group_size)
    k_min = repeat(k_min, 'b s h d -> b s (h g) d', g=group_size)

    score = einsum(q_pos, k_max, 'b h d, b s h d -> b s h') + einsum(q_neg, k_min, 'b h d, b s h d -> b s h')
    score = rearrange(score, 'b s (h g) -> b s h g', g=group_size).sum(dim=-1).transpose(1, 2) # [b, h, s]
    for i in range(lengths.shape[0]):
        seqlen = lengths[i]
        score[..., max(0, seqlen - sliding_blocks):seqlen] = float('inf')
        score[..., seqlen:] = float('-inf')

    topk_score, topk_indices = torch.topk(score, k=topk, dim=-1)

    return score, topk_score, topk_indices


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

@torch.no_grad()
def test():
    torch.manual_seed(42)
    
    batch = 1
    seqlen = 32
    num_heads = 1
    num_kv_heads = 1
    dim_k = 128
    topk = 16
    sliding_blocks = 3
    q = torch.randn(batch, num_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    k_min = torch.randn(batch, seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    k_max = torch.randn(batch, seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([seqlen] * batch, dtype=torch.int32, device="cuda")

    # ref_score, ref_indices
    ref_o, ref_score, ref_indices = ref_indexer_topk(q, k_min, k_max, lengths, topk, sliding_blocks)

    print("ref_o:", ref_o)
    print("ref_score:", ref_score)
    print("ref_indices:", ref_indices)

    indexer_wrapper = KVO_MHAIndexerLevel1Interface(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        topk=topk,
        sliding_blocks=sliding_blocks,
    )
    # score, indices
    o, score, indices = indexer_wrapper.forward(q.unsqueeze(1), k_min, k_max, lengths)

    print("o:", o)
    print("score:", score)
    print("indices:", indices)

    # print(f"err: {get_abs_err(ref_o, o)}, ratio: {get_err_ratio(ref_o, o)}")
    # print(ref_score)
    # print(ref_indices)
    # print(score)
    # print(indices)

if __name__ == '__main__':
    test()
