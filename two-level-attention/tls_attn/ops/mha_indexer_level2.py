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
    level1_topk: int,
    level2_topk: int,
    block_TK: int,
    block_S: int,
    sliding_blocks: int,
    threads: int = 256,
):
    batch = T.symbolic("batch")
    max_batch = T.symbolic("max_batch")
    score_length = level1_topk * block_S
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
                    flip[i] = (order == desc) & (scores_shared[0, i] != scores_shared[0, j])
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
        Scores: T.Tensor([batch, num_kv_heads, score_length], T.float32),
        Indices: T.Tensor([batch, num_kv_heads, score_length], T.int32),
        TopkScores: T.Tensor([batch, num_kv_heads, level2_topk], T.float32),
        TopkIndices: T.Tensor([batch, num_kv_heads, level2_topk], T.int32),
        Lengths: T.Tensor([max_batch], T.int32),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by,):
            scores_shared = T.alloc_shared([2, block_TK2], T.float32)
            indices_shared = T.alloc_shared([2, block_TK2], T.int32)

            i_b = bx
            i_h = by
            seqlen = Lengths[i_b]

            T.fill(scores_shared, float('-inf'))
            T.fill(indices_shared, -1)
            T.sync_threads()

            start_block_idx = (seqlen - 1) // block_S - sliding_blocks + 1
            for i in T.Parallel(sliding_blocks * block_S):
                idx = start_block_idx * block_S + i
                if idx >= 0 and idx < seqlen:
                    scores_shared[0, i] = float('inf')
                    indices_shared[0, i] = idx
            T.sync_threads()

            loop_range = T.ceildiv(score_length, block_TK)
            for i_s in T.serial(loop_range): # loop_range
                T.fill(scores_shared[0, block_TK:], float('-inf'))
                T.fill(indices_shared[0, block_TK:], -1)
                T.sync_threads()

                T.copy(Scores[i_b, i_h, i_s * block_TK:(i_s + 1) * block_TK], scores_shared[0, block_TK:])
                T.copy(Indices[i_b, i_h, i_s * block_TK:(i_s + 1) * block_TK], indices_shared[0, block_TK:])
                T.sync_threads()
                
                bitonic_sort(scores_shared, indices_shared, False, True)
                T.sync_threads()

            T.copy(scores_shared[0, :level2_topk], TopkScores[i_b, i_h, :level2_topk])
            T.copy(indices_shared[0, :level2_topk], TopkIndices[i_b, i_h, :level2_topk])

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
    score_length = topk * block_S
    
    q_shape = [batch, 1, num_heads, dim_k]
    k_shape = [max_batch, seqlen_kv, num_kv_heads, dim_k]
    block_indices_shape = [batch, num_kv_heads, topk]
    scores_shape = [batch, num_kv_heads, score_length]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),
        K: T.Tensor(k_shape, T.bfloat16),
        BlockIndices: T.Tensor(block_indices_shape, T.int32),
        Scores: T.Tensor(scores_shape, T.float32),
        CummaxScores: T.Tensor(scores_shape, T.float32),
        Indices: T.Tensor(scores_shape, T.int32),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_G, dim_k], T.bfloat16)
            K_shared = T.alloc_shared([block_S, dim_k], T.bfloat16)
            cummax_shared = T.alloc_shared([block_S], T.float32)
            indices_shared = T.alloc_shared([block_S], T.int32)

            acc_s = T.alloc_fragment([block_G, block_S], T.float32)
            column_max_i = T.alloc_fragment([block_S], T.float32)
            row_max = T.alloc_fragment([1], T.float32)
            sum_scores_i = T.alloc_fragment([block_S], T.float32)

            i_b = bx
            i_h = by

            T.fill(row_max, float('-inf'))
            T.fill(Q_shared, 0)
            T.copy(Q[i_b, 0, i_h * group_size:(i_h + 1) * group_size, :], Q_shared[:group_size, :])
            for i, j in T.Parallel(block_G, dim_k):
                Q_shared[i, j] = Q_shared[i, j] * scaling

            for i_i in T.Pipelined(topk, num_stages=num_stages):
                idx = BlockIndices[i_b, i_h, i_i]
                if idx >= 0:
                    T.fill(K_shared, 0)
                    T.copy(K[i_b, idx * block_S:(idx + 1) * block_S, i_h, :], K_shared)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, clear_accum=True)
                    T.reduce_max(acc_s, column_max_i, dim=0, clear=True)
                    T.reduce_max(column_max_i, row_max, dim=-1, clear=False)
                    for i, j in T.Parallel(block_G, block_S):
                        if i < group_size:
                            acc_s[i, j] = T.exp(acc_s[i, j] - row_max[0])
                        else:
                            acc_s[i, j] = 0
                    T.reduce_sum(acc_s, sum_scores_i, dim=0, clear=True)
                    
                    T.copy(sum_scores_i, Scores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])

                    for i in T.Parallel(block_S):
                        cummax_shared[i] = row_max[0]
                    T.copy(cummax_shared, CummaxScores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])

                    for i in T.Parallel(block_S):
                        indices_shared[i] = idx * block_S + i
                    T.copy(indices_shared, Indices[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])
                else:
                    for i in T.Parallel(block_S):
                        sum_scores_i[i] = float('-inf')
                        cummax_shared[i] = row_max[0]
                        indices_shared[i] = -1
                    T.copy(sum_scores_i, Scores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])
                    T.copy(cummax_shared, CummaxScores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])
                    T.copy(indices_shared, Indices[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])
            
            for i_i in T.Pipelined(topk, num_stages=num_stages):
                idx = BlockIndices[i_b, i_h, i_i]
                if idx >= 0:
                    T.copy(Scores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S], sum_scores_i)
                    T.copy(CummaxScores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S], cummax_shared)
                    for i in T.Parallel(block_S):
                        sum_scores_i[i] = sum_scores_i[i] * T.exp(cummax_shared[i] - row_max[0])
                    T.copy(sum_scores_i, Scores[i_b, i_h, i_i * block_S:(i_i + 1) * block_S])

    return main


class MHAIndexerLevel2Interface:
    
    def __init__(self,
        num_heads: int,
        num_kv_heads: int,
        dim_k: int,
        level1_topk: int,
        level2_topk: int,
        block_size: int,
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
        self.block_S = block_size
        self.scaling = scaling
        self.sliding_blocks = sliding_blocks
        self.level1_topk = level1_topk
        self.level2_topk = level2_topk
        self.block_TK = max(128, tl.next_power_of_2(level2_topk))

        self.fwd_kernel = mha_indexer_fwd_kernel(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            group_size=self.group_size,
            dim_k=self.dim_k,
            topk=self.level1_topk,
            block_G=self.block_G,
            block_S=self.block_S,
            scaling=self.scaling,
        )
        self.topk_kernel = mha_indexer_topk_kernel(
            num_kv_heads=self.num_kv_heads,
            level1_topk=self.level1_topk,
            level2_topk=self.level2_topk,
            block_TK=self.block_TK,
            block_S=self.block_S,
            sliding_blocks=self.sliding_blocks,
        )

    def forward(self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_indices: torch.Tensor,
        lengths: torch.Tensor,
    ):
        batch = q.shape[0]
        seqlen = self.level1_topk * self.block_S
        scores = torch.full((batch, self.num_kv_heads, seqlen), float('-inf'), dtype=torch.float32, device=q.device)
        indices = torch.full((batch, self.num_kv_heads, seqlen), -1, dtype=torch.int32, device=q.device)
        cummax_scores = torch.full((batch, self.num_kv_heads, seqlen), float('-inf'), dtype=torch.float32, device=q.device)
        topk_scores = torch.zeros((batch, self.num_kv_heads, self.level2_topk), dtype=torch.float32, device=q.device)
        topk_indices = torch.zeros((batch, self.num_kv_heads, self.level2_topk), dtype=torch.int32, device=q.device)

        self.fwd_kernel(q, k, block_indices, scores, cummax_scores, indices)
        self.topk_kernel(scores, indices, topk_scores, topk_indices, lengths)
        return scores, indices, topk_scores, topk_indices

    def forward_with_buffer(self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_indices: torch.Tensor,
        lengths: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        cummax_scores: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        self.fwd_kernel(q, k, block_indices, scores, cummax_scores, indices)
        self.topk_kernel(scores, indices, topk_scores, topk_indices, lengths)

@torch.no_grad()
def ref_indexer_topk(
    q: torch.Tensor,             # [B, H, D]
    k: torch.Tensor,             # [B, S, H, D]
    block_indices: torch.Tensor, # [B, H, K] block index 或 -1
    lengths: torch.Tensor,
    block_size: int,
    sliding_blocks: int,
    topk: int,
    scaling: float = None
):
    from einops import einsum, rearrange, repeat
    B, S, H, D = k.shape
    HQ = q.shape[-2]
    G = HQ // H
    if scaling is None:
        scaling = D ** -0.5
    assert block_indices.shape[:2] == (B, H)
    K_topk = block_indices.shape[-1]
    assert S % block_size == 0
    num_blocks = S // block_size

    device = q.device
    dtype_q = q.dtype

    # 初始化输出
    scores_h = torch.full(
        (B, HQ, K_topk * block_size), 
        fill_value=float('-inf'),
        device=device, 
        dtype=dtype_q
    )
    scores = torch.full(
        (B, H, K_topk * block_size + sliding_blocks * block_size), 
        fill_value=float('-inf'),
        device=device, 
        dtype=dtype_q
    )
    indices = torch.full(
        (B, H, K_topk * block_size + sliding_blocks * block_size),
        fill_value=-1,
        device=device,
        dtype=torch.long
    )

    # 遍历 batch 和 head
    for b in range(B):
        seqlen = lengths[b].item()
        for h in range(H):
            q_vec = q[b, h*G:h*G+G, :] * scaling  # [G, D]
            for k_idx in range(K_topk):
                block_id = block_indices[b, h, k_idx].item()

                # 若 block_id == -1，则这一整块输出保持默认值
                if block_id == -1:
                    continue

                # 正常 block 索引检查
                if not (0 <= block_id < num_blocks):
                    raise ValueError(
                        f"Invalid block index {block_id} at (b={b}, h={h}, k={k_idx}), "
                        f"should be in [0, {num_blocks-1}] or -1."
                    )

                # 在原 seqlen 维度上的 token 起止位置
                seq_start = block_id * block_size
                seq_end = seq_start + block_size

                # 取出该 block 的 k 向量：[block_size, D]
                k_block = k[b, seq_start:seq_end, h, :]  # [block_size, D]

                # 计算 q 与该 block 所有 token 的内积
                # q_vec: [D], k_block: [block_size, D]
                # -> [block_size]
                scores_i = einsum(q_vec, k_block, 'g d, s d -> g s')

                # 写入 output 对应位置
                out_start = k_idx * block_size
                out_end = out_start + block_size
                scores_h[b, h*G:h*G+G, out_start:out_end].copy_(scores_i)
        
                # 写入对应的 token 原始下标
                token_indices = torch.arange(
                    seq_start, seq_end, device=device, dtype=torch.long
                )  # [block_size]
                indices[b, h, out_start:out_end] = token_indices
        
        b_scores_h = rearrange(scores_h[b], '(h g) s -> h g s', g=G)
        max_scores = b_scores_h.amax(dim=-1).amax(dim=-1)
        b_scores_h = (b_scores_h - max_scores[:, None, None]).exp().sum(dim=-2)
        scores[b, :, :b_scores_h.shape[-1]] = b_scores_h

        start_block_idx = (seqlen - 1) // block_size - sliding_blocks + 1
        for idx in range(sliding_blocks):
            out_start = (K_topk + idx) * block_size
            out_end = out_start + block_size
            seq_start = (start_block_idx + idx) * block_size
            seq_end = seq_start + block_size
            token_indices = torch.arange(
                seq_start, seq_end, device=device, dtype=torch.long
            )  # [block_size]
            scores_i = torch.full([block_size], fill_value=float('inf')).type_as(scores)
            scores_i = torch.where(token_indices < seqlen, scores_i, float('-inf'))
            token_indices = torch.where(token_indices < seqlen, token_indices, -1)
            scores[b, h, out_start:out_end] = scores_i
            indices[b, h, out_start:out_end] = token_indices

    topk_scores, topk_scores_indices = torch.topk(scores, k=topk, dim=-1)
    topk_indices = torch.gather(indices, dim=-1, index=topk_scores_indices)

    return scores, indices, topk_scores, topk_indices


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
    
    batch = 2
    num_heads = 4
    num_kv_heads = 1
    dim_k = 128
    level1_topk = 2
    level2_topk = 64 + 16
    block_size = 64
    sliding_blocks = 1
    max_seqlen = (4) * block_size
    seqlen = (4 - 1) * block_size + 4

    q = torch.randn(batch, num_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, max_seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
    block_indices = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32, device="cuda").view(batch, num_kv_heads, -1)
    lengths = torch.tensor([seqlen] * batch, dtype=torch.int32, device="cuda")

    # ref_score, ref_indices
    ref_scores, ref_indices, ref_topk_scores, ref_topk_indices \
        = ref_indexer_topk(q, k, block_indices, lengths, block_size, sliding_blocks, level2_topk)

    print("ref_scores:", ref_scores)
    print("ref_indices:", ref_indices)
    print("ref_topk_scores:", ref_topk_scores)
    print("ref_topk_indices:", ref_topk_indices)

    indexer_wrapper = MHAIndexerLevel2Interface(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        level1_topk=level1_topk,
        level2_topk=level2_topk,
        block_size=block_size,
        sliding_blocks=sliding_blocks,
    )
    scores, indices, topk_scores, topk_indices = indexer_wrapper.forward(q, k, block_indices, lengths)

    print("scores:", scores)
    print("indices:", indices)
    print("topk_scores:", topk_scores)
    print("topk_indices:", topk_indices)


if __name__ == '__main__':
    test()
