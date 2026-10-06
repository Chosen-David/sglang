import torch
import tilelang as tl
import tilelang.language as T
from typing import Optional
from einops import rearrange, repeat

@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_prefill_store_kv_kernel(
    max_batch: int,
    max_seqlen: int,
    num_kv_heads: int,
    dim_k: int,
    dim_v: int,
    block_S: int = 128,
    threads: int = 256,
    num_stages: int = 0
):
    batch = T.symbolic("batch")
    seqlen_kv = T.symbolic("seqlen_kv")

    k_shape = [batch, seqlen_kv, num_kv_heads, dim_k]
    v_shape = [batch, seqlen_kv, num_kv_heads, dim_v]
    offsets_shape = [batch]
    k_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_k]
    v_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_v]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        V: T.Tensor(v_shape, T.bfloat16),
        Offsets: T.Tensor(offsets_shape, T.int32),
        KCache: T.Tensor(k_cache_shape, T.bfloat16),
        VCache: T.Tensor(v_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            offset = Offsets[i_b]
            K_shared = T.alloc_shared([block_S, dim_k], dtype=T.bfloat16)
            V_shared = T.alloc_shared([block_S, dim_v], dtype=T.bfloat16)

            loop_range = T.ceildiv(seqlen_kv - offset, block_S)
            for i_s in T.Pipelined(loop_range, num_stages=num_stages):
                T.fill(K_shared, 0)
                T.fill(V_shared, 0)
                T.copy(K[i_b, offset + i_s * block_S: offset + i_s * block_S + block_S, i_h, :], K_shared)
                T.copy(V[i_b, offset + i_s * block_S: offset + i_s * block_S + block_S, i_h, :], V_shared)
                T.copy(K_shared, KCache[i_b, i_s * block_S:i_s * block_S + block_S, i_h, :])
                T.copy(V_shared, VCache[i_b, i_s * block_S:i_s * block_S + block_S, i_h, :])
    return main


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_decode_store_kv_kernel(
    max_batch: int,
    topk: int,
    block_size: int,
    num_kv_heads: int,
    dim_k: int,
    dim_v: int,
    threads: int = 256,
):
    batch = T.symbolic("batch")
    max_seqlen = topk * block_size

    k_shape = [batch, 1, num_kv_heads, dim_k]
    v_shape = [batch, 1, num_kv_heads, dim_v]
    offsets_shape = [batch]
    block_indices_shape = [batch, num_kv_heads, topk]
    k_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_k]
    v_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_v]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        V: T.Tensor(v_shape, T.bfloat16),
        Lengths: T.Tensor(offsets_shape, T.int32),
        BlockIndices: T.Tensor(block_indices_shape, T.int32),
        KCache: T.Tensor(k_cache_shape, T.bfloat16),
        VCache: T.Tensor(v_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            index = Lengths[i_b] - 1
            block_index = index // block_size
            K_shared = T.alloc_shared([dim_k], dtype=T.bfloat16)
            V_shared = T.alloc_shared([dim_v], dtype=T.bfloat16)

            for i_s in T.serial(topk):
                if BlockIndices[i_b, i_h, i_s] == block_index:
                    T.copy(K[i_b, 0, i_h, :], K_shared)
                    T.copy(V[i_b, 0, i_h, :], V_shared)
                    v = i_s * block_size + index % block_size
                    T.copy(K_shared, KCache[i_b, v, i_h, :])
                    T.copy(V_shared, VCache[i_b, v, i_h, :])

    return main

@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tl.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
})
def mha_update_block_indices_kernel(
    topk: int,
    max_blocks: int,
    num_kv_heads: int,
    threads: int = 256,
):
    batch = T.symbolic("batch")

    block_indices_shape = [batch, num_kv_heads, topk]
    load_shape = [batch, num_kv_heads]

    @T.prim_func
    def main(
        PreBlockIndices: T.Tensor(block_indices_shape, T.int32),
        CurBlockIndices: T.Tensor(block_indices_shape, T.int32),
        NextBlockIndices: T.Tensor(block_indices_shape, T.int32),
        SrcIndices: T.Tensor(block_indices_shape, T.int32),
        DstOffsets: T.Tensor(block_indices_shape, T.int32),
        LoadBlocks: T.Tensor(load_shape, T.int32),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            pre_indices_shared = T.alloc_shared([topk], T.int32)
            cur_indices_shared = T.alloc_shared([topk], T.int32)
            nxt_indices_shared = T.alloc_shared([topk], T.int32)
            mask = T.alloc_fragment([topk, topk], T.int32)
            cur_mask = T.alloc_fragment([topk], T.int32)
            pre_mask = T.alloc_fragment([topk], T.int32)
            load_blocks = T.alloc_fragment([1], T.int32)
            cur_mask_shared = T.alloc_shared([topk], T.int32)
            pre_mask_shared = T.alloc_shared([topk], T.int32)
            cur_offset_shared = T.alloc_shared([topk], T.int32)
            pre_offset_shared = T.alloc_shared([topk], T.int32)
            T.copy(PreBlockIndices[i_b, i_h, :], pre_indices_shared)
            T.copy(CurBlockIndices[i_b, i_h, :], cur_indices_shared)
            T.copy(pre_indices_shared, nxt_indices_shared)

            T.sync_threads() # BUG: without T.sync_threads() cause bug (currently)
            for i, j in T.Parallel(topk, topk):
                mask[i, j] = T.if_then_else(
                    (cur_indices_shared[i] == pre_indices_shared[j]) \
                        & (cur_indices_shared[i] != -1) \
                        & (pre_indices_shared[j] != -1),
                    1, 0
                )

            T.reduce_bitor(mask, cur_mask, dim=1, clear=True)
            T.reduce_bitor(mask, pre_mask, dim=0, clear=True)
            T.copy(cur_mask, cur_mask_shared)
            T.copy(pre_mask, pre_mask_shared)
            for i in T.Parallel(topk):
                cur_mask_shared[i] = (cur_mask_shared[i] ^ 1) & (cur_indices_shared[i] != -1)
                pre_mask_shared[i] = (pre_mask_shared[i] ^ 1) | (pre_indices_shared[i] == -1)

            T.copy(cur_mask_shared, cur_mask)
            T.reduce_sum(cur_mask, load_blocks, clear=True)

            T.cumsum(cur_mask_shared, cur_offset_shared)
            T.cumsum(pre_mask_shared, pre_offset_shared)
            for i in T.Parallel(topk):
                cur_offset_shared[i] -= 1
                pre_offset_shared[i] -= 1

            src_indices_shared = T.alloc_shared([topk], T.int32)
            dst_offsets_shared = T.alloc_shared([topk], T.int32)
            T.fill(src_indices_shared, -1)
            T.fill(dst_offsets_shared, -1)

            LoadBlocks[i_b, i_h] = load_blocks[0]
            for i in T.Parallel(topk):
                if cur_mask_shared[i] == 1:
                    src_indices_shared[cur_offset_shared[i]] = cur_indices_shared[i]
            for i in T.Parallel(topk):
                if pre_mask_shared[i] == 1:
                    dst_offsets_shared[pre_offset_shared[i]] = i
            for i in T.Parallel(topk):
                if dst_offsets_shared[i] != -1:
                    nxt_indices_shared[dst_offsets_shared[i]] = T.if_then_else(
                        i < load_blocks[0],
                        src_indices_shared[i], -1
                    )

            for i in T.Parallel(topk):
                src_indices_shared[i] = T.if_then_else(
                    src_indices_shared[i] != -1,
                    i_b * (num_kv_heads * max_blocks) + i_h * max_blocks + src_indices_shared[i],
                    -1
                )

            T.copy(src_indices_shared, SrcIndices[i_b, i_h, :])
            T.copy(dst_offsets_shared, DstOffsets[i_b, i_h, :])
            T.copy(nxt_indices_shared, NextBlockIndices[i_b, i_h, :])

    return main


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_update_kv_kernel(
    max_batch: int,
    topk: int,
    block_size: int,
    num_kv_heads: int,
    dim_k: int,
    dim_v: int,
    threads: int = 256,
):
    num_blocks = T.symbolic("num_blocks")
    batch = T.symbolic("batch")
    seqlen_kv = topk * block_size

    k_shape = [num_blocks, block_size, dim_k]
    v_shape = [num_blocks, block_size, dim_v]
    load_shape = [batch, num_kv_heads]
    offset_shape = [batch, num_kv_heads, topk]
    k_cache_shape = [max_batch, seqlen_kv, num_kv_heads, dim_k]
    v_cache_shape = [max_batch, seqlen_kv, num_kv_heads, dim_v]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        V: T.Tensor(v_shape, T.bfloat16),
        CumsumLoadBlocks: T.Tensor(load_shape, T.int32),
        LoadBlocks: T.Tensor(load_shape, T.int32),
        DstOffsets: T.Tensor(offset_shape, T.int32),
        KCache: T.Tensor(k_cache_shape, T.bfloat16),
        VCache: T.Tensor(v_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            K_shared = T.alloc_shared([block_size, dim_k], T.bfloat16)
            V_shared = T.alloc_shared([block_size, dim_v], T.bfloat16)
            start = CumsumLoadBlocks[i_b, i_h]
            loads = LoadBlocks[i_b, i_h]
            for i_s in T.Pipelined(loads):
                offset = DstOffsets[i_b, i_h, i_s]
                T.copy(K[start + i_s, :, :], K_shared)
                T.copy(V[start + i_s, :, :], V_shared)
                T.copy(K_shared, KCache[i_b, offset * block_size:(offset + 1) * block_size, i_h, :])
                T.copy(V_shared, VCache[i_b, offset * block_size:(offset + 1) * block_size, i_h, :])

    return main


class KVO_MHAStoreKVInterface:

    def __init__(self,
        max_batch: int,
        max_seqlen: int,
        num_kv_heads: int,
        dim_k: int,
        dim_v: int,
        topk: int,
        block_size: int,
    ) -> None:
        max_blocks = max_seqlen // block_size
        self.topk = topk
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.max_blocks = max_seqlen // block_size

        self.prefill_store_kv_kernel = mha_prefill_store_kv_kernel(
            max_batch,
            max_seqlen,
            num_kv_heads,
            dim_k,
            dim_v,
        )

        self.decode_store_kv_kernel = mha_decode_store_kv_kernel(
            max_batch,
            topk,
            block_size,
            num_kv_heads,
            dim_k,
            dim_v,
        )

        self.update_block_indices_kernel = mha_update_block_indices_kernel(
            topk,
            max_blocks,
            num_kv_heads,
        )

        self.update_kv_kernel = mha_update_kv_kernel(
            max_batch,
            topk,
            block_size,
            num_kv_heads,
            dim_k,
            dim_v
        )

    def prefill_forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        offsets: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        self.prefill_store_kv_kernel(k, v, offsets, k_cache, v_cache)

    def decode_forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lengths: torch.Tensor,
        block_indices: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        self.decode_store_kv_kernel(k, v, lengths, block_indices, k_cache, v_cache)

    def update_block_indices(
        self,
        pre_block_indices: torch.Tensor,
        cur_block_indices: torch.Tensor,
    ):
        device = cur_block_indices.device
        load_blocks = torch.zeros(cur_block_indices.shape[:2], dtype=torch.int32, device=device)
        next_block_indices = torch.zeros_like(cur_block_indices)
        src_indices = torch.zeros_like(cur_block_indices)
        dst_offsets = torch.zeros_like(cur_block_indices)
        self.update_block_indices_kernel(
            pre_block_indices,
            cur_block_indices,
            next_block_indices,
            src_indices,
            dst_offsets,
            load_blocks,
        )
        return next_block_indices, src_indices, dst_offsets, load_blocks

    def update_kv(
        self,
        keys,
        values,
        cumsum_load_blocks,
        load_blocks,
        dst_offsets,
        k_cache,
        v_cache,
    ):
        self.update_kv_kernel(
            keys,
            values,
            cumsum_load_blocks,
            load_blocks,
            dst_offsets,
            k_cache,
            v_cache,
        )

if __name__ == "__main__":
    import torch
    import numpy as np
    
    def test_prefill_forward():
        print("Testing prefill_forward...")
        
        # 设置参数
        max_batch = 4
        max_seqlen = 1024
        num_kv_heads = 8
        dim_k = 64
        dim_v = 64
        topk = 16
        block_size = 128
        
        # 创建接口实例
        interface = KVO_MHAStoreKVInterface(
            max_batch=max_batch,
            max_seqlen=max_seqlen,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            topk=topk,
            block_size=block_size
        )
        
        # 创建测试数据
        batch_size = 1
        seqlen_kv = 512
        
        k = torch.randn(batch_size, seqlen_kv, num_kv_heads, dim_k, dtype=torch.bfloat16).cuda()
        v = torch.randn(batch_size, seqlen_kv, num_kv_heads, dim_v, dtype=torch.bfloat16).cuda()
        offsets = torch.tensor([256], dtype=torch.int32).cuda()  # 不同batch有不同的offset
        k_cache = torch.zeros(max_batch, max_seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16).cuda()
        v_cache = torch.zeros(max_batch, max_seqlen, num_kv_heads, dim_v, dtype=torch.bfloat16).cuda()
        
        # 执行prefill
        interface.prefill_forward(k, v, offsets, k_cache, v_cache)
        
        # 验证结果
        print(f"k_cache shape: {k_cache.shape}")
        print(f"v_cache shape: {v_cache.shape}")
        
        # 检查第一个batch的写入
        batch_idx = 0
        offset = offsets[batch_idx].item()
        expected_k = k[batch_idx, offset:, :, :]
        actual_k = k_cache[batch_idx, :seqlen_kv-offset, :, :]
        
        # 使用allclose检查，考虑到浮点误差
        if torch.allclose(expected_k, actual_k, rtol=1e-3, atol=1e-3):
            print("✓ prefill_forward test passed for batch 0")
        else:
            print("✗ prefill_forward test failed for batch 0")
            print(f"Max diff: {(expected_k - actual_k).abs().max().item()}")
        
        print("Prefill forward test completed\n")
    
    def test_decode_forward():
        from einops import rearrange, repeat
        print("Testing decode_forward...")
        
        # 设置参数
        max_batch = 4
        max_seqlen = 1024
        num_kv_heads = 8
        dim_k = 64
        dim_v = 64
        block_size = 64
        topk = max_seqlen // block_size
        
        # 创建接口实例
        interface = KVO_MHAStoreKVInterface(
            max_batch=max_batch,
            max_seqlen=max_seqlen,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            topk=topk,
            block_size=block_size
        )
        
        # 创建测试数据
        batch_size = 1
        
        # decode阶段，每个token的k/v
        k = torch.randn(batch_size, 1, num_kv_heads, dim_k, dtype=torch.bfloat16).cuda()
        v = torch.randn(batch_size, 1, num_kv_heads, dim_v, dtype=torch.bfloat16).cuda()
        
        # 当前长度（假设已经有一些历史token）
        lengths = torch.tensor([64 + 10], dtype=torch.int32).cuda()
        
        # block indices (topk个block)
        block_indices = repeat(
            torch.arange(0, topk, dtype=torch.int32),
            'k -> b h k', 
            b=batch_size, h=num_kv_heads
        ).contiguous().cuda()
        
        # 初始化cache
        k_cache = torch.zeros(max_batch, max_seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16).cuda()
        v_cache = torch.zeros(max_batch, max_seqlen, num_kv_heads, dim_v, dtype=torch.bfloat16).cuda()
        
        # 执行decode
        interface.decode_forward(k, v, lengths, block_indices, k_cache, v_cache)
        
        print(f"k_cache shape: {k_cache.shape}")
        print(f"v_cache shape: {v_cache.shape}")
        print(f"lengths: {lengths}")
        print(f"block_indices shape: {block_indices.shape}")
        
        # 验证：检查是否写入了正确的位置
        for b in range(batch_size):
            length = lengths[b].item()
            block_index = (length - 1) // block_size
            pos_in_block = (length - 1) % block_size
            
            # 找到应该写入的位置
            for h in range(num_kv_heads):
                for s in range(topk):
                    if block_indices[b, h, s] == block_index:
                        cache_pos = s * block_size + pos_in_block
                        # 检查是否写入了cache
                        # if torch.any(k_cache[b, cache_pos, h, :] != 0):
                            # print(f"✓ Batch {b}, head {h}: wrote to position {cache_pos}")
                        if torch.allclose(k_cache[b, cache_pos, h, :], k[b, 0, h, :], rtol=1e-3, atol=1e-3):
                            print(f"✓ Batch {b}, head {h}: wrote to position {cache_pos}")
        
        print("Decode forward test completed\n")
    
    def test_update_block_indices():
        print("Testing update_block_indices...")
        
        # 设置参数
        max_batch = 4
        max_seqlen = 1024
        num_kv_heads = 1
        dim_k = 64
        dim_v = 64
        topk = 8
        block_size = 128
        
        # 创建接口实例
        interface = KVO_MHAStoreKVInterface(
            max_batch=max_batch,
            max_seqlen=max_seqlen,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            topk=topk,
            block_size=block_size
        )
        
        # 创建测试数据
        batch_size = 1
        
        # 前一个时间步的block indices
        pre_block_indices = torch.tensor(
            [1, 3, 5, 7, -1, -1, -1, -1] + [-1] * (topk - 8), 
            dtype=torch.int32
        ).cuda().view(1, 1, -1)
        
        # 当前时间步的block indices（模拟一些变化）
        cur_block_indices = torch.tensor(
            [7, 4, 6, 1, 5, -1, -1, -1] + [-1] * (topk - 8), 
            dtype=torch.int32
        ).cuda().view(1, 1, -1)
        
        print(f"pre_block_indices shape: {pre_block_indices.shape}")
        print(f"cur_block_indices shape: {cur_block_indices.shape}")
        
        # 执行更新
        next_block_indices, src_indices, dst_offsets, load_blocks = interface.update_block_indices(
            pre_block_indices, cur_block_indices
        )
        
        print(f"next_block_indices shape: {next_block_indices.shape}")
        print(f"src_indices shape: {src_indices.shape}")
        print(f"dst_offsets shape: {dst_offsets.shape}")
        print(f"load_blocks shape: {load_blocks.shape}")
        
        # 验证结果
        print(f"Sample next_block_indices[0, 0]: {next_block_indices[0, 0]}")
        print(f"Sample src_indices[0, 0]: {src_indices[0, 0]}")
        print(f"Sample dst_offsets[0, 0]: {dst_offsets[0, 0]}")
        print(f"load_blocks: {load_blocks}")
        
        # 检查load_blocks是否正确计算了需要加载的block数量
        expected_loads = torch.zeros(batch_size, num_kv_heads, dtype=torch.int32)
        for b in range(batch_size):
            for h in range(num_kv_heads):
                # 计算cur中不在pre中的block数量
                cur_set = set(cur_block_indices[b, h].cpu().numpy().tolist())
                pre_set = set(pre_block_indices[b, h].cpu().numpy().tolist())
                expected_loads[b, h] = len(cur_set - pre_set)
        
        print(f"Expected load_blocks: {expected_loads}")
        print(f"Actual load_blocks: {load_blocks.cpu()}")
        
        if torch.allclose(load_blocks.cpu().float(), expected_loads.float()):
            print("✓ update_block_indices test passed")
        else:
            print("✗ update_block_indices test failed")
        
        print("Update block indices test completed\n")
    
    def test_update_kv():
        print("Testing update_kv...")
        
        # 设置参数
        max_batch = 4
        max_seqlen = 1024
        num_kv_heads = 1
        dim_k = 64
        dim_v = 64
        topk = 8
        block_size = 128
        
        # 创建接口实例
        interface = KVO_MHAStoreKVInterface(
            max_batch=max_batch,
            max_seqlen=max_seqlen,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            topk=topk,
            block_size=block_size
        )

        # Sample next_block_indices[0, 0]: tensor([ 1,  3,  5,  7,  4,  6, -1, -1], device='cuda:0', dtype=torch.int32)
        # Sample src_indices[0, 0]: tensor([ 4,  6, -1, -1, -1, -1, -1, -1], device='cuda:0', dtype=torch.int32)
        # Sample dst_offsets[0, 0]: tensor([ 4,  5,  6,  7, -1, -1, -1, -1], device='cuda:0', dtype=torch.int32)
        
        # 创建测试数据
        batch_size = 1
        
        # 模拟需要更新的block数据
        num_blocks_to_load = 2  # 假设有3个block需要更新
        keys = torch.randn(num_blocks_to_load, block_size, dim_k, dtype=torch.bfloat16).cuda()
        values = torch.randn(num_blocks_to_load, block_size, dim_v, dtype=torch.bfloat16).cuda()
        
        # load_blocks: 每个batch/head需要加载的block数量
        load_blocks = torch.tensor([[2]], dtype=torch.int32).cuda()
        
        # cumsum_load_blocks: load_blocks的累积和
        cumsum_load_blocks = torch.zeros_like(load_blocks)
        total = 0
        for b in range(batch_size):
            for h in range(num_kv_heads):
                cumsum_load_blocks[b, h] = total
                total += load_blocks[b, h].item()
        
        # dst_offsets: 目标位置
        dst_offsets = torch.tensor(
            [1,  4,  5,  6,  7, -1, -1, -1], dtype=torch.int32
        ).cuda().view(1, 1, -1)
        
        # 初始化cache
        k_cache = torch.zeros(batch_size, topk * block_size, num_kv_heads, dim_k, dtype=torch.bfloat16).cuda()
        v_cache = torch.zeros(batch_size, topk * block_size, num_kv_heads, dim_v, dtype=torch.bfloat16).cuda()
        
        print(f"keys shape: {keys.shape}")
        print(f"values shape: {values.shape}")
        print(f"load_blocks: {load_blocks}")
        print(f"cumsum_load_blocks: {cumsum_load_blocks}")
        print(f"dst_offsets shape: {dst_offsets.shape}")
        
        # 执行更新
        interface.update_kv(
            keys, values, cumsum_load_blocks, load_blocks, dst_offsets, k_cache, v_cache
        )
        
        # 验证结果
        print(f"k_cache shape: {k_cache.shape}")
        print(f"v_cache shape: {v_cache.shape}")
        
        # 检查是否写入了正确的位置
        # batch0, head0: 应该写入keys[0]到offset 2
        expected_k_block0 = keys[0]
        actual_k_block0 = k_cache[0, 1*block_size:2*block_size, 0, :]

        expected_k_block1 = keys[1]
        actual_k_block1 = k_cache[0, 4*block_size:5*block_size, 0, :]
        
        if torch.allclose(expected_k_block0, actual_k_block0, rtol=1e-3, atol=1e-3):
            print("✓ Batch0, head0: kv updated correctly at offset 1")
        else:
            print("✗ Batch0, head0: kv update failed")

        if torch.allclose(expected_k_block1, actual_k_block1, rtol=1e-3, atol=1e-3):
            print("✓ Batch0, head0: kv updated correctly at offset 4")
        else:
            print("✗ Batch0, head0: kv update failed")


        print("Update kv test completed\n")
    
    def run_all_tests():
        print("=" * 60)
        print("Running KVO_MHAStoreKVInterface tests")
        print("=" * 60)
                
        # test_prefill_forward()
        # test_decode_forward()
        # test_update_block_indices()
        test_update_kv()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    
    # 运行所有测试
    run_all_tests()
