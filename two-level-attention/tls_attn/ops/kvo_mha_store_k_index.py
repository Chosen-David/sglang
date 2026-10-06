import torch
import tilelang as tl
import tilelang.language as T
from typing import Optional

@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_prefill_store_k_index_kernel(
    max_batch: int,
    max_seqlen: int,
    num_kv_heads: int,
    dim_k: int,
    sfa_block_size: int,
    sfa_cmp_ratio: int,
    block_S: int = 128,
    threads: int = 256,
    num_stages: int = 0
):
    batch = T.symbolic("batch")
    seqlen_kv = T.symbolic("seqlen_kv")

    block_CMP_S = block_S // sfa_block_size
    level1_seqlen = max_seqlen // sfa_block_size
    level2_seqlen = max_seqlen
    cmp_dim_k = dim_k // sfa_cmp_ratio

    half_dim_k = dim_k // 2
    half_cmp_dim_k = half_dim_k // sfa_cmp_ratio

    k_shape = [batch, seqlen_kv, num_kv_heads, dim_k]
    offsets_shape = [batch]

    local_k_cache_shape = [max_batch, sfa_block_size, num_kv_heads, dim_k]
    level1_k_cache_shape = [max_batch, level1_seqlen, num_kv_heads, dim_k]
    level2_k_cache_shape = [max_batch, level2_seqlen, num_kv_heads, cmp_dim_k]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        Offsets: T.Tensor(offsets_shape, T.int32),
        LocalKCache: T.Tensor(local_k_cache_shape, T.bfloat16),
        Level1KMinCache: T.Tensor(level1_k_cache_shape, T.bfloat16),
        Level1KMaxCache: T.Tensor(level1_k_cache_shape, T.bfloat16),
        Level2KCache: T.Tensor(level2_k_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            offset = Offsets[i_b]
            K_shared = T.alloc_shared([block_S, dim_k], dtype=T.bfloat16)
            K_frag = T.alloc_fragment([block_CMP_S, sfa_block_size, dim_k], dtype=T.bfloat16)
            K_min = T.alloc_fragment([block_CMP_S, dim_k], dtype=T.bfloat16)
            K_max = T.alloc_fragment([block_CMP_S, dim_k], dtype=T.bfloat16)

            loop_range = T.ceildiv(seqlen_kv - offset, block_S)
            for i_s in T.Pipelined(loop_range, num_stages=num_stages):
                T.fill(K_shared, 0)
                T.copy(K[i_b, offset + i_s * block_S: offset + i_s * block_S + block_S, i_h, :], K_shared)
                for i, j in T.Parallel(block_S, dim_k):
                    K_frag[i // sfa_block_size, i % sfa_block_size, j] = K_shared[i, j]
                T.reduce_min(K_frag, K_min, dim=1, clear=True)
                T.reduce_max(K_frag, K_max, dim=1, clear=True)
                T.copy(K_min, Level1KMinCache[i_b, i_s * block_CMP_S:(i_s + 1) * block_CMP_S, i_h, :])
                T.copy(K_max, Level1KMaxCache[i_b, i_s * block_CMP_S:(i_s + 1) * block_CMP_S, i_h, :])
                for i, j in T.Parallel(block_S, half_cmp_dim_k):
                    Level2KCache[i_b, i_s * block_S + i, i_h, j] = K_shared[i, half_dim_k - half_cmp_dim_k + j]
                for i, j in T.Parallel(block_S, half_cmp_dim_k):
                    Level2KCache[i_b, i_s * block_S + i, i_h, half_cmp_dim_k + j] = K_shared[i, dim_k - half_cmp_dim_k + j]
            
            K_shared2 = T.alloc_shared([sfa_block_size, dim_k], dtype=T.bfloat16)
            for i, j in T.Parallel(sfa_block_size, dim_k):
                K_shared2[i, j] = T.if_then_else(
                    seqlen_kv - sfa_block_size + i >= 0,
                    K[i_b, seqlen_kv - sfa_block_size + i, i_h, j], 0
                )
            for i, j in T.Parallel(sfa_block_size, dim_k):
                v = (seqlen_kv - offset - sfa_block_size + i) % sfa_block_size
                if v >= 0:
                    LocalKCache[i_b, v, i_h, j] = K_shared2[i, j]

    return main


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def mha_decode_store_k_index_kernel(
    max_batch: int,
    max_seqlen: int,
    num_kv_heads: int,
    dim_k: int,
    sfa_block_size: int,
    sfa_cmp_ratio: int,
    threads: int = 256,
):
    batch = T.symbolic("batch")
    level1_seqlen = max_seqlen // sfa_block_size
    level2_seqlen = max_seqlen
    cmp_dim_k = dim_k // sfa_cmp_ratio

    half_dim_k = dim_k // 2
    half_cmp_dim_k = half_dim_k // sfa_cmp_ratio

    offsets_shape = [batch]
    k_shape = [batch, 1, num_kv_heads, dim_k]
    local_k_cache_shape = [max_batch, sfa_block_size, num_kv_heads, dim_k]
    level1_k_cache_shape = [max_batch, level1_seqlen, num_kv_heads, dim_k]
    level2_k_cache_shape = [max_batch, level2_seqlen, num_kv_heads, cmp_dim_k]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        Lengths: T.Tensor(offsets_shape, T.int32),
        LocalKCache: T.Tensor(local_k_cache_shape, T.bfloat16),
        Level1KMinCache: T.Tensor(level1_k_cache_shape, T.bfloat16),
        Level1KMaxCache: T.Tensor(level1_k_cache_shape, T.bfloat16),
        Level2KCache: T.Tensor(level2_k_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            index = Lengths[i_b] - 1
            i_s = (index // sfa_block_size)

            K_shared = T.alloc_shared([dim_k], T.bfloat16)
            Level1K_shared = T.alloc_shared([sfa_block_size, dim_k], dtype=T.bfloat16)
            K_frag = T.alloc_fragment([sfa_block_size, dim_k], dtype=T.bfloat16)
            K_min = T.alloc_fragment([dim_k], dtype=T.bfloat16)
            K_max = T.alloc_fragment([dim_k], dtype=T.bfloat16)

            T.copy(K[i_b, 0, i_h, :], K_shared)
            T.copy(K_shared, LocalKCache[i_b, index % sfa_block_size, i_h, :])

            if (index + 1) % sfa_block_size == 0:
                T.fill(Level1K_shared, 0)
                T.copy(LocalKCache[i_b, :, i_h, :], Level1K_shared)
                T.copy(Level1K_shared, K_frag)
                T.reduce_min(K_frag, K_min, dim=0, clear=True)
                T.reduce_max(K_frag, K_max, dim=0, clear=True)
                for j in T.Parallel(dim_k):
                    Level1KMinCache[i_b, i_s, i_h, j] = K_min[j]
                for j in T.Parallel(dim_k):
                    Level1KMaxCache[i_b, i_s, i_h, j] = K_max[j]

            for j in T.Parallel(half_cmp_dim_k):
                Level2KCache[i_b, index, i_h, j] = K_shared[half_dim_k - half_cmp_dim_k + j]
            for j in T.Parallel(half_cmp_dim_k):
                Level2KCache[i_b, index, i_h, half_cmp_dim_k + j] = K_shared[dim_k - half_cmp_dim_k + j]

    return main


class KVO_MHAStoreKIndexInterface:

    def __init__(self,
        max_batch: int,
        max_seqlen: int,
        num_kv_heads: int,
        dim_k: int,
        sfa_block_size: int,
        sfa_cmp_ratio: int,
    ) -> None:
        self.max_batch = max_batch
        self.max_seqlen = max_seqlen
        self.num_kv_heads = num_kv_heads
        self.dim_k = dim_k
        self.sfa_block_size = sfa_block_size
        self.sfa_cmp_ratio = sfa_cmp_ratio
        
        self.prefill_store_k_index_kernel = mha_prefill_store_k_index_kernel(
            max_batch,
            max_seqlen,
            num_kv_heads,
            dim_k,
            sfa_block_size,
            sfa_cmp_ratio,
        )

        self.decode_store_k_index_kernel = mha_decode_store_k_index_kernel(
            max_batch,
            max_seqlen,
            num_kv_heads,
            dim_k,
            sfa_block_size,
            sfa_cmp_ratio,
        )

    
    def prefill_forward(
        self,
        k: torch.Tensor,
        offsets: torch.Tensor,
        local_keys: torch.Tensor,
        level1_k_min_cache: torch.Tensor,
        level1_k_max_cache: torch.Tensor,
        level2_k_cache: torch.Tensor,
    ):
        self.prefill_store_k_index_kernel(k, offsets, local_keys, level1_k_min_cache, level1_k_max_cache, level2_k_cache)

    
    def decode_forward(
        self,
        key_states: torch.Tensor,  # Changed from k
        lengths: torch.Tensor,  # Changed from v
        local_keys: torch.Tensor,
        level1_k_min_cache: torch.Tensor,  # Changed from indices
        level1_k_max_cache: torch.Tensor,  # Changed from indices
        level2_k_cache: torch.Tensor,  # Changed from k_cache
    ):
        self.decode_store_k_index_kernel(key_states, lengths, local_keys, level1_k_min_cache, level1_k_max_cache, level2_k_cache)


# ============ PyTorch Reference 实现 ============

def pytorch_prefill_reference(
    k: torch.Tensor,
    offsets: torch.Tensor,
    local_k_cache: torch.Tensor,
    level1_k_min_cache: torch.Tensor,
    level1_k_max_cache: torch.Tensor,
    level2_k_cache: torch.Tensor,
    sfa_block_size: int = 32,
    sfa_cmp_ratio: int = 4
):
    """
    PyTorch实现的prefill reference
    """
    batch, seqlen_kv, num_kv_heads, dim_k = k.shape
    max_batch = local_k_cache.shape[0]
    
    half_dim_k = dim_k // 2
    half_cmp_dim_k = half_dim_k // sfa_cmp_ratio
    
    # 清空缓存
    local_k_cache.zero_()
    level1_k_min_cache.zero_()
    level1_k_max_cache.zero_()
    level2_k_cache.zero_()
    
    for b in range(batch):
        offset = offsets[b].item()
        
        # 处理每个block
        block_S = 128  # 与kernel保持一致
        block_CMP_S = block_S // sfa_block_size
        
        for i_s in range((seqlen_kv - offset + block_S - 1) // block_S):
            start_idx = offset + i_s * block_S
            end_idx = min(start_idx + block_S, seqlen_kv)
            
            if start_idx >= seqlen_kv:
                break
                
            # 获取当前block的K数据
            k_block = k[b, start_idx:end_idx, :, :]  # [block_len, num_kv_heads, dim_k]
            block_len = k_block.shape[0]
            
            # 填充到sfa_block_size的倍数
            if block_len < block_S:
                padding = block_S - block_len
                k_block = torch.cat([k_block, torch.zeros(padding, num_kv_heads, dim_k, 
                                                         dtype=k.dtype, device=k.device)], dim=0)
            
            for h in range(num_kv_heads):
                # 计算每个sfa_block的最小值和最大值
                for cmp_idx in range(block_CMP_S):
                    block_start = cmp_idx * sfa_block_size
                    block_end = block_start + sfa_block_size
                    
                    k_frag = k_block[block_start:block_end, h, :]  # [sfa_block_size, dim_k]
                    
                    if block_len >= block_start + sfa_block_size:
                        # 计算最小值和最大值
                        k_min = torch.min(k_frag, dim=0)[0]  # [dim_k]
                        k_max = torch.max(k_frag, dim=0)[0]  # [dim_k]
                        
                        # 存储到level1缓存
                        level1_k_min_cache[b, i_s * block_CMP_S + cmp_idx, h, :] = k_min
                        level1_k_max_cache[b, i_s * block_CMP_S + cmp_idx, h, :] = k_max
                
                # 存储压缩后的K到level2缓存
                for i in range(block_S):
                    global_idx = i_s * block_S + i
                    if global_idx < seqlen_kv:
                        # 存储中间部分（压缩）
                        for j in range(half_cmp_dim_k):
                            level2_k_cache[b, global_idx, h, j] = k_block[i, h, half_dim_k - half_cmp_dim_k + j]
                            level2_k_cache[b, global_idx, h, half_cmp_dim_k + j] = k_block[i, h, dim_k - half_cmp_dim_k + j]
        
        # 处理local_k_cache（最后sfa_block_size个token）
        for i in range(sfa_block_size):
            src_idx = seqlen_kv - sfa_block_size + i
            if src_idx >= 0:
                for h in range(num_kv_heads):
                    v = (seqlen_kv - offset - sfa_block_size + i) % sfa_block_size
                    if v >= 0:
                        local_k_cache[b, v, h, :] = k[b, src_idx, h, :]


def pytorch_decode_reference(
    k: torch.Tensor,
    lengths: torch.Tensor,
    local_k_cache: torch.Tensor,
    level1_k_min_cache: torch.Tensor,
    level1_k_max_cache: torch.Tensor,
    level2_k_cache: torch.Tensor,
    sfa_block_size: int = 32,
    sfa_cmp_ratio: int = 4
):
    """
    PyTorch实现的decode reference
    """
    batch, _, num_kv_heads, dim_k = k.shape
    max_batch = local_k_cache.shape[0]
    
    half_dim_k = dim_k // 2
    half_cmp_dim_k = half_dim_k // sfa_cmp_ratio
    
    for b in range(batch):
        index = lengths[b].item() - 1
        i_s = index // sfa_block_size
        
        for h in range(num_kv_heads):
            # 存储到local_k_cache
            local_k_cache[b, index % sfa_block_size, h, :] = k[b, 0, h, :]
            
            # 当block填满时，计算最小值和最大值
            if (index + 1) % sfa_block_size == 0:
                # 获取当前block的所有K值
                block_start = i_s * sfa_block_size
                block_k = local_k_cache[b, :, h, :]  # [sfa_block_size, dim_k]
                
                # 计算最小值和最大值
                k_min = torch.min(block_k, dim=0)[0]  # [dim_k]
                k_max = torch.max(block_k, dim=0)[0]  # [dim_k]
                
                # 存储到level1缓存
                level1_k_min_cache[b, i_s, h, :] = k_min
                level1_k_max_cache[b, i_s, h, :] = k_max
            
            # 存储压缩后的K到level2缓存
            for j in range(half_cmp_dim_k):
                level2_k_cache[b, index, h, j] = k[b, 0, h, half_dim_k - half_cmp_dim_k + j]
                level2_k_cache[b, index, h, half_cmp_dim_k + j] = k[b, 0, h, dim_k - half_cmp_dim_k + j]


# ============ 增强的测试代码 ============

def test_prefill_with_comparison():
    """测试prefill kernel并与PyTorch reference比较"""
    import torch
    
    # 配置参数
    max_batch = 2
    max_seqlen = 1024
    num_kv_heads = 4
    dim_k = 128
    sfa_block_size = 32
    sfa_cmp_ratio = 4
    
    # 创建kernel实例
    kernel = mha_prefill_store_k_index_kernel(
        max_batch=max_batch,
        max_seqlen=max_seqlen,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
        block_S=128,
        threads=256,
        num_stages=0
    )
    
    # 准备测试数据
    batch = 2
    seqlen_kv = 512
    
    # 创建输入张量
    k = torch.randn(batch, seqlen_kv, num_kv_heads, dim_k, dtype=torch.bfloat16, device='cuda')
    offsets = torch.tensor([0, 256], dtype=torch.int32, device='cuda')
    
    # 创建kernel输出缓存
    local_k_cache_kernel = torch.zeros(max_batch, sfa_block_size, num_kv_heads, dim_k, 
                                      dtype=torch.bfloat16, device='cuda')
    level1_k_min_cache_kernel = torch.zeros(max_batch, max_seqlen // sfa_block_size, num_kv_heads, dim_k,
                                           dtype=torch.bfloat16, device='cuda')
    level1_k_max_cache_kernel = torch.zeros(max_batch, max_seqlen // sfa_block_size, num_kv_heads, dim_k,
                                           dtype=torch.bfloat16, device='cuda')
    level2_k_cache_kernel = torch.zeros(max_batch, max_seqlen, num_kv_heads, dim_k // sfa_cmp_ratio,
                                       dtype=torch.bfloat16, device='cuda')
    
    # 创建reference输出缓存
    local_k_cache_ref = torch.zeros_like(local_k_cache_kernel)
    level1_k_min_cache_ref = torch.zeros_like(level1_k_min_cache_kernel)
    level1_k_max_cache_ref = torch.zeros_like(level1_k_max_cache_kernel)
    level2_k_cache_ref = torch.zeros_like(level2_k_cache_kernel)
    
    # 执行kernel
    kernel(k, offsets, local_k_cache_kernel, level1_k_min_cache_kernel, 
           level1_k_max_cache_kernel, level2_k_cache_kernel)
    
    # 执行PyTorch reference
    pytorch_prefill_reference(
        k, offsets, local_k_cache_ref, level1_k_min_cache_ref,
        level1_k_max_cache_ref, level2_k_cache_ref,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio
    )
    
    print("Prefill测试完成")
    print("=" * 50)
    
    # 比较结果
    def compare_tensors(name, tensor1, tensor2, rtol=1e-3, atol=1e-3):
        close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"{name}:")
        print(f"  是否一致: {'✓' if close else '✗'}")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        print(f"  非零元素数: {torch.count_nonzero(tensor1).item()}")
        return close
    
    print("\n结果比较:")
    compare_tensors("LocalKCache", local_k_cache_kernel, local_k_cache_ref)
    compare_tensors("Level1KMinCache", level1_k_min_cache_kernel, level1_k_min_cache_ref)
    compare_tensors("Level1KMaxCache", level1_k_max_cache_kernel, level1_k_max_cache_ref)
    compare_tensors("Level2KCache", level2_k_cache_kernel, level2_k_cache_ref)
    
    # 详细检查差异
    print("\n详细检查:")
    for b in range(batch):
        print(f"\nBatch {b}:")
        # 检查local_k_cache
        diff_local = torch.abs(local_k_cache_kernel[b] - local_k_cache_ref[b])
        max_diff_idx = torch.argmax(diff_local.view(-1))
        max_diff_val = diff_local.view(-1)[max_diff_idx].item()
        print(f"  LocalKCache最大差异: {max_diff_val:.6f}")
        
        # 检查level2缓存
        diff_level2 = torch.abs(level2_k_cache_kernel[b] - level2_k_cache_ref[b])
        if torch.any(diff_level2 > 1e-3):
            print(f"  Level2KCache有显著差异的位置:")
            indices = torch.where(diff_level2 > 1e-3)
            for idx in zip(*indices):
                if len(idx) == 3:  # [seq, head, dim]
                    print(f"    位置[{idx[0]}, {idx[1]}, {idx[2]}]: "
                          f"kernel={level2_k_cache_kernel[b, idx[0], idx[1], idx[2]].item():.4f}, "
                          f"ref={level2_k_cache_ref[b, idx[0], idx[1], idx[2]].item():.4f}")
    
    return True


def test_decode_with_comparison():
    """测试decode kernel并与PyTorch reference比较"""
    import torch
    
    # 配置参数
    max_batch = 2
    max_seqlen = 1024
    num_kv_heads = 4
    dim_k = 128
    sfa_block_size = 32
    sfa_cmp_ratio = 4
    
    # 创建kernel实例
    kernel = mha_decode_store_k_index_kernel(
        max_batch=max_batch,
        max_seqlen=max_seqlen,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
        threads=256
    )
    
    # 准备测试数据
    batch = 2
    
    # 创建输入张量
    k = torch.randn(batch, 1, num_kv_heads, dim_k, dtype=torch.bfloat16, device='cuda')
    lengths = torch.tensor([32, 64], dtype=torch.int32, device='cuda')
    
    # 创建kernel输出缓存
    local_k_cache_kernel = torch.zeros(max_batch, sfa_block_size, num_kv_heads, dim_k,
                                      dtype=torch.bfloat16, device='cuda')
    level1_k_min_cache_kernel = torch.zeros(max_batch, max_seqlen // sfa_block_size, num_kv_heads, dim_k,
                                           dtype=torch.bfloat16, device='cuda')
    level1_k_max_cache_kernel = torch.zeros(max_batch, max_seqlen // sfa_block_size, num_kv_heads, dim_k,
                                           dtype=torch.bfloat16, device='cuda')
    level2_k_cache_kernel = torch.zeros(max_batch, max_seqlen, num_kv_heads, dim_k // sfa_cmp_ratio,
                                       dtype=torch.bfloat16, device='cuda')
    
    # 创建reference输出缓存
    local_k_cache_ref = torch.zeros_like(local_k_cache_kernel)
    level1_k_min_cache_ref = torch.zeros_like(level1_k_min_cache_kernel)
    level1_k_max_cache_ref = torch.zeros_like(level1_k_max_cache_kernel)
    level2_k_cache_ref = torch.zeros_like(level2_k_cache_kernel)
    
    # 执行kernel
    kernel(k, lengths, local_k_cache_kernel, level1_k_min_cache_kernel,
           level1_k_max_cache_kernel, level2_k_cache_kernel)
    
    # 执行PyTorch reference
    pytorch_decode_reference(
        k, lengths, local_k_cache_ref, level1_k_min_cache_ref,
        level1_k_max_cache_ref, level2_k_cache_ref,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio
    )
    
    print("\nDecode测试完成")
    print("=" * 50)
    
    # 比较结果
    def compare_tensors(name, tensor1, tensor2, rtol=1e-3, atol=1e-3):
        close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"{name}:")
        print(f"  是否一致: {'✓' if close else '✗'}")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        return close
    
    print("\n结果比较:")
    compare_tensors("LocalKCache", local_k_cache_kernel, local_k_cache_ref)
    compare_tensors("Level1KMinCache", level1_k_min_cache_kernel, level1_k_min_cache_ref)
    compare_tensors("Level1KMaxCache", level1_k_max_cache_kernel, level1_k_max_cache_ref)
    compare_tensors("Level2KCache", level2_k_cache_kernel, level2_k_cache_ref)
    
    # 检查特定位置的更新
    print("\n详细检查:")
    for b in range(batch):
        index = lengths[b].item() - 1
        pos_in_block = index % sfa_block_size
        print(f"\nBatch {b}: index={index}, pos_in_block={pos_in_block}")
        
        # 检查local_k_cache
        kernel_val = local_k_cache_kernel[b, pos_in_block, 0, :5]
        ref_val = local_k_cache_ref[b, pos_in_block, 0, :5]
        print(f"  LocalKCache[:5]: kernel={kernel_val}, ref={ref_val}")
        print(f"  是否一致: {torch.allclose(kernel_val, ref_val, rtol=1e-3, atol=1e-3)}")
        
        # 检查level2缓存
        kernel_val = level2_k_cache_kernel[b, index, 0, :5]
        ref_val = level2_k_cache_ref[b, index, 0, :5]
        print(f"  Level2KCache[:5]: kernel={kernel_val}, ref={ref_val}")
        print(f"  是否一致: {torch.allclose(kernel_val, ref_val, rtol=1e-3, atol=1e-3)}")
    
    return True


def run_comprehensive_tests():
    """运行全面的测试和比较"""
    print("开始全面的KVO MHA Store K Index测试...")
    print("=" * 50)
    
    # 测试prefill kernel
    print("\n1. 测试Prefill Kernel (与PyTorch reference比较):")
    if test_prefill_with_comparison():
        print("✓ Prefill kernel测试通过")
    
    # 测试decode kernel  
    print("\n2. 测试Decode Kernel (与PyTorch reference比较):")
    if test_decode_with_comparison():
        print("✓ Decode kernel测试通过")
            
    print("\n" + "=" * 50)
    print("所有测试完成!")
    return True


if __name__ == "__main__":
    # 当直接运行此文件时执行测试
    run_comprehensive_tests()
