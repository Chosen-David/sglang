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

    level1_k_cache_shape = [max_batch, level1_seqlen, num_kv_heads, dim_k]
    level2_k_cache_shape = [max_batch, level2_seqlen, num_kv_heads, cmp_dim_k]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        Offsets: T.Tensor(offsets_shape, T.int32),
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

    k_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_k]
    offsets_shape = [batch]
    level1_k_cache_shape = [max_batch, level1_seqlen, num_kv_heads, dim_k]
    level2_k_cache_shape = [max_batch, level2_seqlen, num_kv_heads, cmp_dim_k]

    @T.prim_func
    def main(
        KCache: T.Tensor(k_cache_shape, T.bfloat16),
        Lengths: T.Tensor(offsets_shape, T.int32),
        Level1KMinCache: T.Tensor(level1_k_cache_shape, T.bfloat16),
        Level1KMaxCache: T.Tensor(level1_k_cache_shape, T.bfloat16),
        Level2KCache: T.Tensor(level2_k_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, num_kv_heads, threads=threads) as (bx, by):
            i_b = bx
            i_h = by
            index = Lengths[i_b] - 1
            i_s = (index // sfa_block_size)

            Level1K_shared = T.alloc_shared([sfa_block_size, dim_k], dtype=T.bfloat16)
            K_frag = T.alloc_fragment([sfa_block_size, dim_k], dtype=T.bfloat16)
            K_min = T.alloc_fragment([dim_k], dtype=T.bfloat16)
            K_max = T.alloc_fragment([dim_k], dtype=T.bfloat16)

            if (index + 1) % sfa_block_size == 0:
                T.fill(Level1K_shared, 0)
                T.copy(KCache[i_b, i_s * sfa_block_size:(i_s + 1) * sfa_block_size, i_h, :], Level1K_shared)
                T.copy(Level1K_shared, K_frag)
                T.reduce_min(K_frag, K_min, dim=0, clear=True)
                T.reduce_max(K_frag, K_max, dim=0, clear=True)
                for j in T.Parallel(dim_k):
                    Level1KMinCache[i_b, i_s, i_h, j] = K_min[j]
                for j in T.Parallel(dim_k):
                    Level1KMaxCache[i_b, i_s, i_h, j] = K_max[j]

            Level2K_shared = T.alloc_shared([dim_k], dtype=T.bfloat16)
            T.copy(KCache[i_b, index, i_h, :], Level2K_shared)
            for j in T.Parallel(half_cmp_dim_k):
                Level2KCache[i_b, index, i_h, j] = Level2K_shared[half_dim_k - half_cmp_dim_k + j]
            for j in T.Parallel(half_cmp_dim_k):
                Level2KCache[i_b, index, i_h, half_cmp_dim_k + j] = Level2K_shared[dim_k - half_cmp_dim_k + j]

    return main


class MHAStoreKIndexInterface:

    def __init__(self,
        max_batch: int,
        max_seqlen: int,
        num_kv_heads: int,
        dim_k: int,
        sfa_block_size: int,
        sfa_cmp_ratio: int,
    ) -> None:
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
        level1_k_min_cache: torch.Tensor,
        level1_k_max_cache: torch.Tensor,
        level2_k_cache: torch.Tensor,
    ):
        self.prefill_store_k_index_kernel(k, offsets, level1_k_min_cache, level1_k_max_cache, level2_k_cache)
    
    def decode_forward(
        self,
        k_cache: torch.Tensor,  # Changed from k
        lengths: torch.Tensor,  # Changed from v
        level1_k_min_cache: torch.Tensor,  # Changed from indices
        level1_k_max_cache: torch.Tensor,  # Changed from indices
        level2_k_cache: torch.Tensor,  # Changed from k_cache
    ):
        self.decode_store_k_index_kernel(k_cache, lengths, level1_k_min_cache, level1_k_max_cache, level2_k_cache)


class MHAStoreKIndexReference:
    """PyTorch reference implementation for testing kernels"""
    
    @staticmethod
    def prefill_forward(
        k: torch.Tensor,
        offsets: torch.Tensor,
        level1_k_min_cache: torch.Tensor,
        level1_k_max_cache: torch.Tensor,
        level2_k_cache: torch.Tensor,
        sfa_block_size: int,
        sfa_cmp_ratio: int,
    ):
        """
        Reference implementation for prefill K index storage
        
        Args:
            k: [batch, seqlen_kv, num_kv_heads, dim_k]
            offsets: [batch] - starting position in cache for each batch
            level1_k_cache: [max_batch, level1_seqlen, num_kv_heads, 2, dim_k]
            level2_k_cache: [max_batch, level2_seqlen, num_kv_heads, cmp_dim_k]
            sfa_block_size: block size for SFA
            sfa_cmp_ratio: compression ratio for SFA
        """
        batch_size = k.shape[0]
        seqlen_kv = k.shape[1]
        num_kv_heads = k.shape[2]
        dim_k = k.shape[3]
        
        half_dim_k = dim_k // 2
        half_cmp_dim_k = half_dim_k // sfa_cmp_ratio
        
        for b in range(batch_size):
            offset = offsets[b].item()
            for h in range(num_kv_heads):
                # Process in blocks of sfa_block_size
                for block_start in range(0, seqlen_kv - offset, sfa_block_size):
                    block_end = min(block_start + sfa_block_size, seqlen_kv - offset)
                    block_size = block_end - block_start
                                        
                    # Get the current block
                    k_block = k[b, offset + block_start:offset + block_end, h, :]
                    if k_block.shape[0] != sfa_block_size:
                        k_block = torch.nn.functional.pad(k_block, (0, 0, 0, sfa_block_size - k_block.shape[0]), value=0)
                    
                    # Compute min and max for level1 cache
                    k_min = torch.min(k_block, dim=0).values
                    k_max = torch.max(k_block, dim=0).values
                    
                    # Store in level1 cache
                    level1_idx = block_start // sfa_block_size
                    level1_k_min_cache[b, level1_idx, h, :] = k_min
                    level1_k_max_cache[b, level1_idx, h, :] = k_max
                    
                    # Store partial dimensions in level2 cache
                    for i in range(block_size):
                        # First half of compressed dimensions
                        for j in range(half_cmp_dim_k):
                            level2_k_cache[b, block_start + i, h, j] = k_block[i, half_dim_k - half_cmp_dim_k + j]
                        # Second half of compressed dimensions
                        for j in range(half_cmp_dim_k):
                            level2_k_cache[b, block_start + i, h, half_cmp_dim_k + j] = k_block[i, dim_k - half_cmp_dim_k + j]
    
    @staticmethod
    def decode_forward(
        k_cache: torch.Tensor,
        lengths: torch.Tensor,
        level1_k_min_cache: torch.Tensor,
        level1_k_max_cache: torch.Tensor,
        level2_k_cache: torch.Tensor,
        sfa_block_size: int,
        sfa_cmp_ratio: int,
    ):
        """
        Reference implementation for decode K index storage
        
        Args:
            k_cache: [max_batch, max_seqlen, num_kv_heads, dim_k]
            indices: [batch] - position in cache for each batch
            level1_k_cache: [max_batch, level1_seqlen, num_kv_heads, 2, dim_k]
            level2_k_cache: [max_batch, level2_seqlen, num_kv_heads, cmp_dim_k]
            sfa_block_size: block size for SFA
            sfa_cmp_ratio: compression ratio for SFA
        """
        batch_size = lengths.shape[0]
        max_seqlen = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        dim_k = k_cache.shape[3]
        
        cmp_dim_k = dim_k // sfa_cmp_ratio
        half_dim_k = dim_k // 2
        half_cmp_dim_k = half_dim_k // sfa_cmp_ratio
        
        for b in range(batch_size):
            index = lengths[b].item() - 1
            block_idx = index // sfa_block_size
            
            # Check if we need to update level1 cache (when completing a block)
            if (index + 1) % sfa_block_size == 0:
                block_start = block_idx * sfa_block_size
                block_end = block_start + sfa_block_size
                
                for h in range(num_kv_heads):
                    # Get the completed block
                    k_block = k_cache[b, block_start:block_end, h, :]
                    
                    # Compute min and max
                    k_min = torch.min(k_block, dim=0).values
                    k_max = torch.max(k_block, dim=0).values
                    
                    # Store in level1 cache
                    level1_k_min_cache[b, block_idx, h, :] = k_min
                    level1_k_max_cache[b, block_idx, h, :] = k_max
            
            # Always update level2 cache for the current position
            for h in range(num_kv_heads):
                k_vector = k_cache[b, index, h, :]
                
                # Store partial dimensions in level2 cache
                for j in range(half_cmp_dim_k):
                    level2_k_cache[b, index, h, j] = k_vector[half_dim_k - half_cmp_dim_k + j]
                for j in range(half_cmp_dim_k):
                    level2_k_cache[b, index, h, half_cmp_dim_k + j] = k_vector[dim_k - half_cmp_dim_k + j]

def test_mha_prefill_store_k_index(
    max_batch: int = 4,
    max_seqlen: int = 1024,
    num_kv_heads: int = 8,
    dim_k: int = 128,
    sfa_block_size: int = 32,
    sfa_cmp_ratio: int = 4,
    batch_size: int = 4,
    seqlen_kv: int = 1024,
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Test prefill K index storage kernel against reference implementation
    
    Args:
        max_batch: Maximum batch size for cache
        max_seqlen: Maximum sequence length for cache
        num_kv_heads: Number of KV heads
        dim_k: Key dimension
        sfa_block_size: block size for SFA
        sfa_cmp_ratio: compression ratio for SFA
        batch_size: Actual batch size for test
        seqlen_kv: Actual sequence length for test
        device: Device to run test on
        verbose: Whether to print detailed results
    """
    import torch
    
    # Create kernel interface
    interface = MHAStoreKIndexInterface(
        max_batch=max_batch,
        max_seqlen=max_seqlen,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
    )
    
    # Create random inputs
    k = torch.randn(batch_size, seqlen_kv, num_kv_heads, dim_k,
                   dtype=torch.bfloat16, device=device)
    offsets = torch.randint(0, seqlen_kv,
                           (batch_size,), dtype=torch.int32, device=device)
    
    # Calculate cache dimensions
    level1_seqlen = max_seqlen // sfa_block_size
    cmp_dim_k = dim_k // sfa_cmp_ratio
    
    # Create cache tensors
    level1_k_min_cache = torch.zeros(max_batch, level1_seqlen, num_kv_heads, dim_k,
                                     dtype=torch.bfloat16, device=device)
    level1_k_max_cache = torch.zeros(max_batch, level1_seqlen, num_kv_heads, dim_k,
                                     dtype=torch.bfloat16, device=device)
    level2_k_cache = torch.zeros(max_batch, max_seqlen, num_kv_heads, cmp_dim_k,
                                dtype=torch.bfloat16, device=device)
    
    # Create copies for reference
    level1_k_min_cache_ref = level1_k_min_cache.clone()
    level1_k_max_cache_ref = level1_k_max_cache.clone()
    level2_k_cache_ref = level2_k_cache.clone()
    
    # Run kernel
    interface.prefill_forward(k, offsets, level1_k_min_cache, level1_k_max_cache, level2_k_cache)
    
    # Run reference
    MHAStoreKIndexReference.prefill_forward(
        k, offsets, level1_k_min_cache_ref, level1_k_max_cache_ref, level2_k_cache_ref,
        sfa_block_size, sfa_cmp_ratio
    )

    # Compare results
    level1_match_0 = torch.allclose(level1_k_min_cache, level1_k_min_cache_ref, rtol=1e-3, atol=1e-3)
    level1_match_1 = torch.allclose(level1_k_max_cache, level1_k_max_cache_ref, rtol=1e-3, atol=1e-3)
    level1_match = level1_match_0 & level1_match_1
    level2_match = torch.allclose(level2_k_cache, level2_k_cache_ref, rtol=1e-3, atol=1e-3)

    if verbose:
        print(f"=== Prefill K Index Storage Test ===")
        print(f"Batch size: {batch_size}, Seqlen: {seqlen_kv}")
        print(f"SFA block size: {sfa_block_size}, CMP ratio: {sfa_cmp_ratio}")
        print(f"Offsets: {offsets.cpu().tolist()}")
        print(f"Level1 cache match: {level1_match}")
        print(f"Level2 cache match: {level2_match}")
        
        if not level1_match:
            level1_diff = torch.abs(level1_k_min_cache - level1_k_min_cache_ref).max().item()
            print(f"Max Level1 difference: {level1_diff}")
            
        if not level2_match:
            level2_diff = torch.abs(level2_k_cache - level2_k_cache_ref).max().item()
            print(f"Max Level2 difference: {level2_diff}")
    
    return level1_match and level2_match, level1_k_min_cache, level1_k_min_cache_ref, level2_k_cache, level2_k_cache_ref

def test_mha_decode_store_k_index(
    max_batch: int = 4,
    max_seqlen: int = 1024,
    num_kv_heads: int = 2,
    dim_k: int = 128,
    sfa_block_size: int = 32,
    sfa_cmp_ratio: int = 4,
    batch_size: int = 4,
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Test decode K index storage kernel against reference implementation
    
    Args:
        max_batch: Maximum batch size for cache
        max_seqlen: Maximum sequence length for cache
        num_kv_heads: Number of KV heads
        dim_k: Key dimension
        sfa_block_size: block size for SFA
        sfa_cmp_ratio: compression ratio for SFA
        batch_size: Actual batch size for test
        device: Device to run test on
        verbose: Whether to print detailed results
    """
    import torch
    
    # Create kernel interface
    interface = MHAStoreKIndexInterface(
        max_batch=max_batch,
        max_seqlen=max_seqlen,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
    )
    
    # Create random inputs
    k_cache = torch.randn(max_batch, max_seqlen, num_kv_heads, dim_k,
                         dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, max_seqlen,
                           (batch_size,), dtype=torch.int32, device=device)
    lengths = indices + 1
    
    # Calculate cache dimensions
    level1_seqlen = max_seqlen // sfa_block_size
    cmp_dim_k = dim_k // sfa_cmp_ratio
    
    # Create cache tensors
    level1_k_min_cache = torch.zeros(max_batch, level1_seqlen, num_kv_heads, dim_k,
                                dtype=torch.bfloat16, device=device)
    level1_k_max_cache = torch.zeros(max_batch, level1_seqlen, num_kv_heads, dim_k,
                                dtype=torch.bfloat16, device=device)
    level2_k_cache = torch.zeros(max_batch, max_seqlen, num_kv_heads, cmp_dim_k,
                                dtype=torch.bfloat16, device=device)
    
    # Create copies for reference
    level1_k_min_cache_ref = level1_k_min_cache.clone()
    level1_k_max_cache_ref = level1_k_max_cache.clone()
    level2_k_cache_ref = level2_k_cache.clone()
    
    # Run kernel - note: decode_forward signature needs to be updated in MHAStoreKIndexInterface
    # For now, we'll create a temporary k tensor to match the current interface
    k = torch.randn(batch_size, 1, num_kv_heads, dim_k,
                   dtype=torch.bfloat16, device=device)
    
    # First update k_cache with the new k values at indices
    for b in range(batch_size):
        k_cache[b, indices[b], :, :] = k[b, 0, :, :]
    
    # Run kernel
    interface.decode_forward(k_cache, lengths, level1_k_min_cache, level1_k_max_cache, level2_k_cache)
    
    # Run reference
    MHAStoreKIndexReference.decode_forward(
        k_cache, lengths, level1_k_min_cache_ref, level1_k_max_cache_ref, level2_k_cache_ref,
        sfa_block_size, sfa_cmp_ratio
    )

    # Compare results
    level1_match_0 = torch.allclose(level1_k_min_cache, level1_k_min_cache_ref, rtol=1e-3, atol=1e-3)
    level1_match_1 = torch.allclose(level1_k_max_cache, level1_k_max_cache_ref, rtol=1e-3, atol=1e-3)
    level1_match = level1_match_0 & level1_match_1
    level2_match = torch.allclose(level2_k_cache, level2_k_cache_ref, rtol=1e-3, atol=1e-3)
    
    if verbose:
        print(f"=== Decode K Index Storage Test ===")
        print(f"Batch size: {batch_size}")
        print(f"SFA block size: {sfa_block_size}, CMP ratio: {sfa_cmp_ratio}")
        print(f"Indices: {indices.cpu().tolist()}")
        print(f"Level1 cache match: {level1_match}")
        print(f"Level2 cache match: {level2_match}")
        
        if not level1_match:
            level1_diff = torch.abs(level1_k_min_cache - level1_k_min_cache_ref).max().item()
            print(f"Max Level1 difference: {level1_diff}")
            
        if not level2_match:
            level2_diff = torch.abs(level2_k_cache - level2_k_cache_ref).max().item()
            print(f"Max Level2 difference: {level2_diff}")
    
    return level1_match and level2_match, level1_k_min_cache - level1_k_min_cache_ref, level2_k_cache, level2_k_cache_ref

@torch.no_grad()
def run_all_tests(
    device: str = "cuda:0",
    verbose: bool = True,
):
    """
    Run comprehensive tests for both prefill and decode kernels
    
    Args:
        device: Device to run tests on
        verbose: Whether to print detailed results
    """
    torch.manual_seed(42)
    print("Running MHA Store K Index Kernel Tests...")
    
    # Test parameters
    sfa_block_size = 32
    sfa_cmp_ratio = 4
    
    # Test 1: Basic prefill test
    print("\n1. Testing Prefill Kernel (basic)...")
    success1, _, _, _, _ = test_mha_prefill_store_k_index(
        batch_size=2,
        seqlen_kv=96,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
        device=device,
        verbose=verbose,
    )
    
    # Test 2: Prefill with different offsets
    print("\n2. Testing Prefill Kernel (various offsets)...")
    success2, _, _, _, _ = test_mha_prefill_store_k_index(
        batch_size=3,
        seqlen_kv=256,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
        device=device,
        verbose=verbose,
    )

    # Test 3: Basic decode test
    print("\n3. Testing Decode Kernel (basic)...")
    success3, _, _, _ = test_mha_decode_store_k_index(
        batch_size=2,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
        device=device,
        verbose=verbose,
    )
    
    # Test 4: Decode with random indices
    print("\n4. Testing Decode Kernel (random indices)...")
    success4, _, _, _ = test_mha_decode_store_k_index(
        batch_size=3,
        sfa_block_size=sfa_block_size,
        sfa_cmp_ratio=sfa_cmp_ratio,
        device=device,
        verbose=verbose,
    )
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Prefill Test 1: {'PASS' if success1 else 'FAIL'}")
    print(f"Prefill Test 2: {'PASS' if success2 else 'FAIL'}")
    print(f"Decode Test 1:  {'PASS' if success3 else 'FAIL'}")
    print(f"Decode Test 2:  {'PASS' if success4 else 'FAIL'}")
    
    all_passed = success1 and success2 and success3 and success4
    print(f"\nAll tests: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == '__main__':
    run_all_tests()