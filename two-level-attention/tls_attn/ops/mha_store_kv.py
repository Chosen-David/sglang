import torch
import tilelang as tl
import tilelang.language as T
from typing import Optional

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
    max_seqlen: int,
    num_kv_heads: int,
    dim_k: int,
    dim_v: int,
    threads: int = 256,
):
    batch = T.symbolic("batch")
    seqlen_kv = T.symbolic("seqlen_kv")

    k_shape = [batch, 1, num_kv_heads, dim_k]
    v_shape = [batch, 1, num_kv_heads, dim_v]
    offsets_shape = [batch]
    k_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_k]
    v_cache_shape = [max_batch, max_seqlen, num_kv_heads, dim_v]

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, T.bfloat16),
        V: T.Tensor(v_shape, T.bfloat16),
        Lengths: T.Tensor(offsets_shape, T.int32),
        KCache: T.Tensor(k_cache_shape, T.bfloat16),
        VCache: T.Tensor(v_cache_shape, T.bfloat16),
    ):
        with T.Kernel(batch, threads=threads) as (bx,):
            i_b = bx
            index = Lengths[i_b] - 1
            K_shared = T.alloc_shared([num_kv_heads, dim_k], dtype=T.bfloat16)
            V_shared = T.alloc_shared([num_kv_heads, dim_v], dtype=T.bfloat16)

            T.copy(K[i_b, 0, :, :], K_shared)
            T.copy(V[i_b, 0, :, :], V_shared)
            T.copy(K_shared, KCache[i_b, index, :, :])
            T.copy(V_shared, VCache[i_b, index, :, :])

    return main


class MHAStoreKVInterface:

    def __init__(self,
        max_batch: int,
        max_seqlen: int,
        num_kv_heads: int,
        dim_k: int,
        dim_v: int,
    ) -> None:
        self.prefill_store_kv_kernel = mha_prefill_store_kv_kernel(
            max_batch,
            max_seqlen,
            num_kv_heads,
            dim_k,
            dim_v,
        )

        self.decode_store_kv_kernel = mha_decode_store_kv_kernel(
            max_batch,
            max_seqlen,
            num_kv_heads,
            dim_k,
            dim_v,
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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        self.decode_store_kv_kernel(k, v, lengths, k_cache, v_cache)


class MHAStoreKVReference:
    """PyTorch reference implementation for testing kernels"""
    
    @staticmethod
    def prefill_forward(
        k: torch.Tensor,
        v: torch.Tensor,
        offsets: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        """
        Reference implementation for prefill KV storage
        
        Args:
            k: [batch, seqlen_kv, num_kv_heads, dim_k]
            v: [batch, seqlen_kv, num_kv_heads, dim_v]
            offsets: [batch] - starting position in cache for each batch
            k_cache: [max_batch, max_seqlen, num_kv_heads, dim_k]
            v_cache: [max_batch, max_seqlen, num_kv_heads, dim_k]
        """
        batch_size = k.shape[0]
        seqlen_kv = k.shape[1]
        num_kv_heads = k.shape[2]
        
        for b in range(batch_size):
            offset = offsets[b].item()
            # Copy K values
            k_cache[b, :seqlen_kv-offset, :, :] = k[b, offset:, :, :]
            # Copy V values
            v_cache[b, :seqlen_kv-offset, :, :] = v[b, offset:, :, :]
    
    @staticmethod
    def decode_forward(
        k: torch.Tensor,
        v: torch.Tensor,
        lengths: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        """
        Reference implementation for decode KV storage
        
        Args:
            k: [batch, 1, num_kv_heads, dim_k]
            v: [batch, 1, num_kv_heads, dim_v]
            lengths: [batch] - position in cache for each batch
            k_cache: [max_batch, max_seqlen, num_kv_heads, dim_k]
            v_cache: [max_batch, max_seqlen, num_kv_heads, dim_k]
        """
        batch_size = k.shape[0]
        
        for b in range(batch_size):
            index = lengths[b].item() - 1
            # Copy K values (single position)
            k_cache[b, index, :, :] = k[b, 0, :, :]
            # Copy V values (single position)
            v_cache[b, index, :, :] = v[b, 0, :, :]


def test_mha_prefill_store_kv(
    max_batch: int = 4,
    max_seqlen: int = 1024,
    num_kv_heads: int = 8,
    dim_k: int = 128,
    dim_v: int = 128,
    batch_size: int = 4,
    seqlen_kv: int = 1024,
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Test prefill KV storage kernel against reference implementation
    
    Args:
        max_batch: Maximum batch size for cache
        max_seqlen: Maximum sequence length for cache
        num_kv_heads: Number of KV heads
        dim_k: Key dimension
        dim_v: Value dimension
        batch_size: Actual batch size for test
        seqlen_kv: Actual sequence length for test
        device: Device to run test on
        verbose: Whether to print detailed results
    """
    import torch
    
    # Create kernel interface
    interface = MHAStoreKVInterface(
        max_batch=max_batch,
        max_seqlen=max_seqlen,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        dim_v=dim_v,
    )
    
    # Create random inputs
    k = torch.randn(batch_size, seqlen_kv, num_kv_heads, dim_k,
                   dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, seqlen_kv, num_kv_heads, dim_v,
                   dtype=torch.bfloat16, device=device)
    offsets = torch.randint(0, seqlen_kv,
                           (batch_size,), dtype=torch.int32, device=device)
    
    # Create cache tensors
    k_cache = torch.zeros(max_batch, max_seqlen,
                         num_kv_heads, dim_k,
                         dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(max_batch, max_seqlen,
                         num_kv_heads, dim_v,
                         dtype=torch.bfloat16, device=device)
    
    # Create copies for reference
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    
    # Run kernel
    interface.prefill_forward(k, v, offsets, k_cache, v_cache)
    
    # Run reference
    MHAStoreKVReference.prefill_forward(k, v, offsets, k_cache_ref, v_cache_ref)

    # Compare results
    k_match = torch.allclose(k_cache, k_cache_ref, rtol=1e-3, atol=1e-3)
    v_match = torch.allclose(v_cache, v_cache_ref, rtol=1e-3, atol=1e-3)

    if verbose:
        print(f"=== Prefill KV Storage Test ===")
        print(f"Batch size: {batch_size}, Seqlen: {seqlen_kv}")
        print(f"Offsets: {offsets.cpu().tolist()}")
        print(f"K cache match: {k_match}")
        print(f"V cache match: {v_match}")
        
        if not k_match:
            k_diff = torch.abs(k_cache - k_cache_ref).max().item()
            print(f"Max K difference: {k_diff}")
            
        if not v_match:
            v_diff = torch.abs(v_cache - v_cache_ref).max().item()
            print(f"Max V difference: {v_diff}")
    
    return k_match and v_match, k_cache, k_cache_ref, v_cache, v_cache_ref


def test_mha_decode_store_kv(
    max_batch: int = 4,
    max_seqlen: int = 1024,
    num_kv_heads: int = 8,
    dim_k: int = 128,
    dim_v: int = 128,
    batch_size: int = 4,
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Test decode KV storage kernel against reference implementation
    
    Args:
        max_batch: Maximum batch size for cache
        max_seqlen: Maximum sequence length for cache
        num_kv_heads: Number of KV heads
        dim_k: Key dimension
        dim_v: Value dimension
        batch_size: Actual batch size for test
        device: Device to run test on
        verbose: Whether to print detailed results
    """
    import torch
    
    # Create kernel interface
    interface = MHAStoreKVInterface(
        max_batch=max_batch,
        max_seqlen=max_seqlen,
        num_kv_heads=num_kv_heads,
        dim_k=dim_k,
        dim_v=dim_v,
    )
    
    # Create random inputs (single step)
    k = torch.randn(batch_size, 1, num_kv_heads, dim_k,
                   dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, 1, num_kv_heads, dim_v,
                   dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, max_seqlen,
                           (batch_size,), dtype=torch.int32, device=device)
    lengths = indices + 1
    
    # Create cache tensors
    k_cache = torch.zeros(max_batch, max_seqlen,
                         num_kv_heads, dim_k,
                         dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(max_batch, max_seqlen,
                         num_kv_heads, dim_v,
                         dtype=torch.bfloat16, device=device)
    
    # Create copies for reference
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    
    # Run kernel
    interface.decode_forward(k, v, lengths, k_cache, v_cache)
    
    # Run reference
    MHAStoreKVReference.decode_forward(k, v, lengths, k_cache_ref, v_cache_ref)
    
    # Compare results
    k_match = torch.allclose(k_cache, k_cache_ref, rtol=1e-3, atol=1e-3)
    v_match = torch.allclose(v_cache, v_cache_ref, rtol=1e-3, atol=1e-3)
    
    if verbose:
        print(f"=== Decode KV Storage Test ===")
        print(f"Batch size: {batch_size}")
        print(f"Indices: {indices.cpu().tolist()}")
        print(f"K cache match: {k_match}")
        print(f"V cache match: {v_match}")
        
        if not k_match:
            k_diff = torch.abs(k_cache - k_cache_ref).max().item()
            print(f"Max K difference: {k_diff}")
            
        if not v_match:
            v_diff = torch.abs(v_cache - v_cache_ref).max().item()
            print(f"Max V difference: {v_diff}")
    
    return k_match and v_match, k_cache, k_cache_ref, v_cache, v_cache_ref

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
    print("Running MHA Store KV Kernel Tests...")
    
    # Test 1: Basic prefill test
    print("\n1. Testing Prefill Kernel (basic)...")
    success1, _, _, _, _ = test_mha_prefill_store_kv(
        batch_size=2,
        seqlen_kv=512,
        device=device,
        verbose=verbose,
    )
    
    # Test 2: Prefill with different offsets
    print("\n2. Testing Prefill Kernel (various offsets)...")
    success2, _, _, _, _ = test_mha_prefill_store_kv(
        batch_size=3,
        seqlen_kv=256,
        device=device,
        verbose=verbose,
    )

    # Test 3: Basic decode test
    print("\n3. Testing Decode Kernel (basic)...")
    success3, _, _, _, _ = test_mha_decode_store_kv(
        batch_size=2,
        device=device,
        verbose=verbose,
    )
    
    # Test 4: Decode with random indices
    print("\n4. Testing Decode Kernel (random indices)...")
    success4, _, _, _, _ = test_mha_decode_store_kv(
        batch_size=3,
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
