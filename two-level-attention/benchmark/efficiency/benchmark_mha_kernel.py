import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import json
import os
from datetime import datetime
from tls_attn.ops import (
    MHAInterface,
    SparseMHAInterface,
    MHAIndexerLevel1Interface,
    MHAIndexerLevel2Interface,
)

@torch.no_grad()
def benchmark():
    """测试kernel速度的基准测试函数"""
    import time

    # 测试配置
    configs = [
        # (batch, num_heads, num_kv_heads, dim_k, dim_v, seqlen)
        (4, 4, 1, 128, 128, 32 * 1024),
        (4, 4, 1, 128, 128, 64 * 1024),
        (4, 4, 1, 128, 128, 128 * 1024),
    ]

    warmup_iters = 10
    measure_iters = 100
    sfa_block_size = 64
    sfa_cmp_ratio = 4
    sfa_level1_topk = 128
    sfa_level2_topk = 1024
    sfa_sliding_blocks = 3

    # 创建结果保存目录
    results_dir = "exp/results_efficiency"
    os.makedirs(results_dir, exist_ok=True)

    # 生成结果文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(results_dir, f"benchmark_mha_{timestamp}.json")

    # 用于存储所有结果的列表
    all_results = []

    print("=" * 80)
    print("Kernel 性能基准测试 (MHA vs SparseMHA)")
    print("=" * 80)

    for batch, num_heads, num_kv_heads, dim_k, dim_v, seqlen in configs:
        torch.cuda.empty_cache()
        time.sleep(1)

        print(f"\n配置: batch={batch}, heads={num_heads}, kv_heads={num_kv_heads}, "
              f"dim_k={dim_k}, dim_v={dim_v}, seqlen={seqlen}")

        # 创建测试数据
        q = torch.randn(batch, 1, num_heads, dim_k, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen, num_kv_heads, dim_v, dtype=torch.bfloat16, device="cuda")
        o_mha = torch.zeros(batch, 1, num_heads, dim_v, dtype=torch.bfloat16, device="cuda")
        o_sparse_mha = torch.zeros(batch, 1, num_heads, dim_v, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([seqlen] * batch, dtype=torch.int32, device="cuda")

        level1_q = torch.randn_like(q)
        level1_k_min = torch.randn(batch, seqlen // sfa_block_size, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
        level1_k_max = torch.randn(batch, seqlen // sfa_block_size, num_kv_heads, dim_k, dtype=torch.bfloat16, device="cuda")
        level1_scores = torch.full((batch, num_kv_heads, seqlen // sfa_block_size), float('-inf'), dtype=torch.float32, device=q.device)
        level1_topk_scores = torch.zeros((batch, num_kv_heads, sfa_level1_topk), dtype=torch.float32, device=q.device)
        level1_topk_indices = torch.zeros((batch, num_kv_heads, sfa_level1_topk), dtype=torch.int32, device=q.device)
        level1_lengths = lengths // sfa_block_size

        level2_q = torch.randn(batch, 1, num_heads, dim_k // sfa_cmp_ratio, dtype=torch.bfloat16, device="cuda")
        level2_k = torch.randn(batch, seqlen, num_kv_heads, dim_k // sfa_cmp_ratio, dtype=torch.bfloat16, device="cuda")
        level2_scores = torch.full((batch, num_kv_heads, sfa_level1_topk * sfa_block_size), float('-inf'), dtype=torch.float32, device=q.device)
        level2_indices = torch.full((batch, num_kv_heads, sfa_level1_topk * sfa_block_size), -1, dtype=torch.int32, device=q.device)
        level2_cummax_scores = torch.full((batch, num_kv_heads, sfa_level1_topk * sfa_block_size), float('-inf'), dtype=torch.float32, device=q.device)
        level2_topk_scores = torch.zeros((batch, num_kv_heads, sfa_level2_topk), dtype=torch.float32, device=q.device)
        level2_topk_indices = torch.zeros((batch, num_kv_heads, sfa_level2_topk), dtype=torch.int32, device=q.device)
        level2_lengths = lengths

        # 测试MHA
        print("\n  [MHA测试]")
        mha = MHAInterface(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            dim_v=dim_v,
        )

        # Warmup
        for _ in range(warmup_iters):
            mha.forward_with_buffer(q, k, v, lengths, o_mha)
            torch.cuda.synchronize()

        # 创建CUDA事件用于精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 同步确保所有操作完成
        torch.cuda.synchronize()

        # 测量时间
        start_event.record()
        for _ in range(measure_iters):
            mha.forward_with_buffer(q, k, v, lengths, o_mha)
        end_event.record()

        # 等待事件完成
        torch.cuda.synchronize()

        # 计算时间
        elapsed_time_mha = start_event.elapsed_time(end_event) / measure_iters  # 毫秒

        # 计算吞吐量 (tokens/秒)
        total_tokens = batch
        throughput_mha = total_tokens / (elapsed_time_mha / 1000)  # tokens/秒

        # 计算FLOPs估计
        qk_flops = batch * num_heads * seqlen * dim_k
        ov_flops = batch * num_heads * seqlen * dim_v
        total_flops = qk_flops + ov_flops
        flops_per_sec_mha = total_flops / (elapsed_time_mha / 1000)  # FLOPS

        print(f"    平均执行时间: {elapsed_time_mha:.3f} ms")
        print(f"    吞吐量: {throughput_mha:,.0f} tokens/秒")
        print(f"    计算量: {total_flops / 1e9:.2f} GFLOPs")
        print(f"    计算性能: {flops_per_sec_mha / 1e12:.2f} TFLOPS")

        # 测试SparseMHA
        print("\n  [SparseMHA测试]")
        level1_indexer = MHAIndexerLevel1Interface(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            topk=sfa_level1_topk,
            sliding_blocks=sfa_sliding_blocks,
        )
        level2_indexer = MHAIndexerLevel2Interface(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k // sfa_cmp_ratio,
            level1_topk=sfa_level1_topk,
            level2_topk=sfa_level2_topk,
            block_size=sfa_block_size,
            sliding_blocks=sfa_sliding_blocks,
        )
        sparse_mha = SparseMHAInterface(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            topk=sfa_level2_topk,
        )

        # Warmup
        for _ in range(warmup_iters):
            level1_indexer.forward_with_buffer(
                level1_q, level1_k_min, level1_k_max, level1_lengths,
                level1_scores, level1_topk_scores, level1_topk_indices,
            )
            level2_indexer.forward_with_buffer(
                level2_q, level2_k, level1_topk_indices, level2_lengths,
                level2_scores, level2_indices, level2_cummax_scores,
                level2_topk_scores, level2_topk_indices,
            )
            sparse_mha.forward_with_buffer(q, k, v, level2_topk_indices, o_sparse_mha)

        # 同步确保所有操作完成
        torch.cuda.synchronize()

        # 测量时间
        start_event.record()
        for _ in range(measure_iters):
            level1_indexer.forward_with_buffer(
                level1_q, level1_k_min, level1_k_max, level1_lengths,
                level1_scores, level1_topk_scores, level1_topk_indices,
            )
            level2_indexer.forward_with_buffer(
                level2_q, level2_k, level1_topk_indices, level2_lengths,
                level2_scores, level2_indices, level2_cummax_scores,
                level2_topk_scores, level2_topk_indices,
            )
            sparse_mha.forward_with_buffer(q, k, v, level2_topk_indices, o_sparse_mha)
        end_event.record()

        # 等待事件完成
        torch.cuda.synchronize()

        # 计算时间
        elapsed_time_sparse = start_event.elapsed_time(end_event) / measure_iters  # 毫秒

        # 计算吞吐量 (tokens/秒)
        # 注意：稀疏版本处理的token数量减少
        sparse_total_tokens = batch  # 假设保留50%的token
        throughput_sparse = sparse_total_tokens / (elapsed_time_sparse / 1000)  # tokens/秒

        # 计算FLOPs估计（稀疏版本）
        sparse_qk_flops = batch * num_heads * sfa_level2_topk * dim_k
        sparse_ov_flops = batch * num_heads * sfa_level2_topk * dim_v
        sparse_total_flops = sparse_qk_flops + sparse_ov_flops
        flops_per_sec_sparse = sparse_total_flops / (elapsed_time_sparse / 1000)  # FLOPS

        print(f"    平均执行时间: {elapsed_time_sparse:.3f} ms")
        print(f"    吞吐量: {throughput_sparse:,.0f} tokens/秒")
        print(f"    计算量: {sparse_total_flops / 1e9:.2f} GFLOPs")
        print(f"    计算性能: {flops_per_sec_sparse / 1e12:.2f} TFLOPS")

        # 性能对比
        print("\n  [性能对比]")
        speedup = elapsed_time_mha / elapsed_time_sparse
        throughput_ratio = throughput_sparse / throughput_mha
        print(f"    加速比: {speedup:.2f}x")
        print(f"    吞吐量提升: {throughput_ratio:.2f}x")
        print(f"    时间减少: {(elapsed_time_mha - elapsed_time_sparse) / elapsed_time_mha * 100:.1f}%")

        # 保存当前配置的结果
        result = {
            "timestamp": timestamp,
            "config": {
                "batch": batch,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "dim_k": dim_k,
                "dim_v": dim_v,
                "seqlen": seqlen,
                "sfa_block_size": sfa_block_size,
                "sfa_cmp_ratio": sfa_cmp_ratio,
                "sfa_level1_topk": sfa_level1_topk,
                "sfa_level2_topk": sfa_level2_topk,
                "sfa_sliding_blocks": sfa_sliding_blocks,
                "warmup_iters": warmup_iters,
                "measure_iters": measure_iters,
            },
            "mha": {
                "avg_time_ms": elapsed_time_mha,
                "throughput_tokens_per_sec": throughput_mha,
                "total_gflops": total_flops / 1e9,
                "tflops": flops_per_sec_mha / 1e12,
            },
            "sparse_mha": {
                "avg_time_ms": elapsed_time_sparse,
                "throughput_tokens_per_sec": throughput_sparse,
                "total_gflops": sparse_total_flops / 1e9,
                "tflops": flops_per_sec_sparse / 1e12,
            },
            "comparison": {
                "speedup": speedup,
                "throughput_ratio": throughput_ratio,
                "time_reduction_percent": (elapsed_time_mha - elapsed_time_sparse) / elapsed_time_mha * 100,
            }
        }
        all_results.append(result)

    # 保存所有结果到JSON文件
    with open(json_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump({
            "timestamp": timestamp,
            "configs": all_results
        }, jsonfile, indent=2, ensure_ascii=False)

    print(f"JSON结果已保存到: {json_filename}")
    print("=" * 80)

if __name__ == '__main__':
    benchmark()