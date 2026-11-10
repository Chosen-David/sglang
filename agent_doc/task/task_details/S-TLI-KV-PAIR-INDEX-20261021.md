# S-TLI-KV-PAIR-INDEX-20261021

Second user-requested review round. Base: PR1 head057163fbb8c81dc8e98c70cd0664dd47b7974179, on main5ba097eb3c15cb3d265bf33d9da474854ec52fe2. Preserve the first-round host snapshot fix.

_sparse_attn_batched fallback formed integer addresses[n,Hkv,K2,D] then flattened the KV buffer for gathering. Replace only that construction with broadcasting[slot,head] indexing, retaining D as the last dimension. The fused early return, sentinel clamp/valid mask, GQA einsum, softmax, budgets and row lifetime are unchanged. K/V float tensors still materialize.

CPU TorchDispatch observes old KV indices[(3,1024)] and new[(3,2,4),(1,2,1)] for n3/Hkv2/K2=4/D128. The old explicit address tensor contains3072 int64 elements(24576bytes); new indexing reuses24 slot IDs plus2 head IDs. This is index-volume evidence, not total peak memory or measured GPU latency.

Three new test methods include48 exact numerical combinations across n1/3,Hkv1/2,G1/4,D8/32 andfp32/fp16/bf16, reordered requests, unequal lengths, duplicate positions, per-row/capacity sentinels. K/V gather values and final outputs match an independent reference exactly. A fused stub verifies the eager gathers remain bypassed. Baseline passes numerical/fused checks and fails only the reduced-index-shape assertion; it is not a numerical bug. With the first-round six cache tests, ordinary and -O each pass9 methods; independent review reran both modes and approved.

Command: python [-O] -m unittest discover -s python/sglang/srt/layers/attention/tli -p 'test_*.py' (isolated candidate contains these two CPU test files). Existing CUDA-only suites not run. This fallback may also run under Graph: actual capture/replay, service integration, sparse kernels, total memory and GPU speed remain unverified. No default/config/experiment changes.
