# S-TLI-HOST-SNAPSHOT-20261021

Base: 5ba097eb3c15cb3d265bf33d9da474854ec52fe2. User requested confirmed correctness/performance fixes in both repositories, with separate draft PRs and no automatic merge. No GPU execution, experiments, budget changes or method changes.

The eager metadata hook could return early while retaining its previous request/length snapshot. In forward_decode, request identity mismatch correctly fell back to current requests, but cached lengths were independently consumed. That could use stale token slices and dense/sparse dispatch. Normal initialized decode refreshes already; production incidence of early-exit/mismatch cases is unknown.

Clear both cache fields on eager metadata entry. Consume cached lengths only with the matching request snapshot. When seq_lens_cpu is absent or unusable, take one seq_lens.tolist snapshot during metadata preparation instead of once per layer. Graph body and row-recycling rules are unchanged.

Actual CPU regression executes the production init_forward_metadata and forward_decode bodies via AST extraction; attention is a token-slice probe, not full model execution. Six test methods cover four metadata early exits, request-identity mismatch, repeated layers without CPU mirror, CPU mirror and next-forward refresh, Graph dispatch bypass, and existing row recycling. The pinned pre-fix source gives six failures (four early-exit subcases plus mismatch and redundant reads); fixed source passes six methods normally and under -O. For four layers with no CPU mirror, observed length tolist calls decrease from four to one. This is not a CUDA timing result.

Command: python [-O] python/sglang/srt/layers/attention/tli/test_host_metadata_cache.py
Environment: Python 3.12, PyTorch 2.8.0+cpu. No model weights or GPU packages downloaded. Existing test_b1_b2_fix.py requires CUDA at module level and was not run. Full SGLang imports, sparse kernels, capture/replay, real service output, quality and throughput remain unverified. Independent read-only review reran all six methods in both modes and approved the bounded fix; full CUDA validation remains pending.
