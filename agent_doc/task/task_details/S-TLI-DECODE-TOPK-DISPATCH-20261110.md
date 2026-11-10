# S-TLI-DECODE-TOPK-DISPATCH-20261110

## Scope and confirmed defect

User-authorized two-repository correctness/performance-waste audit, existing draft PR #1.
Baseline PR head: `0f54e322eb8a742d5348d6b304d485f85ef4e99b`.
Target main: `5ba097eb3c15cb3d265bf33d9da474854ec52fe2`; remote currently has no `two-level-indexer` branch.

`TLIIndexer.select_decode_batched` computed `torch.topk(sc1, K1)` unconditionally, then discarded its result whenever the optional eager DeepSelect branch supplied L1 candidates. Move that Torch operation into the existing fallback branch. DeepSelect disabled/unavailable and capture paths retain exactly one Torch L1 topk. Taskmd returns through its separate branch unchanged. No budgets, scores, masks, caches, selection policy, experiment files or guides change.

## Validation

- New full AST-extracted production method CPU test: old source fails four DS-active subcases (batch 1/3, skip false/true), normal and `-O`; one fallback test method passes.
- Candidate: two new methods pass in both modes. Exact output comparison against the CPU Torch fallback uses a deterministic DS stand-in; this is not actual DeepSelect correctness validation.
- Thirty-seven focused test methods pass in each mode: profile/PCA validation 14, metadata 6, gather 3 (48 numerical subcases), short mirrors 6, timer 3, geometry 3, dispatch 2. Compileall and whitespace checks pass.
- Independent read-only reviewer reproduced old four failures and new normal/-O passes, and approved the narrow dispatch change.
- Actual CUDA Graph, DeepSelect extension, GPU kernels, service/distributed execution, quality, GPU time/memory and full repository hooks were not run. No speedup is claimed from this CPU call-count reduction. Ruff/isort/pre-commit are not installed in this executor.

## Remaining device acceptance

In the existing authorized GPU test environment, compare matched DS-off/DS-on eager runs on the same model, prompts, batch/lengths and outputs; profile that DS-on no longer launches the discarded L1 Torch topk. Separately check capture/replay stays on the unchanged Torch fallback. No GPU work was started by this audit.
