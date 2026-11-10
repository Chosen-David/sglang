# S-TLI-PCA-VALIDATION-20261110

Date: 2026-11-10

## Scope and source

Source: main 5ba097eb, existing draft PR #1 head ed8302db. No remote
`two-level-indexer` branch. This task continues the same PR; no main merge.
Only PCA basis initialization validation changes, not the selector or runtime kernels.

## Reproduction and change

Actual `_init_from_runner` is AST-loaded with a CPU Tensor returned by a bounded
mock of `torch.load` and a controlled distributed metadata provider. Under
`python -O`, the previous production method accepts all four invalid cases:
wrong layer count, TP=1 wrong KV head count, nondivisible global KV head count,
and wrong per-rank KV head count. Ordinary Python rejects them with assertions.
The four checks now raise `ValueError` explicitly in both modes. Valid PCA
slicing, rank checks, and model results are not intentionally changed.

## Verification

- Four new CPU tests: scalar rejection plus four layout rejection subcases, 15 TP rank slices over TP 1/2/4/8
  with exact reconstruction, and existing rank-mismatch rejection.
- Existing profile 4, metadata 6, sparse gather 3 and short-mirror 6 tests pass
  normally and under -O with Torch 2.8.0+cpu. Sparse gather includes 48
  dtype/shape numerical subcases. This reruns earlier PR tests that were not
  runnable in the previous environment without Torch.
- Changed-file ruff F401/F821/UP037 and whitespace checks pass.
- Full service, full repository hooks, CUDA, Graph capture/replay and GPU
  performance were not run. No historical affected experiment is identified.
- Independent review and publication receipt are tracked in the PR update.

Independent parent review accepted the change and requested a scalar-basis regression;
that case now raises ValueError rather than an incidental IndexError. TP rank
metadata comes from get_parallel; rank-range correctness is not newly validated.
