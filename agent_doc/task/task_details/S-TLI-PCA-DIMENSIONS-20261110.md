# S-TLI-PCA-DIMENSIONS-20261110

Source: PR #1 head 02fb7a84 on main 5ba097eb. Bounded CPU-only continuation;
no model experiments, GPU execution, baseline changes, or main merge.

The PCA loader checked layers, KV heads and projection rank but not tensor rank
or the input feature dimension. Actual initializer accepted shapes (2,2,2),
(2,2,5,2), and (2,2,4,1,2) for head_dim=4 in normal and optimized Python;
the subsequent real einsum then failed. D=1 can also broadcast in K projection.
Reject any basis not shaped [layers, KV heads, head_dim, rank] before TP slicing.
This matches research/results/calibrate_pca_basis.py's stacked output and the
actual indexer's _k_refine/_q_refine contracts. Valid tensors are unchanged.

Five PCA tests (one new method with four negative cases), plus 19 existing
profile/host/gather tests pass normally and under -O with CPU PyTorch. Existing
valid tests include 15 TP rank slices and 48 gather shape/dtype cases.
Changed-file ruff F401/F821/UP037 and git diff --check pass. Parent independently
reviewed the narrow diff. Full repository hooks, service, GPU/Graph, MLA model
execution and actual speed/quality effects were not tested. No historical
invalid-basis run has been identified. This is fail-fast validation, not evidence
of a performance improvement or invalidation of correctly shaped old results.
