# E65 dimension-reduction ablation (mean of 7 held-out samples)

A=far coarse / B=near coarse / C=fine / D=full pipeline; values = far(near) mass capture; trunc_d32 = no-reduction baseline

## (A) far-coarse score reduction
| 方法 | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| trunc (tail-cut) | 0.365 | 0.523 | 0.649 | **0.804** |
| sup_wsvd | 0.781 | 0.788 | 0.795 | nan |
| sup_grad | 0.813 | 0.801 | 0.819 | nan |
| pca (full-dim) | 0.346 | 0.379 | 0.423 | 0.585 |
| pca (in-subspace) | 0.346 | 0.379 | 0.423 | 0.585 |
| jl (full-dim) | 0.219 | 0.245 | 0.300 | 0.383 |
| jl (in-subspace) | 0.219 | 0.245 | 0.300 | 0.383 |

## (C) fine (refine) score reduction
| 方法 | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| trunc (tail-cut) | 0.513 | 0.595 | 0.666 | **0.804** |
| sup_wsvd | 0.803 | 0.804 | 0.804 | nan |
| sup_grad | 0.758 | 0.758 | 0.759 | nan |
| pca (full-dim) | 0.678 | 0.723 | 0.766 | 0.802 |
| pca (in-subspace) | 0.678 | 0.723 | 0.766 | 0.802 |
| jl (full-dim) | 0.701 | 0.728 | 0.776 | 0.780 |
| jl (in-subspace) | 0.701 | 0.728 | 0.776 | 0.780 |

## (D) full pipeline, same d
| 方法 | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| trunc (tail-cut) | 0.311 | 0.430 | 0.565 | **0.804** |
| sup_wsvd | 0.781 | 0.788 | 0.795 | nan |
| sup_grad | 0.779 | 0.764 | 0.773 | nan |
| pca (full-dim) | 0.332 | 0.368 | 0.425 | 0.597 |
| pca (in-subspace) | 0.332 | 0.368 | 0.425 | 0.597 |
| jl (full-dim) | 0.198 | 0.215 | 0.282 | 0.377 |
| jl (in-subspace) | 0.198 | 0.215 | 0.282 | 0.377 |

## (B) near-coarse score reduction
| 方法 | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| trunc (tail-cut) | 0.296 | 0.305 | 0.329 | **0.306** |
| sup_wsvd | 0.389 | 0.335 | 0.319 | nan |
| sup_grad | 0.302 | 0.308 | 0.320 | nan |
| pca (full-dim) | 0.289 | 0.317 | 0.306 | 0.293 |
| pca (in-subspace) | 0.289 | 0.317 | 0.306 | 0.293 |
| jl (full-dim) | 0.322 | 0.303 | 0.314 | 0.323 |
| jl (in-subspace) | 0.322 | 0.303 | 0.314 | 0.323 |

