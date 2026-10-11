# Static cluster tail coverage: verify and stop affected runs

## Public-source finding

Audited source: [tli_indexer.py at e74072be2a3ed478e58ffa4149f9437794727d4c](https://github.com/Chosen-David/sglang/blob/e74072be2a3ed478e58ffa4149f9437794727d4c/two-level-attention/sparse_attn/indexer/tli_indexer.py).

With alpha=beta=0, far_select=sim_greedy or cluster, enable_kmeans=true, skip_far=false, and an initialized cluster cache, the producer at lines 390–399 aligns the true sequence boundary downward. The consumer at lines 916–924 and 1136–1139 uses the padded sequence boundary. Far selection at lines 1230–1238 only uses cached members; the near branch and protected-window updates at lines 1269–1303 do not fill the resulting gap.

Integer counterexample: S=4097, block size=64, prefix=128, sliding window=128. Clusters cover [128,3968), while the consumer far boundary is 4032. The near branch is skipped; the protected sliding window starts at 3969. Token 3968 is therefore never selectable, regardless of its score. Under these conditions the uncovered middle tail has S%64 tokens. Numeric gamma=0 is affected; this is not restricted to gamma=off.

This is a source-level boundary finding independently checked twice, not an observation of a live process or a GPU reproduction.

## Required response

1. Immediately identify any running job using the affected loaded code and effective configuration. Record job identity and relevant source/configuration hashes; do not dump credentials or unrelated process environments.
2. Once the defect is confirmed to affect that job, stop that job immediately, even near completion. Do not stop unrelated jobs solely because this code path exists.
3. Remove affected outputs from valid result sets and invalidate their summaries/cache/downstream references. Use recoverable quarantine or ordinary Git deletion with recoverable history. Keep a minimal invalidation record with cause, scope, hashes and recovery location. Do not irreversibly purge data.
4. Fix the boundary mismatch, pass targeted correctness checks, then rerun only the affected work with new source/run identity within existing authorized resources. Do not reuse contaminated completion markers.
5. Report actual stop, cleanup, fix and rerun outcomes separately. This note does not establish that any job has stopped.

## Minimum acceptance

Test S=4096,4097,4159,4160 with the same block/protection sizes; give the uncovered middle tail high scores and verify it is eligible for selection. Verify true prefix/window coverage, no padding or duplicate selections, and the declared token budget. Cover numeric gamma=0 and gamma=off separately and regress alpha/beta-positive paths. A block-aligned-only test cannot exclude this defect.

No new experiment matrix or additional compute authorization is implied.
