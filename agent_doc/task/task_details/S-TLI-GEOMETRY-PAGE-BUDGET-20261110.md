# S-TLI-GEOMETRY-PAGE-BUDGET-20261110

Source: draft PR #1 head 7f610505c96e4ca554b4fae7f194ea1f068b80aa,
main 5ba097eb. Remote has no two-level-indexer branch. No PR comments or
new remote writer observed. Isolated checkout; human TASK and guides unchanged.

Fixed geometry accepted K1=2, NEAR_PAGES=3, NEAR_LEN=256, NEAR_TOKENS=0
(or off). The actual host _taskmd_regions returned near=3, far=0, with four
available near pages. Thus the configured near quota alone exceeded the total
K1 budget. The three selection paths all clamp the far difference to zero,
not the near quota. The contract is near+far=K1 excluding boundary/protected
bypass pages; this example has aligned boundaries and no bypass page.

Reject NEAR_PAGES>K1 at profile construction rather than silently clamping a
user's requested geometry. Valid numeric/off quotas, zero/endpoints and legacy
configuration remain unchanged. No scoring, kernels, experiment records or
attention-quality results changed. Historical affected runs are not identified.

New tests execute the actual config module and AST-loaded scalar region method.
Before fix: two negative subcases fail in normal and -O; after fix all three
methods pass. Existing profile (4), phase timer (3), short host mirror (6) tests
also pass in normal and -O: 16 methods in each mode. compileall and diff whitespace
checks pass. No Torch in this environment: Torch numerical/TP tests not rerun.
Full repository hooks, full service, CUDA Graph, GPU, quality and speed unverified.
Independent review/publication receipt is recorded with the PR follow-up.
