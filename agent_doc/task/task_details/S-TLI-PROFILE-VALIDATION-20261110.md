# S-TLI-PROFILE-VALIDATION-20261110

Date: 2026-11-10
Source: authorized two-repository correctness/performance audit; CPU only.

## Verified scope

Target Chosen-David/sglang. Remote two-level-indexer does not exist (branch
lookup and ls-remote); main remains 5ba097eb3c15cb3d265bf33d9da474854ec52fe2.
Continue existing draft PR #1, head 8d553336749a3c31432c4c0e52523e802f98fd10.
No PR comments or newer remote writer were observed. Human root TASK.md and
agent_doc/guide are read-only; this is the existing AI task index.

## Reproduction and change

Actual config module, without Torch/CUDA:
- coarse_dim=31, head_dim=128 returns only 30 coordinates, despite pool width
  being coarse_dim. This invalid configuration was accepted in normal Python.
- python -O accepts unknown FAR_METHOD because assert disappears; downstream
  comparisons then silently use another scorer. delta=65, head_dim=128 also
  passes and emits a negative coordinate under -O.

Use explicit ValueError for invalid method names and coordinate geometry.
Require positive even head_dim for both rotate_half coordinate builders;
coarse_dim must be positive even and <= head_dim; delta in [1, head_dim/2].
Valid default and boundary coordinates are unchanged. No kernel/scoring,
protection, budget, cache or numerical attention changes.

## Checks

New four-method test executes the actual config module (not an extracted or
copied implementation), with invalid names/shape cases and exact valid indices.
Before fix: ordinary run fails with 20 subtest failures. Manual optimized
reproduction independently demonstrates admitted invalid name/negative index.
After fix: 4/4 normal and -O. Existing short-host-length suite: 6/6 each mode.
compileall of TLI directory and git diff --check pass. Independent review runs
separately; see PR follow-up for its result and exact publication head.

No installed Torch; prior Torch numerical suites, full SGLang imports, service,
GPU capture/replay/quality/latency and broad CI are not verified. This fix does
not claim GPU speedup or invalidate any historical result: no affected run was
identified. Remaining backend/indexer assertions are outside this narrow patch.

Independent review: both suites pass normal/-O; old/new production modules
match across four legal method combinations and 1,324 legal dimension cases.
Independent old-version negative controls reproduce all three reported faults.
Pinned ruff 0.15.1 F401/F821/UP037 checks pass for both changed Python files;
isort 7.0.0 and ruff formatted the new test only. Full repository pre-commit
hooks not run; pre-existing config.py formatting was not rewritten wholesale.
