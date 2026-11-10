# S-TLI-TIMING-INTERVAL-20261110

Date: 2026-11-10

Scope: diagnostic report arithmetic only, based on PR #1 head 0fa221a5;
main remains 5ba097eb. No selector, budget, cache or kernel changes.

The `_PhaseTimer` clears phase accumulators every 64 reporting calls but divided
the interval total by its lifetime counter. Repeated identical 1 ms samples
therefore printed 1.00, 0.50 and 0.33 ms/step. Use one report interval constant for
both cadence and denominator; retain the cumulative `steps=` counter and existing
log format. Enable conditions and call sites are unchanged.

Regression: new three-test CPU suite AST-loads the actual production timer.
The old code fails two tests and passes one; the fix passes all three normally
and with Python -O. Adjacent profile/PCA/host-metadata/short-mirror/gather suites
bring the total to 27 test methods in each mode using CPU Torch 2.15.0.
Checks: changed-file F401/F821/UP037, formatted new test, compileall and git diff
whitespace checks. Parent independently reviewed arithmetic and call-site scope.

Important limits: the counter increments once per eager decode-layer batch call,
not once per complete model token; Graph bypasses this timer. Prefill index work
can contribute to its phase accumulator. Existing phase timing uses wall-clock
measurements and synchronization and is not a GPU event benchmark. This patch
only fixes report denominator drift, not those instrumentation limitations. No
historical performance result is reinterpreted; no GPU, model-quality, serving,
full repository or CUDA Graph validation was run. No speedup claim.
