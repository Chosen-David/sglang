# S-TLI-SHORT-MIRROR-20261022

## Scope
Base: 363a1087e025e977d0da5964c043539685a7c8b0, extending the existing TLI host snapshot branch. A CPU length mirror shorter than the request list cannot serve the current batch. Previously metadata cached that incomplete list; every layer then used the existing device-length fallback. Resolve the same fallback once in metadata. Complete/padded mirrors, request identity invalidation, row recycling and graph dispatch retain their existing behavior. No budget, protection, selection or attention arithmetic changes.

## Evidence
The normal caller pads/crops request indices, sequence lengths and CPU mirrors together. This is a defensive abnormal-mirror case, not evidence of a frequent production bottleneck. Tests execute the two actual production host-routing method bodies using NumPy tensor doubles, without importing CUDA. Before the fix: 6 methods, 4 failed assertions/subcases in each normal and -O mode. After: 6 methods pass in each mode; four-layer short-mirror example reads device lengths 4 to 1 times, with identical observed lengths/routes/output doubles. Tests include empty/short/complete/padded/missing/malformed mirrors, next-forward refresh, graph bypass and snapshot clearing.

## Limits
No installed Torch in this recovered environment. Real Torch adjacent tests, CUDA Graph execution, numerical attention, full service/full suite and GPU latency were not run. Host read counts are not measured GPU speedup. Independent review reran normal/-O and inspected caller padding/cropping. Existing PR head/ref inconsistency is tracked separately; a branch update alone is not proof PR metadata refreshed.
