# Research pipeline architecture contract

This contract describes the implemented Python interfaces, including the
research-grid and boolean-telemetry extension. It is not a clinical validation
specification, treatment authorization, or claim of empirical evidence.

## 1. Intake validation

Reject invalid input explicitly; do not silently coerce it into valid telemetry.

- `EvidenceGatekeeper.validate_inputs` and its `validate_controller_inputs` alias
  accept built-in dictionaries with exactly `metric_name`, `value`, and `unit`.
  Unknown metrics, missing/extra fields and incompatible units are invalid.
- Numeric contracts accept only finite built-in int/float values, excluding bool.
  Integers that cannot be represented as finite floats are invalid. Bounds depend
  on the named metric, not an arbitrary global unit threshold. Signed stress and
  pressure are permitted. Unit conversion is not implemented.
- `feature_present` alone accepts exact bool values under the `boolean` marker.
  This marker is a schema type, not a physical unit. Integers 0/1 and strings are
  not bool telemetry. Missing, unknown and indeterminate observations are invalid,
  never implicitly False. Numeric metrics do not accept bool by changing units.
- Confidence is always numeric, finite and within [0,1], excluding bool.
- Constructor violations raise explicit ValueError/TypeError. Request-level
  validation failures return `INVALID_INPUT` and a reason; unsupported runtime
  models return `UNSUPPORTED_MODEL`, without computation.

All operation inputs must pass their respective contracts before downstream
processing. The runtime validates both named quantities and its loading
convention. Grid logic validates both complete arrays before processing either;
scalar fields validate all samples under a shared metric/unit contract. Rejection
can short-circuit validation, but cannot start partial downstream computation.
Callers must not concurrently mutate input dictionaries or arrays during a call.

## 2. Policy evaluation

Input validity, arithmetic permission and evidence sufficiency are distinct.
Caller confidence is metadata, not scientific proof or authorization. A finite
value such as 0.999 remains caller-declared confidence: additional decimal
precision cannot turn it into certainty, establish calibration, or confer
clinical validity.

Every valid gate decision is `HELD_FOR_REVIEW` with `evidence_sufficient: false`,
`projection_status: UNVERIFIED` and `clinical_use: false`. There is no acceptance
threshold, verified-provenance workflow or clinical approval state. Boolean
observations have exactly the same evidence limitation.

The runtime may perform its specifically documented research arithmetic on valid
inputs while evidence remains insufficient. Grid computations likewise do not
clear the evidence gate. All operation outputs remain explicitly nonclinical.
Current validation and policy run before every gate cache lookup, including hits.

## 3. Research computation

`MasterRuntimeEngine.execute` supports only `stress_to_reference_ratio`: signed
stress divided by strictly positive caller-supplied reference strength, both in
MPa, with `tension_positive` loading. Zero load yields zero, not a sentinel safety
factor. Unsupported atomic-surgery and biological models do not execute. There
are no default material strengths or universal safety criteria.

`TensorEngine` is an optional, separately validated research entry point using
the same gatekeeper; it is not registered in the master runtime dispatcher.
Its historical name does not imply tensor calculus.

- A cube has odd integer dimension n from 3 through 31. The maximum contains
  29,791 points; the cap bounds cubic materialization and shell-sort work.
- For r=(n-1)/2, original coordinates run lexicographically through [-r,r]^3,
  z fastest. Index mapping is `((x+r)*n + (y+r))*n + (z+r)`. The origin index is
  `(n**3-1)//2`; arrays and returned sample values use this flat order.
- Center-out traversal sorts flat indices by exact integer squared radius, then
  original (x,y,z). The point set is antipodally symmetric; adjacent traversal
  positions are not required to be antipodes.
- Fixed rotation rows are e1=(1,-1,0)/sqrt(2), e2=(1,1,-2)/sqrt(6),
  e3=(1,1,1)/sqrt(3). Rotated components are the dot products with these rows.
  This orthonormal right-handed basis has e1 cross e2=e3 and maps (1,1,1) to
  (0,0,sqrt(3)), along +Z. Native floating precision is retained, not rounded.
- Coordinates use abstract grid-step units, with no inferred physical scale.
  Geometry tests allow floating error around 1e-13 on the tested bounded points;
  irrational rotated coordinates are not expected to compare exactly.
- AND/OR/XOR use explicit integer 0/1 masks of exactly n^3 samples, not bool
  telemetry, arbitrary numeric activation or the prototype's `value > 0`.
  Traversal visits center-out indices but never changes elementwise truth tables
  or output alignment.
- Field analysis returns absolute deviations from the center sample, in the
  shared input unit. The center is only a spatial reference, not a healthy
  baseline. Finite input can overflow subtraction/division; such results return
  `CALCULATION_ERROR` without a numeric result/deviation array.

There is no physical vapour-phase/reverse-time model, diagnostic interpretation,
synchronization, honeycomb, Flower of Life, tesseract or hardware integration.

## 4. Result construction and cache isolation

Only **gate decisions** are cached. Runtime calculations, grid generation, mask
outputs and scalar-deviation outputs are recomputed on every call; they have no
computation-result cache. The gatekeeper currently constructs its policy decision
before looking up the cache, so it does not skip a model computation on a miss.

The general design invariant is:

```text
Validate -> Evaluate Current Policy -> Cache Lookup
  hit:  refresh LRU -> return isolated cached data
  miss: compute if applicable -> construct result -> cache isolated copy -> return
```

The actual gatekeeper path specializes it to:

```text
Snapshot -> Validate Payload -> Validate Confidence -> Construct Current Policy Decision
  -> Canonical Cache Identity -> Locked Lookup
     hit:  refresh LRU -> deepcopy cached decision -> return
     miss: deepcopy current decision into cache -> bounded eviction -> return current decision
```

Invalid input returns before lookup and is not cached. Hits do not require
recaching or redundant computation. This intentionally differs from a literal
"Compute on Miss" mandate: no runtime/grid computation cache exists, and gate
policy is deliberately evaluated before lookup. Adding a computation cache would
require an explicit design change with complete operation identities, not an
assumption that scalar gate entries cache array-level results.

Gate identities are SHA256 of canonical JSON with sorted keys, compact
separators and `allow_nan=False`. They include:

- The gate operation and validated metric/value/unit plus confidence.
- Model, metric-contract and evidence-policy versions and cache configuration.
- The complete current policy decision, so changed current decisions cannot
  return a stale differently qualified result.

Boolean JSON values remain distinct from numeric 0/1, and metric/unit identity
also separates their contracts. Reordered dictionary keys have the same identity;
int/float spellings may occupy separate entries. Identity covers every input that
affects the cached gate decision, **not** unsupported claims of full-operation
caching. Internal contracts/configuration are frozen and public configuration
properties are read-only; version changes require corresponding source changes
or a new instance. Private Python attributes are not a security boundary.

The per-instance LRU is bounded by a strictly positive integer capacity. Lookup,
promotion, insertion, eviction and counters are locked. Deep copies isolate
cached state on writes and hits; returned dictionaries themselves are still
mutable, caller-owned data, not immutable dictionaries. Changing a returned
result cannot modify a later cached result. `cache_info()` is detached metadata.
No persistent or cross-instance cache exists.

Runtime results expose ordered execution metadata and versions. Grid outputs
expose operation/model/metric/grid/policy versions and explicit status fields.
These are inspectable records, not signed, tamper-proof audits or independently
verified scientific provenance.

## Verification and change provenance

The original gatekeeper/runtime was introduced in
[PR #1](https://github.com/makm-asset-management/phased-pulse-dx/pull/1).
The all-branch push/pull-request Python 3.10/3.11/3.12 unittest workflow was
introduced in [PR #2](https://github.com/makm-asset-management/phased-pulse-dx/pull/2).
Both changes are merged. Grid and boolean contracts extend that research-only
baseline; their tests exercise rotation, indexing, truth tables, scalar-reference
arithmetic, cache isolation, strict telemetry types and controller rejection
before downstream processing.

Run `python -m unittest discover -s tests -v` for the current suite. Test success
verifies implemented software contracts, not scientific truth or clinical
performance. Consult the current commit's Actions results for CI status rather
than treating documentation as a live deployment or branch-status service.
