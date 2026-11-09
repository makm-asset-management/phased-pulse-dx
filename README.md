# phased-pulse-dx

Research-only deterministic input validation and evidence gating. **Not a clinical
diagnostic device, treatment controller, or validated scientific model.** A
caller-provided confidence score is not empirical proof. All returned decisions
include `clinical_use: false`; valid measurements remain `HELD_FOR_REVIEW` with
`evidence_sufficient: false` and `projection_status: "UNVERIFIED"`.

See the [architecture contract](docs/architecture_contract.md) for implemented
validation/policy layers, actual cache paths, and research-only guarantees.

## Requirements and tests

Python 3.10+; standard library only. From the repository root:

```powershell
python -m unittest discover -s tests -v
```

## Gatekeeper API

```python
from src.core.evidence_gatekeeper import EvidenceGatekeeper

gate = EvidenceGatekeeper(max_cache_size=1000, model_version="v1.0.0")
measurement = {"metric_name": "stress", "value": -12.0, "unit": "MPa"}
valid, reason = gate.validate_inputs(measurement)
decision = gate.evaluate_evidence_gate(measurement, confidence_score=0.92)
assert decision["status"] == "HELD_FOR_REVIEW"
assert decision["clinical_use"] is False
```

`validate_controller_inputs` aliases `validate_inputs`. Payloads must be built-in
dictionaries containing **exactly** `metric_name`, `value`, and `unit`. Names and
units are case-sensitive; unknown fields/metrics and mismatched units are
explicitly rejected. Values are built-in `int` or `float`, excluding `bool`, and
must be representable as finite floats. Huge integers, NaN and infinity are
rejected. The separate `feature_present` boolean contract described below is the
only exception to numeric values. Confidence always meets the numeric constraints and lies in [0, 1].
No confidence threshold constitutes evidence sufficiency.

| Metric | Required unit | Range |
|---|---|---|
| `pressure` | `MPa` | Signed finite value (relative to a caller-defined reference) |
| `stress` | `MPa` | Signed finite value |
| `reference_strength` | `MPa` | Strictly greater than zero |
| `photon_energy` | `eV` | Greater than or equal to zero |
| `frequency` | `Hz` | Greater than or equal to zero |
| `duration` | `ms` | Greater than or equal to zero |
| `percentage` | `percent` | 0 through 100 inclusive |
| `ratio` | `dimensionless` | Greater than or equal to zero |
| `feature_present` | `boolean` | Exact Python `True` or `False` only |

These are input contracts, not biological material limits. No unit conversions
are implemented: e.g. `Pa` is rejected rather than silently interpreted as `MPa`.
No arbitrary upper bound is invented for signed quantities. Finite representation
and metric-specific bounds are enforced.

`boolean` is a schema marker, not a physical unit. `feature_present` is a neutral
research flag, not a clinical test. Integers 0/1, strings, missing/unknown or
indeterminate values are rejected, never coerced to False. Changing a numeric
metric's unit to `boolean` does not permit bool values. Boolean telemetry remains
held for review and cannot establish diagnosis or evidence quality. Boolean and
numeric cache entries remain distinct. These contracts are versioned `metrics-v2`.

## Controller-first runtime

```python
from src.core.master_runtime import MasterRuntimeEngine

runtime = MasterRuntimeEngine(gate)
result = runtime.execute(
    "stress_to_reference_ratio",
    {
        "stress": {"metric_name": "stress", "value": -12.0, "unit": "MPa"},
        "reference_strength": {
            "metric_name": "reference_strength", "value": 24.0, "unit": "MPa"
        },
        "loading_convention": "tension_positive",
    },
    confidence_score=0.92,
)
assert result["result"]["value"] == -0.5
assert result["status"] == "COMPUTED_RESEARCH_ONLY"
```

This is the only supported operation. All operation fields are required; unknown
fields fail. The two quantities must name their respective metrics. Validation
and nonclinical policy execute before calculation, including when gate decisions
are cached. `tension_positive` means compression is negative. The calculation
is signed stress / positive caller-supplied reference strength; zero load returns
0 with `zero_load: true`, never a sentinel safety factor. Nonfinite calculation
results produce `CALCULATION_ERROR` without a numeric result.

Arithmetic is permitted for research on validated inputs, **not** because evidence
is sufficient. The gate remains held for review. There is no material lookup,
default tissue strength, universal safety threshold, or inference about failure.
`atomic_surgery`, biological operations, and all other unknown operations return
`UNSUPPORTED_MODEL` without computation. No electron changes, hardware execution,
blueprint constants, or prior speculative subsystems are implemented.

Results carry deterministic status codes, model/contract/policy and operation
versions, gate decisions, validated quantities, assumptions/limitations, and an
ordered `audit_trail`. This is inspectable execution metadata, not a signed,
tamper-proof audit log or independently verified provenance.

## Cache and policy

`max_cache_size` must be a positive built-in integer (not bool); `model_version`
must be a nonblank string. Configuration properties are read-only and internal
configuration/contracts are frozen. There are no mutable confidence thresholds
or evidence metadata acceptance switches. A new instance is required for a new
model version; contract/policy changes require a corresponding version change
in source.

Valid gate decisions use an instance-local bounded LRU. SHA256 keys cover
canonical JSON (`sort_keys=True`, `allow_nan=False`) including the operation,
validated metric/value/unit, confidence, model/contract/policy versions, cache
configuration and current policy decision. Reordered dictionary keys produce
the same identity; numeric spellings such as 1 and 1.0 may occupy separate
entries. Invalid inputs are not cached. Validation and policy run before every
lookup. Reads/writes/eviction and counters are locked; all cached results are
deep-copied on storage/return. `cache_info()` returns a detached size/hit/miss
snapshot. There is no persistent cache and the runtime does not cache calculations.

Cache operations are thread-safe. Callers must not mutate supplied dictionaries
concurrently with an API call. Returned dictionaries are caller-owned. This
library is not a security boundary against callers modifying private Python
attributes or source code.

## Center-origin research grid

`src.core.tensor_engine.TensorEngine` is a separately validated research entry
point; `MasterRuntimeEngine.execute` retains its existing single-operation
dispatch. The historical module name does **not** imply tensor calculus.

```python
from src.core.tensor_engine import TensorEngine

engine = TensorEngine(dimension=3, gatekeeper=gate)
grid = engine.generate_vertex_grid()
mask = [0] * engine.point_count
mask[engine.center_index] = 1
logic = engine.evaluate_center_out_logic(mask, mask, "XOR", confidence_score=0.9)
field = engine.analyze_field(
    list(range(engine.point_count)), "stress", "MPa", confidence_score=0.9
)
assert field["center_value"] == 13
assert field["absolute_deviations"][engine.center_index] == 0
```

Dimension must be an odd built-in integer from 3 through **31** (bool excluded).
The cap limits a materialized cube to 29,791 points; point storage grows as n^3
and shell sorting costs O(n^3 log(n^3)). This is a conservative software resource
bound, not a physical limit or a claim that all environments meet a memory/time
budget. Construction is lazy: inspecting the maximum dimension does not allocate
the cube. Dimension properties are read-only.

For radius r=(n-1)/2, original integer coordinates span [-r,r] on each axis,
in lexicographic (x,y,z) order: z varies fastest. The flat index of (x,y,z) is
`((x+r)*n + (y+r))*n + (z+r)`; the origin's flat index is `(n**3-1)//2`.
Input arrays and returned values/deviations use **this original flat order**.
`traversal_indices` lists these flat indices sorted by exact integer squared
radius, then original (x,y,z), with the origin first. The traversal is not a 1D
midpoint walk or a promise of antipodal adjacency; the point set is antipodally
symmetric.

`generate_vertex_grid()` returns original/rotated coordinates in flat order.
The fixed right-handed orthonormal rotation has rows:

```text
e1 = ( 1, -1,  0) / sqrt(2)
e2 = ( 1,  1, -2) / sqrt(6)
e3 = ( 1,  1,  1) / sqrt(3)
rotated(p) = (dot(e1,p), dot(e2,p), dot(e3,p))
```

Thus e1 cross e2 = e3 and (1,1,1) maps onto positive Z at (0,0,sqrt(3)).
Coordinates retain native floating-point precision without decimal rounding.
Coordinates are in abstract `grid_step` units; spacing and physical scale are
not inferred. Tests use absolute floating comparison tolerance around 1e-13
for this bounded grid, not exact equality of irrational rotated coordinates.

Logic accepts two lists/tuples of exactly n^3 **integer 0/1** samples and a
case-sensitive `AND`, `OR`, or `XOR`. This is an explicit binary-mask contract,
not the prototype's numeric `value > 0` activation and not boolean telemetry:
bool, float 0.0/1.0, arbitrary numeric activations and indeterminate values fail.
Both masks are fully validated before processing begins. Samples pass through
the gatekeeper's dimensionless ratio contract as well as the stricter binary
mask check. Computation visits center-out indices but stores results by original
flat index.

Field analysis accepts exactly n^3 homogeneous finite numeric samples under one
shared gatekeeper metric/unit contract (no boolean fields). Every sample passes
the gatekeeper before calculation. The output is `abs(value-center_value)` in
the input unit; it is **absolute deviation from a spatial reference**, not
pathology, a healthy baseline, synchronization, or a diagnosis. Nonfinite
subtraction results return `CALCULATION_ERROR` without deviations.

Malformed arrays, invalid confidence or failed controller validation produce
`INVALID_INPUT` without downstream processing, even after cache hits. Valid
logic/field operations report `COMPUTED_RESEARCH_ONLY` with
`gate_status: HELD_FOR_REVIEW`; all outputs retain `clinical_use: false`,
`evidence_sufficient: false`, operation/model/contract/policy versions, and
`projection_status: UNVERIFIED`. Geometry performs its dimension validation
before generation and makes no measurement-evidence claim. Caller confidence is
not evidence. No reverse vapour-phase physics, honeycomb, Flower of Life,
tesseract, or clinical diagnostic model is implemented.
