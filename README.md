# phased-pulse-dx

Research-only deterministic input validation and evidence gating. **Not a clinical
diagnostic device, treatment controller, or validated scientific model.** A
caller-provided confidence score is not empirical proof. All returned decisions
include `clinical_use: false`; valid measurements remain `HELD_FOR_REVIEW` with
`evidence_sufficient: false` and `projection_status: "UNVERIFIED"`.

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
rejected. Confidence must meet the same numeric constraints and lie in [0, 1].
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

These are input contracts, not biological material limits. No unit conversions
are implemented: e.g. `Pa` is rejected rather than silently interpreted as `MPa`.
No arbitrary upper bound is invented for signed quantities. Finite representation
and metric-specific bounds are enforced.

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
