"""Input contracts and research-only policy; confidence is not evidence."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock
from types import MappingProxyType


@dataclass(frozen=True)
class _Contract:
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False


_CONTRACTS = MappingProxyType({
    "pressure": _Contract("MPa"),
    "stress": _Contract("MPa"),
    "reference_strength": _Contract("MPa", 0, exclusive_minimum=True),
    "photon_energy": _Contract("eV", 0),
    "frequency": _Contract("Hz", 0),
    "duration": _Contract("ms", 0),
    "percentage": _Contract("percent", 0, 100),
    "ratio": _Contract("dimensionless", 0),
})


@dataclass(frozen=True)
class _Configuration:
    max_cache_size: int
    model_version: str
    contract_version: str = "metrics-v1"
    evidence_policy_version: str = "research-only-v1"


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class EvidenceGatekeeper:
    """Thread-safe bounded LRU with read-only public configuration.

    Only built-in dictionaries, strings, ints and floats are accepted. Caller
    dictionaries must not be modified concurrently while an API call is running.
    """

    def __init__(self, max_cache_size=1000, model_version="v1.0.0"):
        if type(max_cache_size) is not int or max_cache_size <= 0:
            raise ValueError("max_cache_size must be a strictly positive integer")
        if type(model_version) is not str or not model_version.strip():
            raise ValueError("model_version must be a nonempty string")
        self._configuration = _Configuration(max_cache_size, model_version)
        self._cache = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    @property
    def model_version(self):
        return self._configuration.model_version

    @property
    def contract_version(self):
        return self._configuration.contract_version

    @property
    def evidence_policy_version(self):
        return self._configuration.evidence_policy_version

    @property
    def max_cache_size(self):
        return self._configuration.max_cache_size

    def cache_info(self):
        with self._lock:
            return {
                "size": len(self._cache), "max_size": self.max_cache_size,
                "hits": self._hits, "misses": self._misses,
            }

    def validate_inputs(self, payload):
        if type(payload) is not dict:
            return False, "payload must be a built-in dictionary"
        if set(payload) != {"metric_name", "value", "unit"}:
            return False, "payload must contain exactly metric_name, value, and unit"
        metric = payload["metric_name"]
        if type(metric) is not str or not metric.strip():
            return False, "metric_name must be a nonempty string"
        if metric not in _CONTRACTS:
            return False, f"unknown metric_name: {metric!r}"
        unit = payload["unit"]
        if type(unit) is not str or not unit:
            return False, "unit must be an explicit nonempty string"
        contract = _CONTRACTS[metric]
        if unit != contract.unit:
            return False, f"{metric} requires unit {contract.unit!r}; conversions are unsupported"
        value = payload["value"]
        if not _finite_number(value):
            return False, "value must be a finite built-in int or float, excluding bool"
        if contract.minimum is not None and (
            value < contract.minimum
            or (contract.exclusive_minimum and value == contract.minimum)
        ):
            relation = "greater than" if contract.exclusive_minimum else "at least"
            return False, f"{metric} must be {relation} {contract.minimum}"
        if contract.maximum is not None and value > contract.maximum:
            return False, f"{metric} must be at most {contract.maximum}"
        return True, "valid input"

    def validate_controller_inputs(self, payload):
        """Recognizable alias for callers using the controller-first prototype."""
        return self.validate_inputs(payload)

    def _base_result(self, status, reason, input_valid):
        return {
            "status": status, "reason": reason, "input_valid": input_valid,
            "evidence_sufficient": False, "clinical_use": False,
            "projection_status": "UNVERIFIED",
            "model_version": self.model_version,
            "contract_version": self.contract_version,
            "evidence_policy_version": self.evidence_policy_version,
        }

    def evaluate_evidence_gate(self, payload, confidence_score):
        snapshot = dict(payload) if type(payload) is dict else payload
        valid, reason = self.validate_controller_inputs(snapshot)
        if not valid:
            return self._base_result("INVALID_INPUT", reason, False)
        if not _finite_number(confidence_score) or not 0 <= confidence_score <= 1:
            return self._base_result(
                "INVALID_INPUT", "confidence_score must be finite and in [0, 1], excluding bool", False
            )
        # Policy is applied before consulting the cache, even on a hit.
        result = self._base_result(
            "HELD_FOR_REVIEW",
            "Caller-declared confidence is not empirical evidence or clinical validation.",
            True,
        )
        result["validated_quantity"] = snapshot
        result["declared_confidence"] = confidence_score
        canonical = json.dumps({
            "operation": "evaluate_evidence_gate",
            "configuration": self._configuration.__dict__,
            "parameters": {"payload": snapshot, "confidence_score": confidence_score},
            "policy_decision": result,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return deepcopy(self._cache[key])
            self._misses += 1
            self._cache[key] = deepcopy(result)
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
            return result
