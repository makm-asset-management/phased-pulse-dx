"""Controller-first dispatch for one explicitly bounded research calculation."""

import math

from .evidence_gatekeeper import EvidenceGatekeeper


class MasterRuntimeEngine:
    """No clinical, biological, atomic-surgery, or hardware execution."""

    OPERATION = "stress_to_reference_ratio"
    OPERATION_VERSION = "stress-ratio-v1"

    def __init__(self, gatekeeper=None):
        if gatekeeper is not None and type(gatekeeper) is not EvidenceGatekeeper:
            raise TypeError("gatekeeper must be an EvidenceGatekeeper")
        self._gatekeeper = gatekeeper if gatekeeper is not None else EvidenceGatekeeper()

    def execute(self, operation, payload, confidence_score):
        gate = self._gatekeeper
        result = {
            "operation": operation if type(operation) is str else None,
            "operation_version": self.OPERATION_VERSION if operation == self.OPERATION else None,
            "model_version": gate.model_version,
            "contract_version": gate.contract_version,
            "evidence_policy_version": gate.evidence_policy_version,
            "status": "INVALID_INPUT",
            "input_valid": False,
            "clinical_use": False,
            "projection_status": "UNVERIFIED",
            "gate_decisions": [],
            "validated_quantities": {},
            "audit_trail": [],
            "limitations": [
                "Arithmetic only; no material, geometry, fatigue, or failure model.",
                "Reference strength is caller-supplied, not independently verified.",
                "No safety threshold, clinical approval, or physical intervention.",
            ],
        }
        if type(operation) is not str or not operation.strip():
            result["reason"] = "operation must be a nonempty string"
            return result
        if operation != self.OPERATION:
            result["status"] = "UNSUPPORTED_MODEL"
            result["reason"] = "Only stress_to_reference_ratio is supported"
            result["audit_trail"].append({"step": "dispatch_validation", "status": "REJECTED"})
            return result
        if type(payload) is not dict or set(payload) != {
            "stress", "reference_strength", "loading_convention"
        }:
            result["reason"] = (
                "payload must contain exactly stress, reference_strength, and loading_convention"
            )
            return result
        if type(payload["loading_convention"]) is not str or (
            payload["loading_convention"] != "tension_positive"
        ):
            result["reason"] = "loading_convention must be 'tension_positive'"
            return result
        for metric in ("stress", "reference_strength"):
            quantity = payload[metric]
            decision = gate.evaluate_evidence_gate(quantity, confidence_score)
            result["gate_decisions"].append(decision)
            result["audit_trail"].append({
                "step": f"validate_{metric}", "status": decision["status"],
                "model_version": decision["model_version"],
                "contract_version": decision["contract_version"],
                "evidence_policy_version": decision["evidence_policy_version"],
            })
            if not decision["input_valid"]:
                result["reason"] = f"{metric}: {decision['reason']}"
                return result
            validated = decision["validated_quantity"]
            if validated["metric_name"] != metric:
                result["reason"] = f"{metric} must use metric_name {metric!r}"
                return result
            result["validated_quantities"][metric] = validated
        result["input_valid"] = True
        result["research_arithmetic_permitted"] = True
        result["audit_trail"].append({
            "step": "research_arithmetic_policy", "status": "PERMITTED_NONCLINICAL_ONLY",
            "evidence_policy_version": gate.evidence_policy_version,
        })
        stress = result["validated_quantities"]["stress"]["value"]
        strength = result["validated_quantities"]["reference_strength"]["value"]
        ratio = self._compute_ratio(stress, strength)
        if not math.isfinite(ratio):
            result["status"] = "CALCULATION_ERROR"
            result["reason"] = "ratio exceeds finite floating-point representation"
            result["audit_trail"].append({"step": "calculate", "status": "NONFINITE_RESULT"})
            return result
        result["status"] = "COMPUTED_RESEARCH_ONLY"
        result["reason"] = "Signed stress divided by caller-supplied reference strength"
        result["result"] = {"value": ratio, "unit": "dimensionless", "zero_load": stress == 0}
        result["assumptions"] = {
            "loading_convention": "tension_positive",
            "compression_is_negative": True,
            "reference_strength_is_positive": True,
        }
        result["audit_trail"].append({
            "step": "calculate", "status": "COMPUTED_RESEARCH_ONLY",
            "operation_version": self.OPERATION_VERSION,
        })
        return result

    @staticmethod
    def _compute_ratio(stress, reference_strength):
        return stress / reference_strength
