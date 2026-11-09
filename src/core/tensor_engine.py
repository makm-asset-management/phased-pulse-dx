"""Research-only centered cubic geometry, binary masks and scalar deviations."""

import math

from .evidence_gatekeeper import EvidenceGatekeeper


_MAX_DIMENSION = 31
_BASIS = (
    (1 / math.sqrt(2), -1 / math.sqrt(2), 0.0),
    (1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)),
    (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
)


class TensorEngine:
    """A scalar-grid utility, not tensor calculus or a clinical model.

    Inputs are in original lexicographic grid order. Methods return detached
    results. Callers must not mutate supplied arrays during a call.
    """

    def __init__(self, dimension=3, gatekeeper=None):
        if type(dimension) is not int or not 3 <= dimension <= _MAX_DIMENSION or dimension % 2 == 0:
            raise ValueError("dimension must be an odd built-in integer from 3 through 31")
        if gatekeeper is not None and type(gatekeeper) is not EvidenceGatekeeper:
            raise TypeError("gatekeeper must be an EvidenceGatekeeper")
        self._dimension = dimension
        self._gatekeeper = gatekeeper if gatekeeper is not None else EvidenceGatekeeper()

    @property
    def dimension(self):
        return self._dimension

    @property
    def point_count(self):
        return self.dimension ** 3

    @property
    def center_index(self):
        return (self.point_count - 1) // 2

    @property
    def basis(self):
        return _BASIS

    def _response(self, operation):
        return {
            "operation": operation,
            "operation_version": "center-grid-v1",
            "model_version": self._gatekeeper.model_version,
            "contract_version": self._gatekeeper.contract_version,
            "grid_contract_version": "binary-scalar-grid-v1",
            "evidence_policy_version": self._gatekeeper.evidence_policy_version,
            "status": "INVALID_INPUT",
            "input_valid": False,
            "clinical_use": False,
            "projection_status": "UNVERIFIED",
            "evidence_sufficient": False,
            "gate_status": "HELD_FOR_REVIEW",
            "dimension": self.dimension,
            "center_index": self.center_index,
            "limitations": [
                "Research geometry and arithmetic only, not a physical or clinical model.",
                "The center is a spatial reference, not a healthy baseline.",
            ],
        }

    def _original_coordinates(self):
        radius = self.dimension // 2
        return tuple(
            (x, y, z)
            for x in range(-radius, radius + 1)
            for y in range(-radius, radius + 1)
            for z in range(-radius, radius + 1)
        )

    def _traversal_indices(self):
        points = self._original_coordinates()
        return tuple(sorted(
            range(self.point_count),
            key=lambda index: (sum(value * value for value in points[index]), points[index]),
        ))

    def generate_vertex_grid(self):
        """Return coordinates in lexicographic order and a center-out index map."""
        result = self._response("vertex_aligned_grid")
        original = self._original_coordinates()
        rotated = tuple(
            tuple(math.fsum(component * axis for component, axis in zip(point, row))
                  for row in _BASIS)
            for point in original
        )
        result.update(
            status="COMPUTED_RESEARCH_ONLY", input_valid=True,
            original_coordinates=original, rotated_coordinates=rotated,
            traversal_indices=self._traversal_indices(), basis=_BASIS,
            coordinate_unit="grid_step",
            reason="Right-handed coordinate rotation; no physical scale is inferred.",
        )
        return result

    def _validate_array(self, values, metric_name, unit, confidence_score, binary=False):
        if type(values) not in (list, tuple) or len(values) != self.point_count:
            return None, f"values must be a list or tuple of exactly {self.point_count} samples"
        snapshot = tuple(values)
        for index, value in enumerate(snapshot):
            if type(value) is bool:
                return None, f"sample {index}: grid arrays require numeric samples, excluding bool"
            if binary and (type(value) is not int or value not in (0, 1)):
                return None, f"sample {index}: binary masks require integer 0 or 1, excluding bool"
            decision = self._gatekeeper.evaluate_evidence_gate(
                {"metric_name": metric_name, "value": value, "unit": unit}, confidence_score
            )
            if not decision["input_valid"]:
                return None, f"sample {index}: {decision['reason']}"
        return snapshot, None

    def evaluate_center_out_logic(self, input_a, input_b, operation, confidence_score=0.0):
        """Elementwise AND/OR/XOR on explicit integer 0/1 masks."""
        result = self._response("binary_grid_logic")
        if type(operation) is not str or operation not in ("AND", "OR", "XOR"):
            result["reason"] = "logic operation must be AND, OR, or XOR"
            return result
        a, error = self._validate_array(
            input_a, "ratio", "dimensionless", confidence_score, binary=True
        )
        if error is not None:
            result["reason"] = f"input_a: {error}"
            return result
        b, error = self._validate_array(
            input_b, "ratio", "dimensionless", confidence_score, binary=True
        )
        if error is not None:
            result["reason"] = f"input_b: {error}"
            return result
        indices = self._traversal_indices()
        values = self._process_logic(a, b, operation, indices)
        result.update(
            status="COMPUTED_RESEARCH_ONLY", input_valid=True, values=values,
            logic_operation=operation, unit="dimensionless", traversal_indices=indices,
            gate_status="HELD_FOR_REVIEW", declared_confidence=confidence_score,
            reason="Validated binary-mask computation; confidence is not evidence.",
        )
        return result

    @staticmethod
    def _process_logic(a, b, operation, indices):
        values = [0] * len(a)
        for index in indices:
            if operation == "AND":
                values[index] = a[index] & b[index]
            elif operation == "OR":
                values[index] = a[index] | b[index]
            else:
                values[index] = a[index] ^ b[index]
        return tuple(values)

    def analyze_field(self, values, metric_name, unit, confidence_score=0.0):
        """Absolute deviations from the center sample, in the shared input unit."""
        result = self._response("scalar_center_deviation")
        snapshot, error = self._validate_array(values, metric_name, unit, confidence_score)
        if error is not None:
            result["reason"] = error
            return result
        deviations = self._process_deviations(snapshot)
        if any(not math.isfinite(value) for value in deviations):
            result.update(
                status="CALCULATION_ERROR", input_valid=True,
                reason="Absolute deviation exceeds finite floating-point representation.",
            )
            return result
        result.update(
            status="COMPUTED_RESEARCH_ONLY", input_valid=True,
            absolute_deviations=deviations,
            center_value=snapshot[self.center_index], metric_name=metric_name, unit=unit,
            traversal_indices=self._traversal_indices(),
            gate_status="HELD_FOR_REVIEW", declared_confidence=confidence_score,
            reason="Absolute spatial-reference deviations, not pathology or diagnosis.",
        )
        return result

    def _process_deviations(self, values):
        center = values[self.center_index]
        return tuple(abs(float(value) - float(center)) for value in values)
