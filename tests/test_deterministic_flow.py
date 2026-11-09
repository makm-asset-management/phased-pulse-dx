import unittest
from unittest.mock import patch

from src.core.evidence_gatekeeper import EvidenceGatekeeper
from src.core.master_runtime import MasterRuntimeEngine


def payload(stress=10, strength=20):
    return {
        "stress": {"metric_name": "stress", "value": stress, "unit": "MPa"},
        "reference_strength": {
            "metric_name": "reference_strength", "value": strength, "unit": "MPa",
        },
        "loading_convention": "tension_positive",
    }


class DeterministicFlowTests(unittest.TestCase):
    def test_signed_and_zero_load(self):
        runtime = MasterRuntimeEngine()
        for stress, expected in ((10, 0.5), (-10, -0.5), (0, 0)):
            result = runtime.execute(runtime.OPERATION, payload(stress), 1)
            self.assertEqual(result["status"], "COMPUTED_RESEARCH_ONLY")
            self.assertEqual(result["result"]["value"], expected)
            self.assertEqual(result["result"]["zero_load"], stress == 0)
            self.assertFalse(result["clinical_use"])
            self.assertEqual(result["gate_decisions"][0]["status"], "HELD_FOR_REVIEW")
            self.assertEqual([step["step"] for step in result["audit_trail"]], [
                "validate_stress", "validate_reference_strength",
                "research_arithmetic_policy", "calculate",
            ])

    def test_all_invalid_fields_stop_processor(self):
        malformed = [None, [], {}, dict(payload(), extra=True)]
        for field in payload():
            item = payload()
            del item[field]
            malformed.append(item)
        for field, value in (
            ("loading_convention", []), ("loading_convention", "unknown"),
            ("stress", None), ("reference_strength", {}),
        ):
            item = payload()
            item[field] = value
            malformed.append(item)
        for metric, field, value in (
            ("stress", "value", True), ("stress", "value", float("nan")),
            ("stress", "unit", []), ("stress", "unit", "eV"),
            ("stress", "metric_name", "pressure"),
            ("reference_strength", "value", 0), ("reference_strength", "value", -1),
            ("reference_strength", "unit", "Pa"),
            ("reference_strength", "metric_name", "stress"),
        ):
            item = payload()
            item[metric][field] = value
            malformed.append(item)
        runtime = MasterRuntimeEngine()
        with patch.object(runtime, "_compute_ratio") as compute:
            for item in malformed:
                with self.subTest(item=item):
                    result = runtime.execute(runtime.OPERATION, item, 0.99)
                    self.assertEqual(result["status"], "INVALID_INPUT")
                    self.assertFalse(result["clinical_use"])
                    self.assertNotIn("result", result)
            compute.assert_not_called()

    def test_unsupported_models_never_compute(self):
        runtime = MasterRuntimeEngine()
        with patch.object(runtime, "_compute_ratio") as compute:
            for operation in ("atomic_surgery", "biological", "unknown", [], None, ""):
                result = runtime.execute(operation, payload(), 1)
                self.assertIn(result["status"], ("UNSUPPORTED_MODEL", "INVALID_INPUT"))
                self.assertFalse(result["clinical_use"])
                self.assertNotIn("result", result)
            compute.assert_not_called()

    def test_cache_never_bypasses_controller_or_confidence(self):
        gate = EvidenceGatekeeper()
        runtime = MasterRuntimeEngine(gate)
        first = runtime.execute(runtime.OPERATION, payload(), 0.92)
        second = runtime.execute(runtime.OPERATION, payload(), 0.75)
        self.assertEqual(first["gate_decisions"][0]["declared_confidence"], 0.92)
        self.assertEqual(second["gate_decisions"][0]["declared_confidence"], 0.75)
        runtime.execute(runtime.OPERATION, payload(), 0.75)
        self.assertEqual(gate.cache_info()["hits"], 2)
        with patch.object(runtime, "_compute_ratio") as compute:
            for confidence in (True, -1, float("inf"), 10 ** 1000):
                result = runtime.execute(runtime.OPERATION, payload(), confidence)
                self.assertEqual(result["status"], "INVALID_INPUT")
            with patch.object(gate, "validate_controller_inputs",
                              return_value=(False, "controller rejected")):
                result = runtime.execute(runtime.OPERATION, payload(), 0.75)
                self.assertEqual(result["status"], "INVALID_INPUT")
            compute.assert_not_called()

    def test_determinism_and_runtime_mutation_isolation(self):
        runtime = MasterRuntimeEngine()
        first = runtime.execute(runtime.OPERATION, payload(), 0.9)
        self.assertEqual(first, runtime.execute(runtime.OPERATION, payload(), 0.9))
        first["validated_quantities"]["stress"]["value"] = 999
        first["gate_decisions"][0]["clinical_use"] = True
        later = runtime.execute(runtime.OPERATION, payload(), 0.9)
        self.assertEqual(later["result"]["value"], 0.5)
        self.assertFalse(later["gate_decisions"][0]["clinical_use"])

    def test_nonfinite_calculation_is_explicit_error(self):
        runtime = MasterRuntimeEngine()
        result = runtime.execute(runtime.OPERATION, payload(1e308, 1e-308), 1)
        self.assertEqual(result["status"], "CALCULATION_ERROR")
        self.assertNotIn("result", result)
        self.assertFalse(result["clinical_use"])

    def test_gatekeeper_constructor_type(self):
        with self.assertRaises(TypeError):
            MasterRuntimeEngine(gatekeeper=False)


if __name__ == "__main__":
    unittest.main()
