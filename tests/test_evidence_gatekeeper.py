import concurrent.futures
import unittest

from src.core.evidence_gatekeeper import EvidenceGatekeeper


def quantity(metric="stress", value=10, unit="MPa"):
    return {"metric_name": metric, "value": value, "unit": unit}


class GatekeeperTests(unittest.TestCase):
    def test_explicit_boolean_contract(self):
        gate = EvidenceGatekeeper()
        for value in (True, False):
            result = gate.evaluate_evidence_gate(quantity("feature_present", value, "boolean"), 0.9)
            self.assertTrue(result["input_valid"])
            self.assertIs(result["validated_quantity"]["value"], value)
            self.assertFalse(result["clinical_use"])
            self.assertFalse(result["evidence_sufficient"])
        for value in (0, 1, 0.0, 1.0, "true", "false", None, [], {}, "unknown"):
            self.assertFalse(gate.validate_inputs(quantity("feature_present", value, "boolean"))[0])
        for metric, value, unit in (
            ("stress", True, "boolean"), ("ratio", False, "boolean"),
            ("feature_present", True, "dimensionless"), ("unknown", False, "boolean"),
            ("stress", False, "MPa"), ("ratio", True, "dimensionless"),
        ):
            self.assertFalse(gate.validate_inputs(quantity(metric, value, unit))[0])
        self.assertEqual(gate.evaluate_evidence_gate(
            quantity("feature_present", True, "boolean"), True)["status"], "INVALID_INPUT")

    def test_boolean_cache_identity_and_isolation(self):
        gate = EvidenceGatekeeper()
        for metric, value, unit in (
            ("feature_present", True, "boolean"), ("feature_present", False, "boolean"),
            ("ratio", 1, "dimensionless"), ("ratio", 0, "dimensionless"),
        ):
            gate.evaluate_evidence_gate(quantity(metric, value, unit), 0.9)
        self.assertEqual(gate.cache_info()["size"], 4)
        result = gate.evaluate_evidence_gate(quantity("feature_present", True, "boolean"), 0.9)
        result["validated_quantity"]["value"] = False
        result["clinical_use"] = True
        fresh = gate.evaluate_evidence_gate(quantity("feature_present", True, "boolean"), 0.9)
        self.assertIs(fresh["validated_quantity"]["value"], True)
        self.assertFalse(fresh["clinical_use"])

    def test_invalid_constructors(self):
        for size in (0, -1, True, 1.5, "2", None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                EvidenceGatekeeper(size)
        for version in ("", "  ", None, True, [], 2):
            with self.subTest(version=version), self.assertRaises(ValueError):
                EvidenceGatekeeper(model_version=version)

    def test_malformed_payloads(self):
        circular = quantity()
        circular["extra"] = circular
        for payload in (
            None, [], "stress", {}, {"value": 1}, circular,
            dict(quantity(), extra=object()),
            quantity(metric=[]), quantity(metric={}), quantity(metric=True),
            quantity(metric=""), quantity(metric="  "), quantity(metric="unknown"),
            quantity(unit=[]), quantity(unit={}), quantity(unit=True), quantity(unit=""),
            quantity(unit="Pa"), quantity(metric="duration"),
        ):
            with self.subTest(payload=repr(payload)):
                valid, reason = EvidenceGatekeeper().validate_inputs(payload)
                self.assertFalse(valid)
                self.assertTrue(reason)
                result = EvidenceGatekeeper().evaluate_evidence_gate(payload, 0.9)
                self.assertEqual(result["status"], "INVALID_INPUT")
                self.assertFalse(result["clinical_use"])

    def test_nonfinite_and_nonnumeric_values(self):
        for value in (True, False, None, "10", [], {}, float("nan"),
                      float("inf"), float("-inf"), 10 ** 1000):
            with self.subTest(value=type(value)):
                self.assertFalse(EvidenceGatekeeper().validate_inputs(quantity(value=value))[0])

    def test_contract_boundaries_and_signed_values(self):
        gate = EvidenceGatekeeper()
        for metric in ("pressure", "stress"):
            for value in (-100, 0, 100):
                self.assertTrue(gate.validate_inputs(quantity(metric, value))[0])
        for metric, unit in (
            ("photon_energy", "eV"), ("frequency", "Hz"), ("duration", "ms"),
            ("percentage", "percent"), ("ratio", "dimensionless"),
        ):
            self.assertTrue(gate.validate_inputs(quantity(metric, 0, unit))[0])
            self.assertFalse(gate.validate_inputs(quantity(metric, -1, unit))[0])
        self.assertTrue(gate.validate_inputs(quantity("percentage", 100, "percent"))[0])
        self.assertFalse(gate.validate_inputs(quantity("percentage", 100.01, "percent"))[0])
        for value in (0, -1):
            self.assertFalse(gate.validate_inputs(quantity("reference_strength", value))[0])
        self.assertTrue(gate.validate_inputs(quantity("reference_strength", 0.001))[0])

    def test_confidence_is_validated_and_not_evidence(self):
        gate = EvidenceGatekeeper()
        for score in (True, False, None, [], "0.9", -0.01, 1.01, float("nan"),
                      float("inf"), 10 ** 1000):
            with self.subTest(score=type(score)):
                self.assertEqual(gate.evaluate_evidence_gate(quantity(), score)["status"],
                                 "INVALID_INPUT")
        for score in (0, 0.75, 0.85, 0.92, 1):
            result = gate.evaluate_evidence_gate(quantity(), score)
            self.assertEqual(result["status"], "HELD_FOR_REVIEW")
            self.assertTrue(result["input_valid"])
            self.assertFalse(result["evidence_sufficient"])
            self.assertFalse(result["clinical_use"])
            self.assertEqual(result["projection_status"], "UNVERIFIED")

    def test_confidence_and_order_in_cache_key(self):
        gate = EvidenceGatekeeper()
        self.assertEqual(gate.evaluate_evidence_gate(quantity(), 0.92)["declared_confidence"], 0.92)
        self.assertEqual(gate.evaluate_evidence_gate(quantity(), 0.75)["declared_confidence"], 0.75)
        reordered = {"unit": "MPa", "value": 10, "metric_name": "stress"}
        gate.evaluate_evidence_gate(reordered, 0.75)
        self.assertEqual(gate.cache_info(), {"size": 2, "max_size": 1000, "hits": 1, "misses": 2})

    def test_lru_eviction(self):
        gate = EvidenceGatekeeper(2)
        for value in (1, 2, 1, 3, 1, 2):
            gate.evaluate_evidence_gate(quantity(value=value), 0.9)
        self.assertEqual(gate.cache_info(), {"size": 2, "max_size": 2, "hits": 2, "misses": 4})

    def test_mutation_isolation_on_miss_and_hit(self):
        gate = EvidenceGatekeeper()
        payload = quantity()
        result = gate.evaluate_evidence_gate(payload, 0.9)
        payload["value"] = -20
        result["clinical_use"] = True
        result["validated_quantity"]["unit"] = "Pa"
        hit = gate.evaluate_evidence_gate(quantity(), 0.9)
        self.assertFalse(hit["clinical_use"])
        self.assertEqual(hit["validated_quantity"], quantity())
        hit["validated_quantity"]["value"] = 99
        self.assertEqual(gate.evaluate_evidence_gate(quantity(), 0.9)["validated_quantity"], quantity())

    def test_metrics_and_versions_do_not_alias(self):
        gate = EvidenceGatekeeper()
        gate.evaluate_evidence_gate(quantity(), 0.9)
        gate.evaluate_evidence_gate(quantity("pressure"), 0.9)
        self.assertEqual(gate.cache_info()["size"], 2)
        other = EvidenceGatekeeper(model_version="v2")
        self.assertEqual(other.evaluate_evidence_gate(quantity(), 0.9)["model_version"], "v2")
        self.assertEqual(other.cache_info()["misses"], 1)
        for field in ("model_version", "contract_version", "evidence_policy_version",
                      "max_cache_size"):
            with self.assertRaises(AttributeError):
                setattr(gate, field, "changed")

    def test_concurrent_access_is_bounded_and_isolated(self):
        gate = EvidenceGatekeeper(3)

        def evaluate(index):
            result = gate.evaluate_evidence_gate(quantity(value=index % 5), 0.9)
            result["clinical_use"] = True
            return result["validated_quantity"]["value"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            self.assertEqual(list(executor.map(evaluate, range(100))), [i % 5 for i in range(100)])
        self.assertLessEqual(gate.cache_info()["size"], 3)
        self.assertFalse(gate.evaluate_evidence_gate(quantity(), 0.9)["clinical_use"])


if __name__ == "__main__":
    unittest.main()
