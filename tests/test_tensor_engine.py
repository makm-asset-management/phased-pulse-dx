import math
import unittest
from unittest.mock import patch

from src.core.evidence_gatekeeper import EvidenceGatekeeper
from src.core.tensor_engine import TensorEngine


class TensorEngineTests(unittest.TestCase):
    def test_dimension_contract_and_lazy_boundary(self):
        for dimension in (None, True, 1, 2, 4, 33, -3, 3.0, "3", [], 10 ** 1000):
            with self.subTest(dimension=type(dimension)), self.assertRaises(ValueError):
                TensorEngine(dimension)
        with patch.object(TensorEngine, "_original_coordinates") as generate:
            maximum = TensorEngine(31)
            self.assertEqual(maximum.point_count, 29791)
            self.assertEqual(maximum.center_index, 14895)
            generate.assert_not_called()
        with self.assertRaises(TypeError):
            TensorEngine(gatekeeper=False)
        with self.assertRaises(AttributeError):
            TensorEngine().dimension = 5

    def test_basis_is_orthonormal_and_right_handed(self):
        e1, e2, e3 = TensorEngine().basis
        for i, row in enumerate((e1, e2, e3)):
            for j, other in enumerate((e1, e2, e3)):
                self.assertAlmostEqual(sum(a * b for a, b in zip(row, other)),
                                       1.0 if i == j else 0.0, places=14)
        cross = (e1[1] * e2[2] - e1[2] * e2[1],
                 e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0])
        for actual, expected in zip(cross, e3):
            self.assertAlmostEqual(actual, expected, places=14)
        diagonal = tuple(math.fsum(row) for row in (e1, e2, e3))
        for actual, expected in zip(diagonal, (0, 0, math.sqrt(3))):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_grid_mapping_order_unique_origin_and_antipodes(self):
        engine = TensorEngine(5)
        grid = engine.generate_vertex_grid()
        points = grid["original_coordinates"]
        indices = grid["traversal_indices"]
        self.assertEqual(len(set(points)), 125)
        self.assertEqual(sorted(indices), list(range(125)))
        self.assertEqual(points[indices[0]], (0, 0, 0))
        self.assertEqual(points[engine.center_index], (0, 0, 0))
        self.assertEqual(points[0], (-2, -2, -2))
        self.assertEqual(points[-1], (2, 2, 2))
        for index, (x, y, z) in enumerate(points):
            self.assertEqual(index, ((x + 2) * 5 + y + 2) * 5 + z + 2)
        keys = [(sum(v * v for v in points[i]), points[i]) for i in indices]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual([points[i] for i in indices[1:7]],
                         [(-1, 0, 0), (0, -1, 0), (0, 0, -1),
                          (0, 0, 1), (0, 1, 0), (1, 0, 0)])
        self.assertEqual(set(points), {tuple(-v for v in point) for point in points})
        self.assertEqual(grid, engine.generate_vertex_grid())
        self.assertFalse(grid["clinical_use"])

    def test_rotation_preserves_norms_and_distances(self):
        grid = TensorEngine(5).generate_vertex_grid()
        original, rotated = grid["original_coordinates"], grid["rotated_coordinates"]
        for point, image in zip(original, rotated):
            self.assertAlmostEqual(math.dist(point, (0, 0, 0)),
                                   math.dist(image, (0, 0, 0)), places=13)
        for index in range(len(original) - 1):
            self.assertAlmostEqual(math.dist(original[index], original[index + 1]),
                                   math.dist(rotated[index], rotated[index + 1]), places=13)

    def test_binary_truth_tables_and_index_alignment(self):
        engine = TensorEngine()
        a = ([0, 0, 1, 1] * 7)[:27]
        b = ([0, 1, 0, 1] * 7)[:27]
        for operation, truth in (("AND", [0, 0, 0, 1]),
                                 ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])):
            result = engine.evaluate_center_out_logic(a, b, operation, 1)
            self.assertEqual(result["status"], "COMPUTED_RESEARCH_ONLY")
            self.assertEqual(result["values"], tuple((truth * 7)[:27]))
            self.assertFalse(result["clinical_use"])
            self.assertFalse(result["evidence_sufficient"])
            self.assertEqual(result["gate_status"], "HELD_FOR_REVIEW")

    def test_full_mask_validation_before_processing(self):
        engine = TensorEngine()
        valid = [0] * 27
        with patch.object(engine, "_process_logic") as process:
            for invalid in ([], [0] * 26, [0] * 28, {}, "0" * 27,
                            valid[:-1] + [True], valid[:-1] + [1.0],
                            valid[:-1] + [-1], valid[:-1] + [2],
                            valid[:-1] + [float("nan")], valid[:-1] + [[]],
                            valid[:-1] + [10 ** 1000]):
                for a, b in ((invalid, valid), (valid, invalid)):
                    self.assertEqual(engine.evaluate_center_out_logic(a, b, "AND")["status"],
                                     "INVALID_INPUT")
            for operation in (None, [], "and", "NOT"):
                self.assertEqual(engine.evaluate_center_out_logic(valid, valid, operation)["status"],
                                 "INVALID_INPUT")
            process.assert_not_called()

    def test_field_center_reference_and_units(self):
        engine = TensorEngine()
        values = [float(i - 13) for i in range(27)]
        result = engine.analyze_field(values, "stress", "MPa", 0.99)
        self.assertEqual(result["center_value"], 0)
        self.assertEqual(result["absolute_deviations"], tuple(abs(v) for v in values))
        self.assertEqual(result["absolute_deviations"][13], 0)
        self.assertEqual(result["unit"], "MPa")
        self.assertEqual(result["traversal_indices"][0], 13)
        self.assertFalse(result["clinical_use"])
        self.assertEqual(result["gate_status"], "HELD_FOR_REVIEW")
        values[0] = 100
        self.assertEqual(result["absolute_deviations"][0], 13)

    def test_field_invalid_inputs_never_process(self):
        engine = TensorEngine()
        with patch.object(engine, "_process_deviations") as process:
            for invalid in ([], None, {}, [0] * 26, [0] * 28):
                self.assertEqual(engine.analyze_field(invalid, "stress", "MPa")["status"],
                                 "INVALID_INPUT")
            for bad in (True, None, [], {}, float("inf"), float("nan"), 10 ** 1000):
                values = [0] * 26 + [bad]
                self.assertEqual(engine.analyze_field(values, "stress", "MPa")["status"],
                                 "INVALID_INPUT")
            for metric, unit in (([], "MPa"), ("unknown", "MPa"), ("stress", []),
                                 ("stress", "Pa"), ("percentage", "MPa")):
                self.assertEqual(engine.analyze_field([0] * 27, metric, unit)["status"],
                                 "INVALID_INPUT")
            self.assertEqual(engine.analyze_field([101] * 27, "percentage", "percent")["status"],
                             "INVALID_INPUT")
            for values in ([True] * 27, [0] * 27):
                self.assertEqual(engine.analyze_field(
                    values, "feature_present", "boolean")["status"], "INVALID_INPUT")
            process.assert_not_called()

    def test_gate_failures_and_cache_cannot_bypass_validation(self):
        gate = EvidenceGatekeeper()
        engine = TensorEngine(gatekeeper=gate)
        mask = [0] * 27
        engine.evaluate_center_out_logic(mask, mask, "AND", 0.92)
        later = engine.evaluate_center_out_logic(mask, mask, "AND", 0.75)
        self.assertEqual(later["declared_confidence"], 0.75)
        self.assertGreater(gate.cache_info()["hits"], 0)
        with (
            patch.object(engine, "_process_logic") as logic,
            patch.object(engine, "_process_deviations") as field,
        ):
            for confidence in (True, [], float("nan"), float("inf"), 10 ** 1000, -1, 2):
                self.assertEqual(engine.evaluate_center_out_logic(
                    mask, mask, "AND", confidence)["status"], "INVALID_INPUT")
                self.assertEqual(engine.analyze_field(
                    mask, "stress", "MPa", confidence)["status"], "INVALID_INPUT")
            with patch.object(gate, "validate_controller_inputs",
                              return_value=(False, "controller rejected")):
                self.assertEqual(engine.evaluate_center_out_logic(
                    mask, mask, "AND", 0.75)["status"], "INVALID_INPUT")
                self.assertEqual(engine.analyze_field(
                    mask, "stress", "MPa", 0.75)["status"], "INVALID_INPUT")
            logic.assert_not_called()
            field.assert_not_called()

    def test_field_subtraction_overflow_is_explicit(self):
        values = [1e308] * 27
        values[13] = -1e308
        result = TensorEngine().analyze_field(values, "stress", "MPa")
        self.assertEqual(result["status"], "CALCULATION_ERROR")
        self.assertNotIn("absolute_deviations", result)
        self.assertFalse(result["clinical_use"])


if __name__ == "__main__":
    unittest.main()
