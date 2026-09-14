from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from readmission_audit.evaluation import export_private_predictions, patient_cluster_bootstrap, threshold_workload_table, validate_predictions


def example():
    return pd.DataFrame({"encounter_id": range(1, 7), "patient_id": ["a", "a", "b", "b", "c", "c"],
                         "y_true": [0, 1, 0, 1, 0, 1], "y_prob": [.1, .8, .4, .6, .55, .45]})


class EvaluationTests(unittest.TestCase):
    def test_cluster_bootstrap_is_deterministic_and_threshold_stays_fixed(self):
        data = example()
        first = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=100, seed=7)
        second = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertFalse(first["threshold_retuned"])
        self.assertEqual(first["metrics"]["recall"]["estimate"], 2 / 3)
        self.assertEqual(first["metrics"]["precision"]["estimate"], 2 / 3)
        self.assertEqual(first["single_class_replicates"], 0)
        self.assertTrue(all(metric["valid_replicates"] == 100 for metric in first["metrics"].values()))
        json.dumps(first, allow_nan=False)

    def test_brier_percentiles_match_independent_whole_patient_resampling(self):
        data = example()
        result = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=100, seed=9)
        cluster_losses = np.array([np.mean((data.iloc[index:index + 2].y_prob - data.iloc[index:index + 2].y_true) ** 2)
                                  for index in (0, 2, 4)])
        rng = np.random.default_rng(9)
        values = [cluster_losses[rng.integers(0, 3, size=3)].mean() for _ in range(100)]
        expected = np.quantile(values, [.025, .975])
        self.assertAlmostEqual(result["metrics"]["brier_score"]["ci_low"], expected[0])
        self.assertAlmostEqual(result["metrics"]["brier_score"]["ci_high"], expected[1])

    def test_threshold_includes_exact_boundary_without_searching(self):
        data = example()
        data["y_prob"] = .5
        result = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=30)
        self.assertEqual(result["metrics"]["flagged_rate"]["estimate"], 1)
        self.assertEqual(result["metrics"]["recall"]["estimate"], 1)

    def test_unequal_cluster_sizes_and_single_class_replicates_are_accounted(self):
        data = pd.DataFrame({"encounter_id": range(5), "patient_id": ["a", "a", "b", "b", "b"],
                             "y_true": [0, 0, 1, 1, 1], "y_prob": [.1, .2, .6, .7, .8]})
        result = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=100, seed=5)
        self.assertEqual(result["replicate_row_count_range"], [4, 6])
        self.assertGreater(result["single_class_replicates"], 0)
        auc = result["metrics"]["roc_auc"]
        self.assertEqual(auc["undefined_replicates"], result["single_class_replicates"])
        self.assertEqual(auc["valid_replicates"] + auc["undefined_replicates"], 100)
        self.assertIsNone(auc["ci_low"])
        self.assertEqual(result["metrics"]["brier_score"]["valid_replicates"], 100)

    def test_missing_rows_error_or_explicit_exclusion(self):
        data = example()
        data.loc[0, "y_prob"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=30)
        result = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=30, missing_policy="drop")
        self.assertEqual(result["input_accounting"]["input_rows"], 6)
        self.assertEqual(result["input_accounting"]["included_rows"], 5)
        self.assertEqual(result["input_accounting"]["missing_rows_excluded"], 1)

    def test_invalid_prediction_values_rejected(self):
        for column, value in (("y_prob", float("inf")), ("y_prob", -1), ("y_prob", 1.1), ("y_true", 2), ("y_true", .5)):
            with self.subTest(column=column, value=value):
                data = example()
                data[column] = data[column].astype(float)
                data.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_predictions(data)

    def test_duplicate_encounters_and_missing_identifiers_rejected(self):
        data = example()
        data.loc[0, "encounter_id"] = 2
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_predictions(data)
        data = example()
        data.loc[0, "patient_id"] = " "
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_predictions(data)
        with self.assertRaisesRegex(ValueError, "columns"):
            validate_predictions(example().drop(columns="patient_id"))

    def test_single_class_dataset_keeps_defined_metrics_only(self):
        data = example()
        data["y_true"] = 0
        data["y_prob"] = .1
        result = patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=30)
        self.assertEqual(result["single_class_replicates"], 30)
        for metric in ("roc_auc", "average_precision", "recall", "precision"):
            self.assertIsNone(result["metrics"][metric]["estimate"])
            self.assertEqual(result["metrics"][metric]["undefined_replicates"], 30)
        self.assertAlmostEqual(result["metrics"]["brier_score"]["estimate"], .01)

    def test_config_validation_and_one_cluster(self):
        for params in ({"frozen_threshold": -1}, {"frozen_threshold": float("nan")}, {"n_resamples": 1}, {"seed": -1}, {"confidence_level": 1}, {"min_valid_fraction": 0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                patient_cluster_bootstrap(example(), **{"frozen_threshold": .5, "n_resamples": 30, **params})
        data = example()
        data["patient_id"] = "same"
        with self.assertRaisesRegex(ValueError, "two patient"):
            patient_cluster_bootstrap(data, frozen_threshold=.5, n_resamples=30)

    def test_export_keeps_rows_aligned_and_validates_lengths(self):
        data = example()
        ids = data[["encounter_id", "patient_id"]].rename(columns={"patient_id": "patient_nbr"})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "predictions.csv"
            metadata = export_private_predictions(ids, data.y_true, data.y_prob, destination)
            self.assertEqual(metadata["included_rows"], 6)
            exported = pd.read_csv(destination)
            self.assertEqual(exported.y_prob.tolist(), data.y_prob.tolist())
            self.assertEqual(exported.patient_id.tolist(), data.patient_id.tolist())
            with self.assertRaisesRegex(ValueError, "equal lengths"):
                export_private_predictions(ids, data.y_true.iloc[:2], data.y_prob, destination)

    def test_workload_derived_from_counts_without_threshold_selection(self):
        saved = {"rows": 100, "threshold": .5, "true_positives": 10, "false_positives": 20, "false_negatives": 10, "true_negatives": 60}
        candidate = {**saved, "threshold": .3, "true_positives": 15, "false_positives": 35, "false_negatives": 5, "true_negatives": 45}
        result = threshold_workload_table(pd.DataFrame([candidate]), saved)
        reference = result[result.saved_validation_threshold].iloc[0]
        self.assertEqual(reference.flags_per_1000_encounters, 300)
        self.assertEqual(reference.recall, .5)
        self.assertEqual(reference.precision, 1 / 3)
        self.assertEqual(result.saved_validation_threshold.sum(), 1)
        self.assertNotIn("recommended_threshold", result)

    def test_confusion_counts_and_cohort_consistency_validated(self):
        saved = {"rows": 10, "threshold": .5, "true_positives": 1, "false_positives": 2, "false_negatives": 1, "true_negatives": 6}
        for bad in ({**saved, "rows": 11}, {**saved, "true_positives": -1}, {**saved, "true_positives": 2, "false_positives": 1}, {**saved, "true_positives": 2, "false_negatives": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                threshold_workload_table(pd.DataFrame([bad]), saved)


if __name__ == "__main__":
    unittest.main()
