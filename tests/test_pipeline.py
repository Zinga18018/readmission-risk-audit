import unittest

import numpy as np
import pandas as pd
import torch

from readmission_audit.pipeline import (
    FTTransformer,
    ReadmissionDNN,
    catboost_blend_sweep,
    clean_diabetes_data,
    engineer_clinical_features,
    make_features,
    make_patient_order_split,
    mixed_type_feature_relevance,
    positive_probability_logits,
    softmax_probabilities,
    threshold_table,
)


class PipelineTests(unittest.TestCase):
    def test_cleaning_creates_binary_target_and_excludes_expired(self):
        frame = pd.DataFrame(
            {
                "encounter_id": [1, 2, 3],
                "patient_nbr": [10, 20, 30],
                "discharge_disposition_id": [1, 11, 1],
                "readmitted": ["<30", "NO", ">30"],
                "race": ["Caucasian", "?", "AfricanAmerican"],
            }
        )

        cleaned, audit = clean_diabetes_data(frame)

        self.assertEqual(cleaned["early_readmission"].tolist(), [1, 0])
        self.assertEqual(audit["excluded_hospice_or_expired_rows"], 1)
        self.assertEqual(cleaned["race"].isna().sum(), 0)

    def test_patient_order_split_has_zero_patient_overlap(self):
        frame = pd.DataFrame(
            {
                "encounter_id": np.arange(1, 21),
                "patient_nbr": np.arange(101, 121),
                "discharge_disposition_id": 1,
                "readmitted": ["<30", "NO"] * 10,
            }
        )
        cleaned, _ = clean_diabetes_data(frame)
        splits, audit = make_patient_order_split(cleaned)

        patients = {
            name: set(split["patient_nbr"]) for name, split in splits.items()
        }
        self.assertFalse(patients["train"] & patients["validation"])
        self.assertFalse(patients["train"] & patients["test"])
        self.assertFalse(patients["validation"] & patients["test"])
        self.assertEqual(sum(audit["patient_overlap_counts"].values()), 0)

    def test_features_remove_target_and_identifiers(self):
        frame = pd.DataFrame(
            {
                "encounter_id": [1],
                "patient_nbr": [10],
                "early_readmission": [1],
                "admission_type_id": [1],
                "diag_1": ["250.1"],
                "num_medications": [3],
            }
        )
        features, target = make_features(frame)
        self.assertNotIn("early_readmission", features.columns)
        self.assertNotIn("encounter_id", features.columns)
        self.assertNotIn("patient_nbr", features.columns)
        self.assertEqual(features.loc[0, "diag_1_family"], "Diabetes")
        self.assertEqual(features.loc[0, "diag_1_three_digit"], "250")
        self.assertEqual(target.tolist(), [1])

    def test_patient_history_uses_only_strictly_prior_encounters(self):
        frame = pd.DataFrame(
            {
                "encounter_id": [30, 10, 20, 15],
                "patient_nbr": [1, 1, 1, 2],
                "time_in_hospital": [100, 2, 4, 9],
                "num_medications": [30, 10, 20, 5],
                "diag_1": ["428", "250.1", "786", "401"],
                "discharge_disposition_id": [1, 2, 3, 1],
            }
        )

        engineered = engineer_clinical_features(frame)

        self.assertEqual(engineered["prior_encounter_count"].tolist(), [2, 0, 1, 0])
        self.assertEqual(engineered.loc[2, "prior_mean_time_in_hospital"], 2.0)
        self.assertEqual(engineered.loc[0, "prior_mean_time_in_hospital"], 3.0)
        self.assertTrue(np.isnan(engineered.loc[1, "prior_mean_time_in_hospital"]))
        self.assertEqual(
            engineered.loc[2, "previous_primary_diagnosis_family"], "Diabetes"
        )

    def test_dnn_returns_raw_logits_and_softmax_is_separate(self):
        torch.manual_seed(42)
        model = ReadmissionDNN(input_dim=4, dropout=0.05)
        logits = model(torch.ones((3, 4))).detach().numpy()
        probabilities = softmax_probabilities(logits)

        self.assertEqual(logits.shape, (3, 2))
        self.assertFalse(np.allclose(logits.sum(axis=1), 1.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_ft_transformer_returns_two_raw_logits(self):
        torch.manual_seed(42)
        model = FTTransformer(
            numeric_features=3,
            category_cardinalities=[4, 5],
            token_dimension=16,
            transformer_blocks=2,
            attention_heads=4,
            feedforward_dimension=32,
            attention_dropout=0.05,
        )
        numeric = torch.ones((4, 3))
        categorical = torch.tensor([[1, 2], [2, 3], [0, 1], [3, 4]])
        logits = model(numeric, categorical).detach().numpy()
        probabilities = softmax_probabilities(logits)

        self.assertEqual(logits.shape, (4, 2))
        self.assertFalse(np.allclose(logits.sum(axis=1), 1.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_threshold_table_counts_confusion_matrix_values(self):
        y_true = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.8, 0.1])

        table = threshold_table(y_true, probabilities, thresholds=[0.5])
        row = table.iloc[0]

        self.assertEqual(row["true_positives"], 1)
        self.assertEqual(row["false_positives"], 1)
        self.assertEqual(row["false_negatives"], 1)
        self.assertEqual(row["true_negatives"], 1)

    def test_probability_logits_round_trip_and_blend_sweep(self):
        probabilities = np.array([0.1, 0.8, 0.3, 0.9])
        logits = positive_probability_logits(probabilities)
        restored = softmax_probabilities(logits)[:, 1]
        np.testing.assert_allclose(restored, probabilities)

        y_true = np.array([0, 1, 0, 1])
        primary = np.array([0.2, 0.7, 0.3, 0.8])
        secondary = np.array([0.4, 0.6, 0.1, 0.9])
        sweep = catboost_blend_sweep(y_true, primary, secondary)
        self.assertEqual(len(sweep), 21)
        self.assertTrue(sweep["primary_weight"].between(0.0, 1.0).all())

    def test_mixed_type_relevance_ranks_informative_features(self):
        target = np.array([0, 1] * 100)
        frame = pd.DataFrame(
            {
                "numeric_signal": target.astype(float),
                "categorical_signal": np.where(target == 1, "yes", "no"),
                "constant_noise": 1.0,
            }
        )

        ranking = mixed_type_feature_relevance(frame, target)

        self.assertIn(ranking.iloc[0]["feature"], {
            "numeric_signal",
            "categorical_signal",
        })
        constant = ranking.loc[ranking["feature"] == "constant_noise"].iloc[0]
        self.assertEqual(constant["mutual_information"], 0.0)


if __name__ == "__main__":
    unittest.main()
