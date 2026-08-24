import unittest

import numpy as np
import pandas as pd
import torch

from readmission_audit.pipeline import (
    FTTransformer,
    ReadmissionDNN,
    clean_diabetes_data,
    make_features,
    make_patient_order_split,
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
        self.assertEqual(features.loc[0, "diag_1"], "Diabetes")
        self.assertEqual(target.tolist(), [1])

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


if __name__ == "__main__":
    unittest.main()
