import unittest

import numpy as np
import pandas as pd

from readmission_audit.pipeline import clean_diabetes_data, threshold_table


class PipelineTests(unittest.TestCase):
    def test_clean_diabetes_data_creates_binary_target_and_drops_ids(self):
        df = pd.DataFrame(
            {
                "encounter_id": [1, 2],
                "patient_nbr": [10, 20],
                "age": ["[50-60)", "[60-70)"],
                "readmitted": ["<30", "NO"],
                "race": ["Caucasian", "?"],
            }
        )

        cleaned = clean_diabetes_data(df)

        self.assertNotIn("encounter_id", cleaned.columns)
        self.assertNotIn("patient_nbr", cleaned.columns)
        self.assertEqual(cleaned["early_readmission"].tolist(), [1, 0])
        self.assertEqual(cleaned["race"].isna().sum(), 1)

    def test_threshold_table_counts_confusion_matrix_values(self):
        y_true = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.8, 0.1])

        table = threshold_table(y_true, probabilities, thresholds=[0.5])
        row = table.iloc[0]

        self.assertEqual(row["true_positives"], 1)
        self.assertEqual(row["false_positives"], 1)
        self.assertEqual(row["false_negatives"], 1)
        self.assertEqual(row["true_negatives"], 1)
        self.assertEqual(row["precision"], 0.5)
        self.assertEqual(row["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
