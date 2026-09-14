# Readmission evaluation and workload readout

**Historical retrospective/exploratory evaluation.** Patient-disjoint splitting does not make a repeatedly inspected test partition an untouched final evaluation.

No model was retrained, refit or recalibrated by this readout. It derives workload from the existing aggregate files.

## Saved threshold

At the saved validation-selected threshold 0.122, 1,904 of 10,822 encounters are flagged (175.94 per 1,000).
The confusion counts are TP=312, FP=1592, FN=637, TN=8281. Recall is 32.88%; precision is 16.39%.

These are encounter-level audit flags, not unique patient workload, clinical alerts or evidence that intervention changes outcomes.

## Historical sensitivity analysis

The table explains workload versus missed positives. **Do not choose a replacement threshold by optimizing this historical test table.** Agree on a workload objective, choose on validation and evaluate on a newly designated cohort.

| Threshold | Flags / 1,000 | Recall | Precision | Missed positives / 1,000 | Saved threshold |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.040 | 827.48 | 93.26% | 9.88% | 5.91 | False |
| 0.060 | 557.38 | 75.55% | 11.89% | 21.44 | False |
| 0.080 | 375.44 | 59.75% | 13.96% | 35.30 | False |
| 0.100 | 262.52 | 44.78% | 14.96% | 48.42 | False |
| 0.120 | 183.24 | 33.83% | 16.19% | 58.03 | False |
| 0.122 | 175.94 | 32.88% | 16.39% | 58.86 | True |
| 0.140 | 118.83 | 25.08% | 18.51% | 65.70 | False |
| 0.160 | 78.73 | 19.92% | 22.18% | 70.23 | False |
| 0.180 | 55.17 | 15.49% | 24.62% | 74.11 | False |
| 0.200 | 38.16 | 11.91% | 27.36% | 77.25 | False |
| 0.220 | 27.72 | 9.27% | 29.33% | 79.56 | False |
| 0.240 | 19.87 | 7.27% | 32.09% | 81.32 | False |
| 0.260 | 14.78 | 5.80% | 34.38% | 82.61 | False |
| 0.280 | 11.18 | 4.95% | 38.84% | 83.35 | False |
| 0.300 | 9.06 | 4.21% | 40.82% | 84.00 | False |

## Uncertainty

Status: **unavailable**.

No per-encounter predictions with patient IDs were supplied. Aggregate metrics cannot reconstruct patient-cluster intervals.
Required input: encounter ID, patient ID, binary outcome and frozen-model probability for each evaluated encounter.

The [future protocol](../config/evaluation_protocol.json) is a draft with unfilled cohort and artifact-lock fields. The utility resamples whole patient clusters while retaining encounter weighting, freezes the supplied threshold, and reports missing rows and undefined/single-class replicates.
