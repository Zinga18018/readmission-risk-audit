# Clinical Readmission Risk Modeling, Calibration & Drift

This project predicts **binary readmission within 30 days** for diabetes-related
hospital encounters. One shared experiment runner compares logistic regression,
histogram gradient boosting, CatBoost, a shallow DNN, and FT-Transformer using
the same leakage-safe split and evaluation protocol.

The project is an educational model audit, not a clinical product.

## Dataset and target

- Source: [UCI Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
- Raw size: 101,766 encounters, 50 CSV columns
- UCI description: 47 predictive features from 130 US hospitals and delivery
  networks across 1999-2008
- Binary target: `1` when `readmitted == "<30"`; otherwise `0`
- License: CC BY 4.0; dataset DOI: `10.24432/C5230J`

The public file does **not** contain a row-level date or year. This project
therefore does not pretend that it can recover literal annual cohorts.

## Leakage-safe evaluation

The primary split is patient-disjoint and ordered by `encounter_id`:

1. use the lower 70% of encounter IDs as the training-order boundary;
2. use 70-85% as the validation-order interval;
3. use the upper 15% as the test-order interval;
4. exclude patients who cross a boundary.

This is an **encounter-order deployment proxy**, not a calendar-time split.
`patient_nbr` and `encounter_id` are used only to construct the split and are
removed from model features. Preprocessing is fit on train only, dropout and
threshold selection use validation only, and test is used once after selection.

Hospice and expired discharge dispositions are excluded because those
encounters are not eligible for subsequent readmission. Diagnosis codes are
grouped into broad ICD-9 families; categorical variables are imputed and
one-hot encoded, while numeric variables are median-imputed and standardized.

## Models

### Logistic regression

A regularized linear probability baseline using the same train-fitted one-hot
preprocessing as the DNN.

### Histogram gradient boosting

A nonlinear tree-based baseline using train-fitted imputation and ordinal
categorical encoding.

### CatBoost

CatBoost receives train-fitted missing-value handling while retaining native
categorical columns. Validation log loss controls early stopping; PR-AUC is
reported by the common evaluator rather than used as CatBoost's stopping rule.

### FT-Transformer

```text
numerical feature projections + categorical embeddings
  -> [CLS] + 44 feature tokens
  -> 3 pre-normalized Transformer blocks
  -> 8-head self-attention, ReGLU feed-forward layers
  -> LayerNorm + ReLU + Linear(2 raw logits)
```

- token dimension: `64`
- attention dropout sweep: `0.00`, `0.05`
- feed-forward dropout: `0.10`
- loss: `CrossEntropyLoss` on raw logits
- softmax: evaluation/inference only
- selection metric: validation PR-AUC
- calibration: validation-only temperature scaling

### Shallow PyTorch DNN

```text
encoded input
  -> Linear(input_dim, 128)
  -> ReLU
  -> Dropout(p)
  -> Linear(128, 64)
  -> ReLU
  -> Dropout(p)
  -> Linear(64, 2 raw logits)
```

- loss: `CrossEntropyLoss` on raw logits
- softmax: evaluation/inference only
- dropout sweep: `0.00`, `0.05`, `0.10`, `0.15`
- selection metric: validation PR-AUC
- regularization: AdamW weight decay plus early stopping
- calibration: one temperature fit on validation logits only

## Evaluation

Every model is evaluated with ROC-AUC, PR-AUC, F1, recall, accuracy, Brier
score, log loss, 10-bin expected calibration error, and calibration curves.
The classification threshold is chosen on validation data to maximize F1 and
then frozen for test.

Distribution shift is measured from train to validation/test with:

- PSI and KS statistic for numeric features;
- Jensen-Shannon divergence for categorical features;
- validation-to-test changes in discrimination, threshold metrics, and
  calibration.

## Verified local results

The checked-in values below are refreshed only after a successful full training
run. Machine-readable proof is in `outputs/metrics.json`,
`outputs/model_comparison.csv`, `outputs/performance_drift.csv`, and
`outputs/leakage_audit.json`.

<!-- VERIFIED_RESULTS_START -->
Verified locally on 2026-08-23 with seed 42:

```text
raw encounters:                 101,766
eligible after hospice/death:    99,343
modeling rows after boundaries:  83,621
train / validation / test:       63,512 / 9,287 / 10,822
patient overlap across splits:   0
encoded DNN input features:       165
FT feature tokens:                 44 + CLS
selected DNN dropout:             0.15
selected FT attention dropout:    0.00
FT temperature:                   0.7795
```

Held-out test results; each threshold was selected on validation data:

| Model | ROC-AUC | PR-AUC | F1 | Recall | Accuracy | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.6301 | 0.1443 | 0.1959 | 0.2192 | 0.8422 | 0.0790 | 0.0179 |
| HistGradientBoosting | **0.6406** | 0.1497 | **0.2112** | **0.3667** | 0.7597 | 0.0787 | 0.0167 |
| CatBoost | 0.6403 | 0.1514 | 0.2060 | 0.2234 | **0.8490** | 0.0785 | 0.0157 |
| DNN, temperature-scaled | 0.6365 | **0.1519** | 0.1920 | 0.2719 | 0.7993 | **0.0784** | **0.0121** |
| FT-Transformer, temperature-scaled | 0.6345 | 0.1481 | 0.2066 | 0.2561 | 0.8276 | 0.0789 | 0.0199 |

No single model dominated. Histogram gradient boosting had the best ROC-AUC,
F1, and recall; CatBoost had the best accuracy; the shallow DNN had the best
PR-AUC, Brier score, and ECE. FT-Transformer did not beat the simpler models on
this split despite being substantially more expensive to train.

For FT-Transformer, 0% attention dropout narrowly beat 5% validation PR-AUC:
0.1529 versus 0.1528. Temperature scaling improved FT validation Brier from
0.0774 to 0.0745 and ECE from 0.0517 to 0.0181. Test Brier improved from 0.0810
to 0.0789 and ECE from 0.0432 to 0.0199.

From validation to the later encounter-order test cohort, calibrated FT
ROC-AUC fell by 0.0046, PR-AUC by 0.0048, F1 by 0.0107, and Brier worsened by
0.0044. `number_diagnoses` showed the strongest numeric shift (PSI 0.2631,
KS 0.2491). FT train-minus-validation gaps were 0.0701 PR-AUC and 0.0401
ROC-AUC, so overfitting remains documented rather than hidden.
<!-- VERIFIED_RESULTS_END -->

## Run

```powershell
$ReadmissionVenv = "$env:USERPROFILE\Documents\Codex\.venvs\readmission-risk"
python -m venv $ReadmissionVenv
& "$ReadmissionVenv\Scripts\python.exe" -m pip install -r requirements.txt
& "$ReadmissionVenv\Scripts\python.exe" scripts\build_readmission_audit.py
& "$ReadmissionVenv\Scripts\python.exe" -m streamlit run app.py
```

The short environment path avoids Windows' legacy path-length limit when
installing PyTorch from this deeply nested project directory.

## Test

```powershell
& "$ReadmissionVenv\Scripts\python.exe" -m unittest discover -s tests -v
```

## Generated artifacts

```text
artifacts/
  dnn_state.pt
  ft_transformer_state.pt
  ft_preprocessor.joblib
  ft_transformer_config.json
  catboost_model.cbm
  preprocessor.joblib
  logistic_model.joblib
  tree_model.joblib
  model_config.json
outputs/
  metrics.json
  model_comparison.csv
  dropout_sweep.csv
  ft_attention_dropout_sweep.csv
  ft_train_history.csv
  calibration_curve.csv
  feature_drift.csv
  performance_drift.csv
  leakage_audit.json
```

## Responsible use

The output is not a diagnosis, treatment recommendation, or validated hospital
workflow. The dataset is historical and sensitive, hospital identifiers and
timestamps are unavailable, and performance on this public benchmark does not
establish present-day clinical validity.
