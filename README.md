# Hospital Readmission Risk Audit

This is a practical data science project built around the UCI Diabetes
130-hospitals dataset. The goal is not to make a clinical product. The goal is
to show a clean model-audit workflow for a high-stakes prediction problem:
early hospital readmission within 30 days.

The project focuses on the parts interviewers can actually ask about:

- target definition
- imbalanced classification
- threshold tradeoffs
- calibration
- feature drivers
- subgroup summaries
- reproducible scripts and tests

## Data

- Source: UCI Machine Learning Repository
- Dataset: Diabetes 130-US Hospitals for Years 1999-2008
- Rows: 101,766 hospital encounters in the raw file
- Task: classify whether a patient was readmitted in less than 30 days

The raw dataset is downloaded by the build script and is not committed to the
repo.

## Run

```powershell
pip install -r requirements.txt
python scripts\build_readmission_audit.py
streamlit run app.py
```

## Test

```powershell
python -m unittest discover -s tests
```

## Verified Snapshot

Generated with:

```powershell
python scripts\build_readmission_audit.py
```

The generated metrics are written to `outputs/metrics.json`.

Current local result:

```text
raw_rows=101766 raw_columns=50
rows=101766
positive_rate=0.1116
roc_auc=0.6464
average_precision=0.2010
brier_score=0.2288
best_threshold=0.50 precision=0.1637 recall=0.5632 f1=0.2536
```

At the selected 0.50 threshold, the model flags 38.40% of test encounters and
recalls 56.32% of early readmissions. This is intentionally reported as an
audit result, not as a claim of clinical readiness.

## Resume-Safe Wording

> Built a hospital readmission risk audit using the UCI Diabetes 130-hospitals
> dataset, modeling 101,766 encounters for early 30-day readmission and
> reporting ROC-AUC, average precision, threshold tradeoffs, calibration,
> feature drivers, and subgroup summaries.

## Responsible Framing

This is an educational model-audit project. It should not be used for clinical
decision-making. The value is in the evaluation workflow, not in claiming a
deployable healthcare system.
