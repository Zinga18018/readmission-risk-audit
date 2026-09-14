# Evaluation repair verification — September 14, 2026

## Scope and preserved evidence

This repair updates evaluation disclosure and adds uncertainty/workload tooling. It does not retrain, refit or recalibrate a model. The historical `metrics.json`, `leakage_audit.json`, `threshold_table.csv` and saved CatBoost configuration remain unchanged.

The checked-in historical evidence still reports 83,621 modeling encounters, 10,822 test encounters, ROC-AUC 0.6612406434211856, average precision 0.1746536707606301 and Brier score 0.07730718522952124. These numbers were read from saved files, not independently reproduced by training during this repair.

## Verification

Environment: Python 3.14.7, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1 and SciPy 1.18.1. Only the lightweight evaluation requirements were installed in `.venv-evaluation`.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -p test_evaluation.py -v` | 12 tests passed in 1.341 seconds |
| `python scripts/evaluation_readout.py` | Wrote aggregate-only readout with uncertainty explicitly unavailable |
| Synthetic private CSV through the CLI, 100 resamples | Completed for 6 constructed encounters in 3 fake patient clusters; caller label `synthetic_demo` retained |
| Syntax compilation | `app.py`, pipeline, evaluation module, build script and readout script passed |
| Git ignore verification | Private synthetic CSV and its result JSON are ignored |
| `git diff --check` | Passed |

The tests cover deterministic whole-patient resampling, unequal cluster sizes, exact threshold boundaries, fixed-threshold behavior, missing-input accounting, single-class and undefined metric counts, invalid probabilities/labels/IDs, duplicate encounters, export alignment and threshold workload consistency. The Brier percentile test independently reconstructs whole-cluster draws.

## Workload proof

At the saved threshold 0.122, TP=312, FP=1,592, FN=637 and TN=8,281, summing to 10,822. Therefore 1,904 encounters are flagged (175.94 per 1,000); recall is 32.88% and precision 16.39%. The 15-row workload table contains the historical 14-threshold grid plus the exact saved threshold. No new threshold was selected.

## Remaining limits

- Actual historical cluster intervals remain unavailable because no patient-linked prediction rows were supplied. Synthetic intervals are not presented as model evidence.
- The optional full-training export path was syntax checked; its independent export helper was runtime tested. Full training and the legacy Torch/CatBoost test suite were not run.
- The edited dashboard was syntax checked, not exercised with saved models or redeployed in this repair.
- The future protocol is a draft, with no new cohort, artifact lock, approved threshold objective or executed confirmatory evaluation.
- Bootstrap intervals are conditional on the frozen model and observed cohort. They cannot undo test reuse, account for model-fitting uncertainty, establish clinical utility or replace external validation.
