from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from readmission_audit.pipeline import (
    SEED,
    calibration_table,
    clean_diabetes_data,
    evaluate_probabilities,
    feature_types,
    make_features,
    make_patient_order_split,
    prepare_catboost_features,
    select_f1_threshold,
)


def main() -> None:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv")
    cleaned, _ = clean_diabetes_data(raw)
    splits, _ = make_patient_order_split(cleaned)
    features = {}
    targets = {}
    for name, frame in splits.items():
        features[name], target = make_features(frame)
        targets[name] = target.to_numpy()

    _, categorical = feature_types(features["train"])
    prepared = {
        name: prepare_catboost_features(frame, categorical)
        for name, frame in features.items()
    }
    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        custom_metric=["PRAUC"],
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        prepared["train"],
        targets["train"],
        cat_features=categorical,
        eval_set=(prepared["validation"], targets["validation"]),
        use_best_model=True,
        early_stopping_rounds=75,
        verbose=False,
    )
    probabilities = {
        name: model.predict_proba(frame)[:, 1]
        for name, frame in prepared.items()
    }
    threshold = select_f1_threshold(
        targets["validation"], probabilities["validation"]
    )

    comparison_path = PROJECT_ROOT / "outputs" / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    comparison = comparison[comparison["model"] != "CatBoost"]
    model_rows = []
    calibration_rows = []
    for split_name in ("train", "validation", "test"):
        row = evaluate_probabilities(
            targets[split_name], probabilities[split_name], threshold
        )
        row.update({"model": "CatBoost", "split": split_name})
        model_rows.append(row)
        calibration_rows.append(
            calibration_table(
                targets[split_name],
                probabilities[split_name],
                "CatBoost",
                split_name,
            )
        )
    comparison = pd.concat(
        [comparison, pd.DataFrame(model_rows)], ignore_index=True
    )
    comparison.to_csv(comparison_path, index=False)

    calibration_path = PROJECT_ROOT / "outputs" / "calibration_curve.csv"
    calibration = pd.read_csv(calibration_path)
    calibration = calibration[calibration["model"] != "CatBoost"]
    pd.concat([calibration, *calibration_rows], ignore_index=True).to_csv(
        calibration_path, index=False
    )

    validation = pd.Series(model_rows[1])
    test = pd.Series(model_rows[2])
    drift_row: dict[str, object] = {"model": "CatBoost"}
    for metric in (
        "roc_auc",
        "pr_auc",
        "f1",
        "recall",
        "accuracy",
        "brier_score",
        "ece_10_bin",
    ):
        drift_row[f"validation_{metric}"] = float(validation[metric])
        drift_row[f"test_{metric}"] = float(test[metric])
        drift_row[f"test_minus_validation_{metric}"] = float(
            test[metric] - validation[metric]
        )
    drift_path = PROJECT_ROOT / "outputs" / "performance_drift.csv"
    drift = pd.read_csv(drift_path)
    drift = drift[drift["model"] != "CatBoost"]
    pd.concat([drift, pd.DataFrame([drift_row])], ignore_index=True).to_csv(
        drift_path, index=False
    )

    model.save_model(str(PROJECT_ROOT / "artifacts" / "catboost_model.cbm"))
    print(f"trees={model.tree_count_} threshold={threshold:.3f}")
    print(
        "catboost_test "
        f"roc_auc={test['roc_auc']:.4f} "
        f"pr_auc={test['pr_auc']:.4f} "
        f"f1={test['f1']:.4f} "
        f"recall={test['recall']:.4f} "
        f"brier={test['brier_score']:.4f}"
    )


if __name__ == "__main__":
    main()
