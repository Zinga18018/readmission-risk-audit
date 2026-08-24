from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid

from readmission_audit.pipeline import (
    SEED,
    catboost_blend_sweep,
    clean_diabetes_data,
    evaluate_probabilities,
    feature_types,
    make_features,
    make_patient_order_split,
    positive_probability_logits,
    prepare_catboost_features,
    select_f1_threshold,
    softmax_probabilities,
    tune_temperature,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"


def load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "diabetic_data.csv")
    cleaned, _ = clean_diabetes_data(raw)
    splits, _ = make_patient_order_split(cleaned)
    features: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    for split, frame in splits.items():
        X, y = make_features(frame)
        features[split] = X
        targets[split] = y.to_numpy(dtype=int)

    selected_features = features["train"].columns.tolist()
    _, categorical = feature_types(features["train"])
    prepared = {
        split: prepare_catboost_features(frame, categorical)
        for split, frame in features.items()
    }

    grids = [
        {
            "depth": [5], "learning_rate": [0.04], "l2_leaf_reg": [5.0],
            "random_strength": [0.5], "bagging_temperature": [1.0],
            "border_count": [128], "has_time": [False],
        },
        {
            "depth": [6], "learning_rate": [0.035], "l2_leaf_reg": [8.0],
            "random_strength": [0.5], "bagging_temperature": [0.5],
            "border_count": [128], "has_time": [False],
        },
        {
            "depth": [7], "learning_rate": [0.03], "l2_leaf_reg": [10.0],
            "random_strength": [1.0], "bagging_temperature": [1.0],
            "border_count": [128], "has_time": [False],
        },
        {
            "depth": [8], "learning_rate": [0.025], "l2_leaf_reg": [12.0],
            "random_strength": [1.0], "bagging_temperature": [0.5],
            "border_count": [128], "has_time": [False],
        },
        {
            "depth": [6], "learning_rate": [0.05], "l2_leaf_reg": [12.0],
            "random_strength": [0.25], "bagging_temperature": [1.0],
            "border_count": [254], "has_time": [True],
        },
        {
            "depth": [7], "learning_rate": [0.04], "l2_leaf_reg": [15.0],
            "random_strength": [0.5], "bagging_temperature": [1.5],
            "border_count": [254], "has_time": [True],
        },
        {
            "depth": [6], "learning_rate": [0.05], "l2_leaf_reg": [8.0],
            "random_strength": [0.5], "bagging_temperature": [1.0],
            "border_count": [128], "has_time": [False],
        },
        {
            "depth": [5], "learning_rate": [0.05], "l2_leaf_reg": [5.0],
            "random_strength": [0.5], "bagging_temperature": [1.0],
            "border_count": [128], "has_time": [False],
        },
    ]
    candidates = list(ParameterGrid(grids))
    candidate_models: list[CatBoostClassifier] = []
    candidate_probabilities: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, object]] = []
    for index, params in enumerate(candidates, start=1):
        model = CatBoostClassifier(
            iterations=650,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=SEED,
            allow_writing_files=False,
            verbose=False,
            thread_count=8,
            bootstrap_type="Bayesian",
            cat_features=categorical,
            **params,
        )
        model.fit(prepared["train"], targets["train"], verbose=False)
        probability = {
            split: model.predict_proba(frame)[:, 1]
            for split, frame in prepared.items()
        }
        candidate_models.append(model)
        candidate_probabilities.append(probability)
        validation = evaluate_probabilities(
            targets["validation"], probability["validation"], 0.5
        )
        rows.append(
            {
                "candidate": index,
                **params,
                "validation_pr_auc": validation["pr_auc"],
                "validation_roc_auc": validation["roc_auc"],
                "validation_brier": validation["brier_score"],
            }
        )
        print(
            f"candidate {index}/{len(candidates)} validation "
            f"PR-AUC={validation['pr_auc']:.5f} "
            f"ROC-AUC={validation['roc_auc']:.5f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )
    results.to_csv(OUTPUTS / "catboost_later_validation_gate.csv", index=False)
    best_indices = (results.head(2)["candidate"].astype(int) - 1).tolist()
    first_index, second_index = best_indices
    weight_sweep = catboost_blend_sweep(
        targets["validation"],
        candidate_probabilities[first_index]["validation"],
        candidate_probabilities[second_index]["validation"],
    )
    weight_sweep.to_csv(
        OUTPUTS / "catboost_improved_pair_blend_sweep.csv", index=False
    )
    first_weight = float(weight_sweep.iloc[0]["primary_weight"])
    blended_raw = {
        split: (
            first_weight * candidate_probabilities[first_index][split]
            + (1.0 - first_weight) * candidate_probabilities[second_index][split]
        )
        for split in ("train", "validation", "test")
    }
    logits = {
        split: positive_probability_logits(probability)
        for split, probability in blended_raw.items()
    }
    temperature = tune_temperature(logits["validation"], targets["validation"])
    blended = {
        split: softmax_probabilities(split_logits, temperature)[:, 1]
        for split, split_logits in logits.items()
    }
    threshold = select_f1_threshold(targets["validation"], blended["validation"])
    comparison_rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        metrics = evaluate_probabilities(targets[split], blended[split], threshold)
        metrics.update(
            {"model": "CatBoost later-validation-gated pair", "split": split}
        )
        comparison_rows.append(metrics)

    old_config = json.loads(
        (ARTIFACTS / "catboost_config.json").read_text(encoding="utf-8")
    )
    old_primary = load_model(ARTIFACTS / "catboost_model.cbm")
    old_secondary = load_model(ARTIFACTS / "catboost_secondary_model.cbm")
    old_columns = old_config["feature_columns"]
    old_categorical = old_config["categorical_features"]
    old_frames = {
        split: prepare_catboost_features(
            features[split].reindex(columns=old_columns), old_categorical
        )
        for split in ("train", "validation", "test")
    }
    old_raw = {
        split: (
            old_config["primary_weight"]
            * old_primary.predict_proba(frame)[:, 1]
            + old_config["secondary_weight"]
            * old_secondary.predict_proba(frame)[:, 1]
        )
        for split, frame in old_frames.items()
    }
    old_scaled = {
        split: softmax_probabilities(
            positive_probability_logits(probability), old_config["temperature"]
        )[:, 1]
        for split, probability in old_raw.items()
    }
    for split in ("train", "validation", "test"):
        metrics = evaluate_probabilities(
            targets[split], old_scaled[split], old_config["threshold"]
        )
        metrics.update({"model": "Existing CatBoost ensemble", "split": split})
        comparison_rows.append(metrics)

    cross_sweep = catboost_blend_sweep(
        targets["validation"], old_scaled["validation"], blended["validation"]
    )
    cross_sweep.to_csv(
        OUTPUTS / "catboost_existing_new_blend_sweep.csv", index=False
    )
    old_weight = float(cross_sweep.iloc[0]["primary_weight"])
    cross_blend = {
        split: old_weight * old_scaled[split] + (1 - old_weight) * blended[split]
        for split in ("train", "validation", "test")
    }
    cross_threshold = select_f1_threshold(
        targets["validation"], cross_blend["validation"]
    )
    for split in ("train", "validation", "test"):
        metrics = evaluate_probabilities(
            targets[split], cross_blend[split], cross_threshold
        )
        metrics.update(
            {"model": "Existing + later-gated CatBoost blend", "split": split}
        )
        comparison_rows.append(metrics)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        OUTPUTS / "catboost_final_improvement_comparison.csv", index=False
    )
    first_model = candidate_models[first_index]
    second_model = candidate_models[second_index]
    first_model.save_model(ARTIFACTS / "catboost_later_gated_primary.cbm")
    second_model.save_model(ARTIFACTS / "catboost_later_gated_secondary.cbm")
    config = {
        "selection": "grouped CV audit followed by later-cohort validation gate",
        "selected_candidate_ids": [first_index + 1, second_index + 1],
        "primary_params": candidates[first_index],
        "secondary_params": candidates[second_index],
        "primary_weight": first_weight,
        "secondary_weight": 1.0 - first_weight,
        "temperature": temperature,
        "threshold": threshold,
        "feature_columns": selected_features,
        "categorical_features": categorical,
        "old_ensemble_blend_weight": old_weight,
        "new_ensemble_blend_weight": 1.0 - old_weight,
        "cross_blend_threshold": cross_threshold,
    }
    (ARTIFACTS / "catboost_later_gated_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        comparison.loc[comparison["split"].isin(["validation", "test"]), [
            "model", "split", "roc_auc", "pr_auc", "f1", "recall",
            "accuracy", "brier_score", "ece_10_bin", "threshold"
        ]].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

