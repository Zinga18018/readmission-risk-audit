from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from tabm import TabM
from xgboost import XGBClassifier

from readmission_audit.pipeline import (
    FTPreprocessor,
    SEED,
    clean_diabetes_data,
    evaluate_probabilities,
    make_features,
    make_linear_preprocessor,
    make_patient_order_split,
    positive_probability_logits,
    prepare_catboost_features,
    select_f1_threshold,
    set_reproducible_seed,
    softmax_probabilities,
    tune_temperature,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"


def load_catboost(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def old_catboost_probabilities(
    features: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    config = json.loads(
        (ARTIFACTS / "catboost_config.json").read_text(encoding="utf-8")
    )
    primary = load_catboost(ARTIFACTS / "catboost_model.cbm")
    secondary = load_catboost(ARTIFACTS / "catboost_secondary_model.cbm")
    probabilities: dict[str, np.ndarray] = {}
    for split, frame in features.items():
        prepared = prepare_catboost_features(
            frame.reindex(columns=config["feature_columns"]),
            config["categorical_features"],
        )
        raw = (
            config["primary_weight"] * primary.predict_proba(prepared)[:, 1]
            + config["secondary_weight"]
            * secondary.predict_proba(prepared)[:, 1]
        )
        probabilities[split] = softmax_probabilities(
            positive_probability_logits(raw), config["temperature"]
        )[:, 1]
    return probabilities


def tune_and_scale(
    probabilities: dict[str, np.ndarray], y_validation: np.ndarray
) -> tuple[dict[str, np.ndarray], float]:
    logits = {
        split: positive_probability_logits(probability)
        for split, probability in probabilities.items()
    }
    temperature = tune_temperature(logits["validation"], y_validation)
    return (
        {
            split: softmax_probabilities(split_logits, temperature)[:, 1]
            for split, split_logits in logits.items()
        },
        temperature,
    )


def evaluate_model(
    name: str,
    probabilities: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    threshold = select_f1_threshold(
        targets["validation"], probabilities["validation"]
    )
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        row = evaluate_probabilities(
            targets[split], probabilities[split], threshold
        )
        row.update({"model": name, "split": split})
        rows.append(row)
    return rows


def predict_tabm(
    model: TabM,
    numeric: np.ndarray,
    categorical: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            logits = model(
                torch.from_numpy(numeric[start : start + batch_size]),
                torch.from_numpy(categorical[start : start + batch_size]),
            ).squeeze(-1)
            rows.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
    return np.concatenate(rows)


def train_tabm(
    train_numeric: np.ndarray,
    train_categorical: np.ndarray,
    y_train: np.ndarray,
    validation_numeric: np.ndarray,
    validation_categorical: np.ndarray,
    y_validation: np.ndarray,
    category_cardinalities: list[int],
) -> tuple[TabM, pd.DataFrame]:
    set_reproducible_seed(SEED)
    model = TabM.make(
        n_num_features=train_numeric.shape[1],
        cat_cardinalities=category_cardinalities,
        d_out=1,
        arch_type="tabm-mini",
        k=16,
        n_blocks=3,
        d_block=256,
        dropout=0.10,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=1e-4
    )
    rng = np.random.default_rng(SEED)
    best_state = copy.deepcopy(model.state_dict())
    best_pr_auc = -math.inf
    stale = 0
    history: list[dict[str, float]] = []
    batch_size = 512

    for epoch in range(1, 21):
        model.train()
        order = rng.permutation(len(y_train))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            numeric = torch.from_numpy(train_numeric[indices])
            categorical = torch.from_numpy(train_categorical[indices])
            target = torch.from_numpy(
                y_train[indices].astype(np.float32, copy=False)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric, categorical).squeeze(-1)
            expanded_target = target[:, None].expand_as(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, expanded_target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation_probability = predict_tabm(
            model, validation_numeric, validation_categorical
        )
        validation_pr_auc = float(
            average_precision_score(y_validation, validation_probability)
        )
        validation_roc_auc = float(
            roc_auc_score(y_validation, validation_probability)
        )
        history.append(
            {
                "epoch": epoch,
                "train_binary_cross_entropy": float(np.mean(losses)),
                "validation_pr_auc": validation_pr_auc,
                "validation_roc_auc": validation_roc_auc,
            }
        )
        print(
            f"TabM epoch {epoch}: validation PR-AUC={validation_pr_auc:.5f}",
            flush=True,
        )
        if validation_pr_auc > best_pr_auc + 1e-5:
            best_pr_auc = validation_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def three_model_stack_sweep(
    y_validation: np.ndarray,
    catboost_probability: np.ndarray,
    xgboost_probability: np.ndarray,
    tabm_probability: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for catboost_weight in np.linspace(0.0, 1.0, 21):
        for xgboost_weight in np.linspace(
            0.0, 1.0 - catboost_weight, int(round((1 - catboost_weight) * 20)) + 1
        ):
            tabm_weight = 1.0 - catboost_weight - xgboost_weight
            probability = (
                catboost_weight * catboost_probability
                + xgboost_weight * xgboost_probability
                + tabm_weight * tabm_probability
            )
            rows.append(
                {
                    "catboost_weight": float(catboost_weight),
                    "xgboost_weight": float(xgboost_weight),
                    "tabm_weight": float(tabm_weight),
                    "validation_pr_auc": float(
                        average_precision_score(y_validation, probability)
                    ),
                    "validation_roc_auc": float(
                        roc_auc_score(y_validation, probability)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )


def main() -> None:
    set_reproducible_seed(SEED)
    raw = pd.read_csv(ROOT / "data" / "raw" / "diabetic_data.csv")
    cleaned, _ = clean_diabetes_data(raw)
    splits, split_audit = make_patient_order_split(cleaned)
    features: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    for split, frame in splits.items():
        X, y = make_features(frame)
        features[split] = X
        targets[split] = y.to_numpy(dtype=int)

    print("[1/4] XGBoost validation search", flush=True)
    preprocessor = make_linear_preprocessor(features["train"])
    matrices = {
        "train": preprocessor.fit_transform(features["train"]),
        "validation": None,
        "test": None,
    }
    matrices["validation"] = preprocessor.transform(features["validation"])
    matrices["test"] = preprocessor.transform(features["test"])
    imbalance = float(
        (targets["train"] == 0).sum() / (targets["train"] == 1).sum()
    )
    candidates = [
        {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
         "gamma": 0.0, "scale_pos_weight": 1.0},
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5,
         "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 10.0,
         "gamma": 0.0, "scale_pos_weight": 1.0},
        {"max_depth": 5, "learning_rate": 0.02, "min_child_weight": 10,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0,
         "gamma": 0.1, "scale_pos_weight": 1.0},
        {"max_depth": 4, "learning_rate": 0.04, "min_child_weight": 10,
         "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 10.0,
         "gamma": 0.05, "scale_pos_weight": math.sqrt(imbalance)},
        {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 15,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 15.0,
         "gamma": 0.1, "scale_pos_weight": math.sqrt(imbalance)},
        {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 10,
         "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 10.0,
         "gamma": 0.0, "scale_pos_weight": imbalance},
    ]
    xgb_models: list[XGBClassifier] = []
    xgb_probabilities: list[dict[str, np.ndarray]] = []
    xgb_rows: list[dict[str, object]] = []
    for index, params in enumerate(candidates, start=1):
        model = XGBClassifier(
            n_estimators=2500,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=100,
            tree_method="hist",
            max_bin=256,
            random_state=SEED,
            n_jobs=8,
            **params,
        )
        model.fit(
            matrices["train"],
            targets["train"],
            eval_set=[(matrices["validation"], targets["validation"])],
            verbose=False,
        )
        probability = {
            split: model.predict_proba(matrix)[:, 1]
            for split, matrix in matrices.items()
        }
        xgb_models.append(model)
        xgb_probabilities.append(probability)
        xgb_rows.append(
            {
                "candidate": index,
                **params,
                "best_iteration": int(model.best_iteration),
                "validation_pr_auc": float(
                    average_precision_score(
                        targets["validation"], probability["validation"]
                    )
                ),
                "validation_roc_auc": float(
                    roc_auc_score(
                        targets["validation"], probability["validation"]
                    )
                ),
            }
        )
        print(
            f"XGBoost {index}/{len(candidates)}: validation "
            f"PR-AUC={xgb_rows[-1]['validation_pr_auc']:.5f}",
            flush=True,
        )
    xgb_search = pd.DataFrame(xgb_rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )
    xgb_search.to_csv(OUTPUTS / "xgboost_validation_search.csv", index=False)
    xgb_index = int(xgb_search.iloc[0]["candidate"]) - 1
    xgb_model = xgb_models[xgb_index]
    xgb_scaled, xgb_temperature = tune_and_scale(
        xgb_probabilities[xgb_index], targets["validation"]
    )
    joblib.dump(preprocessor, ARTIFACTS / "xgboost_preprocessor.joblib")
    xgb_model.save_model(ARTIFACTS / "xgboost_model.json")

    print("[2/4] TabM parameter-efficient ensemble", flush=True)
    tabm_preprocessor = FTPreprocessor(features["train"]).fit(features["train"])
    tabm_arrays = {
        split: tabm_preprocessor.transform(frame)
        for split, frame in features.items()
    }
    tabm_model, tabm_history = train_tabm(
        *tabm_arrays["train"],
        targets["train"],
        *tabm_arrays["validation"],
        targets["validation"],
        tabm_preprocessor.category_cardinalities,
    )
    tabm_history.to_csv(OUTPUTS / "tabm_history.csv", index=False)
    tabm_raw = {
        split: predict_tabm(tabm_model, *arrays)
        for split, arrays in tabm_arrays.items()
    }
    tabm_scaled, tabm_temperature = tune_and_scale(
        tabm_raw, targets["validation"]
    )
    joblib.dump(tabm_preprocessor, ARTIFACTS / "tabm_preprocessor.joblib")
    torch.save(tabm_model.state_dict(), ARTIFACTS / "tabm_state.pt")

    print("[3/4] Validation-only three-model stack", flush=True)
    catboost = old_catboost_probabilities(features)
    sweep = three_model_stack_sweep(
        targets["validation"],
        catboost["validation"],
        xgb_scaled["validation"],
        tabm_scaled["validation"],
    )
    sweep.to_csv(OUTPUTS / "catboost_xgboost_tabm_stack_sweep.csv", index=False)
    selected_weights = sweep.iloc[0]
    stack_raw = {
        split: (
            selected_weights["catboost_weight"] * catboost[split]
            + selected_weights["xgboost_weight"] * xgb_scaled[split]
            + selected_weights["tabm_weight"] * tabm_scaled[split]
        )
        for split in ("train", "validation", "test")
    }
    stack, stack_temperature = tune_and_scale(
        stack_raw, targets["validation"]
    )

    print("[4/4] Frozen test comparison", flush=True)
    rows: list[dict[str, object]] = []
    for name, probability in (
        ("Existing CatBoost ensemble", catboost),
        ("XGBoost temperature-scaled", xgb_scaled),
        ("TabM temperature-scaled", tabm_scaled),
        ("CatBoost + XGBoost + TabM stack", stack),
    ):
        rows.extend(evaluate_model(name, probability, targets))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUTS / "xgboost_tabm_stack_comparison.csv", index=False)

    config = {
        "xgboost": {
            "candidate": xgb_index + 1,
            "params": candidates[xgb_index],
            "temperature": xgb_temperature,
        },
        "tabm": {
            "architecture": "tabm-mini",
            "k": 16,
            "n_blocks": 3,
            "d_block": 256,
            "dropout": 0.10,
            "temperature": tabm_temperature,
            "category_cardinalities": tabm_preprocessor.category_cardinalities,
            "numeric_features": len(tabm_preprocessor.numeric_features),
        },
        "stack": {
            "catboost_weight": float(selected_weights["catboost_weight"]),
            "xgboost_weight": float(selected_weights["xgboost_weight"]),
            "tabm_weight": float(selected_weights["tabm_weight"]),
            "temperature": stack_temperature,
        },
        "patient_overlap_counts": split_audit["patient_overlap_counts"],
        "test_used_for_selection": False,
    }
    (ARTIFACTS / "xgboost_tabm_stack_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        comparison[comparison["split"].isin(["validation", "test"])][
            [
                "model", "split", "roc_auc", "pr_auc", "f1", "recall",
                "accuracy", "brier_score", "ece_10_bin", "threshold"
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

