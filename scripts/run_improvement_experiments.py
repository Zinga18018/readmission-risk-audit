from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from readmission_audit.experiments import (
    AutoencoderCandidate,
    autocorrelation_audit,
    candidate_pruned_features,
    categorical_chi_square_audit,
    encode_matrix,
    feature_quality_audit,
    train_autoencoder,
)
from readmission_audit.pipeline import (
    FTPreprocessor,
    SEED,
    _predict_ft_logits,
    _train_one_ft_transformer,
    catboost_blend_sweep,
    clean_diabetes_data,
    evaluate_probabilities,
    feature_types,
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
RAW_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def score_pair(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return (
        float(average_precision_score(y, probability)),
        float(roc_auc_score(y, probability)),
    )


def evaluate_three_splits(
    name: str,
    probabilities: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], float]:
    threshold = select_f1_threshold(targets["validation"], probabilities["validation"])
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        row = evaluate_probabilities(targets[split], probabilities[split], threshold)
        row.update({"model": name, "split": split})
        rows.append(row)
    return rows, threshold


def main() -> None:
    started = time.perf_counter()
    set_reproducible_seed(SEED)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and splitting data", flush=True)
    raw = pd.read_csv(RAW_DATA)
    cleaned, cohort_audit = clean_diabetes_data(raw)
    splits, split_audit = make_patient_order_split(cleaned)
    features: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    for split, frame in splits.items():
        X, y = make_features(frame)
        features[split] = X
        targets[split] = y.to_numpy(dtype=int)

    X_train = features["train"]
    y_train = targets["train"]
    quality = feature_quality_audit(X_train)
    quality.to_csv(OUTPUTS / "feature_quality_audit.csv", index=False)

    raw_missing = (
        raw.replace("?", np.nan)
        .isna()
        .agg(["sum", "mean"])
        .T.reset_index(names="feature")
        .rename(columns={"sum": "missing_rows", "mean": "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    raw_missing["decision"] = "retain"
    raw_missing.loc[
        raw_missing["feature"].isin(["weight", "payer_code", "medical_specialty"]),
        "decision",
    ] = "drop_high_missing_raw_field"
    raw_missing.loc[
        raw_missing["feature"].isin(["max_glu_serum", "A1Cresult"]),
        "decision",
    ] = "retain_as_category_plus_recorded_indicator"
    raw_missing.to_csv(OUTPUTS / "raw_missingness_audit.csv", index=False)

    print("[2/7] Chi-square and train-only feature-quality audits", flush=True)
    chi_square = categorical_chi_square_audit(X_train, y_train)
    chi_square.to_csv(OUTPUTS / "categorical_chi_square_audit.csv", index=False)
    ordered_train = splits["train"].sort_values("encounter_id")
    autocorrelation_audit(
        ordered_train["early_readmission"],
        "train_target_encounter_order_proxy",
    ).to_csv(OUTPUTS / "autocorrelation_diagnostics.csv", index=False)

    print("[3/7] Validation-only feature pruning ablation", flush=True)
    feature_candidates = candidate_pruned_features(quality)
    ablation_rows: list[dict[str, object]] = []
    ablation_models: dict[str, CatBoostClassifier] = {}
    for candidate_name, dropped in feature_candidates.items():
        selected = [column for column in X_train.columns if column not in dropped]
        _, categorical = feature_types(X_train[selected])
        prepared = {
            split: prepare_catboost_features(features[split][selected], categorical)
            for split in ("train", "validation")
        }
        model = CatBoostClassifier(
            iterations=450,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="PRAUC",
            l2_leaf_reg=8.0,
            random_strength=0.5,
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            random_seed=SEED,
            allow_writing_files=False,
            verbose=False,
            thread_count=8,
            cat_features=categorical,
        )
        model.fit(prepared["train"], y_train, verbose=False)
        probability = model.predict_proba(prepared["validation"])[:, 1]
        pr_auc, roc_auc = score_pair(targets["validation"], probability)
        ablation_rows.append(
            {
                "candidate": candidate_name,
                "feature_count": len(selected),
                "dropped_feature_count": len(dropped),
                "dropped_features": "|".join(dropped),
                "validation_pr_auc": pr_auc,
                "validation_roc_auc": roc_auc,
            }
        )
        ablation_models[candidate_name] = model
        print(
            f"  {candidate_name}: {len(selected)} features, PR-AUC={pr_auc:.5f}",
            flush=True,
        )

    ablation = pd.DataFrame(ablation_rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )
    ablation.to_csv(OUTPUTS / "feature_pruning_ablation.csv", index=False)
    selected_candidate = str(ablation.iloc[0]["candidate"])
    dropped_features = feature_candidates[selected_candidate]
    selected_features = [
        column for column in X_train.columns if column not in dropped_features
    ]
    selected_frames = {
        split: features[split][selected_features].copy()
        for split in ("train", "validation", "test")
    }
    _, categorical_features = feature_types(selected_frames["train"])
    catboost_frames = {
        split: prepare_catboost_features(frame, categorical_features)
        for split, frame in selected_frames.items()
    }

    print(
        f"[4/7] Patient-grouped CatBoost GridSearchCV on {selected_candidate}",
        flush=True,
    )
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
    ]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    groups = splits["train"].loc[selected_frames["train"].index, "patient_nbr"]
    fold_indices = list(
        cv.split(catboost_frames["train"], y_train, groups=groups)
    )
    grid_rows: list[dict[str, object]] = []
    for candidate_index, params in enumerate(ParameterGrid(grids), start=1):
        fold_rows: list[dict[str, float]] = []
        for fold, (fit_indices, score_indices) in enumerate(fold_indices, start=1):
            fold_model = CatBoostClassifier(
                iterations=650,
                loss_function="Logloss",
                eval_metric="PRAUC",
                random_seed=SEED,
                allow_writing_files=False,
                verbose=False,
                thread_count=8,
                bootstrap_type="Bayesian",
                cat_features=categorical_features,
                **params,
            )
            fold_model.fit(
                catboost_frames["train"].iloc[fit_indices],
                y_train[fit_indices],
                verbose=False,
            )
            train_probability = fold_model.predict_proba(
                catboost_frames["train"].iloc[fit_indices]
            )[:, 1]
            score_probability = fold_model.predict_proba(
                catboost_frames["train"].iloc[score_indices]
            )[:, 1]
            train_pr, train_roc = score_pair(y_train[fit_indices], train_probability)
            score_pr, score_roc = score_pair(y_train[score_indices], score_probability)
            fold_rows.append(
                {
                    "train_pr_auc": train_pr,
                    "train_roc_auc": train_roc,
                    "test_pr_auc": score_pr,
                    "test_roc_auc": score_roc,
                }
            )
        fold_frame = pd.DataFrame(fold_rows)
        row: dict[str, object] = {
            "candidate": candidate_index,
            **params,
            "mean_train_pr_auc": float(fold_frame["train_pr_auc"].mean()),
            "mean_test_pr_auc": float(fold_frame["test_pr_auc"].mean()),
            "std_test_pr_auc": float(fold_frame["test_pr_auc"].std(ddof=1)),
            "mean_train_roc_auc": float(fold_frame["train_roc_auc"].mean()),
            "mean_test_roc_auc": float(fold_frame["test_roc_auc"].mean()),
            "std_test_roc_auc": float(fold_frame["test_roc_auc"].std(ddof=1)),
            "pr_auc_train_cv_gap": float(
                fold_frame["train_pr_auc"].mean()
                - fold_frame["test_pr_auc"].mean()
            ),
        }
        for fold, metrics in enumerate(fold_rows, start=1):
            row[f"fold_{fold}_test_pr_auc"] = metrics["test_pr_auc"]
            row[f"fold_{fold}_test_roc_auc"] = metrics["test_roc_auc"]
        grid_rows.append(row)
        print(
            f"  grid {candidate_index}/{len(list(ParameterGrid(grids)))}: "
            f"CV PR-AUC={row['mean_test_pr_auc']:.5f}",
            flush=True,
        )
    search_frame = pd.DataFrame(grid_rows).sort_values(
        ["mean_test_pr_auc", "mean_test_roc_auc"], ascending=False
    )
    search_frame.to_csv(OUTPUTS / "catboost_group_grid_search.csv", index=False)

    parameter_names = list(next(iter(ParameterGrid(grids))).keys())
    best_params = {
        name: search_frame.iloc[0][name].item()
        if hasattr(search_frame.iloc[0][name], "item")
        else search_frame.iloc[0][name]
        for name in parameter_names
    }
    best_cv_pr_auc = float(search_frame.iloc[0]["mean_test_pr_auc"])
    final_catboost = CatBoostClassifier(
        iterations=3000,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
        thread_count=8,
        bootstrap_type="Bayesian",
        cat_features=categorical_features,
        **best_params,
    )
    final_catboost.fit(
        catboost_frames["train"],
        y_train,
        eval_set=(catboost_frames["validation"], targets["validation"]),
        early_stopping_rounds=120,
        use_best_model=True,
        verbose=False,
    )
    grid_probabilities_raw = {
        split: final_catboost.predict_proba(frame)[:, 1]
        for split, frame in catboost_frames.items()
    }
    grid_logits = {
        split: positive_probability_logits(probability)
        for split, probability in grid_probabilities_raw.items()
    }
    grid_temperature = tune_temperature(
        grid_logits["validation"], targets["validation"]
    )
    grid_probabilities = {
        split: softmax_probabilities(logits, grid_temperature)[:, 1]
        for split, logits in grid_logits.items()
    }
    final_catboost.save_model(ARTIFACTS / "catboost_grid_model.cbm")

    print("[5/7] Dense zero-dropout FT-Transformer", flush=True)
    ft_preprocessor = FTPreprocessor(selected_frames["train"]).fit(
        selected_frames["train"]
    )
    ft_arrays = {
        split: ft_preprocessor.transform(frame)
        for split, frame in selected_frames.items()
    }
    ft_model, ft_history, ft_summary = _train_one_ft_transformer(
        *ft_arrays["train"],
        y_train,
        *ft_arrays["validation"],
        targets["validation"],
        ft_preprocessor.category_cardinalities,
        attention_dropout=0.0,
        token_dimension=96,
        transformer_blocks=4,
        attention_heads=8,
        feedforward_dimension=256,
        feedforward_dropout=0.0,
        max_epochs=16,
        patience=4,
        batch_size=384,
    )
    ft_history.to_csv(OUTPUTS / "ft_dense_zero_dropout_history.csv", index=False)
    ft_logits = {
        split: _predict_ft_logits(ft_model, *arrays)
        for split, arrays in ft_arrays.items()
    }
    ft_temperature = tune_temperature(ft_logits["validation"], targets["validation"])
    ft_probabilities = {
        split: softmax_probabilities(logits, ft_temperature)[:, 1]
        for split, logits in ft_logits.items()
    }
    joblib.dump(ft_preprocessor, ARTIFACTS / "ft_dense_preprocessor.joblib")
    torch.save(ft_model.state_dict(), ARTIFACTS / "ft_dense_zero_dropout_state.pt")

    print("[6/7] Autoencoder bottleneck experiments", flush=True)
    ae_preprocessor = make_linear_preprocessor(selected_frames["train"])
    ae_matrices = {
        "train": ae_preprocessor.fit_transform(selected_frames["train"]),
        "validation": None,
        "test": None,
    }
    ae_matrices["validation"] = ae_preprocessor.transform(
        selected_frames["validation"]
    )
    ae_matrices["test"] = ae_preprocessor.transform(selected_frames["test"])
    ae_candidates: list[AutoencoderCandidate] = []
    ae_histories: list[pd.DataFrame] = []
    ae_models: dict[int, object] = {}
    ae_latent: dict[int, dict[str, np.ndarray]] = {}
    for latent_dim in (32, 64):
        autoencoder, history = train_autoencoder(
            ae_matrices["train"], ae_matrices["validation"], latent_dim
        )
        ae_histories.append(history)
        ae_models[latent_dim] = autoencoder
        latent = {
            split: encode_matrix(autoencoder, matrix)
            for split, matrix in ae_matrices.items()
        }
        ae_latent[latent_dim] = latent
        classifiers = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=SEED
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=50,
                l2_regularization=2.0,
                early_stopping=True,
                random_state=SEED,
            ),
        }
        for classifier_name, classifier in classifiers.items():
            classifier.fit(latent["train"], y_train)
            probability = classifier.predict_proba(latent["validation"])[:, 1]
            pr_auc, roc_auc = score_pair(targets["validation"], probability)
            ae_candidates.append(
                AutoencoderCandidate(
                    latent_dim=latent_dim,
                    model=autoencoder,
                    classifier_name=classifier_name,
                    classifier=classifier,
                    validation_pr_auc=pr_auc,
                    validation_roc_auc=roc_auc,
                )
            )
    pd.concat(ae_histories, ignore_index=True).to_csv(
        OUTPUTS / "autoencoder_history.csv", index=False
    )
    best_ae = max(
        ae_candidates,
        key=lambda item: (item.validation_pr_auc, item.validation_roc_auc),
    )
    best_latent = ae_latent[best_ae.latent_dim]
    ae_probabilities_raw = {
        split: best_ae.classifier.predict_proba(matrix)[:, 1]
        for split, matrix in best_latent.items()
    }
    ae_logits = {
        split: positive_probability_logits(probability)
        for split, probability in ae_probabilities_raw.items()
    }
    ae_temperature = tune_temperature(ae_logits["validation"], targets["validation"])
    ae_probabilities = {
        split: softmax_probabilities(logits, ae_temperature)[:, 1]
        for split, logits in ae_logits.items()
    }
    ae_comparison = pd.DataFrame(
        [
            {
                "latent_dim": candidate.latent_dim,
                "classifier": candidate.classifier_name,
                "validation_pr_auc": candidate.validation_pr_auc,
                "validation_roc_auc": candidate.validation_roc_auc,
                "selected": candidate is best_ae,
            }
            for candidate in ae_candidates
        ]
    ).sort_values("validation_pr_auc", ascending=False)
    ae_comparison.to_csv(OUTPUTS / "autoencoder_comparison.csv", index=False)
    torch.save(best_ae.model.state_dict(), ARTIFACTS / "autoencoder_state.pt")
    joblib.dump(ae_preprocessor, ARTIFACTS / "autoencoder_preprocessor.joblib")
    joblib.dump(best_ae.classifier, ARTIFACTS / "autoencoder_classifier.joblib")

    print("[7/7] Frozen test evaluation and ordering-proxy residual ACF", flush=True)
    comparison_rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    for name, probabilities in (
        ("CatBoost grouped-grid temperature-scaled", grid_probabilities),
        ("FT-Transformer dense zero-dropout temperature-scaled", ft_probabilities),
        (
            f"Autoencoder-{best_ae.latent_dim} + {best_ae.classifier_name}",
            ae_probabilities,
        ),
    ):
        rows, threshold = evaluate_three_splits(name, probabilities, targets)
        comparison_rows.extend(rows)
        thresholds[name] = threshold

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUTS / "improvement_model_comparison.csv", index=False)

    ordered_test_indices = splits["test"].sort_values("encounter_id").index
    test_position = pd.Series(
        np.arange(len(splits["test"])), index=splits["test"].index
    ).loc[ordered_test_indices].to_numpy()
    test_residuals = (
        targets["test"][test_position]
        - grid_probabilities["test"][test_position]
    )
    acf_existing = pd.read_csv(OUTPUTS / "autocorrelation_diagnostics.csv")
    residual_acf = autocorrelation_audit(
        test_residuals, "catboost_test_residual_encounter_order_proxy"
    )
    pd.concat([acf_existing, residual_acf], ignore_index=True).to_csv(
        OUTPUTS / "autocorrelation_diagnostics.csv", index=False
    )

    config = {
        "selected_feature_candidate": selected_candidate,
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "categorical_features": categorical_features,
        "best_params": best_params,
        "best_iteration": int(final_catboost.get_best_iteration()),
        "temperature": grid_temperature,
        "threshold": thresholds["CatBoost grouped-grid temperature-scaled"],
        "raw_feature_columns": [
            column
            for column in splits["train"].drop(
                columns=["early_readmission", "encounter_id", "patient_nbr"],
                errors="ignore",
            ).columns
        ],
    }
    (ARTIFACTS / "catboost_grid_config.json").write_text(
        json.dumps(json_ready(config), indent=2), encoding="utf-8"
    )
    ft_config = {
        **ft_summary,
        "temperature": ft_temperature,
        "threshold": thresholds[
            "FT-Transformer dense zero-dropout temperature-scaled"
        ],
        "numeric_features": len(ft_preprocessor.numeric_features),
        "category_cardinalities": ft_preprocessor.category_cardinalities,
        "selected_features": selected_features,
    }
    (ARTIFACTS / "ft_dense_zero_dropout_config.json").write_text(
        json.dumps(json_ready(ft_config), indent=2), encoding="utf-8"
    )
    ae_config = {
        "latent_dim": best_ae.latent_dim,
        "classifier": best_ae.classifier_name,
        "temperature": ae_temperature,
        "threshold": thresholds[
            f"Autoencoder-{best_ae.latent_dim} + {best_ae.classifier_name}"
        ],
        "input_dim": int(ae_matrices["train"].shape[1]),
        "selected_features": selected_features,
    }
    (ARTIFACTS / "autoencoder_config.json").write_text(
        json.dumps(json_ready(ae_config), indent=2), encoding="utf-8"
    )

    validation_rank = comparison.loc[comparison["split"] == "validation"].sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )
    test_rank = comparison.loc[comparison["split"] == "test"].sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )
    summary = {
        "runtime_seconds": time.perf_counter() - started,
        "cohort": cohort_audit,
        "split": split_audit,
        "selection_rule": "validation PR-AUC, then validation ROC-AUC",
        "test_not_used_for_feature_or_hyperparameter_selection": True,
        "selected_feature_candidate": selected_candidate,
        "feature_count_before": int(X_train.shape[1]),
        "feature_count_after": len(selected_features),
        "dropped_features": dropped_features,
        "catboost_group_cv_best_pr_auc": best_cv_pr_auc,
        "catboost_best_params": best_params,
        "catboost_best_iteration": int(final_catboost.get_best_iteration()),
        "dense_ft": ft_summary,
        "autoencoder_selected": {
            "latent_dim": best_ae.latent_dim,
            "classifier": best_ae.classifier_name,
            "validation_pr_auc": best_ae.validation_pr_auc,
            "validation_roc_auc": best_ae.validation_roc_auc,
        },
        "validation_best": validation_rank.iloc[0].to_dict(),
        "test_results": test_rank.to_dict(orient="records"),
        "chi_square_interpretation": (
            "Target-independence chi-square with Cramer's V is the useful "
            "categorical association diagnostic. Uniform goodness-of-fit does "
            "not establish that clinical data are neat or non-random."
        ),
        "acf_interpretation": (
            "The source has no row-level dates. ACF and Ljung-Box results use "
            "encounter-ID order only and are not claimed as temporal stationarity tests."
        ),
    }
    (OUTPUTS / "improvement_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
