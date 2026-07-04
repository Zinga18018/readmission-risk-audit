from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
}


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)
    cleaned["early_readmission"] = (cleaned["readmitted"] == "<30").astype(int)
    cleaned = cleaned.drop(columns=["readmitted"], errors="ignore")

    for column in DROP_COLUMNS:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=column)

    return cleaned


def split_features(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "early_readmission" not in cleaned.columns:
        raise ValueError("expected early_readmission column after cleaning")

    y = cleaned["early_readmission"].astype(int)
    X = cleaned.drop(columns=["early_readmission"])
    return X, y


def make_model(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def threshold_table(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6),
) -> pd.DataFrame:
    y_arr = np.asarray(list(y_true), dtype=int)
    p_arr = np.asarray(list(probabilities), dtype=float)
    rows = []

    for threshold in thresholds:
        pred = (p_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "flagged_rate": float(pred.mean()),
                "precision": precision_score(y_arr, pred, zero_division=0),
                "recall": recall_score(y_arr, pred, zero_division=0),
                "f1": f1_score(y_arr, pred, zero_division=0),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_group(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    y_true: pd.Series,
    group_col: str,
) -> pd.DataFrame:
    working = frame[[group_col]].copy()
    working["probability"] = probabilities
    working["actual"] = y_true.to_numpy()
    summary = (
        working.groupby(group_col, dropna=False)
        .agg(
            rows=("actual", "size"),
            observed_readmission_rate=("actual", "mean"),
            mean_predicted_risk=("probability", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    return summary


def coefficient_table(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]
    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    )
    return table.sort_values("abs_coefficient", ascending=False).head(top_n)


def train_and_evaluate(df: pd.DataFrame) -> dict[str, object]:
    cleaned = clean_diabetes_data(df)
    X, y = split_features(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_model(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds = threshold_table(y_test, probabilities)
    best_row = thresholds.sort_values(["f1", "recall"], ascending=False).iloc[0]

    metrics = {
        "rows": int(len(df)),
        "features_after_drop": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "best_threshold": float(best_row["threshold"]),
        "best_threshold_precision": float(best_row["precision"]),
        "best_threshold_recall": float(best_row["recall"]),
        "best_threshold_f1": float(best_row["f1"]),
        "flagged_rate_at_best_threshold": float(best_row["flagged_rate"]),
    }

    group_tables = {}
    for group_col in ["age", "race", "gender"]:
        if group_col in X_test.columns:
            group_tables[group_col] = summarize_by_group(
                X_test,
                probabilities,
                y_test,
                group_col,
            )

    return {
        "model": model,
        "metrics": metrics,
        "thresholds": thresholds,
        "top_coefficients": coefficient_table(model),
        "group_tables": group_tables,
    }


def write_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    result["thresholds"].to_csv(output_dir / "threshold_table.csv", index=False)
    result["top_coefficients"].to_csv(
        output_dir / "top_coefficients.csv",
        index=False,
    )

    for name, table in result["group_tables"].items():
        table.to_csv(output_dir / f"group_summary_{name}.csv", index=False)
