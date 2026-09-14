"""Evaluation without model training or heavyweight inference dependencies."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED_COLUMNS = ("encounter_id", "patient_id", "y_true", "y_prob")


def validate_predictions(frame: pd.DataFrame, missing_policy: str = "error") -> tuple[pd.DataFrame, dict]:
    """One encounter per row; patient IDs may repeat and outcomes may vary within patient."""
    if missing_policy not in {"error", "drop"}:
        raise ValueError("missing_policy must be error or drop")
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    data = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    input_rows = len(data)
    for column in ("encounter_id", "patient_id"):
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)
    missing = data.isna().any(axis=1)
    missing_rows = int(missing.sum())
    if missing_rows and missing_policy == "error":
        raise ValueError(f"{missing_rows} rows have missing required values; no rows were silently dropped")
    data = data.loc[~missing].reset_index(drop=True)
    if data.empty:
        raise ValueError("No complete predictions remain")
    if data.encounter_id.duplicated().any():
        raise ValueError("encounter_id must be unique; repeated patients belong in patient_id")
    for column in ("y_true", "y_prob"):
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be numeric") from error
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite")
    if not data.y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 labels")
    if not data.y_prob.between(0, 1).all():
        raise ValueError("y_prob must be between 0 and 1")
    data["y_true"] = data.y_true.astype(int)
    data["y_prob"] = data.y_prob.astype(float)
    return data, {"input_rows": input_rows, "included_rows": len(data),
                  "missing_rows_excluded": missing_rows, "missing_policy": missing_policy,
                  "included_patients": int(data.patient_id.nunique())}


def _metric_values(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    flags = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(flags & positive))
    fp = int(np.sum(flags & ~positive))
    fn = int(np.sum(~flags & positive))
    both_classes = np.unique(labels).size == 2
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "flagged_rate": float(np.mean(flags)),
    }


def patient_cluster_bootstrap(frame: pd.DataFrame, *, frozen_threshold: float,
                              n_resamples: int = 2000, seed: int = 42,
                              confidence_level: float = .95, missing_policy: str = "error",
                              min_valid_fraction: float = .8) -> dict:
    """Resample patients, retaining all of each sampled patient's encounters.

    Metrics remain encounter-weighted. No refitting, recalibration or threshold
    optimization occurs within a replicate. Percentile intervals are conditional
    on the frozen model and observed cohort, not a correction for test reuse.
    """
    if not math.isfinite(frozen_threshold) or not 0 <= frozen_threshold <= 1:
        raise ValueError("frozen_threshold must be finite and between 0 and 1")
    if type(n_resamples) is not int or n_resamples < 20:
        raise ValueError("n_resamples must be an integer of at least 20")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < confidence_level < 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Invalid confidence level or minimum valid fraction")
    data, accounting = validate_predictions(frame, missing_policy)
    groups = list(data.groupby("patient_id", sort=True).indices.values())
    if len(groups) < 2:
        raise ValueError("At least two patient clusters are required")
    labels, probabilities = data.y_true.to_numpy(), data.y_prob.to_numpy()
    point = _metric_values(labels, probabilities, frozen_threshold)
    draws = {metric: [] for metric in point}
    rng = np.random.default_rng(seed)
    single_class = 0
    replicate_rows = []
    for _ in range(n_resamples):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        replicate_rows.append(len(indices))
        y, p = labels[indices], probabilities[indices]
        single_class += int(np.unique(y).size < 2)
        for metric, value in _metric_values(y, p, frozen_threshold).items():
            if value is not None:
                draws[metric].append(value)
    alpha = 1 - confidence_level
    required_valid = max(20, math.ceil(min_valid_fraction * n_resamples))
    intervals = {}
    for metric, values in draws.items():
        available = point[metric] is not None and len(values) >= required_valid
        bounds = np.quantile(values, [alpha / 2, 1 - alpha / 2]) if available else [None, None]
        intervals[metric] = {"estimate": point[metric],
                             "ci_low": float(bounds[0]) if available else None,
                             "ci_high": float(bounds[1]) if available else None,
                             "valid_replicates": len(values),
                             "undefined_replicates": n_resamples - len(values),
                             "status": "available" if available else "unavailable_or_insufficient_valid_replicates"}
    return {"method": "patient-cluster percentile bootstrap",
            "estimand": "encounter-weighted performance conditional on frozen model and observed cohort",
            "frozen_threshold": frozen_threshold, "threshold_retuned": False,
            "seed": seed, "n_resamples": n_resamples, "confidence_level": confidence_level,
            "minimum_valid_replicates": required_valid,
            "single_class_replicates": single_class,
            "replicate_row_count_range": [min(replicate_rows), max(replicate_rows)],
            "input_accounting": accounting, "metrics": intervals,
            "limitations": ["Does not undo historical test inspection or selection bias.",
                            "Does not include model fitting or calibration uncertainty.",
                            "Assumes independent sampled patients; hospital clustering and calendar-time transfer are not addressed.",
                            "Undefined metric replicates are counted; reported intervals use only valid replicates."]}


def export_private_predictions(identifiers: pd.DataFrame, labels, probabilities,
                               destination: Path) -> dict:
    """Export aligned identifiers and predictions; caller must keep destination private."""
    if len(identifiers) != len(labels) or len(labels) != len(probabilities):
        raise ValueError("Identifiers, labels and probabilities must have equal lengths")
    data = identifiers.rename(columns={"patient_nbr": "patient_id"}).reset_index(drop=True).copy()
    data["y_true"] = np.asarray(labels)
    data["y_prob"] = np.asarray(probabilities)
    data, accounting = validate_predictions(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return {**accounting, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def threshold_workload_table(thresholds: pd.DataFrame, saved_result: dict) -> pd.DataFrame:
    """Describe historical workload without selecting a threshold from test data."""
    required = ("rows", "threshold", "true_positives", "false_positives", "false_negatives", "true_negatives")
    if set(required) - set(thresholds.columns) or set(required) - set(saved_result):
        raise ValueError("Threshold readout needs row count, threshold and all confusion counts")
    rows = pd.concat([thresholds.loc[:, list(required)], pd.DataFrame([{key: saved_result[key] for key in required}])], ignore_index=True)
    for column in required:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(rows[column]).all():
            raise ValueError("Threshold readout must contain finite values")
    count_columns = ["rows", "true_positives", "false_positives", "false_negatives", "true_negatives"]
    if ((rows[count_columns] < 0) | (rows[count_columns] % 1 != 0)).any().any() or (rows.rows <= 0).any():
        raise ValueError("Confusion counts must be nonnegative integers and rows must be positive")
    if not rows.threshold.between(0, 1).all():
        raise ValueError("Thresholds must be between 0 and 1")
    if not (rows[count_columns[1:]].sum(axis=1) == rows.rows).all():
        raise ValueError("Confusion counts do not sum to rows")
    if rows.rows.nunique() != 1 or (rows.true_positives + rows.false_negatives).nunique() != 1:
        raise ValueError("Threshold rows must describe the same evaluation cohort")
    if (rows.groupby("threshold")[count_columns].nunique() > 1).any().any():
        raise ValueError("Conflicting confusion counts for the same threshold")
    rows["audit_flags"] = rows.true_positives + rows.false_positives
    rows["flags_per_1000_encounters"] = 1000 * rows.audit_flags / rows.rows
    rows["missed_positives_per_1000_encounters"] = 1000 * rows.false_negatives / rows.rows
    rows["recall"] = rows.true_positives / (rows.true_positives + rows.false_negatives).replace(0, np.nan)
    rows["precision"] = rows.true_positives / rows.audit_flags.replace(0, np.nan)
    rows["saved_validation_threshold"] = np.isclose(rows.threshold, saved_result["threshold"], rtol=0, atol=1e-12)
    return rows.drop_duplicates(subset=list(required)).sort_values("threshold").reset_index(drop=True)
