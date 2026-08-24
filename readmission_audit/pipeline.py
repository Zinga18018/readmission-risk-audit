from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SEED = 42
TARGET = "early_readmission"
ID_COLUMNS = ("encounter_id", "patient_nbr")
NOMINAL_ID_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)
HOSPICE_OR_EXPIRED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
HIGH_MISSING_COLUMNS = {"weight", "payer_code", "medical_specialty"}
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def diagnosis_family(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip()
    if not text or text == "?":
        return "Missing"
    if text.startswith("V"):
        return "Supplemental"
    if text.startswith("E"):
        return "External_injury"
    try:
        number = float(text)
    except ValueError:
        return "Other"
    if 390 <= number <= 459 or number == 785:
        return "Circulatory"
    if 460 <= number <= 519 or number == 786:
        return "Respiratory"
    if 520 <= number <= 579 or number == 787:
        return "Digestive"
    if 250 <= number < 251:
        return "Diabetes"
    if 800 <= number <= 999:
        return "Injury"
    if 710 <= number <= 739:
        return "Musculoskeletal"
    if 580 <= number <= 629 or number == 788:
        return "Genitourinary"
    if 140 <= number <= 239:
        return "Neoplasms"
    return "Other"


def clean_diabetes_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create the binary target and an eligible-for-readmission cohort."""
    required = {"readmitted", "encounter_id", "patient_nbr"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    cleaned = df.copy().replace("?", np.nan)
    raw_rows = len(cleaned)
    excluded_mask = cleaned["discharge_disposition_id"].isin(
        HOSPICE_OR_EXPIRED_DISPOSITIONS
    )
    cleaned = cleaned.loc[~excluded_mask].copy()
    cleaned[TARGET] = (cleaned["readmitted"] == "<30").astype("int64")
    cleaned = cleaned.drop(columns=["readmitted"])

    audit = {
        "raw_rows": int(raw_rows),
        "excluded_hospice_or_expired_rows": int(excluded_mask.sum()),
        "eligible_rows": int(len(cleaned)),
        "unique_patients": int(cleaned["patient_nbr"].nunique()),
        "duplicate_encounter_ids": int(cleaned["encounter_id"].duplicated().sum()),
    }
    return cleaned, audit


def make_patient_order_split(
    cleaned: pd.DataFrame,
    train_quantile: float = 0.70,
    validation_quantile: float = 0.85,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build a patient-disjoint split ordered by the encounter identifier.

    The UCI file has no row-level timestamp. Encounter ID is therefore used only
    as an order proxy, never represented as a calendar date. Patients crossing a
    boundary are excluded so no identity appears in more than one split.
    """
    if not 0 < train_quantile < validation_quantile < 1:
        raise ValueError("expected 0 < train_quantile < validation_quantile < 1")

    train_cut, validation_cut = cleaned["encounter_id"].quantile(
        [train_quantile, validation_quantile]
    )
    patient_range = cleaned.groupby("patient_nbr")["encounter_id"].agg(["min", "max"])
    train_patients = set(patient_range.index[patient_range["max"] <= train_cut])
    validation_patients = set(
        patient_range.index[
            (patient_range["min"] > train_cut)
            & (patient_range["max"] <= validation_cut)
        ]
    )
    test_patients = set(
        patient_range.index[patient_range["min"] > validation_cut]
    )

    splits = {
        "train": cleaned[cleaned["patient_nbr"].isin(train_patients)].copy(),
        "validation": cleaned[
            cleaned["patient_nbr"].isin(validation_patients)
        ].copy(),
        "test": cleaned[cleaned["patient_nbr"].isin(test_patients)].copy(),
    }
    assigned_rows = sum(len(frame) for frame in splits.values())
    assigned_patients = train_patients | validation_patients | test_patients
    pairwise_overlaps = {
        "train_validation": len(train_patients & validation_patients),
        "train_test": len(train_patients & test_patients),
        "validation_test": len(validation_patients & test_patients),
    }
    audit: dict[str, object] = {
        "strategy": "patient-disjoint encounter-ID order proxy",
        "calendar_time_claimed": False,
        "train_encounter_id_cut": float(train_cut),
        "validation_encounter_id_cut": float(validation_cut),
        "excluded_boundary_crossing_rows": int(len(cleaned) - assigned_rows),
        "excluded_boundary_crossing_patients": int(
            cleaned["patient_nbr"].nunique() - len(assigned_patients)
        ),
        "patient_overlap_counts": pairwise_overlaps,
        "splits": {
            name: {
                "rows": int(len(frame)),
                "patients": int(frame["patient_nbr"].nunique()),
                "positive_rate": float(frame[TARGET].mean()),
                "encounter_id_min": int(frame["encounter_id"].min()),
                "encounter_id_max": int(frame["encounter_id"].max()),
            }
            for name, frame in splits.items()
        },
    }
    return splits, audit


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in frame.columns:
        raise ValueError(f"expected {TARGET} after cleaning")
    y = frame[TARGET].astype("int64")
    X = frame.drop(columns=[TARGET, *ID_COLUMNS], errors="ignore").copy()
    X = X.drop(columns=list(HIGH_MISSING_COLUMNS), errors="ignore")

    for column in NOMINAL_ID_COLUMNS:
        if column in X.columns:
            X[column] = X[column].astype("Int64").astype("string")
    for column in DIAGNOSIS_COLUMNS:
        if column in X.columns:
            X[column] = X[column].map(diagnosis_family).astype("string")
    return X, y


def feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical


def make_linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_types(X)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=50,
                                max_categories=100,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.3,
    )


def make_tree_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[bool]]:
    numeric, categorical = feature_types(X)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                max_categories=254,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    return preprocessor, categorical_mask


class ReadmissionDNN(torch.nn.Module):
    """Intentionally shallow tabular network that returns raw logits."""

    def __init__(self, input_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _dense_float32(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _predict_logits(
    model: ReadmissionDNN,
    matrix: object,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            batch = _dense_float32(matrix[start : start + batch_size])
            logits = model(torch.from_numpy(batch))
            rows.append(logits.cpu().numpy())
    return np.vstack(rows)


def softmax_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax only outside the model for evaluation/inference."""
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _train_one_dnn(
    train_matrix: object,
    y_train: np.ndarray,
    validation_matrix: object,
    y_validation: np.ndarray,
    dropout: float,
    max_epochs: int = 20,
    patience: int = 4,
    batch_size: int = 1024,
) -> tuple[ReadmissionDNN, pd.DataFrame, dict[str, float]]:
    set_reproducible_seed(SEED)
    model = ReadmissionDNN(int(train_matrix.shape[1]), dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)

    best_state = copy.deepcopy(model.state_dict())
    best_pr_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(y_train))
        batch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features = torch.from_numpy(_dense_float32(train_matrix[indices]))
            targets = torch.from_numpy(y_train[indices].astype(np.int64, copy=False))
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        validation_logits = _predict_logits(model, validation_matrix)
        validation_probabilities = softmax_probabilities(validation_logits)[:, 1]
        validation_pr_auc = average_precision_score(
            y_validation, validation_probabilities
        )
        validation_loss = log_loss(
            y_validation, validation_probabilities, labels=[0, 1]
        )
        history.append(
            {
                "epoch": epoch,
                "dropout": dropout,
                "train_cross_entropy": float(np.mean(batch_losses)),
                "validation_log_loss": float(validation_loss),
                "validation_pr_auc": float(validation_pr_auc),
            }
        )

        if validation_pr_auc > best_pr_auc + 1e-5:
            best_pr_auc = float(validation_pr_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    summary = {
        "dropout": float(dropout),
        "best_epoch": int(best_epoch),
        "validation_pr_auc": float(best_pr_auc),
        "epochs_run": int(len(history)),
    }
    return model, pd.DataFrame(history), summary


def tune_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    logits_tensor = torch.from_numpy(logits.astype(np.float32, copy=False))
    target_tensor = torch.from_numpy(y_true.astype(np.int64, copy=False))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.25, 4.0)
        loss = loss_fn(logits_tensor / temperature, target_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.25, 4.0))


def select_f1_threshold(
    y_true: Iterable[int], probabilities: Iterable[float]
) -> float:
    y_array = np.asarray(list(y_true), dtype=int)
    probability_array = np.asarray(list(probabilities), dtype=float)
    thresholds = np.linspace(0.02, 0.60, 291)
    f1_values = [
        f1_score(y_array, probability_array >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(f1_values))])


def expected_calibration_error(
    y_true: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> float:
    y_array = np.asarray(list(y_true), dtype=float)
    probability_array = np.asarray(list(probabilities), dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probability_array, edges) - 1, bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            ece += mask.mean() * abs(
                probability_array[mask].mean() - y_array[mask].mean()
            )
    return float(ece)


def evaluate_probabilities(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> dict[str, float | int]:
    y_array = np.asarray(list(y_true), dtype=int)
    probability_array = np.asarray(list(probabilities), dtype=float)
    predictions = (probability_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predictions, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_array)),
        "positive_rate": float(y_array.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_array, probability_array)),
        "pr_auc": float(average_precision_score(y_array, probability_array)),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_array, probability_array)),
        "ece_10_bin": expected_calibration_error(y_array, probability_array),
        "log_loss": float(log_loss(y_array, probability_array, labels=[0, 1])),
        "flagged_rate": float(predictions.mean()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def calibration_table(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    model: str,
    split: str,
    bins: int = 10,
) -> pd.DataFrame:
    y_array = np.asarray(list(y_true), dtype=float)
    probability_array = np.asarray(list(probabilities), dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probability_array, edges) - 1, bins - 1)
    rows = []
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "bin": index + 1,
                    "rows": int(mask.sum()),
                    "mean_predicted_probability": float(probability_array[mask].mean()),
                    "observed_positive_rate": float(y_array[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def threshold_table(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [evaluate_probabilities(y_true, probabilities, threshold) for threshold in thresholds]
    )


def population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(reference_values) == 0 or len(current_values) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference_values, bins=edges)[0].astype(float)
    current_counts = np.histogram(current_values, bins=edges)[0].astype(float)
    reference_rate = np.clip(reference_counts / reference_counts.sum(), 1e-6, None)
    current_rate = np.clip(current_counts / current_counts.sum(), 1e-6, None)
    return float(np.sum((current_rate - reference_rate) * np.log(current_rate / reference_rate)))


def categorical_js_divergence(reference: pd.Series, current: pd.Series) -> float:
    reference_counts = reference.fillna("Missing").astype(str).value_counts(normalize=True)
    current_counts = current.fillna("Missing").astype(str).value_counts(normalize=True)
    categories = reference_counts.index.union(current_counts.index)
    p = reference_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    q = current_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    return float(jensenshannon(p, q, base=2) ** 2)


def feature_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    comparison: str,
) -> pd.DataFrame:
    numeric, categorical = feature_types(reference)
    rows: list[dict[str, object]] = []
    for column in numeric:
        reference_values = pd.to_numeric(reference[column], errors="coerce").dropna()
        current_values = pd.to_numeric(current[column], errors="coerce").dropna()
        ks_statistic = (
            float(ks_2samp(reference_values, current_values).statistic)
            if len(reference_values) and len(current_values)
            else float("nan")
        )
        rows.append(
            {
                "comparison": comparison,
                "feature": column,
                "feature_type": "numeric",
                "psi": population_stability_index(reference[column], current[column]),
                "ks_statistic": ks_statistic,
                "js_divergence": float("nan"),
            }
        )
    for column in categorical:
        rows.append(
            {
                "comparison": comparison,
                "feature": column,
                "feature_type": "categorical",
                "psi": float("nan"),
                "ks_statistic": float("nan"),
                "js_divergence": categorical_js_divergence(
                    reference[column], current[column]
                ),
            }
        )
    return pd.DataFrame(rows)


class FTPreprocessor:
    """Train-fitted numeric scaling and categorical indexing for FT-Transformer."""

    def __init__(self, X: pd.DataFrame) -> None:
        self.numeric_features, self.categorical_features = feature_types(X)
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_scaler = StandardScaler()
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.category_cardinalities: list[int] = []

    def fit(self, X: pd.DataFrame) -> "FTPreprocessor":
        numeric = self.numeric_imputer.fit_transform(X[self.numeric_features])
        self.numeric_scaler.fit(numeric)
        categorical = self.categorical_imputer.fit_transform(
            X[self.categorical_features]
        )
        self.categorical_encoder.fit(categorical)
        self.category_cardinalities = [
            len(categories) + 1
            for categories in self.categorical_encoder.categories_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = self.numeric_imputer.transform(X[self.numeric_features])
        numeric = self.numeric_scaler.transform(numeric).astype(np.float32)
        categorical = self.categorical_imputer.transform(
            X[self.categorical_features]
        )
        categorical = self.categorical_encoder.transform(categorical)
        categorical = (categorical.astype(np.int64) + 1).clip(min=0)
        return numeric, categorical


def prepare_catboost_features(
    X: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    prepared = X.copy()
    for column in categorical_features:
        prepared[column] = prepared[column].fillna("Missing").astype(str)
    return prepared


class FTTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        token_dimension: int,
        attention_heads: int,
        feedforward_dimension: int,
        attention_dropout: float,
        feedforward_dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(token_dimension)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=token_dimension,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.feedforward_norm = torch.nn.LayerNorm(token_dimension)
        self.feedforward_input = torch.nn.Linear(
            token_dimension, feedforward_dimension * 2
        )
        self.feedforward_dropout = torch.nn.Dropout(feedforward_dropout)
        self.feedforward_output = torch.nn.Linear(
            feedforward_dimension, token_dimension
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        normalized = self.feedforward_norm(tokens)
        value, gate = self.feedforward_input(normalized).chunk(2, dim=-1)
        activated = value * torch.nn.functional.relu(gate)
        tokens = tokens + self.feedforward_output(
            self.feedforward_dropout(activated)
        )
        return tokens


class FTTransformer(torch.nn.Module):
    """Feature Tokenizer + Transformer returning two raw class logits."""

    def __init__(
        self,
        numeric_features: int,
        category_cardinalities: list[int],
        token_dimension: int = 64,
        transformer_blocks: int = 3,
        attention_heads: int = 8,
        feedforward_dimension: int = 128,
        attention_dropout: float = 0.05,
        feedforward_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.numeric_features = numeric_features
        self.category_cardinalities = list(category_cardinalities)
        self.token_dimension = token_dimension
        self.numeric_weight = torch.nn.Parameter(
            torch.empty(numeric_features, token_dimension)
        )
        self.numeric_bias = torch.nn.Parameter(
            torch.empty(numeric_features, token_dimension)
        )
        torch.nn.init.normal_(self.numeric_weight, std=0.02)
        torch.nn.init.normal_(self.numeric_bias, std=0.02)

        total_categories = sum(category_cardinalities)
        self.category_embeddings = torch.nn.Embedding(
            total_categories, token_dimension
        )
        torch.nn.init.normal_(self.category_embeddings.weight, std=0.02)
        offsets = np.cumsum([0, *category_cardinalities[:-1]])
        self.register_buffer(
            "category_offsets", torch.tensor(offsets, dtype=torch.long)
        )
        self.category_bias = torch.nn.Parameter(
            torch.zeros(len(category_cardinalities), token_dimension)
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, token_dimension))
        torch.nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = torch.nn.ModuleList(
            [
                FTTransformerBlock(
                    token_dimension=token_dimension,
                    attention_heads=attention_heads,
                    feedforward_dimension=feedforward_dimension,
                    attention_dropout=attention_dropout,
                    feedforward_dropout=feedforward_dropout,
                )
                for _ in range(transformer_blocks)
            ]
        )
        self.head_norm = torch.nn.LayerNorm(token_dimension)
        self.head = torch.nn.Linear(token_dimension, 2)

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        numeric_tokens = (
            numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        )
        category_indices = categorical + self.category_offsets
        categorical_tokens = (
            self.category_embeddings(category_indices) + self.category_bias
        )
        cls = self.cls_token.expand(numeric.shape[0], -1, -1)
        tokens = torch.cat([cls, numeric_tokens, categorical_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        cls_output = torch.nn.functional.relu(self.head_norm(tokens[:, 0]))
        return self.head(cls_output)


def _predict_ft_logits(
    model: FTTransformer,
    numeric: np.ndarray,
    categorical: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            numeric_batch = torch.from_numpy(numeric[start : start + batch_size])
            categorical_batch = torch.from_numpy(
                categorical[start : start + batch_size]
            )
            rows.append(
                model(numeric_batch, categorical_batch).cpu().numpy()
            )
    return np.vstack(rows)


def _train_one_ft_transformer(
    train_numeric: np.ndarray,
    train_categorical: np.ndarray,
    y_train: np.ndarray,
    validation_numeric: np.ndarray,
    validation_categorical: np.ndarray,
    y_validation: np.ndarray,
    category_cardinalities: list[int],
    attention_dropout: float,
    max_epochs: int = 16,
    patience: int = 3,
    batch_size: int = 512,
) -> tuple[FTTransformer, pd.DataFrame, dict[str, float]]:
    set_reproducible_seed(SEED)
    model = FTTransformer(
        numeric_features=train_numeric.shape[1],
        category_cardinalities=category_cardinalities,
        attention_dropout=attention_dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-5
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    best_state = copy.deepcopy(model.state_dict())
    best_pr_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(y_train))
        batch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            numeric_batch = torch.from_numpy(train_numeric[indices])
            categorical_batch = torch.from_numpy(train_categorical[indices])
            targets = torch.from_numpy(
                y_train[indices].astype(np.int64, copy=False)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric_batch, categorical_batch)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        validation_logits = _predict_ft_logits(
            model, validation_numeric, validation_categorical
        )
        validation_probabilities = softmax_probabilities(
            validation_logits
        )[:, 1]
        validation_pr_auc = average_precision_score(
            y_validation, validation_probabilities
        )
        validation_loss = log_loss(
            y_validation, validation_probabilities, labels=[0, 1]
        )
        history.append(
            {
                "epoch": epoch,
                "attention_dropout": attention_dropout,
                "train_cross_entropy": float(np.mean(batch_losses)),
                "validation_log_loss": float(validation_loss),
                "validation_pr_auc": float(validation_pr_auc),
            }
        )
        if validation_pr_auc > best_pr_auc + 1e-5:
            best_pr_auc = float(validation_pr_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    summary = {
        "attention_dropout": float(attention_dropout),
        "best_epoch": int(best_epoch),
        "validation_pr_auc": float(best_pr_auc),
        "epochs_run": int(len(history)),
    }
    return model, pd.DataFrame(history), summary


@dataclass
class TrainingArtifacts:
    linear_preprocessor: ColumnTransformer
    logistic_model: LogisticRegression
    tree_preprocessor: ColumnTransformer
    tree_model: HistGradientBoostingClassifier
    catboost_model: CatBoostClassifier
    dnn_model: ReadmissionDNN
    dnn_temperature: float
    selected_dropout: float
    ft_preprocessor: FTPreprocessor
    ft_model: FTTransformer
    ft_temperature: float
    selected_attention_dropout: float
    feature_columns: list[str]


def train_and_evaluate(df: pd.DataFrame) -> dict[str, object]:
    set_reproducible_seed(SEED)
    cleaned, cohort_audit = clean_diabetes_data(df)
    splits, split_audit = make_patient_order_split(cleaned)
    features: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.Series] = {}
    for name, frame in splits.items():
        features[name], targets[name] = make_features(frame)

    X_train = features["train"]
    y_train = targets["train"].to_numpy()
    X_validation = features["validation"]
    y_validation = targets["validation"].to_numpy()
    X_test = features["test"]
    y_test = targets["test"].to_numpy()

    linear_preprocessor = make_linear_preprocessor(X_train)
    train_matrix = linear_preprocessor.fit_transform(X_train)
    validation_matrix = linear_preprocessor.transform(X_validation)
    test_matrix = linear_preprocessor.transform(X_test)

    logistic_model = LogisticRegression(
        max_iter=500, solver="lbfgs", random_state=SEED
    )
    logistic_model.fit(train_matrix, y_train)

    tree_preprocessor, categorical_mask = make_tree_preprocessor(X_train)
    tree_train = tree_preprocessor.fit_transform(X_train)
    tree_validation = tree_preprocessor.transform(X_validation)
    tree_test = tree_preprocessor.transform(X_test)
    tree_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )
    tree_model.fit(tree_train, y_train)

    _, categorical_features = feature_types(X_train)
    catboost_train = prepare_catboost_features(X_train, categorical_features)
    catboost_validation = prepare_catboost_features(
        X_validation, categorical_features
    )
    catboost_test = prepare_catboost_features(X_test, categorical_features)
    catboost_model = CatBoostClassifier(
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
    catboost_model.fit(
        catboost_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(catboost_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=75,
        verbose=False,
    )

    dropout_rows: list[dict[str, float]] = []
    histories: list[pd.DataFrame] = []
    dnn_candidates: dict[float, ReadmissionDNN] = {}
    for dropout in (0.00, 0.05, 0.10, 0.15):
        model, history, summary = _train_one_dnn(
            train_matrix,
            y_train,
            validation_matrix,
            y_validation,
            dropout=dropout,
        )
        dnn_candidates[dropout] = model
        dropout_rows.append(summary)
        histories.append(history)

    dropout_sweep = pd.DataFrame(dropout_rows).sort_values(
        ["validation_pr_auc", "dropout"], ascending=[False, True]
    )
    selected_dropout = float(dropout_sweep.iloc[0]["dropout"])
    dnn_model = dnn_candidates[selected_dropout]
    train_history = pd.concat(histories, ignore_index=True)

    dnn_train_logits = _predict_logits(dnn_model, train_matrix)
    dnn_validation_logits = _predict_logits(dnn_model, validation_matrix)
    dnn_test_logits = _predict_logits(dnn_model, test_matrix)
    dnn_temperature = tune_temperature(dnn_validation_logits, y_validation)

    ft_preprocessor = FTPreprocessor(X_train).fit(X_train)
    ft_train_numeric, ft_train_categorical = ft_preprocessor.transform(X_train)
    ft_validation_numeric, ft_validation_categorical = ft_preprocessor.transform(
        X_validation
    )
    ft_test_numeric, ft_test_categorical = ft_preprocessor.transform(X_test)
    ft_sweep_rows: list[dict[str, float]] = []
    ft_histories: list[pd.DataFrame] = []
    ft_candidates: dict[float, FTTransformer] = {}
    for attention_dropout in (0.00, 0.05):
        ft_candidate, ft_history, ft_summary = _train_one_ft_transformer(
            ft_train_numeric,
            ft_train_categorical,
            y_train,
            ft_validation_numeric,
            ft_validation_categorical,
            y_validation,
            ft_preprocessor.category_cardinalities,
            attention_dropout=attention_dropout,
        )
        ft_candidates[attention_dropout] = ft_candidate
        ft_sweep_rows.append(ft_summary)
        ft_histories.append(ft_history)
    ft_attention_dropout_sweep = pd.DataFrame(ft_sweep_rows).sort_values(
        ["validation_pr_auc", "attention_dropout"], ascending=[False, True]
    )
    selected_attention_dropout = float(
        ft_attention_dropout_sweep.iloc[0]["attention_dropout"]
    )
    ft_model = ft_candidates[selected_attention_dropout]
    ft_train_history = pd.concat(ft_histories, ignore_index=True)
    ft_train_logits = _predict_ft_logits(
        ft_model, ft_train_numeric, ft_train_categorical
    )
    ft_validation_logits = _predict_ft_logits(
        ft_model, ft_validation_numeric, ft_validation_categorical
    )
    ft_test_logits = _predict_ft_logits(
        ft_model, ft_test_numeric, ft_test_categorical
    )
    ft_temperature = tune_temperature(ft_validation_logits, y_validation)

    probability_sets: dict[str, dict[str, np.ndarray]] = {
        "Logistic Regression": {
            "train": logistic_model.predict_proba(train_matrix)[:, 1],
            "validation": logistic_model.predict_proba(validation_matrix)[:, 1],
            "test": logistic_model.predict_proba(test_matrix)[:, 1],
        },
        "HistGradientBoosting": {
            "train": tree_model.predict_proba(tree_train)[:, 1],
            "validation": tree_model.predict_proba(tree_validation)[:, 1],
            "test": tree_model.predict_proba(tree_test)[:, 1],
        },
        "CatBoost": {
            "train": catboost_model.predict_proba(catboost_train)[:, 1],
            "validation": catboost_model.predict_proba(catboost_validation)[:, 1],
            "test": catboost_model.predict_proba(catboost_test)[:, 1],
        },
        "DNN raw": {
            "train": softmax_probabilities(dnn_train_logits)[:, 1],
            "validation": softmax_probabilities(dnn_validation_logits)[:, 1],
            "test": softmax_probabilities(dnn_test_logits)[:, 1],
        },
        "DNN temperature-scaled": {
            "train": softmax_probabilities(
                dnn_train_logits, dnn_temperature
            )[:, 1],
            "validation": softmax_probabilities(
                dnn_validation_logits, dnn_temperature
            )[:, 1],
            "test": softmax_probabilities(
                dnn_test_logits, dnn_temperature
            )[:, 1],
        },
        "FT-Transformer raw": {
            "train": softmax_probabilities(ft_train_logits)[:, 1],
            "validation": softmax_probabilities(ft_validation_logits)[:, 1],
            "test": softmax_probabilities(ft_test_logits)[:, 1],
        },
        "FT-Transformer temperature-scaled": {
            "train": softmax_probabilities(
                ft_train_logits, ft_temperature
            )[:, 1],
            "validation": softmax_probabilities(
                ft_validation_logits, ft_temperature
            )[:, 1],
            "test": softmax_probabilities(ft_test_logits, ft_temperature)[:, 1],
        },
    }

    model_rows: list[dict[str, object]] = []
    calibration_rows: list[pd.DataFrame] = []
    thresholds: dict[str, float] = {}
    target_arrays = {
        "train": y_train,
        "validation": y_validation,
        "test": y_test,
    }
    for model_name, split_probabilities in probability_sets.items():
        threshold = select_f1_threshold(
            y_validation, split_probabilities["validation"]
        )
        thresholds[model_name] = threshold
        for split_name in ("train", "validation", "test"):
            row = evaluate_probabilities(
                target_arrays[split_name],
                split_probabilities[split_name],
                threshold,
            )
            row.update({"model": model_name, "split": split_name})
            model_rows.append(row)
            calibration_rows.append(
                calibration_table(
                    target_arrays[split_name],
                    split_probabilities[split_name],
                    model_name,
                    split_name,
                )
            )

    model_comparison = pd.DataFrame(model_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    performance_drift_rows: list[dict[str, object]] = []
    for model_name in probability_sets:
        validation_row = model_comparison[
            (model_comparison["model"] == model_name)
            & (model_comparison["split"] == "validation")
        ].iloc[0]
        test_row = model_comparison[
            (model_comparison["model"] == model_name)
            & (model_comparison["split"] == "test")
        ].iloc[0]
        drift_row: dict[str, object] = {"model": model_name}
        for metric in (
            "roc_auc",
            "pr_auc",
            "f1",
            "recall",
            "accuracy",
            "brier_score",
            "ece_10_bin",
        ):
            drift_row[f"validation_{metric}"] = float(validation_row[metric])
            drift_row[f"test_{metric}"] = float(test_row[metric])
            drift_row[f"test_minus_validation_{metric}"] = float(
                test_row[metric] - validation_row[metric]
            )
        performance_drift_rows.append(drift_row)
    performance_drift = pd.DataFrame(performance_drift_rows)

    drift = pd.concat(
        [
            feature_drift_table(X_train, X_validation, "train_to_validation"),
            feature_drift_table(X_train, X_test, "train_to_test"),
        ],
        ignore_index=True,
    )

    dnn_train = model_comparison[
        (model_comparison["model"] == "DNN raw")
        & (model_comparison["split"] == "train")
    ].iloc[0]
    dnn_validation = model_comparison[
        (model_comparison["model"] == "DNN raw")
        & (model_comparison["split"] == "validation")
    ].iloc[0]
    ft_train = model_comparison[
        (model_comparison["model"] == "FT-Transformer raw")
        & (model_comparison["split"] == "train")
    ].iloc[0]
    ft_validation = model_comparison[
        (model_comparison["model"] == "FT-Transformer raw")
        & (model_comparison["split"] == "validation")
    ].iloc[0]
    leakage_audit = {
        **cohort_audit,
        **split_audit,
        "target_removed_from_features": TARGET not in X_train.columns,
        "raw_readmitted_removed_from_features": "readmitted" not in X_train.columns,
        "identifier_features_removed": all(
            column not in X_train.columns for column in ID_COLUMNS
        ),
        "preprocessor_fit_scope": "train only",
        "threshold_selection_scope": "validation only",
        "temperature_scaling_scope": "validation logits only",
        "test_used_for_training_or_selection": False,
        "dnn_train_validation_gap": {
            "pr_auc": float(dnn_train["pr_auc"] - dnn_validation["pr_auc"]),
            "roc_auc": float(dnn_train["roc_auc"] - dnn_validation["roc_auc"]),
            "log_loss_validation_minus_train": float(
                dnn_validation["log_loss"] - dnn_train["log_loss"]
            ),
        },
        "ft_transformer_train_validation_gap": {
            "pr_auc": float(ft_train["pr_auc"] - ft_validation["pr_auc"]),
            "roc_auc": float(ft_train["roc_auc"] - ft_validation["roc_auc"]),
            "log_loss_validation_minus_train": float(
                ft_validation["log_loss"] - ft_train["log_loss"]
            ),
        },
    }

    demo_defaults: dict[str, object] = {}
    demo_options: dict[str, list[str]] = {}
    numeric, categorical = feature_types(X_train)
    for column in numeric:
        demo_defaults[column] = float(pd.to_numeric(X_train[column]).median())
    for column in categorical:
        values = X_train[column].dropna().astype(str)
        demo_defaults[column] = values.mode().iat[0] if len(values) else "Missing"
        if column in {
            "age",
            "race",
            "gender",
            "A1Cresult",
            "max_glu_serum",
            "insulin",
            "change",
            "diabetesMed",
        }:
            demo_options[column] = sorted(values.unique().tolist())

    selected_test = model_comparison[
        (
            model_comparison["model"]
            == "FT-Transformer temperature-scaled"
        )
        & (model_comparison["split"] == "test")
    ].iloc[0].to_dict()
    headline_metrics = {
        "raw_rows": int(len(df)),
        "eligible_rows": int(len(cleaned)),
        "modeling_rows": int(sum(len(frame) for frame in splits.values())),
        "features_before_encoding": int(X_train.shape[1]),
        "encoded_features": int(train_matrix.shape[1]),
        "selected_dropout": selected_dropout,
        "dnn_temperature": dnn_temperature,
        "selected_attention_dropout": selected_attention_dropout,
        "ft_temperature": ft_temperature,
        "ft_test": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in selected_test.items()
            if key not in {"model", "split"}
        },
    }

    artifacts = TrainingArtifacts(
        linear_preprocessor=linear_preprocessor,
        logistic_model=logistic_model,
        tree_preprocessor=tree_preprocessor,
        tree_model=tree_model,
        catboost_model=catboost_model,
        dnn_model=dnn_model,
        dnn_temperature=dnn_temperature,
        selected_dropout=selected_dropout,
        ft_preprocessor=ft_preprocessor,
        ft_model=ft_model,
        ft_temperature=ft_temperature,
        selected_attention_dropout=selected_attention_dropout,
        feature_columns=X_train.columns.tolist(),
    )
    return {
        "metrics": headline_metrics,
        "model_comparison": model_comparison,
        "performance_drift": performance_drift,
        "calibration": calibration,
        "dropout_sweep": dropout_sweep,
        "train_history": train_history,
        "ft_attention_dropout_sweep": ft_attention_dropout_sweep,
        "ft_train_history": ft_train_history,
        "feature_drift": drift,
        "leakage_audit": leakage_audit,
        "artifacts": artifacts,
        "demo_defaults": demo_defaults,
        "demo_options": demo_options,
        "probability_sets": probability_sets,
        "targets": targets,
        "thresholds": thresholds,
    }


def write_outputs(
    result: dict[str, object], output_dir: Path, artifact_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(result["leakage_audit"], indent=2), encoding="utf-8"
    )
    for key, filename in (
        ("model_comparison", "model_comparison.csv"),
        ("performance_drift", "performance_drift.csv"),
        ("calibration", "calibration_curve.csv"),
        ("dropout_sweep", "dropout_sweep.csv"),
        ("train_history", "dnn_train_history.csv"),
        (
            "ft_attention_dropout_sweep",
            "ft_attention_dropout_sweep.csv",
        ),
        ("ft_train_history", "ft_train_history.csv"),
        ("feature_drift", "feature_drift.csv"),
    ):
        result[key].to_csv(output_dir / filename, index=False)

    calibrated_probabilities = result["probability_sets"][
        "FT-Transformer temperature-scaled"
    ]["test"]
    calibrated_threshold = result["thresholds"][
        "FT-Transformer temperature-scaled"
    ]
    threshold_table(
        result["targets"]["test"],
        calibrated_probabilities,
        thresholds=np.linspace(0.04, 0.30, 14),
    ).to_csv(output_dir / "threshold_table.csv", index=False)

    artifacts: TrainingArtifacts = result["artifacts"]
    joblib.dump(artifacts.linear_preprocessor, artifact_dir / "preprocessor.joblib")
    joblib.dump(artifacts.logistic_model, artifact_dir / "logistic_model.joblib")
    joblib.dump(artifacts.tree_preprocessor, artifact_dir / "tree_preprocessor.joblib")
    joblib.dump(artifacts.tree_model, artifact_dir / "tree_model.joblib")
    artifacts.catboost_model.save_model(artifact_dir / "catboost_model.cbm")
    torch.save(artifacts.dnn_model.state_dict(), artifact_dir / "dnn_state.pt")
    model_config = {
        "input_dim": int(artifacts.dnn_model.network[0].in_features),
        "dropout": artifacts.selected_dropout,
        "temperature": artifacts.dnn_temperature,
        "threshold": result["thresholds"]["DNN temperature-scaled"],
        "feature_columns": artifacts.feature_columns,
        "architecture": "input -> Linear(128) -> ReLU -> Dropout -> Linear(64) -> ReLU -> Dropout -> Linear(2)",
        "training_loss": "CrossEntropyLoss on raw logits",
        "probability_transform": "softmax at evaluation/inference only",
    }
    (artifact_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    joblib.dump(
        artifacts.ft_preprocessor,
        artifact_dir / "ft_preprocessor.joblib",
    )
    torch.save(
        artifacts.ft_model.state_dict(), artifact_dir / "ft_transformer_state.pt"
    )
    ft_config = {
        "numeric_features": len(artifacts.ft_preprocessor.numeric_features),
        "categorical_features": artifacts.ft_preprocessor.categorical_features,
        "category_cardinalities": artifacts.ft_preprocessor.category_cardinalities,
        "token_dimension": artifacts.ft_model.token_dimension,
        "transformer_blocks": len(artifacts.ft_model.blocks),
        "attention_heads": artifacts.ft_model.blocks[0].attention.num_heads,
        "feedforward_dimension": artifacts.ft_model.blocks[
            0
        ].feedforward_output.in_features,
        "attention_dropout": artifacts.selected_attention_dropout,
        "feedforward_dropout": artifacts.ft_model.blocks[
            0
        ].feedforward_dropout.p,
        "temperature": artifacts.ft_temperature,
        "threshold": calibrated_threshold,
        "feature_columns": artifacts.feature_columns,
        "architecture": "feature tokens -> CLS + 3 pre-norm Transformer blocks -> Linear(2)",
        "training_loss": "CrossEntropyLoss on raw logits",
        "probability_transform": "softmax at evaluation/inference only",
    }
    (artifact_dir / "ft_transformer_config.json").write_text(
        json.dumps(ft_config, indent=2), encoding="utf-8"
    )
    (artifact_dir / "demo_defaults.json").write_text(
        json.dumps(result["demo_defaults"], indent=2), encoding="utf-8"
    )
    (artifact_dir / "demo_options.json").write_text(
        json.dumps(result["demo_options"], indent=2), encoding="utf-8"
    )
