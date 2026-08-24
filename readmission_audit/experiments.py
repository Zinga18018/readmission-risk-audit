from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import chi2, chi2_contingency, chisquare

from readmission_audit.pipeline import SEED, feature_types, set_reproducible_seed


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(list(p_values), dtype=float)
    count = len(values)
    if count == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def categorical_chi_square_audit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    minimum_category_count: int = 50,
) -> pd.DataFrame:
    """Audit categorical association with the target using training rows only.

    The target-independence test is the useful diagnostic. A separate uniform
    goodness-of-fit result is included only because it was requested; clinical
    categories are not expected to be uniform and that test is not a feature
    selection rule.
    """
    _, categorical = feature_types(X_train)
    rows: list[dict[str, object]] = []
    target = pd.Series(np.asarray(y_train, dtype=int), index=X_train.index)

    for feature in categorical:
        values = X_train[feature].fillna("Missing").astype(str)
        counts = values.value_counts()
        rare = set(counts.index[counts < minimum_category_count])
        pooled = values.where(~values.isin(rare), "Other_rare")
        table = pd.crosstab(pooled, target)
        if table.shape[0] < 2 or table.shape[1] < 2:
            statistic = p_value = cramers_v = float("nan")
            minimum_expected = float("nan")
            expected_at_least_five_share = float("nan")
            chi_square_assumptions_met = False
        else:
            statistic, p_value, _, expected = chi2_contingency(
                table.to_numpy(), correction=False
            )
            n = table.to_numpy().sum()
            denominator = max(1, min(table.shape[0] - 1, table.shape[1] - 1))
            cramers_v = math.sqrt((statistic / n) / denominator)
            minimum_expected = float(expected.min())
            expected_at_least_five_share = float((expected >= 5).mean())
            chi_square_assumptions_met = bool(
                minimum_expected >= 1 and expected_at_least_five_share >= 0.80
            )

        observed = pooled.value_counts().to_numpy(dtype=float)
        if len(observed) >= 2:
            gof = chisquare(observed)
            gof_statistic = float(gof.statistic)
            gof_p_value = float(gof.pvalue)
        else:
            gof_statistic = gof_p_value = float("nan")

        rows.append(
            {
                "feature": feature,
                "categories_after_pooling": int(table.shape[0]),
                "pooled_rare_levels": int(len(rare)),
                "target_independence_chi2": float(statistic),
                "target_independence_p_value": float(p_value),
                "cramers_v": float(cramers_v),
                "minimum_expected_count": minimum_expected,
                "expected_count_at_least_five_share": expected_at_least_five_share,
                "chi_square_assumptions_met": chi_square_assumptions_met,
                "uniform_gof_chi2": gof_statistic,
                "uniform_gof_p_value": gof_p_value,
                "uniform_null_rejected_0_05": bool(gof_p_value < 0.05)
                if np.isfinite(gof_p_value)
                else False,
            }
        )

    result = pd.DataFrame(rows)
    result["target_independence_fdr_p_value"] = benjamini_hochberg(
        result["target_independence_p_value"].fillna(1.0)
    )
    result["target_associated_fdr_0_05"] = (
        result["target_independence_fdr_p_value"] < 0.05
    )
    return result.sort_values(
        ["cramers_v", "target_independence_fdr_p_value"],
        ascending=[False, True],
    ).reset_index(drop=True)


def feature_quality_audit(X_train: pd.DataFrame) -> pd.DataFrame:
    """Describe missingness and train-only constant/ultra-rare fields."""
    rows: list[dict[str, object]] = []
    for feature in X_train.columns:
        values = X_train[feature]
        counts = values.value_counts(dropna=False)
        top_count = int(counts.iloc[0])
        non_mode_count = int(len(values) - top_count)
        missing_rate = float(values.isna().mean())
        if feature.startswith("prior_mean_"):
            missing_type = "structural_no_prior_encounter"
        elif feature in {"max_glu_serum", "A1Cresult"}:
            missing_type = "test_result_not_recorded"
        elif missing_rate > 0:
            missing_type = "ordinary_missing"
        else:
            missing_type = "none"
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric"
                if pd.api.types.is_numeric_dtype(values)
                else "categorical",
                "missing_rate": missing_rate,
                "missing_type": missing_type,
                "cardinality_including_missing": int(values.nunique(dropna=False)),
                "top_value_share": float(top_count / len(values)),
                "non_mode_rows": non_mode_count,
                "exact_constant": bool(values.nunique(dropna=False) <= 1),
                "ultra_rare_non_mode_lt_50": bool(
                    values.nunique(dropna=False) > 1 and non_mode_count < 50
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "top_value_share"], ascending=False
    ).reset_index(drop=True)


def candidate_pruned_features(quality: pd.DataFrame) -> dict[str, list[str]]:
    constants = quality.loc[quality["exact_constant"], "feature"].tolist()
    ultra_rare = quality.loc[
        quality["exact_constant"] | quality["ultra_rare_non_mode_lt_50"],
        "feature",
    ].tolist()
    lab_results = ["max_glu_serum", "A1Cresult"]
    return {
        "all_features": [],
        "drop_exact_constants": constants,
        "drop_constants_and_ultra_rare": ultra_rare,
        "drop_ultra_rare_and_sparse_lab_results": sorted(
            set(ultra_rare + lab_results)
        ),
    }


def autocorrelation_audit(
    values: Iterable[float], series: str, max_lag: int = 20
) -> pd.DataFrame:
    """Compute ACF and cumulative Ljung-Box diagnostics for an ordered series."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    centered = array - array.mean()
    denominator = float(np.dot(centered, centered))
    n = len(centered)
    rows: list[dict[str, object]] = []
    cumulative = 0.0
    for lag in range(1, min(max_lag, n - 1) + 1):
        autocorrelation = (
            float(np.dot(centered[lag:], centered[:-lag]) / denominator)
            if denominator > 0
            else float("nan")
        )
        if np.isfinite(autocorrelation):
            cumulative += autocorrelation**2 / (n - lag)
            statistic = n * (n + 2) * cumulative
            p_value = float(chi2.sf(statistic, df=lag))
        else:
            statistic = p_value = float("nan")
        rows.append(
            {
                "series": series,
                "lag": lag,
                "acf": autocorrelation,
                "ljung_box_q": float(statistic),
                "ljung_box_p_value": p_value,
                "reject_no_autocorrelation_0_05": bool(p_value < 0.05)
                if np.isfinite(p_value)
                else False,
            }
        )
    return pd.DataFrame(rows)


class TabularAutoencoder(torch.nn.Module):
    """Small reconstruction model with a deterministic bottleneck."""

    def __init__(self, input_dim: int, latent_dim: int = 64) -> None:
        super().__init__()
        hidden_dim = min(256, max(96, input_dim // 2))
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        return self.decoder(latent), latent


def _dense_batch(matrix: object, indices: np.ndarray) -> np.ndarray:
    batch = matrix[indices]
    if sparse.issparse(batch):
        batch = batch.toarray()
    return np.asarray(batch, dtype=np.float32)


def reconstruction_loss(
    model: TabularAutoencoder, matrix: object, batch_size: int = 1024
) -> float:
    model.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(start + batch_size, matrix.shape[0])
            indices = np.arange(start, stop)
            features = torch.from_numpy(_dense_batch(matrix, indices))
            reconstructed, _ = model(features)
            total += float(
                torch.nn.functional.mse_loss(
                    reconstructed, features, reduction="sum"
                )
            )
            rows += int(features.numel())
    return total / rows


def train_autoencoder(
    train_matrix: object,
    validation_matrix: object,
    latent_dim: int,
    max_epochs: int = 15,
    patience: int = 3,
    batch_size: int = 1024,
) -> tuple[TabularAutoencoder, pd.DataFrame]:
    set_reproducible_seed(SEED)
    model = TabularAutoencoder(int(train_matrix.shape[1]), latent_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-5
    )
    rng = np.random.default_rng(SEED)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(train_matrix.shape[0])
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features = torch.from_numpy(_dense_batch(train_matrix, indices))
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(features)
            loss = torch.nn.functional.mse_loss(reconstructed, features)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = reconstruction_loss(model, validation_matrix)
        history.append(
            {
                "latent_dim": int(latent_dim),
                "epoch": int(epoch),
                "train_reconstruction_mse": float(np.mean(losses)),
                "validation_reconstruction_mse": float(validation_loss),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def encode_matrix(
    model: TabularAutoencoder, matrix: object, batch_size: int = 2048
) -> np.ndarray:
    model.eval()
    encoded: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(start + batch_size, matrix.shape[0])
            indices = np.arange(start, stop)
            features = torch.from_numpy(_dense_batch(matrix, indices))
            encoded.append(model.encoder(features).cpu().numpy())
    return np.vstack(encoded)


@dataclass
class AutoencoderCandidate:
    latent_dim: int
    model: TabularAutoencoder
    classifier_name: str
    classifier: object
    validation_pr_auc: float
    validation_roc_auc: float
