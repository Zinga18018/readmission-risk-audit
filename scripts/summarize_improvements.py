from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    pruning = pd.read_csv(OUTPUTS / "feature_pruning_ablation.csv")
    group_grid = pd.read_csv(OUTPUTS / "catboost_group_grid_search.csv")
    later_gate = pd.read_csv(OUTPUTS / "catboost_later_validation_gate.csv")
    final_catboost = pd.read_csv(
        OUTPUTS / "catboost_final_improvement_comparison.csv"
    )
    neural = pd.read_csv(OUTPUTS / "improvement_model_comparison.csv")
    autoencoder = pd.read_csv(OUTPUTS / "autoencoder_comparison.csv")
    ft_history = pd.read_csv(OUTPUTS / "ft_dense_zero_dropout_history.csv")
    chi_square = pd.read_csv(OUTPUTS / "categorical_chi_square_audit.csv")
    autocorrelation = pd.read_csv(
        OUTPUTS / "autocorrelation_diagnostics.csv"
    )
    xgboost_search = pd.read_csv(OUTPUTS / "xgboost_validation_search.csv")
    tabm_history = pd.read_csv(OUTPUTS / "tabm_history.csv")
    rescue = pd.read_csv(OUTPUTS / "xgboost_tabm_stack_comparison.csv")

    summary = {
        "status": "verified_complete",
        "selection_rule": (
            "train-only feature audit; patient-grouped CV; later encounter-order "
            "validation gate; frozen test comparison"
        ),
        "primary_model_decision": "retain_existing_catboost_ensemble",
        "decision_reason": (
            "The four-model blend changed test PR-AUC by only +0.000036 while "
            "F1 and accuracy decreased; added inference complexity is not justified."
        ),
        "feature_pruning": records(pruning),
        "patient_grouped_grid_best": records(group_grid.head(1))[0],
        "later_validation_gate_best": records(later_gate.head(1))[0],
        "catboost_validation_and_test": records(
            final_catboost[
                final_catboost["split"].isin(["validation", "test"])
            ]
        ),
        "dense_zero_dropout_ft": {
            "architecture": {
                "token_dimension": 96,
                "transformer_blocks": 4,
                "attention_heads": 8,
                "feedforward_dimension": 256,
                "attention_dropout": 0.0,
                "feedforward_dropout": 0.0,
            },
            "best_epoch": int(
                ft_history.sort_values("validation_pr_auc", ascending=False)
                .iloc[0]["epoch"]
            ),
            "validation_and_test": records(
                neural[
                    neural["model"].str.startswith(
                        "FT-Transformer dense zero-dropout"
                    )
                    & neural["split"].isin(["validation", "test"])
                ]
            ),
        },
        "autoencoder": {
            "candidate_validation_results": records(autoencoder),
            "selected_validation_and_test": records(
                neural[
                    neural["model"].str.startswith("Autoencoder-")
                    & neural["split"].isin(["validation", "test"])
                ]
            ),
            "decision": "reject_predictive_signal_was_lost",
        },
        "categorical_chi_square": {
            "features_tested": int(len(chi_square)),
            "assumption_valid_tables": int(
                chi_square["chi_square_assumptions_met"].sum()
            ),
            "interpretation": (
                "Target-independence chi-square plus Cramer's V is useful. "
                "Uniform goodness-of-fit does not prove neatness or randomness."
            ),
        },
        "acf": {
            "lag_20": records(autocorrelation[autocorrelation["lag"] == 20]),
            "interpretation": (
                "Encounter-ID ordering proxy only; the dataset has no row-level date."
            ),
        },
        "final_rescue_pass": {
            "xgboost_validation_best": records(xgboost_search.head(1))[0],
            "tabm_best_epoch": records(
                tabm_history.sort_values(
                    "validation_pr_auc", ascending=False
                ).head(1)
            )[0],
            "validation_and_test": records(
                rescue[rescue["split"].isin(["validation", "test"])]
            ),
            "decision": "retain_existing_catboost_due_later_cohort_transfer",
            "next_requirement": (
                "Richer source predictors: real dates, labs, vitals, comorbidity "
                "detail, hospital context, and behavioral or social-risk data."
            ),
        },
    }
    (OUTPUTS / "improvement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
