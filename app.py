from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier

from readmission_audit.pipeline import (
    positive_probability_logits,
    prepare_catboost_features,
    prepare_model_features,
    softmax_probabilities,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="Clinical Readmission Risk, Calibration & Drift",
    page_icon="🏥",
    layout="wide",
)
st.title("Clinical Readmission Risk, Calibration & Drift")
st.caption(
    "UCI Diabetes 130-US Hospitals · binary <30-day readmission · "
    "educational audit only, not a clinical decision system"
)

required = [
    OUTPUTS / "metrics.json",
    OUTPUTS / "model_comparison.csv",
    OUTPUTS / "feature_relevance.csv",
    OUTPUTS / "top_k_feature_ablation.csv",
    OUTPUTS / "top_k_final_comparison.csv",
    OUTPUTS / "raw_missingness_audit.csv",
    OUTPUTS / "feature_quality_audit.csv",
    OUTPUTS / "categorical_chi_square_audit.csv",
    OUTPUTS / "autocorrelation_diagnostics.csv",
    OUTPUTS / "feature_pruning_ablation.csv",
    OUTPUTS / "catboost_group_grid_search.csv",
    OUTPUTS / "catboost_later_validation_gate.csv",
    OUTPUTS / "catboost_final_improvement_comparison.csv",
    OUTPUTS / "autoencoder_comparison.csv",
    OUTPUTS / "ft_dense_zero_dropout_history.csv",
    ARTIFACTS / "catboost_secondary_model.cbm",
    ARTIFACTS / "catboost_config.json",
]
if not all(path.exists() for path in required):
    st.warning("Run `python scripts/build_readmission_audit.py` first.")
    st.stop()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_catboost_artifacts():
    config = read_json(ARTIFACTS / "catboost_config.json")
    primary = CatBoostClassifier()
    primary.load_model(ARTIFACTS / "catboost_model.cbm")
    secondary = CatBoostClassifier()
    secondary.load_model(ARTIFACTS / "catboost_secondary_model.cbm")
    return config, primary, secondary


metrics = read_json(OUTPUTS / "metrics.json")
leakage = read_json(OUTPUTS / "leakage_audit.json")
comparison = pd.read_csv(OUTPUTS / "model_comparison.csv")
performance_drift = pd.read_csv(OUTPUTS / "performance_drift.csv")
calibration = pd.read_csv(OUTPUTS / "calibration_curve.csv")
feature_drift = pd.read_csv(OUTPUTS / "feature_drift.csv")
dropout_sweep = pd.read_csv(OUTPUTS / "dropout_sweep.csv")
ft_dropout_sweep = pd.read_csv(OUTPUTS / "ft_attention_dropout_sweep.csv")
catboost_blend_sweep = pd.read_csv(OUTPUTS / "catboost_blend_sweep.csv")
thresholds = pd.read_csv(OUTPUTS / "threshold_table.csv")
feature_relevance = pd.read_csv(OUTPUTS / "feature_relevance.csv")
top_k_ablation = pd.read_csv(OUTPUTS / "top_k_feature_ablation.csv")
top_k_final = pd.read_csv(OUTPUTS / "top_k_final_comparison.csv")
raw_missingness = pd.read_csv(OUTPUTS / "raw_missingness_audit.csv")
feature_quality = pd.read_csv(OUTPUTS / "feature_quality_audit.csv")
chi_square_audit = pd.read_csv(OUTPUTS / "categorical_chi_square_audit.csv")
autocorrelation = pd.read_csv(OUTPUTS / "autocorrelation_diagnostics.csv")
feature_pruning = pd.read_csv(OUTPUTS / "feature_pruning_ablation.csv")
catboost_group_grid = pd.read_csv(OUTPUTS / "catboost_group_grid_search.csv")
catboost_later_gate = pd.read_csv(OUTPUTS / "catboost_later_validation_gate.csv")
catboost_final_comparison = pd.read_csv(
    OUTPUTS / "catboost_final_improvement_comparison.csv"
)
autoencoder_comparison = pd.read_csv(OUTPUTS / "autoencoder_comparison.csv")
ft_dense_history = pd.read_csv(OUTPUTS / "ft_dense_zero_dropout_history.csv")

(
    overview_tab,
    models_tab,
    feature_audit_tab,
    experiments_tab,
    drift_tab,
    demo_tab,
    leakage_tab,
) = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Feature & missingness audit",
        "Improvement experiments",
        "Calibration & drift",
        "Try tuned ensemble",
        "Leakage audit",
    ]
)

with overview_tab:
    test = metrics["best_model_test"]
    columns = st.columns(7)
    columns[0].metric("Raw encounters", f"{metrics['raw_rows']:,}")
    columns[1].metric("Modeled encounters", f"{metrics['modeling_rows']:,}")
    columns[2].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    columns[3].metric("PR-AUC", f"{test['pr_auc']:.3f}")
    columns[4].metric("F1", f"{test['f1']:.3f}")
    columns[5].metric("Recall", f"{test['recall']:.3f}")
    columns[6].metric("Brier", f"{test['brier_score']:.3f}")

    st.subheader("Selected model")
    st.code(
        "engineered clinical features → tuned CatBoost depth-5/depth-8 blend "
        "→ validation-only temperature scaling"
    )
    st.write(
        "The selected ensemble uses richer diagnosis groups and three-digit codes, "
        "medication and utilization intensity, and strictly prior encounter history. "
        "Hyperparameters, blend weight, probability temperature, and threshold were "
        "selected without using the test set."
    )
    st.info(
        "The original tuned CatBoost ensemble remains primary. A new grouped-grid "
        "search, denser zero-dropout FT-Transformer, and autoencoder were audited; "
        "none produced a meaningful held-out PR-AUC improvement."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("CatBoost blend sweep")
        st.dataframe(
            catboost_blend_sweep.head(10), width="stretch", hide_index=True
        )
        with st.expander("Neural dropout experiments"):
            st.write("FT-Transformer attention dropout")
            st.dataframe(ft_dropout_sweep, width="stretch", hide_index=True)
            st.write("Simple DNN dropout")
            st.dataframe(dropout_sweep, width="stretch", hide_index=True)
    with right:
        st.subheader("Patient-aware ordered split")
        split_table = pd.DataFrame(leakage["splits"]).T.reset_index(
            names="split"
        )
        st.dataframe(split_table, width="stretch", hide_index=True)
        st.caption(
            "Encounter ID is an order proxy because the public file has no row-level date. "
            "These are not claimed as literal calendar-year cohorts."
        )

with models_tab:
    st.subheader("Held-out test comparison")
    test_comparison = comparison[comparison["split"] == "test"].copy()
    display_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "f1",
        "recall",
        "accuracy",
        "brier_score",
        "ece_10_bin",
        "threshold",
    ]
    st.dataframe(
        test_comparison[display_columns].style.format(
            {column: "{:.4f}" for column in display_columns[1:]}
        ),
        width="stretch",
        hide_index=True,
    )
    chart_frame = test_comparison.melt(
        id_vars="model",
        value_vars=["roc_auc", "pr_auc", "f1", "recall", "accuracy"],
        var_name="metric",
        value_name="value",
    )
    st.plotly_chart(
        px.bar(
            chart_frame,
            x="metric",
            y="value",
            color="model",
            barmode="group",
            range_y=[0, 1],
        ),
        width="stretch",
    )
    st.subheader("Selected ensemble threshold tradeoff on test")
    st.dataframe(thresholds, width="stretch", hide_index=True)

    st.subheader("Train-only feature ranking and validation top-k ablation")
    st.dataframe(top_k_ablation, width="stretch", hide_index=True)
    st.caption(
        "Mutual information ranked numeric and categorical feature groups using "
        "training data only. Validation selected k=50 by ROC-AUC, but its frozen "
        "held-out result was worse than the full 71-feature ensemble."
    )
    left, right = st.columns(2)
    with left:
        st.write("Top 20 mixed-type relevance scores")
        st.dataframe(
            feature_relevance.head(20), width="stretch", hide_index=True
        )
    with right:
        st.write("Frozen held-out comparison")
        st.dataframe(top_k_final, width="stretch", hide_index=True)

with feature_audit_tab:
    st.subheader("What is missing, and what was done")
    st.dataframe(
        raw_missingness.loc[raw_missingness["missing_rate"] > 0],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Weight, payer code, and medical specialty remain excluded for high "
        "missingness. A1C and glucose results remain categorical because 'not "
        "recorded' is meaningful; explicit recorded/not-recorded indicators were added."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Constant and ultra-rare training fields")
        sparse_fields = feature_quality[
            feature_quality["exact_constant"]
            | feature_quality["ultra_rare_non_mode_lt_50"]
        ]
        st.dataframe(
            sparse_fields[
                [
                    "feature",
                    "exact_constant",
                    "non_mode_rows",
                    "top_value_share",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Validation-only pruning result")
        st.dataframe(feature_pruning, width="stretch", hide_index=True)
        st.caption(
            "All 73 engineered fields won validation PR-AUC. The project therefore "
            "does not drop a feature merely because it is sparse or incomplete."
        )

    st.subheader("Categorical target association: chi-square + Cramér's V")
    st.dataframe(
        chi_square_audit[
            [
                "feature",
                "cramers_v",
                "target_independence_fdr_p_value",
                "minimum_expected_count",
                "chi_square_assumptions_met",
                "uniform_gof_p_value",
            ]
        ].head(25),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "A uniform goodness-of-fit test does not prove that clinical data are "
        "'neatly spread' or non-random. Target-independence chi-square is the useful "
        "association test here; Cramér's V supplies effect size, and invalid low-count "
        "tables are flagged."
    )

    st.subheader("Encounter-order ACF and Ljung-Box diagnostic")
    st.plotly_chart(
        px.line(
            autocorrelation,
            x="lag",
            y="acf",
            color="series",
            markers=True,
        ),
        width="stretch",
    )
    st.caption(
        "The UCI file has no row-level dates. This is an encounter-ID ordering proxy, "
        "not a claim about calendar-time autocorrelation or stationarity."
    )

with experiments_tab:
    st.subheader("Patient-grouped CatBoost grid search")
    st.dataframe(
        catboost_group_grid[
            [
                "candidate",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "bagging_temperature",
                "mean_test_pr_auc",
                "std_test_pr_auc",
                "pr_auc_train_cv_gap",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Patients never cross folds. The large grouped-CV to later-validation gap "
        "shows encounter-order cohort shift, so each candidate also passed through "
        "a separate later-cohort validation gate."
    )
    st.subheader("Later-cohort validation gate")
    st.dataframe(catboost_later_gate, width="stretch", hide_index=True)

    st.subheader("Final CatBoost comparison")
    st.dataframe(
        catboost_final_comparison[
            catboost_final_comparison["split"].isin(["validation", "test"])
        ][
            [
                "model",
                "split",
                "roc_auc",
                "pr_auc",
                "f1",
                "recall",
                "accuracy",
                "brier_score",
                "ece_10_bin",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.success(
        "Decision: retain the existing CatBoost ensemble. The larger blend changed "
        "test PR-AUC from 0.174654 to 0.174690 but reduced F1 and accuracy; the gain "
        "is too small to justify added complexity."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Dense zero-dropout FT-Transformer")
        st.dataframe(ft_dense_history, width="stretch", hide_index=True)
        st.caption(
            "96-dimensional tokens, 4 blocks, 8 heads, 256-unit feed-forward layer, "
            "0 attention dropout, and 0 feed-forward dropout."
        )
    with right:
        st.subheader("Autoencoder bottlenecks")
        st.dataframe(autoencoder_comparison, width="stretch", hide_index=True)
        st.caption(
            "The unsupervised bottleneck removed predictive signal: the selected "
            "64-dimensional representation reached only 0.13364 validation PR-AUC."
        )

with drift_tab:
    st.subheader("Calibration on the later encounter-order test cohort")
    selected_calibration = calibration[
        (calibration["split"] == "test")
        & calibration["model"].isin(
            [
                "Logistic Regression",
                "CatBoost tuned ensemble temperature-scaled",
                "FT-Transformer temperature-scaled",
            ]
        )
    ]
    figure = px.line(
        selected_calibration,
        x="mean_predicted_probability",
        y="observed_positive_rate",
        color="model",
        markers=True,
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="perfect calibration",
        )
    )
    figure.update_xaxes(range=[0, 0.6])
    figure.update_yaxes(range=[0, 0.6])
    st.plotly_chart(figure, width="stretch")

    st.subheader("Validation-to-test performance change")
    st.dataframe(performance_drift, width="stretch", hide_index=True)

    st.subheader("Feature distribution shift")
    test_drift = feature_drift[
        feature_drift["comparison"] == "train_to_test"
    ].copy()
    test_drift["drift_score"] = test_drift[
        ["psi", "ks_statistic", "js_divergence"]
    ].max(axis=1, skipna=True)
    st.dataframe(
        test_drift.sort_values("drift_score", ascending=False).head(20),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "PSI and KS are reported for numeric features; Jensen-Shannon divergence "
        "is reported for categorical features. Values describe this split only."
    )

with demo_tab:
    st.subheader("Single-encounter tuned CatBoost ensemble inference")
    st.info(
        "This illustrates the saved preprocessing and model artifact. It does not "
        "provide medical advice or a clinically validated score."
    )
    config, primary_model, secondary_model = load_catboost_artifacts()
    defaults = read_json(ARTIFACTS / "demo_defaults.json")
    options = read_json(ARTIFACTS / "demo_options.json")
    values = dict(defaults)

    left, middle, right = st.columns(3)
    with left:
        for field, label in (
            ("age", "Age band"),
            ("race", "Race"),
            ("gender", "Gender"),
            ("A1Cresult", "A1C result"),
        ):
            if field in options:
                default_index = options[field].index(str(defaults[field]))
                values[field] = st.selectbox(
                    label, options[field], index=default_index
                )
    with middle:
        for field, label, maximum in (
            ("time_in_hospital", "Days in hospital", 14),
            ("num_lab_procedures", "Lab procedures", 150),
            ("num_medications", "Medications", 80),
            ("number_inpatient", "Prior inpatient visits", 25),
        ):
            if field in defaults:
                values[field] = st.number_input(
                    label,
                    min_value=0,
                    max_value=maximum,
                    value=int(defaults[field]),
                )
    with right:
        for field, label, maximum in (
            ("number_emergency", "Prior emergency visits", 50),
            ("number_outpatient", "Prior outpatient visits", 50),
        ):
            if field in defaults:
                values[field] = st.number_input(
                    label,
                    min_value=0,
                    max_value=maximum,
                    value=int(defaults[field]),
                )
        for field, label in (
            ("insulin", "Insulin status"),
            ("diabetesMed", "Diabetes medication"),
        ):
            if field in options:
                default_index = options[field].index(str(defaults[field]))
                values[field] = st.selectbox(
                    label, options[field], index=default_index
                )

    raw_row = pd.DataFrame([values], columns=config["raw_feature_columns"])
    row = prepare_model_features(raw_row).reindex(
        columns=config["feature_columns"]
    )
    row = prepare_catboost_features(row, config["categorical_features"])
    primary_probability = primary_model.predict_proba(row)[:, 1]
    secondary_probability = secondary_model.predict_proba(row)[:, 1]
    blended_probability = (
        config["primary_weight"] * primary_probability
        + config["secondary_weight"] * secondary_probability
    )
    logits = positive_probability_logits(blended_probability)
    probability = float(softmax_probabilities(logits, config["temperature"])[0, 1])
    flagged = probability >= config["threshold"]
    result_column, explanation_column = st.columns([1, 2])
    result_column.metric("Estimated <30-day probability", f"{probability:.1%}")
    result_column.metric("Validation-selected threshold", f"{config['threshold']:.1%}")
    explanation_column.write(
        "**Audit flag:** " + ("above threshold" if flagged else "below threshold")
    )
    explanation_column.caption(
        "The threshold maximized F1 on validation data. It is not a treatment cutoff."
    )

with leakage_tab:
    st.subheader("Verified safeguards")
    checks = {
        "Zero train/validation patient overlap": leakage["patient_overlap_counts"][
            "train_validation"
        ]
        == 0,
        "Zero train/test patient overlap": leakage["patient_overlap_counts"][
            "train_test"
        ]
        == 0,
        "Zero validation/test patient overlap": leakage["patient_overlap_counts"][
            "validation_test"
        ]
        == 0,
        "Target removed from features": leakage["target_removed_from_features"],
        "Patient and encounter IDs removed from features": leakage[
            "identifier_features_removed"
        ],
        "Test excluded from model selection": not leakage[
            "test_used_for_training_or_selection"
        ],
    }
    for label, passed in checks.items():
        st.write(("✅" if passed else "❌") + f" {label}")
    st.subheader("DNN train/validation gap")
    st.json(leakage["dnn_train_validation_gap"])
    st.subheader("FT-Transformer train/validation gap")
    st.json(leakage["ft_transformer_train_validation_gap"])
    st.subheader("Full audit record")
    st.json(leakage)
