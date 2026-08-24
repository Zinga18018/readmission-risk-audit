from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from readmission_audit.pipeline import FTTransformer, softmax_probabilities


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
    ARTIFACTS / "ft_transformer_state.pt",
]
if not all(path.exists() for path in required):
    st.warning("Run `python scripts/build_readmission_audit.py` first.")
    st.stop()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_ft_artifacts():
    config = read_json(ARTIFACTS / "ft_transformer_config.json")
    preprocessor = joblib.load(ARTIFACTS / "ft_preprocessor.joblib")
    model = FTTransformer(
        numeric_features=config["numeric_features"],
        category_cardinalities=config["category_cardinalities"],
        token_dimension=config["token_dimension"],
        transformer_blocks=config["transformer_blocks"],
        attention_heads=config["attention_heads"],
        feedforward_dimension=config["feedforward_dimension"],
        attention_dropout=config["attention_dropout"],
        feedforward_dropout=config["feedforward_dropout"],
    )
    model.load_state_dict(
        torch.load(
            ARTIFACTS / "ft_transformer_state.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return config, preprocessor, model


metrics = read_json(OUTPUTS / "metrics.json")
leakage = read_json(OUTPUTS / "leakage_audit.json")
comparison = pd.read_csv(OUTPUTS / "model_comparison.csv")
performance_drift = pd.read_csv(OUTPUTS / "performance_drift.csv")
calibration = pd.read_csv(OUTPUTS / "calibration_curve.csv")
feature_drift = pd.read_csv(OUTPUTS / "feature_drift.csv")
dropout_sweep = pd.read_csv(OUTPUTS / "dropout_sweep.csv")
ft_dropout_sweep = pd.read_csv(OUTPUTS / "ft_attention_dropout_sweep.csv")
thresholds = pd.read_csv(OUTPUTS / "threshold_table.csv")

overview_tab, models_tab, drift_tab, demo_tab, leakage_tab = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Calibration & drift",
        "Try FT-Transformer",
        "Leakage audit",
    ]
)

with overview_tab:
    test = metrics["ft_test"]
    columns = st.columns(7)
    columns[0].metric("Raw encounters", f"{metrics['raw_rows']:,}")
    columns[1].metric("Modeled encounters", f"{metrics['modeling_rows']:,}")
    columns[2].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    columns[3].metric("PR-AUC", f"{test['pr_auc']:.3f}")
    columns[4].metric("F1", f"{test['f1']:.3f}")
    columns[5].metric("Recall", f"{test['recall']:.3f}")
    columns[6].metric("Brier", f"{test['brier_score']:.3f}")

    st.subheader("What was trained")
    st.code(
        "numeric + categorical feature tokens → CLS token → "
        "3 Transformer blocks → Linear(2 raw logits)"
    )
    st.write(
        "FT-Transformer training uses `CrossEntropyLoss` on raw logits. Softmax is applied only "
        "during evaluation and inference. A validation-selected temperature "
        "rescales logits before softmax; its held-out calibration effect is reported "
        "rather than assumed."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("FT attention-dropout sweep")
        st.dataframe(ft_dropout_sweep, width="stretch", hide_index=True)
        with st.expander("Simple DNN dropout sweep"):
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
    st.subheader("FT-Transformer threshold tradeoff on test")
    st.dataframe(thresholds, width="stretch", hide_index=True)

with drift_tab:
    st.subheader("Calibration on the later encounter-order test cohort")
    selected_calibration = calibration[
        (calibration["split"] == "test")
        & calibration["model"].isin(
            [
                "Logistic Regression",
                "CatBoost",
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
    st.subheader("Single-encounter FT-Transformer inference")
    st.info(
        "This illustrates the saved preprocessing and model artifact. It does not "
        "provide medical advice or a clinically validated score."
    )
    config, preprocessor, ft_model = load_ft_artifacts()
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

    row = pd.DataFrame([values], columns=config["feature_columns"])
    numeric, categorical = preprocessor.transform(row)
    with torch.no_grad():
        logits = ft_model(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(categorical, dtype=torch.long),
        ).numpy()
    probability = float(
        softmax_probabilities(logits, config["temperature"])[0, 1]
    )
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
