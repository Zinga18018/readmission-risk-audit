from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Hospital Readmission Risk Audit", layout="wide")
st.title("Hospital Readmission Risk Audit")

metrics_path = OUTPUTS / "metrics.json"
if not metrics_path.exists():
    st.warning("Run `python scripts/build_readmission_audit.py` first.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
thresholds = pd.read_csv(OUTPUTS / "threshold_table.csv")
coefficients = pd.read_csv(OUTPUTS / "top_coefficients.csv")

st.caption(
    "Educational audit project using the UCI Diabetes 130-hospitals dataset. "
    "This is not a clinical decision system."
)

cols = st.columns(5)
cols[0].metric("Rows", f"{metrics['rows']:,}")
cols[1].metric("Early readmission rate", f"{metrics['positive_rate']:.1%}")
cols[2].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
cols[3].metric("Avg precision", f"{metrics['average_precision']:.3f}")
cols[4].metric("Brier score", f"{metrics['brier_score']:.3f}")

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Threshold Tradeoff")
    long_thresholds = thresholds.melt(
        id_vars="threshold",
        value_vars=["precision", "recall", "f1", "flagged_rate"],
        var_name="metric",
        value_name="value",
    )
    st.plotly_chart(
        px.line(
            long_thresholds,
            x="threshold",
            y="value",
            color="metric",
            markers=True,
            title="Precision, recall, F1, and flagged workload",
        ),
        use_container_width=True,
    )
    st.dataframe(thresholds, use_container_width=True)

with right:
    st.subheader("Top Model Drivers")
    chart_data = coefficients.sort_values("abs_coefficient", ascending=True)
    st.plotly_chart(
        px.bar(
            chart_data,
            x="coefficient",
            y="feature",
            orientation="h",
            title="Largest linear-model coefficients",
        ),
        use_container_width=True,
    )

st.subheader("Group Summaries")
tabs = st.tabs(["Age", "Race", "Gender"])
for tab, name in zip(tabs, ["age", "race", "gender"]):
    path = OUTPUTS / f"group_summary_{name}.csv"
    with tab:
        if path.exists():
            table = pd.read_csv(path)
            st.dataframe(table, use_container_width=True)
        else:
            st.info(f"No {name} summary found.")

st.subheader("What This Project Is Actually Showing")
st.write(
    "The model is not presented as a medical product. The useful part is the audit workflow: "
    "how the target is defined, how performance changes at different thresholds, which "
    "features drive the model, and where subgroup-level predicted risk differs from observed outcomes."
)
