"""Read historical aggregate evidence and optionally bootstrap real private predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from readmission_audit.evaluation import patient_cluster_bootstrap, threshold_workload_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Private CSV: encounter_id, patient_id, y_true, y_prob")
    parser.add_argument("--model-config", type=Path, default=ROOT / "artifacts" / "catboost_config.json")
    parser.add_argument("--cohort-status", choices=["historical_exploratory", "new_cohort_unverified", "synthetic_demo"], default="historical_exploratory")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-policy", choices=["error", "drop"], default="error")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    metrics_path = ROOT / "outputs" / "metrics.json"
    table_path = ROOT / "outputs" / "threshold_table.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    workload = threshold_workload_table(pd.read_csv(table_path), metrics["best_model_test"])
    uncertainty = {"status": "unavailable", "reason": "No per-encounter predictions with patient IDs were supplied. Aggregate metrics cannot reconstruct patient-cluster intervals.",
                   "required_columns": ["encounter_id", "patient_id", "y_true", "y_prob"]}
    if args.predictions is not None:
        predictions = pd.read_csv(args.predictions, dtype={"encounter_id": "string", "patient_id": "string"})
        uncertainty = {"status": "computed_for_supplied_predictions",
                       "cohort_status_supplied_by_caller": args.cohort_status,
                       "cohort_provenance_independently_verified": False,
                       "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                       **patient_cluster_bootstrap(predictions, frozen_threshold=float(config["threshold"]),
                           n_resamples=args.resamples, seed=args.seed, missing_policy=args.missing_policy)}
    result = {"historical_evaluation_status": "retrospective_exploratory_test_reuse",
              "new_model_training_performed": False,
              "historical_metric_source_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
              "historical_threshold_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
              "bootstrap_model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
              "historical_saved_threshold": metrics["best_model_test"]["threshold"],
              "historical_workload": json.loads(workload.to_json(orient="records")),
              "uncertainty": uncertainty}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_readout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    workload.to_csv(args.output_dir / "historical_threshold_workload.csv", index=False)
    selected = workload[workload.saved_validation_threshold].iloc[0]
    lines = ["# Readmission evaluation and workload readout", "",
             "**Historical retrospective/exploratory evaluation.** Patient-disjoint splitting does not make a repeatedly inspected test partition an untouched final evaluation.", "",
             "No model was retrained, refit or recalibrated by this readout. It derives workload from the existing aggregate files.", "",
             "## Saved threshold", "",
             f"At the saved validation-selected threshold {selected.threshold:.3f}, {int(selected.audit_flags):,} of {int(selected.rows):,} encounters are flagged ({selected.flags_per_1000_encounters:.2f} per 1,000).",
             f"The confusion counts are TP={int(selected.true_positives)}, FP={int(selected.false_positives)}, FN={int(selected.false_negatives)}, TN={int(selected.true_negatives)}. Recall is {selected.recall:.2%}; precision is {selected.precision:.2%}.", "",
             "These are encounter-level audit flags, not unique patient workload, clinical alerts or evidence that intervention changes outcomes.", "",
             "## Historical sensitivity analysis", "",
             "The table explains workload versus missed positives. **Do not choose a replacement threshold by optimizing this historical test table.** Agree on a workload objective, choose on validation and evaluate on a newly designated cohort.", "",
             "| Threshold | Flags / 1,000 | Recall | Precision | Missed positives / 1,000 | Saved threshold |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in workload.itertuples():
        lines.append(f"| {row.threshold:.3f} | {row.flags_per_1000_encounters:.2f} | {row.recall:.2%} | {row.precision:.2%} | {row.missed_positives_per_1000_encounters:.2f} | {row.saved_validation_threshold} |")
    lines += ["", "## Uncertainty", "", f"Status: **{uncertainty['status']}**."]
    if args.predictions is None:
        lines += ["", uncertainty["reason"], "Required input: encounter ID, patient ID, binary outcome and frozen-model probability for each evaluated encounter."]
    else:
        lines += ["", "Intervals for supplied predictions are in the JSON output. The caller-provided cohort label is not independent proof of an untouched cohort; these intervals cannot undo selection bias or historical test reuse."]
    lines += ["", "The [future protocol](../config/evaluation_protocol.json) is a draft with unfilled cohort and artifact-lock fields. The utility resamples whole patient clusters while retaining encounter weighting, freezes the supplied threshold, and reports missing rows and undefined/single-class replicates.", ""]
    (args.output_dir / "evaluation_readout.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Historical flags: {int(selected.audit_flags)}/{int(selected.rows)}; recall={selected.recall:.6f}; uncertainty={uncertainty['status']}")


if __name__ == "__main__":
    main()
