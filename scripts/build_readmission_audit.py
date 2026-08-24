from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from readmission_audit.pipeline import train_and_evaluate, write_outputs

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
ZIP_PATH = RAW_DIR / "diabetes_130_hospitals.zip"
CSV_NAME = "diabetic_data.csv"


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"downloading {DATA_URL}", flush=True)
        urlretrieve(DATA_URL, ZIP_PATH)
    else:
        print(f"using cached {ZIP_PATH}", flush=True)
    return ZIP_PATH


def extract_csv(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(CSV_NAME)]
        if not matches:
            raise FileNotFoundError(f"could not find {CSV_NAME} in {zip_path}")
        target = RAW_DIR / CSV_NAME
        if not target.exists():
            with archive.open(matches[0]) as source, target.open("wb") as destination:
                destination.write(source.read())
        return target


def main() -> None:
    csv_path = extract_csv(download_dataset())
    print(f"reading {csv_path}", flush=True)
    frame = pd.read_csv(csv_path)
    print(f"raw_rows={len(frame)} raw_columns={len(frame.columns)}", flush=True)
    print(
        "training logistic, tree, CatBoost, DNN, and FT-Transformer candidates",
        flush=True,
    )

    result = train_and_evaluate(frame)
    write_outputs(result, OUTPUT_DIR, ARTIFACT_DIR)

    metrics = result["metrics"]
    test = metrics["best_model_test"]
    print(
        f"modeling_rows={metrics['modeling_rows']} "
        f"encoded_features={metrics['encoded_features']}",
        flush=True,
    )
    print(
        f"dnn_dropout={metrics['selected_dropout']:.2f} "
        f"ft_attention_dropout={metrics['selected_attention_dropout']:.2f} "
        f"ft_temperature={metrics['ft_temperature']:.4f}",
        flush=True,
    )
    print(
        "best_model_test "
        f"model={metrics['best_model']} "
        f"roc_auc={test['roc_auc']:.4f} "
        f"pr_auc={test['pr_auc']:.4f} "
        f"f1={test['f1']:.4f} "
        f"recall={test['recall']:.4f} "
        f"accuracy={test['accuracy']:.4f} "
        f"brier={test['brier_score']:.4f} "
        f"ece={test['ece_10_bin']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
