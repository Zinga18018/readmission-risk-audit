from __future__ import annotations

import zipfile
from pathlib import Path
import sys
from urllib.request import urlretrieve

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from readmission_audit.pipeline import train_and_evaluate, write_outputs

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
ZIP_PATH = RAW_DIR / "diabetes_130_hospitals.zip"
CSV_NAME = "diabetic_data.csv"


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"downloading {DATA_URL}")
        urlretrieve(DATA_URL, ZIP_PATH)
    else:
        print(f"using cached {ZIP_PATH}")
    return ZIP_PATH


def extract_csv(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith(CSV_NAME)]
        if not matches:
            raise FileNotFoundError(f"could not find {CSV_NAME} in {zip_path}")
        member = matches[0]
        target = RAW_DIR / CSV_NAME
        if not target.exists():
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
        return target


def main() -> None:
    zip_path = download_dataset()
    csv_path = extract_csv(zip_path)

    print(f"reading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"raw_rows={len(df)} raw_columns={len(df.columns)}")

    result = train_and_evaluate(df)
    write_outputs(result, OUTPUT_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sample_cols = [
        "age",
        "race",
        "gender",
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "readmitted",
    ]
    existing = [c for c in sample_cols if c in df.columns]
    df[existing].head(5000).to_csv(PROCESSED_DIR / "dashboard_sample.csv", index=False)

    metrics = result["metrics"]
    print(f"rows={metrics['rows']}")
    print(f"positive_rate={metrics['positive_rate']:.4f}")
    print(f"roc_auc={metrics['roc_auc']:.4f}")
    print(f"average_precision={metrics['average_precision']:.4f}")
    print(f"brier_score={metrics['brier_score']:.4f}")
    print(
        "best_threshold="
        f"{metrics['best_threshold']:.2f} "
        f"precision={metrics['best_threshold_precision']:.4f} "
        f"recall={metrics['best_threshold_recall']:.4f} "
        f"f1={metrics['best_threshold_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
