"""Plot saved aggregate workload counts; no patient records or model execution."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/evaluation_readout.json"
OUTPUT = ROOT / "docs/images/historical-workload.svg"
SOURCE_SHA256 = "58793475c3f3f133b4d82ea381e9e389977a7da82def6eb11776fd5708f83331"


def checked_rows():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Saved readout changed; review it before updating the plot.")
    record = json.loads(raw)
    assert record["historical_evaluation_status"] == "retrospective_exploratory_test_reuse"
    assert record["new_model_training_performed"] is False
    rows = record["historical_workload"]
    assert len(rows) == 15
    assert [r["threshold"] for r in rows] == sorted(r["threshold"] for r in rows)
    for r in rows:
        tp, fp, fn, tn = (r[k] for k in ("true_positives", "false_positives", "false_negatives", "true_negatives"))
        assert tp + fp + fn + tn == r["rows"] == 10822
        assert tp + fp == r["audit_flags"]
        assert math.isclose(1000 * (tp + fp) / r["rows"], r["flags_per_1000_encounters"], abs_tol=1e-7)
        assert math.isclose(1000 * fn / r["rows"], r["missed_positives_per_1000_encounters"], abs_tol=1e-7)
    saved = [r for r in rows if r["saved_validation_threshold"]]
    assert len(saved) == 1 and math.isclose(saved[0]["threshold"], record["historical_saved_threshold"])
    assert (saved[0]["true_positives"], saved[0]["false_positives"], saved[0]["false_negatives"], saved[0]["true_negatives"]) == (312, 1592, 637, 8281)
    return rows, saved[0]["threshold"]


def render(png=None):
    rows, threshold = checked_rows()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "readmission-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Use requirements-visuals.txt for reproducible rendering.")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "readmission-historical-workload",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
        x = [r["threshold"] for r in rows]
        for key, label, color, marker in [
            ("flags_per_1000_encounters", "Flagged encounters", "#32648e", "o"),
            ("missed_positives_per_1000_encounters", "Missed readmissions", "#b65b24", "s")]:
            ax.plot(x, [r[key] for r in rows], label=label, color=color, marker=marker, markersize=4, linewidth=1.5)
        ax.axvline(threshold, color="#555555", linestyle="--", linewidth=1,
                   label=f"Saved threshold ({threshold:g})")
        ax.set(title="Historical workload (10,822 encounters)", xlabel="Score threshold",
               ylabel="Count per 1,000 encounters", xlim=(.03, .31), ylim=(0, 880))
        ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", metadata={"Date": None}, facecolor="white")
        if png:
            fig.savefig(png, dpi=160, facecolor="white")
        plt.close(fig)
    svg = buf.getvalue().replace('<svg ', '<svg role="img" aria-labelledby="title desc" ', 1)
    match = re.search(r"<svg\b[^>]*>", svg)
    description = ('<title id="title">Historical readmission audit workload</title>'
                   '<desc id="desc">Fifteen recorded score thresholds show flagged encounters '
                   'and missed readmissions per 1,000 encounters. Saved threshold is 0.122. '
                   'Retrospective exploratory test reuse; no new model run, uncertainty interval '
                   'or clinical utility claim.</desc>')
    svg = svg[:match.end()] + description + svg[match.end():]
    ET.fromstring(svg)
    return svg.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--png", type=Path, help="Optional local review image")
    args = parser.parse_args()
    data = render(args.png)
    if args.check:
        assert OUTPUT.read_bytes() == data, "The SVG differs from fresh Matplotlib output."
        print("PASS: pinned aggregate input, 15 thresholds, confusion counts, rates and exact SVG rerender")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(data)
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
