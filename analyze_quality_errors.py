import os
import shutil
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PREDICTION_FILE = Path(
    "reports/quality_baseline_v4_class_weighted/test_predictions.csv"
)

OUTPUT_DIR = Path(
    "reports/quality_baseline_v4_class_weighted/error_analysis"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(PREDICTION_FILE)

errors = df[
    df["correct"] == False
].copy()


# ============================================================
# ERROR GROUPS
# ============================================================

groups = [
    ("old", "dried"),
    ("ripe", "unripe"),
    ("ripe", "old"),
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for true_label, predicted_label in groups:

    folder_name = f"{true_label}_TO_{predicted_label}"

    folder = OUTPUT_DIR / folder_name

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY IMAGES
# ============================================================

print("=" * 80)
print("CIRCULENS ERROR VISUAL AUDIT")
print("=" * 80)

for true_label, predicted_label in groups:

    subset = errors[
        (errors["label"] == true_label)
        &
        (errors["prediction_label"] == predicted_label)
    ].copy()

    folder_name = f"{true_label}_TO_{predicted_label}"

    output_folder = OUTPUT_DIR / folder_name

    print()
    print(
        f"{true_label} -> {predicted_label}: "
        f"{len(subset)} images"
    )

    for i, row in subset.reset_index(drop=True).iterrows():

        source = Path(row["path"])

        if not source.exists():

            print(
                "MISSING:",
                source
            )

            continue

        destination = (
            output_folder
            /
            f"{i+1:03d}_"
            f"{true_label}_"
            f"pred_{predicted_label}_"
            f"conf_{row['confidence']:.3f}_"
            f"{source.name}"
        )

        shutil.copy2(
            source,
            destination
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print()
print("Output:")
print(OUTPUT_DIR)