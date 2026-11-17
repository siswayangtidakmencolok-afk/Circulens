# analyze_quality_errors_v5.py
#
# CircuLens Quality — V5 Error Analysis
#
# Purpose:
#   Analyze the errors produced by train_quality_finetune_v5.py
#   and identify the main bottlenecks before developing V6.
#
# Input:
#   reports/quality_finetune_v5/
#   dataset/circulens_quality_clean_v4/
#
# Outputs:
#   reports/quality_error_analysis_v5/
#
# Main analyses:
#   1. Load V5 predictions
#   2. Validate prediction data
#   3. Overall error statistics
#   4. Error pairs
#   5. Error by true class
#   6. Error by predicted class
#   7. Error by source
#   8. Source × error pair
#   9. Confidence analysis
#   10. High-confidence error analysis
#   11. Save ranked error cases
#   12. Generate visual error contact sheets
#   13. Generate confusion matrix visualization
#   14. Generate diagnostic summary
#
# No model training is performed in this script.


import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

MANIFEST_PATH = Path(
    "dataset/circulens_quality_clean_v4/"
    "quality_clean_v4_manifest_rebalanced.csv"
)

V5_OUTPUT_DIR = Path("reports/quality_finetune_v5")

OUTPUT_DIR = Path("reports/quality_error_analysis_v5")

PREDICTIONS_PATH = V5_OUTPUT_DIR / "test_predictions.csv"
ERRORS_PATH = V5_OUTPUT_DIR / "test_errors.csv"
HIGH_CONF_PATH = V5_OUTPUT_DIR / "high_confidence_errors.csv"
CONFUSION_PATH = V5_OUTPUT_DIR / "confusion_matrix.csv"
ERROR_PAIRS_PATH = V5_OUTPUT_DIR / "error_pairs.csv"
ERROR_PAIRS_SOURCE_PATH = V5_OUTPUT_DIR / "error_pairs_by_source.csv"

IMAGE_SIZE = 224

HIGH_CONFIDENCE_THRESHOLD = 0.80

MAX_VISUAL_ERRORS_PER_GROUP = 20
MAX_HIGH_CONF_ERRORS = 40

FONT_SIZE = 18

warnings.filterwarnings("ignore")

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# CLASS ORDER
# =============================================================================

CLASS_ORDER = [
    "damaged",
    "dried",
    "old",
    "ripe",
    "unripe",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "visual_errors").mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "visual_errors" / "error_pairs").mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "visual_errors" / "high_confidence").mkdir(
        parents=True,
        exist_ok=True,
    )


def find_column(df, candidates, required=True):
    """
    Find the first matching column from a list of candidates.
    Matching is case-insensitive.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_label(value):
    if pd.isna(value):
        return value

    return str(value).strip().lower()


def resolve_image_path(path_value):
    """
    Resolve an image path stored in the CSV.

    Handles:
      - relative paths
      - C:/...
      - C:\\...
      - paths relative to project root
      - paths relative to manifest directory
    """

    if pd.isna(path_value):
        return None

    raw = str(path_value).strip()

    if not raw:
        return None

    # Normalize slash direction
    raw = raw.replace("\\", os.sep)

    candidate_paths = []

    # Direct path
    candidate_paths.append(Path(raw))

    # Relative to current project directory
    candidate_paths.append(Path.cwd() / raw)

    # Relative to manifest directory
    candidate_paths.append(MANIFEST_PATH.parent / raw)

    # Relative to dataset root
    candidate_paths.append(
        Path("dataset") / raw
    )

    # Relative to project root if path starts with dataset/
    if raw.lower().startswith("dataset" + os.sep):
        candidate_paths.append(Path.cwd() / raw)

    # Try candidates
    for candidate in candidate_paths:

        try:
            candidate = candidate.resolve()

            if candidate.exists():
                return candidate

        except Exception:
            continue

    return None


def get_image_filename(row, path_col, filename_col=None):
    """
    Determine best filename for display.
    """

    if filename_col is not None:
        value = row.get(filename_col)

        if not pd.isna(value):
            value = str(value).strip()

            if value:
                return value

    value = row.get(path_col)

    if not pd.isna(value):
        return Path(str(value)).name

    return "unknown.jpg"


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def create_text_image(
    image,
    lines,
    width=900,
    padding=20,
):
    """
    Add text below an image.
    """

    image = image.convert("RGB")

    scale = width / image.width

    height = int(image.height * scale)

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    text_height = 50 + (len(lines) * 30)

    canvas = Image.new(
        "RGB",
        (width, height + text_height),
        "white",
    )

    canvas.paste(
        image,
        (0, 0),
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            FONT_SIZE,
        )
    except Exception:
        font = ImageFont.load_default()

    y = height + 10

    for line in lines:
        draw.text(
            (padding, y),
            str(line),
            fill="black",
            font=font,
        )

        y += 30

    return canvas


def load_image_for_row(
    row,
    path_col,
):
    path = resolve_image_path(row[path_col])

    if path is None:
        return None

    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception:
        return None


# =============================================================================
# START
# =============================================================================

print_header(
    "CIRCULENS QUALITY — V5 ERROR ANALYSIS"
)

print()
print("V5 directory:")
print(V5_OUTPUT_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("High-confidence threshold:")
print(f"{HIGH_CONFIDENCE_THRESHOLD:.2f}")


ensure_output_dirs()


# =============================================================================
# 1. CHECK INPUT FILES
# =============================================================================

print_header("1. CHECKING INPUT FILES")

required_files = [
    PREDICTIONS_PATH,
    MANIFEST_PATH,
]

optional_files = [
    ERRORS_PATH,
    HIGH_CONF_PATH,
    CONFUSION_PATH,
    ERROR_PAIRS_PATH,
    ERROR_PAIRS_SOURCE_PATH,
]

for path in required_files:

    print(
        f"{str(path):80s}",
        "FOUND" if path.exists() else "MISSING",
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


for path in optional_files:

    print(
        f"{str(path):80s}",
        "FOUND" if path.exists() else "OPTIONAL/MISSING",
    )


# =============================================================================
# 2. LOAD PREDICTIONS
# =============================================================================

print_header("2. LOADING V5 PREDICTIONS")

pred = pd.read_csv(PREDICTIONS_PATH)

print("Rows   :", len(pred))
print("Columns:", list(pred.columns))

print()
print("First rows:")
print(pred.head())


# =============================================================================
# 3. IDENTIFY COLUMNS
# =============================================================================

print_header("3. IDENTIFYING PREDICTION COLUMNS")

true_col = find_column(
    pred,
    [
        "label",
        "true_label",
        "true",
        "actual",
        "y_true",
    ],
)

pred_col = find_column(
    pred,
    [
        "prediction_label",
        "predicted_label",
        "prediction",
        "predicted",
        "y_pred",
    ],
)

path_col = find_column(
    pred,
    [
        "path",
        "image_path",
        "filepath",
        "file_path",
    ],
)

source_col = find_column(
    pred,
    [
        "source",
    ],
    required=False,
)

confidence_col = find_column(
    pred,
    [
        "confidence",
        "prediction_confidence",
        "max_probability",
        "probability",
        "max_prob",
    ],
    required=False,
)

filename_col = find_column(
    pred,
    [
        "filename",
        "file_name",
    ],
    required=False,
)

print("True label column :", true_col)
print("Prediction column :", pred_col)
print("Path column       :", path_col)
print("Source column     :", source_col)
print("Confidence column :", confidence_col)
print("Filename column   :", filename_col)


# =============================================================================
# 4. NORMALIZE DATA
# =============================================================================

print_header("4. NORMALIZING PREDICTION DATA")

pred[true_col] = pred[true_col].apply(normalize_label)
pred[pred_col] = pred[pred_col].apply(normalize_label)

if source_col is not None:
    pred[source_col] = pred[source_col].apply(
        lambda x: str(x).strip()
        if not pd.isna(x)
        else "unknown"
    )

if confidence_col is not None:
    pred["analysis_confidence"] = pred[
        confidence_col
    ].apply(safe_float)

else:
    pred["analysis_confidence"] = np.nan


pred["is_error"] = (
    pred[true_col] != pred[pred_col]
)


pred["error_pair"] = (
    pred[true_col]
    + " -> "
    + pred[pred_col]
)


pred["filename_analysis"] = pred.apply(
    lambda row: get_image_filename(
        row,
        path_col,
        filename_col,
    ),
    axis=1,
)


print(
    "Total predictions:",
    len(pred),
)

print(
    "Correct predictions:",
    int((~pred["is_error"]).sum()),
)

print(
    "Errors:",
    int(pred["is_error"].sum()),
)

print(
    "Accuracy:",
    f"{(~pred['is_error']).mean() * 100:.2f}%",
)


# =============================================================================
# 5. OVERALL ERROR SUMMARY
# =============================================================================

print_header("5. OVERALL ERROR SUMMARY")

total = len(pred)

correct = int((~pred["is_error"]).sum())
errors = int(pred["is_error"].sum())

accuracy = correct / total if total else 0
error_rate = errors / total if total else 0

overall_summary = pd.DataFrame(
    [
        {
            "total_samples": total,
            "correct_predictions": correct,
            "errors": errors,
            "accuracy": accuracy,
            "error_rate": error_rate,
        }
    ]
)

overall_summary.to_csv(
    OUTPUT_DIR / "overall_error_summary.csv",
    index=False,
)

print(
    f"Total samples      : {total}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Errors             : {errors}"
)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Error rate         : {error_rate * 100:.2f}%"
)


# =============================================================================
# 6. ERROR PAIRS
# =============================================================================

print_header("6. ERROR PAIRS")

errors_df = pred[
    pred["is_error"]
].copy()

error_pairs = (
    errors_df
    .groupby(
        [true_col, pred_col],
        dropna=False,
    )
    .size()
    .reset_index(
        name="errors"
    )
    .sort_values(
        "errors",
        ascending=False,
    )
)

error_pairs["percentage_of_errors"] = (
    error_pairs["errors"]
    / max(errors, 1)
    * 100
)

error_pairs["percentage_of_test"] = (
    error_pairs["errors"]
    / max(total, 1)
    * 100
)

error_pairs.to_csv(
    OUTPUT_DIR / "error_pairs.csv",
    index=False,
)

print(
    error_pairs.to_string(
        index=False
    )
)


# =============================================================================
# 7. ERROR BY TRUE CLASS
# =============================================================================

print_header("7. ERROR BY TRUE CLASS")

true_class_total = (
    pred
    .groupby(true_col)
    .size()
    .rename("total_samples")
)

true_class_errors = (
    errors_df
    .groupby(true_col)
    .size()
    .rename("errors")
)

true_class_correct = (
    pred[
        ~pred["is_error"]
    ]
    .groupby(true_col)
    .size()
    .rename("correct")
)

error_by_true = pd.concat(
    [
        true_class_total,
        true_class_correct,
        true_class_errors,
    ],
    axis=1,
).fillna(0)

error_by_true["error_rate"] = (
    error_by_true["errors"]
    / error_by_true["total_samples"]
)

error_by_true["accuracy"] = (
    error_by_true["correct"]
    / error_by_true["total_samples"]
)

error_by_true = (
    error_by_true
    .reset_index()
    .rename(
        columns={
            true_col: "true_label"
        }
    )
)

error_by_true.to_csv(
    OUTPUT_DIR / "error_by_true_class.csv",
    index=False,
)

print(
    error_by_true.to_string(
        index=False
    )
)


# =============================================================================
# 8. ERROR BY PREDICTED CLASS
# =============================================================================

print_header("8. ERROR BY PREDICTED CLASS")

prediction_total = (
    pred
    .groupby(pred_col)
    .size()
    .rename("predicted_total")
)

prediction_errors = (
    errors_df
    .groupby(pred_col)
    .size()
    .rename("errors")
)

error_by_prediction = pd.concat(
    [
        prediction_total,
        prediction_errors,
    ],
    axis=1,
).fillna(0)

error_by_prediction["error_fraction"] = (
    error_by_prediction["errors"]
    / error_by_prediction["predicted_total"]
)

error_by_prediction = (
    error_by_prediction
    .reset_index()
    .rename(
        columns={
            pred_col: "predicted_label"
        }
    )
)

error_by_prediction.to_csv(
    OUTPUT_DIR / "error_by_predicted_class.csv",
    index=False,
)

print(
    error_by_prediction.to_string(
        index=False
    )
)


# =============================================================================
# 9. SOURCE ANALYSIS
# =============================================================================

print_header("9. SOURCE ERROR ANALYSIS")

if source_col is not None:

    source_total = (
        pred
        .groupby(source_col)
        .size()
        .rename("total_samples")
    )

    source_errors = (
        errors_df
        .groupby(source_col)
        .size()
        .rename("errors")
    )

    source_correct = (
        pred[
            ~pred["is_error"]
        ]
        .groupby(source_col)
        .size()
        .rename("correct")
    )

    error_by_source = pd.concat(
        [
            source_total,
            source_correct,
            source_errors,
        ],
        axis=1,
    ).fillna(0)

    error_by_source["error_rate"] = (
        error_by_source["errors"]
        / error_by_source["total_samples"]
    )

    error_by_source["accuracy"] = (
        error_by_source["correct"]
        / error_by_source["total_samples"]
    )

    error_by_source = (
        error_by_source
        .reset_index()
        .rename(
            columns={
                source_col: "source"
            }
        )
    )

    error_by_source.to_csv(
        OUTPUT_DIR / "error_by_source.csv",
        index=False,
    )

    print(
        error_by_source.to_string(
            index=False
        )
    )

else:

    print(
        "Source column not available."
    )


# =============================================================================
# 10. SOURCE × ERROR PAIR
# =============================================================================

print_header("10. SOURCE × ERROR PAIR")

if source_col is not None:

    source_error_pairs = (
        errors_df
        .groupby(
            [
                source_col,
                true_col,
                pred_col,
            ]
        )
        .size()
        .reset_index(
            name="errors"
        )
        .sort_values(
            "errors",
            ascending=False,
        )
    )

    source_error_pairs["error_pair"] = (
        source_error_pairs[true_col]
        + " -> "
        + source_error_pairs[pred_col]
    )

    source_error_pairs.to_csv(
        OUTPUT_DIR
        / "source_error_pairs.csv",
        index=False,
    )

    print(
        source_error_pairs[
            [
                source_col,
                true_col,
                pred_col,
                "errors",
            ]
        ].to_string(
            index=False
        )
    )

else:

    source_error_pairs = pd.DataFrame()


# =============================================================================
# 11. CONFIDENCE ANALYSIS
# =============================================================================

print_header("11. CONFIDENCE ANALYSIS")

if confidence_col is not None:

    valid_conf = pred[
        pred["analysis_confidence"].notna()
    ].copy()

    if len(valid_conf) > 0:

        valid_conf["confidence_bin"] = pd.cut(
            valid_conf["analysis_confidence"],
            bins=[
                0.0,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                0.95,
                1.00,
            ],
            include_lowest=True,
        )

        confidence_summary = (
            valid_conf
            .groupby(
                "confidence_bin",
                observed=False,
            )
            .agg(
                samples=("is_error", "size"),
                errors=("is_error", "sum"),
                accuracy=("is_error", lambda x: 1 - x.mean()),
            )
            .reset_index()
        )

        confidence_summary[
            "error_rate"
        ] = (
            confidence_summary["errors"]
            / confidence_summary["samples"]
        )

        confidence_summary.to_csv(
            OUTPUT_DIR
            / "confidence_analysis.csv",
            index=False,
        )

        print(
            confidence_summary.to_string(
                index=False
            )
        )

        correct_conf = valid_conf[
            ~valid_conf["is_error"]
        ]["analysis_confidence"]

        error_conf = valid_conf[
            valid_conf["is_error"]
        ]["analysis_confidence"]

        print()

        print(
            "Correct confidence mean:",
            f"{correct_conf.mean():.4f}"
        )

        print(
            "Error confidence mean  :",
            f"{error_conf.mean():.4f}"
        )

    else:

        print(
            "Confidence column exists "
            "but contains no valid values."
        )

else:

    print(
        "No confidence column detected."
    )


# =============================================================================
# 12. HIGH-CONFIDENCE ERRORS
# =============================================================================

print_header("12. HIGH-CONFIDENCE ERROR ANALYSIS")

if confidence_col is not None:

    high_conf_errors = errors_df[
        errors_df["analysis_confidence"]
        >= HIGH_CONFIDENCE_THRESHOLD
    ].copy()

    high_conf_errors = (
        high_conf_errors
        .sort_values(
            "analysis_confidence",
            ascending=False,
        )
    )

    high_conf_errors[
        [
            true_col,
            pred_col,
            "analysis_confidence",
            path_col,
        ]
    ].to_csv(
        OUTPUT_DIR
        / "high_confidence_errors.csv",
        index=False,
    )

    print(
        f"Threshold : "
        f"{HIGH_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"High-confidence errors: "
        f"{len(high_conf_errors)}"
    )

    if len(high_conf_errors) > 0:

        print()

        display_columns = [
            true_col,
            pred_col,
            "analysis_confidence",
        ]

        if source_col is not None:
            display_columns.append(
                source_col
            )

        display_columns.append(
            path_col
        )

        print(
            high_conf_errors[
                display_columns
            ]
            .head(MAX_HIGH_CONF_ERRORS)
            .to_string(
                index=False
            )
        )

else:

    high_conf_errors = errors_df.copy()

    print(
        "Confidence unavailable."
    )


# =============================================================================
# 13. RANK ERROR CASES
# =============================================================================

print_header("13. RANKING ERROR CASES")

ranked_errors = errors_df.copy()

if confidence_col is not None:

    ranked_errors[
        "analysis_confidence"
    ] = ranked_errors[
        "analysis_confidence"
    ].fillna(0)

    ranked_errors = (
        ranked_errors
        .sort_values(
            "analysis_confidence",
            ascending=False,
        )
    )

else:

    ranked_errors = (
        ranked_errors
        .sort_values(
            [
                true_col,
                pred_col,
            ]
        )
    )


ranked_errors[
    "error_rank"
] = np.arange(
    1,
    len(ranked_errors) + 1,
)


ranked_errors.to_csv(
    OUTPUT_DIR / "ranked_error_cases.csv",
    index=False,
)

print(
    "Saved:",
    OUTPUT_DIR / "ranked_error_cases.csv"
)


# =============================================================================
# 14. TOP ERROR PAIRS
# =============================================================================

print_header("14. TOP ERROR PAIRS")

top_pairs = error_pairs.head(10)

if len(top_pairs) > 0:

    for _, row in top_pairs.iterrows():

        true_label = row[true_col]
        pred_label = row[pred_col]
        count = int(row["errors"])

        percentage = row[
            "percentage_of_errors"
        ]

        print(
            f"{true_label:10s} -> "
            f"{pred_label:10s} : "
            f"{count:3d} errors "
            f"({percentage:.2f}% of all errors)"
        )

else:

    print(
        "No errors found."
    )


# =============================================================================
# 15. GENERATE VISUAL ERROR SHEETS
# =============================================================================

print_header(
    "15. GENERATING VISUAL ERROR SHEETS"
)

visual_records = []


# -------------------------------------------------------------------------
# A. Error pairs
# -------------------------------------------------------------------------

for _, pair in error_pairs.iterrows():

    true_label = pair[true_col]
    pred_label = pair[pred_col]

    pair_df = errors_df[
        (errors_df[true_col] == true_label)
        & (
            errors_df[pred_col]
            == pred_label
        )
    ].copy()

    if confidence_col is not None:

        pair_df = (
            pair_df
            .sort_values(
                "analysis_confidence",
                ascending=False,
            )
            .head(
                MAX_VISUAL_ERRORS_PER_GROUP
            )
        )

    else:

        pair_df = pair_df.head(
            MAX_VISUAL_ERRORS_PER_GROUP
        )

    pair_safe = (
        f"{true_label}_to_{pred_label}"
        .replace(" ", "_")
        .replace("/", "_")
    )

    pair_dir = (
        OUTPUT_DIR
        / "visual_errors"
        / "error_pairs"
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0

    for idx, row in pair_df.iterrows():

        image = load_image_for_row(
            row,
            path_col,
        )

        if image is None:
            continue

        confidence = row[
            "analysis_confidence"
        ]

        confidence_text = (
            f"{confidence:.3f}"
            if not pd.isna(confidence)
            else "N/A"
        )

        filename = (
            f"{saved + 1:02d}_"
            f"{row['filename_analysis']}"
        )

        output_image = create_text_image(
            image,
            [
                f"True: {true_label}",
                f"Predicted: {pred_label}",
                f"Confidence: {confidence_text}",
                f"File: {row['filename_analysis']}",
            ],
        )

        output_path = (
            pair_dir / filename
        )

        try:
            output_image.save(
                output_path,
                quality=95,
            )

            saved += 1

            visual_records.append(
                {
                    "type": "error_pair",
                    "group": (
                        f"{true_label} -> "
                        f"{pred_label}"
                    ),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "filename": row[
                        "filename_analysis"
                    ],
                    "image_path": str(
                        row[path_col]
                    ),
                    "visual_path": str(
                        output_path
                    ),
                }
            )

        except Exception as exc:

            print(
                "Could not save image:",
                exc,
            )

    print(
        f"{true_label:10s} -> "
        f"{pred_label:10s}: "
        f"{saved} visual cases saved"
    )


# -------------------------------------------------------------------------
# B. High-confidence errors
# -------------------------------------------------------------------------

if len(high_conf_errors) > 0:

    high_conf_dir = (
        OUTPUT_DIR
        / "visual_errors"
        / "high_confidence"
    )

    high_conf_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    high_conf_to_visualize = (
        high_conf_errors
        .head(MAX_HIGH_CONF_ERRORS)
    )

    saved = 0

    for _, row in high_conf_to_visualize.iterrows():

        image = load_image_for_row(
            row,
            path_col,
        )

        if image is None:
            continue

        confidence = row[
            "analysis_confidence"
        ]

        true_label = row[true_col]
        pred_label = row[pred_col]

        filename = (
            f"{saved + 1:02d}_"
            f"{row['filename_analysis']}"
        )

        output_image = create_text_image(
            image,
            [
                f"True: {true_label}",
                f"Predicted: {pred_label}",
                f"Confidence: {confidence:.3f}",
                f"File: {row['filename_analysis']}",
            ],
        )

        output_path = (
            high_conf_dir / filename
        )

        try:

            output_image.save(
                output_path,
                quality=95,
            )

            saved += 1

            visual_records.append(
                {
                    "type": "high_confidence",
                    "group": "high_confidence",
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "filename": row[
                        "filename_analysis"
                    ],
                    "image_path": str(
                        row[path_col]
                    ),
                    "visual_path": str(
                        output_path
                    ),
                }
            )

        except Exception as exc:

            print(
                "Could not save image:",
                exc,
            )

    print(
        f"High-confidence visual cases saved: "
        f"{saved}"
    )


visual_records_df = pd.DataFrame(
    visual_records
)

visual_records_df.to_csv(
    OUTPUT_DIR
    / "visual_error_index.csv",
    index=False,
)


# =============================================================================
# 16. CONFUSION MATRIX
# =============================================================================

print_header(
    "16. GENERATING CONFUSION MATRIX"
)

confusion = pd.crosstab(
    pred[true_col],
    pred[pred_col],
)

confusion = confusion.reindex(
    index=CLASS_ORDER,
    columns=CLASS_ORDER,
    fill_value=0,
)

confusion.to_csv(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)

print(confusion)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
)

plt.title(
    "CircuLens Quality V5 — Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()

confusion_plot_path = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    confusion_plot_path
)


# =============================================================================
# 17. ERROR PAIR BAR CHART
# =============================================================================

print_header(
    "17. GENERATING ERROR PAIR CHART"
)

if len(error_pairs) > 0:

    chart_df = error_pairs.copy()

    chart_df["pair"] = (
        chart_df[true_col]
        + " → "
        + chart_df[pred_col]
    )

    chart_df = (
        chart_df
        .head(10)
        .sort_values(
            "errors",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        chart_df["pair"],
        chart_df["errors"],
    )

    plt.xlabel(
        "Number of errors"
    )

    plt.ylabel(
        "Error pair"
    )

    plt.title(
        "CircuLens Quality V5 — Top Error Pairs"
    )

    plt.tight_layout()

    error_chart_path = (
        OUTPUT_DIR
        / "top_error_pairs.png"
    )

    plt.savefig(
        error_chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        error_chart_path
    )


# =============================================================================
# 18. SOURCE ERROR CHART
# =============================================================================

print_header(
    "18. GENERATING SOURCE ERROR CHART"
)

if source_col is not None and len(error_by_source) > 0:

    chart_df = error_by_source.copy()

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        chart_df["source"],
        chart_df["error_rate"] * 100,
    )

    plt.ylabel(
        "Error rate (%)"
    )

    plt.xlabel(
        "Dataset source"
    )

    plt.title(
        "CircuLens Quality V5 — Error Rate by Source"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    source_chart_path = (
        OUTPUT_DIR
        / "error_rate_by_source.png"
    )

    plt.savefig(
        source_chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        source_chart_path
    )


# =============================================================================
# 19. CLASS ERROR RATE CHART
# =============================================================================

print_header(
    "19. GENERATING CLASS ERROR CHART"
)

if len(error_by_true) > 0:

    chart_df = (
        error_by_true
        .sort_values(
            "error_rate",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        chart_df["true_label"],
        chart_df["error_rate"] * 100,
    )

    plt.xlabel(
        "Error rate (%)"
    )

    plt.ylabel(
        "True class"
    )

    plt.title(
        "CircuLens Quality V5 — Error Rate by True Class"
    )

    plt.tight_layout()

    class_chart_path = (
        OUTPUT_DIR
        / "error_rate_by_class.png"
    )

    plt.savefig(
        class_chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        class_chart_path
    )


# =============================================================================
# 20. HIGH-CONFIDENCE ERROR PAIRS
# =============================================================================

print_header(
    "20. HIGH-CONFIDENCE ERROR PAIRS"
)

if len(high_conf_errors) > 0:

    high_conf_pairs = (
        high_conf_errors
        .groupby(
            [
                true_col,
                pred_col,
            ]
        )
        .size()
        .reset_index(
            name="high_confidence_errors"
        )
        .sort_values(
            "high_confidence_errors",
            ascending=False,
        )
    )

    high_conf_pairs[
        "error_pair"
    ] = (
        high_conf_pairs[true_col]
        + " -> "
        + high_conf_pairs[pred_col]
    )

    high_conf_pairs.to_csv(
        OUTPUT_DIR
        / "high_confidence_error_pairs.csv",
        index=False,
    )

    print(
        high_conf_pairs.to_string(
            index=False
        )
    )

else:

    high_conf_pairs = pd.DataFrame()

    print(
        "No high-confidence errors."
    )


# =============================================================================
# 21. ROOT-CAUSE HEURISTIC
# =============================================================================

print_header(
    "21. PRELIMINARY ROOT-CAUSE ANALYSIS"
)

root_causes = []


# -------------------------------------------------------------------------
# Dominant error pair
# -------------------------------------------------------------------------

if len(error_pairs) > 0:

    dominant_pair = error_pairs.iloc[0]

    dominant_true = dominant_pair[
        true_col
    ]

    dominant_pred = dominant_pair[
        pred_col
    ]

    dominant_count = int(
        dominant_pair["errors"]
    )

    root_causes.append(
        {
            "priority": 1,
            "finding": "Dominant error pair",
            "detail": (
                f"{dominant_true} -> "
                f"{dominant_pred} "
                f"({dominant_count} errors)"
            ),
        }
    )


# -------------------------------------------------------------------------
# Class with highest error rate
# -------------------------------------------------------------------------

if len(error_by_true) > 0:

    worst_class = (
        error_by_true
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .iloc[0]
    )

    root_causes.append(
        {
            "priority": 2,
            "finding": "Worst true class",
            "detail": (
                f"{worst_class['true_label']} "
                f"has "
                f"{worst_class['error_rate'] * 100:.2f}% "
                f"error rate "
                f"({int(worst_class['errors'])} errors "
                f"out of "
                f"{int(worst_class['total_samples'])})"
            ),
        }
    )


# -------------------------------------------------------------------------
# Source concentration
# -------------------------------------------------------------------------

if (
    source_col is not None
    and len(error_by_source) > 0
):

    worst_source = (
        error_by_source
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .iloc[0]
    )

    root_causes.append(
        {
            "priority": 3,
            "finding": "Highest source error rate",
            "detail": (
                f"{worst_source['source']} "
                f"has "
                f"{worst_source['error_rate'] * 100:.2f}% "
                f"error rate"
            ),
        }
    )


# -------------------------------------------------------------------------
# High confidence errors
# -------------------------------------------------------------------------

if confidence_col is not None:

    hc_count = len(
        high_conf_errors
    )

    root_causes.append(
        {
            "priority": 4,
            "finding": "High-confidence errors",
            "detail": (
                f"{hc_count} errors have "
                f"confidence >= "
                f"{HIGH_CONFIDENCE_THRESHOLD:.2f}"
            ),
        }
    )


root_causes_df = pd.DataFrame(
    root_causes
)

root_causes_df.to_csv(
    OUTPUT_DIR
    / "preliminary_root_causes.csv",
    index=False,
)

print(
    root_causes_df.to_string(
        index=False
    )
)


# =============================================================================
# 22. AUTOMATIC DIAGNOSTIC SUMMARY
# =============================================================================

print_header(
    "22. GENERATING DIAGNOSTIC SUMMARY"
)

summary_lines = []

summary_lines.append(
    "CIRCULENS QUALITY — V5 ERROR ANALYSIS"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append(
    ""
)

summary_lines.append(
    f"Total test samples : {total}"
)

summary_lines.append(
    f"Correct predictions: {correct}"
)

summary_lines.append(
    f"Errors             : {errors}"
)

summary_lines.append(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

summary_lines.append(
    f"Error rate         : {error_rate * 100:.2f}%"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "TOP ERROR PAIRS"
)

summary_lines.append(
    "-" * 70
)


for _, row in error_pairs.head(5).iterrows():

    pair = (
        f"{row[true_col]} -> "
        f"{row[pred_col]}"
    )

    summary_lines.append(
        f"{pair:25s}: "
        f"{int(row['errors'])} errors "
        f"({row['percentage_of_errors']:.2f}% of errors)"
    )


summary_lines.append(
    ""
)

summary_lines.append(
    "ERROR RATE BY TRUE CLASS"
)

summary_lines.append(
    "-" * 70
)


for _, row in (
    error_by_true
    .sort_values(
        "error_rate",
        ascending=False,
    )
    .iterrows()
):

    summary_lines.append(
        f"{row['true_label']:12s}: "
        f"{row['error_rate'] * 100:.2f}% "
        f"({int(row['errors'])}/"
        f"{int(row['total_samples'])})"
    )


if source_col is not None:

    summary_lines.append(
        ""
    )

    summary_lines.append(
        "ERROR RATE BY SOURCE"
    )

    summary_lines.append(
        "-" * 70
    )

    for _, row in (
        error_by_source
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .iterrows()
    ):

        summary_lines.append(
            f"{row['source']:30s}: "
            f"{row['error_rate'] * 100:.2f}% "
            f"({int(row['errors'])}/"
            f"{int(row['total_samples'])})"
        )


summary_lines.append(
    ""
)

summary_lines.append(
    "HIGH-CONFIDENCE ERRORS"
)

summary_lines.append(
    "-" * 70
)

summary_lines.append(
    f"Threshold: "
    f"{HIGH_CONFIDENCE_THRESHOLD:.2f}"
)

summary_lines.append(
    f"Count   : "
    f"{len(high_conf_errors)}"
)


summary_lines.append(
    ""
)

summary_lines.append(
    "INTERPRETATION"
)

summary_lines.append(
    "-" * 70
)


# -------------------------------------------------------------------------
# Interpretation 1
# -------------------------------------------------------------------------

if errors > 0:

    if len(error_pairs) > 0:

        dominant = error_pairs.iloc[0]

        dominant_count = int(
            dominant["errors"]
        )

        dominant_percentage = (
            dominant[
                "percentage_of_errors"
            ]
        )

        if dominant_percentage >= 30:

            summary_lines.append(
                "A single error pair dominates "
                "the model's mistakes."
            )

            summary_lines.append(
                "This suggests that targeted "
                "investigation of this class boundary "
                "should be prioritized before changing "
                "the architecture."
            )

        elif dominant_percentage >= 20:

            summary_lines.append(
                "One error pair is a major contributor "
                "to total errors."
            )

        else:

            summary_lines.append(
                "Errors are distributed across multiple "
                "class boundaries."
            )


# -------------------------------------------------------------------------
# Interpretation 2
# -------------------------------------------------------------------------

if len(high_conf_errors) > 0:

    hc_ratio = (
        len(high_conf_errors)
        / max(errors, 1)
    )

    if hc_ratio >= 0.30:

        summary_lines.append(
            "A substantial portion of errors are "
            "high-confidence predictions."
        )

        summary_lines.append(
            "These cases should be visually inspected "
            "because they may indicate difficult visual "
            "boundaries, label ambiguity, or systematic "
            "dataset characteristics."
        )

    else:

        summary_lines.append(
            "Most errors are not extremely high-confidence, "
            "suggesting that uncertainty contributes to "
            "part of the remaining error."
        )


# -------------------------------------------------------------------------
# Interpretation 3
# -------------------------------------------------------------------------

if len(error_by_true) > 0:

    worst = (
        error_by_true
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .iloc[0]
    )

    summary_lines.append(
        f"The weakest true class is "
        f"'{worst['true_label']}' with an "
        f"error rate of "
        f"{worst['error_rate'] * 100:.2f}%."
    )


# -------------------------------------------------------------------------
# Interpretation 4
# -------------------------------------------------------------------------

if (
    source_col is not None
    and len(error_by_source) > 1
):

    max_source = (
        error_by_source
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .iloc[0]
    )

    min_source = (
        error_by_source
        .sort_values(
            "error_rate",
            ascending=True,
        )
        .iloc[0]
    )

    source_gap = (
        max_source["error_rate"]
        - min_source["error_rate"]
    )

    summary_lines.append(
        f"Source error-rate gap: "
        f"{source_gap * 100:.2f} percentage points."
    )

    if source_gap >= 0.10:

        summary_lines.append(
            "The source difference is large enough "
            "to justify checking for domain/source bias."
        )

    else:

        summary_lines.append(
            "The source error-rate difference is "
            "relatively small."
        )


summary_lines.append(
    ""
)

summary_lines.append(
    "RECOMMENDED NEXT STEP"
)

summary_lines.append(
    "-" * 70
)

summary_lines.append(
    "Do NOT immediately modify the model."
)

summary_lines.append(
    "First inspect the generated visual error cases, "
    "especially the dominant error pairs and "
    "high-confidence errors."
)

summary_lines.append(
    "After visual inspection, determine whether the "
    "remaining errors are primarily caused by visual "
    "similarity, label ambiguity/noise, source/domain "
    "differences, or insufficient representation."
)

summary_lines.append(
    "Only then design the V6 training strategy."
)


summary_text = "\n".join(
    summary_lines
)

summary_path = (
    OUTPUT_DIR
    / "diagnostic_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        summary_text
    )


print(
    summary_text
)


# =============================================================================
# 23. SAVE FULL ANALYSIS DATA
# =============================================================================

print_header(
    "23. SAVING FULL ANALYSIS DATA"
)

analysis_columns = [
    true_col,
    pred_col,
    path_col,
    "filename_analysis",
    "is_error",
    "error_pair",
    "analysis_confidence",
]

if source_col is not None:
    analysis_columns.insert(
        2,
        source_col,
    )

analysis_df = pred[
    analysis_columns
].copy()

analysis_df.to_csv(
    OUTPUT_DIR
    / "full_v5_error_analysis.csv",
    index=False,
)

print(
    "Saved:",
    OUTPUT_DIR
    / "full_v5_error_analysis.csv"
)


# =============================================================================
# 24. FINAL OUTPUT LIST
# =============================================================================

print_header(
    "24. ANALYSIS COMPLETED"
)

generated_files = [
    "overall_error_summary.csv",
    "error_pairs.csv",
    "error_by_true_class.csv",
    "error_by_predicted_class.csv",
    "error_by_source.csv",
    "source_error_pairs.csv",
    "confidence_analysis.csv",
    "high_confidence_errors.csv",
    "high_confidence_error_pairs.csv",
    "ranked_error_cases.csv",
    "visual_error_index.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    "top_error_pairs.png",
    "error_rate_by_source.png",
    "error_rate_by_class.png",
    "preliminary_root_causes.csv",
    "diagnostic_summary.txt",
    "full_v5_error_analysis.csv",
]

for filename in generated_files:

    path = OUTPUT_DIR / filename

    if path.exists():

        print(
            f"  {filename}"
        )


print()
print(
    "Visual error directories:"
)

print(
    f"  {OUTPUT_DIR / 'visual_errors' / 'error_pairs'}"
)

print(
    f"  {OUTPUT_DIR / 'visual_errors' / 'high_confidence'}"
)


print()
print(
    "=" * 90
)

print(
    "CIRCULENS QUALITY — V5 ERROR ANALYSIS COMPLETED"
)

print(
    "=" * 90
)

print()
print(
    "IMPORTANT:"
)

print(
    "Review the generated visual error cases before "
    "designing V6."
)

print()
print(
    "Primary files to inspect:"
)

print(
    f"1. {OUTPUT_DIR / 'diagnostic_summary.txt'}"
)

print(
    f"2. {OUTPUT_DIR / 'error_pairs.csv'}"
)

print(
    f"3. {OUTPUT_DIR / 'high_confidence_error_pairs.csv'}"
)

print(
    f"4. {OUTPUT_DIR / 'visual_errors'}"
)

print(
    f"5. {OUTPUT_DIR / 'confusion_matrix.png'}"
)

print(
    f"6. {OUTPUT_DIR / 'top_error_pairs.png'}"
)

print()
print(
    "Do NOT train V6 until the dominant visual errors "
    "have been inspected."
)

print()
