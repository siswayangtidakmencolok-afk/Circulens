import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

FINETUNE_DIR = "reports/quality_finetune_v1"
BASELINE_DIR = "reports/quality_baseline_v4_class_weighted"

FINETUNE_PRED = os.path.join(
    FINETUNE_DIR,
    "test_predictions.csv"
)

BASELINE_PRED = os.path.join(
    BASELINE_DIR,
    "test_predictions.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("CIRCULENS QUALITY — FINE-TUNE V1 ERROR ANALYSIS")
print("=" * 90)

# ============================================================
# LOAD
# ============================================================

if not os.path.exists(FINETUNE_PRED):
    raise FileNotFoundError(
        f"Fine-tune predictions not found: {FINETUNE_PRED}"
    )

ft = pd.read_csv(FINETUNE_PRED)

print()
print("Fine-tune predictions:")
print(FINETUNE_PRED)

print()
print("Rows:", len(ft))

# ============================================================
# BASIC METRICS
# ============================================================

print()
print("=" * 90)
print("1. BASIC METRICS")
print("=" * 90)

accuracy = ft["correct"].mean()

errors = ft[ft["correct"] == False].copy()

print()
print(f"Total test samples : {len(ft)}")
print(f"Correct            : {ft['correct'].sum()}")
print(f"Errors             : {len(errors)}")
print(f"Accuracy           : {accuracy * 100:.2f}%")

# ============================================================
# ERROR PAIRS
# ============================================================

print()
print("=" * 90)
print("2. ERROR PAIRS")
print("=" * 90)

error_pairs = (
    errors
    .groupby(
        ["label", "prediction_label"]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print()
print(
    error_pairs.to_string()
)

error_pairs_df = (
    error_pairs
    .reset_index(
        name="errors"
    )
)

error_pairs_df.to_csv(
    os.path.join(
        FINETUNE_DIR,
        "error_pairs_analysis.csv"
    ),
    index=False
)

# ============================================================
# ERROR BY SOURCE
# ============================================================

print()
print("=" * 90)
print("3. ERROR BY SOURCE")
print("=" * 90)

if "source" in ft.columns:

    source_summary = (
        ft
        .groupby("source")
        .apply(
            lambda x: pd.Series({
                "samples": len(x),
                "accuracy": x["correct"].mean(),
                "errors": (~x["correct"]).sum(),
                "mean_confidence": x["confidence"].mean()
            }),
            include_groups=False
        )
    )

    print()
    print(
        source_summary.to_string()
    )

    source_summary.to_csv(
        os.path.join(
            FINETUNE_DIR,
            "error_by_source_analysis.csv"
        )
    )

# ============================================================
# ERROR BY TRUE CLASS
# ============================================================

print()
print("=" * 90)
print("4. ERROR BY TRUE CLASS")
print("=" * 90)

class_summary = (
    ft
    .groupby("label")
    .apply(
        lambda x: pd.Series({
            "samples": len(x),
            "accuracy": x["correct"].mean(),
            "errors": (~x["correct"]).sum(),
            "mean_confidence": x["confidence"].mean()
        }),
        include_groups=False
    )
    .sort_values(
        "accuracy"
    )
)

print()
print(
    class_summary.to_string()
)

class_summary.to_csv(
    os.path.join(
        FINETUNE_DIR,
        "error_by_class_analysis.csv"
    )
)

# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print()
print("=" * 90)
print("5. CONFIDENCE ANALYSIS")
print("=" * 90)

print()
print("Correct predictions:")
print(
    ft.loc[
        ft["correct"] == True,
        "confidence"
    ].describe()
)

print()
print("Incorrect predictions:")
print(
    ft.loc[
        ft["correct"] == False,
        "confidence"
    ].describe()
)

# ============================================================
# HIGH CONFIDENCE ERRORS
# ============================================================

print()
print("=" * 90)
print("6. HIGH-CONFIDENCE ERRORS")
print("=" * 90)

for threshold in [0.70, 0.80, 0.90, 0.95]:

    subset = errors[
        errors["confidence"] >= threshold
    ]

    print()
    print(
        f"Confidence >= {threshold:.2f}: "
        f"{len(subset)} errors"
    )

    if len(subset) > 0:

        print(
            subset[
                [
                    "label",
                    "prediction_label",
                    "confidence"
                ]
            ]
            .sort_values(
                "confidence",
                ascending=False
            )
            .head(20)
            .to_string(
                index=False
            )
        )

# ============================================================
# TOP PROBLEM PAIRS
# ============================================================

print()
print("=" * 90)
print("7. TOP PROBLEM PAIRS")
print("=" * 90)

top_pairs = (
    errors
    .groupby(
        [
            "label",
            "prediction_label"
        ]
    )
    .agg(
        errors=("correct", "size"),
        mean_confidence=("confidence", "mean"),
        max_confidence=("confidence", "max")
    )
    .sort_values(
        "errors",
        ascending=False
    )
)

print()
print(
    top_pairs.head(15).to_string()
)

# ============================================================
# ERROR PAIRS BY SOURCE
# ============================================================

print()
print("=" * 90)
print("8. ERROR PAIRS × SOURCE")
print("=" * 90)

if "source" in errors.columns:

    pair_source = (
        errors
        .groupby(
            [
                "label",
                "prediction_label",
                "source"
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        pair_source.to_string()
    )

    pair_source_df = (
        pair_source
        .reset_index(
            name="errors"
        )
    )

    pair_source_df.to_csv(
        os.path.join(
            FINETUNE_DIR,
            "error_pairs_by_source_analysis.csv"
        ),
        index=False
    )

# ============================================================
# BASELINE COMPARISON
# ============================================================

print()
print("=" * 90)
print("9. BASELINE VS FINE-TUNE V1")
print("=" * 90)

if os.path.exists(BASELINE_PRED):

    baseline = pd.read_csv(
        BASELINE_PRED
    )

    baseline_accuracy = (
        baseline["correct"].mean()
    )

    finetune_accuracy = (
        ft["correct"].mean()
    )

    print()
    print(
        f"Baseline accuracy : "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"Fine-tune accuracy: "
        f"{finetune_accuracy * 100:.2f}%"
    )

    print(
        f"Improvement       : "
        f"{(finetune_accuracy - baseline_accuracy) * 100:+.2f} pp"
    )

    # --------------------------------------------------------
    # SOURCE COMPARISON
    # --------------------------------------------------------

    if (
        "source" in baseline.columns
        and
        "source" in ft.columns
    ):

        baseline_source = (
            baseline
            .groupby("source")["correct"]
            .mean()
        )

        finetune_source = (
            ft
            .groupby("source")["correct"]
            .mean()
        )

        source_compare = pd.DataFrame({
            "baseline_accuracy":
                baseline_source,

            "finetune_accuracy":
                finetune_source
        })

        source_compare["improvement_pp"] = (
            (
                source_compare[
                    "finetune_accuracy"
                ]
                -
                source_compare[
                    "baseline_accuracy"
                ]
            )
            * 100
        )

        print()
        print("SOURCE COMPARISON:")
        print(
            source_compare.to_string()
        )

        source_compare.to_csv(
            os.path.join(
                FINETUNE_DIR,
                "baseline_vs_finetune_source.csv"
            )
        )

    # --------------------------------------------------------
    # CLASS COMPARISON
    # --------------------------------------------------------

    baseline_class = (
        baseline
        .groupby("label")["correct"]
        .mean()
    )

    finetune_class = (
        ft
        .groupby("label")["correct"]
        .mean()
    )

    class_compare = pd.DataFrame({
        "baseline_accuracy":
            baseline_class,

        "finetune_accuracy":
            finetune_class
    })

    class_compare["improvement_pp"] = (
        (
            class_compare[
                "finetune_accuracy"
            ]
            -
            class_compare[
                "baseline_accuracy"
            ]
        )
        * 100
    )

    print()
    print("CLASS COMPARISON:")
    print(
        class_compare.to_string()
    )

    class_compare.to_csv(
        os.path.join(
            FINETUNE_DIR,
            "baseline_vs_finetune_class.csv"
        )
    )

else:

    print()
    print(
        "Baseline predictions not found."
    )

# ============================================================
# MOST UNCERTAIN CORRECT PREDICTIONS
# ============================================================

print()
print("=" * 90)
print("10. LOW-CONFIDENCE CORRECT PREDICTIONS")
print("=" * 90)

low_conf_correct = (
    ft[
        ft["correct"] == True
    ]
    .sort_values(
        "confidence"
    )
    .head(20)
)

print()

print(
    low_conf_correct[
        [
            "label",
            "prediction_label",
            "confidence"
        ]
    ].to_string(
        index=False
    )
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 90)
print("ANALYSIS COMPLETED")
print("=" * 90)

print()
print("Generated analysis files:")

files = [
    "error_pairs_analysis.csv",
    "error_by_source_analysis.csv",
    "error_by_class_analysis.csv",
    "error_pairs_by_source_analysis.csv",
    "baseline_vs_finetune_source.csv",
    "baseline_vs_finetune_class.csv"
]

for filename in files:

    path = os.path.join(
        FINETUNE_DIR,
        filename
    )

    if os.path.exists(path):
        print(" ", filename)

print()
print(
    "Next step: inspect the printed ERROR PAIRS "
    "and BASELINE VS FINE-TUNE tables."
)

print("=" * 90)