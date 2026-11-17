# audit_quality_labels.py
#
# CircuLens Quality Dataset Audit
# Purpose:
#   Audit V5/V6 predictions to identify:
#   1. label x source imbalance
#   2. source-specific error patterns
#   3. ripe / old confusion
#   4. sequence/group consistency
#   5. confidence behavior
#   6. suspicious / ambiguous samples
#   7. samples requiring manual review
#
# Input:
#   reports/quality_finetune_v5/test_predictions.csv
#   reports/quality_finetune_v6/test_predictions.csv
#
# Output:
#   reports/quality_finetune_v6/v5_v6_error_analysis/label_audit/
#
# Run:
#   py audit_quality_labels.py

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

V5_PATH = Path(
    "reports/quality_finetune_v5/test_predictions.csv"
)

V6_PATH = Path(
    "reports/quality_finetune_v6/test_predictions.csv"
)

OUTPUT_DIR = Path(
    "reports/quality_finetune_v6/v5_v6_error_analysis/label_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "damaged",
    "dried",
    "old",
    "ripe",
    "unripe",
]

FOCUS_CLASSES = [
    "ripe",
    "old",
]

HIGH_CONF_THRESHOLD = 0.80
VERY_HIGH_CONF_THRESHOLD = 0.90

REQUIRED_COLUMNS = [
    "path",
    "filename",
    "label",
    "source",
    "sequence_group",
    "timestamp",
    "prediction_label",
    "prediction_confidence",
]


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0


def normalize_label(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()


def load_predictions(path, version):
    if not path.exists():
        raise FileNotFoundError(
            f"{version} prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{version} is missing required columns: {missing}"
        )

    df["label"] = df["label"].map(normalize_label)
    df["prediction_label"] = df["prediction_label"].map(normalize_label)

    df["prediction_confidence"] = pd.to_numeric(
        df["prediction_confidence"],
        errors="coerce"
    )

    df["correct"] = (
        df["label"] == df["prediction_label"]
    )

    df["high_conf_error"] = (
        (~df["correct"])
        & (df["prediction_confidence"] >= HIGH_CONF_THRESHOLD)
    )

    df["very_high_conf_error"] = (
        (~df["correct"])
        & (df["prediction_confidence"] >= VERY_HIGH_CONF_THRESHOLD)
    )

    df["version"] = version

    return df


# =============================================================================
# LOAD
# =============================================================================

section("CIRCULENS QUALITY — DATASET LABEL AUDIT")

print("V5:")
print(V5_PATH)

print()
print("V6:")
print(V6_PATH)

print()
print("Output:")
print(OUTPUT_DIR)

section("1. LOADING V5 / V6")

v5 = load_predictions(V5_PATH, "v5")
v6 = load_predictions(V6_PATH, "v6")

print(f"V5 rows: {len(v5)}")
print(f"V6 rows: {len(v6)}")

if len(v5) != len(v6):
    print("WARNING: V5 and V6 row counts differ.")


# =============================================================================
# MATCH
# =============================================================================

section("2. MATCHING V5 / V6")

v5_match = v5[
    [
        "path",
        "filename",
        "label",
        "source",
        "sequence_group",
        "timestamp",
        "prediction_label",
        "prediction_confidence",
        "correct",
        "high_conf_error",
        "very_high_conf_error",
    ]
].copy()

v6_match = v6[
    [
        "path",
        "filename",
        "label",
        "source",
        "sequence_group",
        "timestamp",
        "prediction_label",
        "prediction_confidence",
        "correct",
        "high_conf_error",
        "very_high_conf_error",
    ]
].copy()

merged = v5_match.merge(
    v6_match,
    on="path",
    how="inner",
    suffixes=("_v5", "_v6"),
)

print(f"Matched images: {len(merged)}")

save_csv(
    merged,
    "audit_master_v5_v6.csv"
)


# =============================================================================
# 3. LABEL DISTRIBUTION
# =============================================================================

section("3. OVERALL LABEL DISTRIBUTION")

label_distribution = (
    merged.groupby("label_v5")
    .size()
    .reset_index(name="samples")
)

label_distribution["percentage"] = (
    label_distribution["samples"]
    / len(merged)
    * 100
)

label_distribution = label_distribution.rename(
    columns={"label_v5": "label"}
)

label_distribution["expected_class"] = label_distribution["label"].isin(
    CLASSES
)

print(label_distribution.to_string(index=False))

save_csv(
    label_distribution,
    "01_overall_label_distribution.csv"
)


# =============================================================================
# 4. LABEL × SOURCE
# =============================================================================

section("4. LABEL × SOURCE DISTRIBUTION")

label_source = (
    merged.groupby(
        ["source_v5", "label_v5"]
    )
    .size()
    .reset_index(name="samples")
)

source_totals = (
    label_source.groupby("source_v5")["samples"]
    .transform("sum")
)

label_source["percentage_within_source"] = (
    label_source["samples"] / source_totals * 100
)

label_source = label_source.rename(
    columns={
        "source_v5": "source",
        "label_v5": "label",
    }
)

print(label_source.to_string(index=False))

save_csv(
    label_source,
    "02_label_by_source.csv"
)


# =============================================================================
# 5. SOURCE × CLASS × V5/V6 PERFORMANCE
# =============================================================================

section("5. SOURCE × CLASS PERFORMANCE")

source_class_rows = []

for source in sorted(merged["source_v5"].dropna().unique()):

    for label in CLASSES:

        subset = merged[
            (merged["source_v5"] == source)
            & (merged["label_v5"] == label)
        ]

        support = len(subset)

        if support == 0:
            continue

        v5_correct = int(subset["correct_v5"].sum())
        v6_correct = int(subset["correct_v6"].sum())

        source_class_rows.append({
            "source": source,
            "label": label,
            "support": support,
            "v5_correct": v5_correct,
            "v5_recall": pct(v5_correct, support),
            "v6_correct": v6_correct,
            "v6_recall": pct(v6_correct, support),
            "recall_delta": pct(v6_correct, support)
            - pct(v5_correct, support),
        })

source_class = pd.DataFrame(source_class_rows)

print(source_class.to_string(index=False))

save_csv(
    source_class,
    "03_source_class_performance.csv"
)


# =============================================================================
# 6. RIPE × SOURCE
# =============================================================================

section("6. RIPE ANALYSIS BY SOURCE")

ripe = merged[
    merged["label_v5"] == "ripe"
].copy()

ripe_source_rows = []

for source in sorted(ripe["source_v5"].dropna().unique()):

    subset = ripe[
        ripe["source_v5"] == source
    ]

    support = len(subset)

    v5_correct = int(
        subset["correct_v5"].sum()
    )

    v6_correct = int(
        subset["correct_v6"].sum()
    )

    ripe_source_rows.append({
        "source": source,
        "support": support,
        "v5_correct": v5_correct,
        "v5_recall": pct(v5_correct, support),
        "v6_correct": v6_correct,
        "v6_recall": pct(v6_correct, support),
        "v5_to_v6_lost": int(
            (
                subset["correct_v5"]
                & ~subset["correct_v6"]
            ).sum()
        ),
        "v5_to_v6_gained": int(
            (
                ~subset["correct_v5"]
                & subset["correct_v6"]
            ).sum()
        ),
    })

ripe_source = pd.DataFrame(
    ripe_source_rows
)

print(ripe_source.to_string(index=False))

save_csv(
    ripe_source,
    "04_ripe_by_source.csv"
)


# =============================================================================
# 7. OLD × SOURCE
# =============================================================================

section("7. OLD ANALYSIS BY SOURCE")

old = merged[
    merged["label_v5"] == "old"
].copy()

old_source_rows = []

for source in sorted(old["source_v5"].dropna().unique()):

    subset = old[
        old["source_v5"] == source
    ]

    support = len(subset)

    v5_correct = int(
        subset["correct_v5"].sum()
    )

    v6_correct = int(
        subset["correct_v6"].sum()
    )

    old_source_rows.append({
        "source": source,
        "support": support,
        "v5_correct": v5_correct,
        "v5_recall": pct(v5_correct, support),
        "v6_correct": v6_correct,
        "v6_recall": pct(v6_correct, support),
        "v5_to_v6_lost": int(
            (
                subset["correct_v5"]
                & ~subset["correct_v6"]
            ).sum()
        ),
        "v5_to_v6_gained": int(
            (
                ~subset["correct_v5"]
                & subset["correct_v6"]
            ).sum()
        ),
    })

old_source = pd.DataFrame(
    old_source_rows
)

print(old_source.to_string(index=False))

save_csv(
    old_source,
    "05_old_by_source.csv"
)


# =============================================================================
# 8. CONFUSION ANALYSIS
# =============================================================================

section("8. V5 / V6 CONFUSION ANALYSIS")

def confusion_table(df, version):
    table = (
        pd.crosstab(
            df[f"label_{version}"],
            df[f"prediction_label_{version}"],
            margins=False
        )
        .reset_index()
        .rename(
            columns={
                f"label_{version}": "true_label"
            }
        )
    )

    return table


v5_confusion = confusion_table(
    merged,
    "v5"
)

v6_confusion = confusion_table(
    merged,
    "v6"
)

save_csv(
    v5_confusion,
    "06_v5_confusion_matrix.csv"
)

save_csv(
    v6_confusion,
    "07_v6_confusion_matrix.csv"
)


# =============================================================================
# 9. RIPE ERROR TYPES
# =============================================================================

section("9. RIPE ERROR PATTERNS")

ripe_errors = ripe[
    ~ripe["correct_v6"]
].copy()

ripe_error_pairs = (
    ripe_errors.groupby(
        [
            "source_v5",
            "prediction_label_v5",
            "prediction_label_v6",
        ]
    )
    .size()
    .reset_index(name="samples")
    .sort_values(
        "samples",
        ascending=False
    )
)

ripe_error_pairs = ripe_error_pairs.rename(
    columns={
        "source_v5": "source",
        "prediction_label_v5": "v5_prediction",
        "prediction_label_v6": "v6_prediction",
    }
)

print(ripe_error_pairs.to_string(index=False))

save_csv(
    ripe_error_pairs,
    "08_ripe_error_patterns.csv"
)


# =============================================================================
# 10. OLD ERROR TYPES
# =============================================================================

section("10. OLD ERROR PATTERNS")

old_errors = old[
    ~old["correct_v6"]
].copy()

old_error_pairs = (
    old_errors.groupby(
        [
            "source_v5",
            "prediction_label_v5",
            "prediction_label_v6",
        ]
    )
    .size()
    .reset_index(name="samples")
    .sort_values(
        "samples",
        ascending=False
    )
)

old_error_pairs = old_error_pairs.rename(
    columns={
        "source_v5": "source",
        "prediction_label_v5": "v5_prediction",
        "prediction_label_v6": "v6_prediction",
    }
)

print(old_error_pairs.to_string(index=False))

save_csv(
    old_error_pairs,
    "09_old_error_patterns.csv"
)


# =============================================================================
# 11. HIGH-CONFIDENCE ERRORS
# =============================================================================

section("11. HIGH-CONFIDENCE ERRORS")

high_conf = merged[
    merged["high_conf_error_v6"]
].copy()

high_conf["confidence_gap"] = (
    high_conf["prediction_confidence_v6"]
)

high_conf = high_conf.sort_values(
    "prediction_confidence_v6",
    ascending=False
)

high_conf_output = high_conf[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
        "correct_v5",
        "correct_v6",
    ]
].copy()

print(
    f"V6 high-confidence errors: {len(high_conf_output)}"
)

save_csv(
    high_conf_output,
    "10_v6_high_confidence_errors.csv"
)


# =============================================================================
# 12. VERY HIGH-CONFIDENCE WRONG RIPE / OLD
# =============================================================================

section("12. VERY HIGH-CONFIDENCE RIPE / OLD ERRORS")

focus_high_conf = merged[
    merged["label_v5"].isin(FOCUS_CLASSES)
    & (~merged["correct_v6"])
    & (
        merged["prediction_confidence_v6"]
        >= VERY_HIGH_CONF_THRESHOLD
    )
].copy()

focus_high_conf = focus_high_conf.sort_values(
    "prediction_confidence_v6",
    ascending=False
)

focus_output = focus_high_conf[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
    ]
].copy()

print(
    f"Very-high-confidence ripe/old errors: "
    f"{len(focus_output)}"
)

save_csv(
    focus_output,
    "11_focus_high_confidence_errors.csv"
)


# =============================================================================
# 13. V5 CORRECT → V6 WRONG
# =============================================================================

section("13. V5 CORRECT → V6 WRONG")

lost = merged[
    merged["correct_v5"]
    & (~merged["correct_v6"])
].copy()

lost["error_type"] = (
    lost["label_v5"]
    + "_to_"
    + lost["prediction_label_v6"]
)

lost = lost.sort_values(
    "prediction_confidence_v6",
    ascending=False
)

lost_output = lost[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
        "error_type",
    ]
].copy()

print(f"Samples: {len(lost_output)}")

print(
    lost_output[
        [
            "label_v5",
            "prediction_label_v6",
            "source_v5",
        ]
    ]
    .value_counts()
    .reset_index(name="samples")
    .to_string(index=False)
)

save_csv(
    lost_output,
    "12_v5_correct_v6_wrong.csv"
)


# =============================================================================
# 14. V5 WRONG → V6 CORRECT
# =============================================================================

section("14. V5 WRONG → V6 CORRECT")

gained = merged[
    (~merged["correct_v5"])
    & merged["correct_v6"]
].copy()

gained["error_type_v5"] = (
    gained["label_v5"]
    + "_to_"
    + gained["prediction_label_v5"]
)

gained_output = gained[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
        "error_type_v5",
    ]
].copy()

print(f"Samples: {len(gained_output)}")

save_csv(
    gained_output,
    "13_v5_wrong_v6_correct.csv"
)


# =============================================================================
# 15. BOTH WRONG
# =============================================================================

section("15. BOTH WRONG")

both_wrong = merged[
    (~merged["correct_v5"])
    & (~merged["correct_v6"])
].copy()

both_wrong["v5_error"] = (
    both_wrong["label_v5"]
    + "_to_"
    + both_wrong["prediction_label_v5"]
)

both_wrong["v6_error"] = (
    both_wrong["label_v5"]
    + "_to_"
    + both_wrong["prediction_label_v6"]
)

both_wrong_output = both_wrong[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
        "v5_error",
        "v6_error",
    ]
].copy()

print(f"Samples: {len(both_wrong_output)}")

save_csv(
    both_wrong_output,
    "14_both_wrong.csv"
)


# =============================================================================
# 16. SEQUENCE GROUP CONSISTENCY
# =============================================================================

section("16. SEQUENCE GROUP CONSISTENCY")

sequence = (
    merged.groupby(
        [
            "source_v5",
            "sequence_group_v5",
        ]
    )
    .agg(
        samples=("path", "count"),
        unique_labels=("label_v5", "nunique"),
        v5_accuracy=("correct_v5", "mean"),
        v6_accuracy=("correct_v6", "mean"),
        ripe_count=(
            "label_v5",
            lambda x: (x == "ripe").sum()
        ),
        old_count=(
            "label_v5",
            lambda x: (x == "old").sum()
        ),
    )
    .reset_index()
)

sequence["v5_accuracy"] *= 100
sequence["v6_accuracy"] *= 100

sequence["accuracy_delta"] = (
    sequence["v6_accuracy"]
    - sequence["v5_accuracy"]
)

sequence["mixed_label_group"] = (
    sequence["unique_labels"] > 1
)

sequence = sequence.sort_values(
    ["mixed_label_group", "samples"],
    ascending=[False, False]
)

print(
    sequence.head(30).to_string(index=False)
)

save_csv(
    sequence,
    "15_sequence_group_consistency.csv"
)


# =============================================================================
# 17. SEQUENCE GROUPS WITH RIPE
# =============================================================================

section("17. RIPE SEQUENCE GROUP AUDIT")

ripe_sequence = sequence[
    sequence["ripe_count"] > 0
].copy()

ripe_sequence["ripe_ratio"] = (
    ripe_sequence["ripe_count"]
    / ripe_sequence["samples"]
    * 100
)

ripe_sequence = ripe_sequence.sort_values(
    [
        "v6_accuracy",
        "samples",
    ],
    ascending=[
        True,
        False,
    ]
)

save_csv(
    ripe_sequence,
    "16_ripe_sequence_audit.csv"
)

print(
    ripe_sequence.head(30).to_string(index=False)
)


# =============================================================================
# 18. SEQUENCE GROUPS WITH OLD
# =============================================================================

section("18. OLD SEQUENCE GROUP AUDIT")

old_sequence = sequence[
    sequence["old_count"] > 0
].copy()

old_sequence["old_ratio"] = (
    old_sequence["old_count"]
    / old_sequence["samples"]
    * 100
)

old_sequence = old_sequence.sort_values(
    [
        "v6_accuracy",
        "samples",
    ],
    ascending=[
        True,
        False,
    ]
)

save_csv(
    old_sequence,
    "17_old_sequence_audit.csv"
)

print(
    old_sequence.head(30).to_string(index=False)
)


# =============================================================================
# 19. POTENTIAL AMBIGUOUS SAMPLES
# =============================================================================

section("19. POTENTIAL AMBIGUOUS SAMPLES")

#
# Heuristic:
#
# A sample is suspicious when:
#
# A. Both models agree but are wrong with high confidence
# OR
# B. V5 and V6 disagree and both confidence values are moderate/high
# OR
# C. V5 was correct but V6 becomes wrong with high confidence
#

ambiguous = merged[
    (
        (~merged["correct_v5"])
        & (~merged["correct_v6"])
        & (
            merged["prediction_confidence_v5"]
            >= HIGH_CONF_THRESHOLD
        )
        & (
            merged["prediction_confidence_v6"]
            >= HIGH_CONF_THRESHOLD
        )
    )
    |
    (
        merged["correct_v5"]
        & (~merged["correct_v6"])
        & (
            merged["prediction_confidence_v6"]
            >= HIGH_CONF_THRESHOLD
        )
    )
].copy()

ambiguous["ambiguity_reason"] = ""

both_wrong_mask = (
    (~ambiguous["correct_v5"])
    & (~ambiguous["correct_v6"])
)

ambiguous.loc[
    both_wrong_mask,
    "ambiguity_reason"
] = "both_models_wrong_high_confidence"

lost_mask = (
    ambiguous["correct_v5"]
    & (~ambiguous["correct_v6"])
)

ambiguous.loc[
    lost_mask,
    "ambiguity_reason"
] = "v5_correct_v6_wrong_high_confidence"

ambiguous["priority"] = "medium"

ambiguous.loc[
    (
        ambiguous["label_v5"].isin(FOCUS_CLASSES)
    ),
    "priority"
] = "high"

ambiguous = ambiguous.sort_values(
    [
        "priority",
        "prediction_confidence_v6",
    ],
    ascending=[
        True,
        False,
    ]
)

ambiguous_output = ambiguous[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
        "ambiguity_reason",
        "priority",
    ]
].copy()

print(
    f"Potential ambiguous samples: "
    f"{len(ambiguous_output)}"
)

save_csv(
    ambiguous_output,
    "18_potential_ambiguous_samples.csv"
)


# =============================================================================
# 20. LABEL QUALITY SCORE
# =============================================================================

section("20. LABEL QUALITY INDICATOR")

quality_rows = []

for label in CLASSES:

    subset = merged[
        merged["label_v5"] == label
    ]

    support = len(subset)

    if support == 0:
        continue

    v5_acc = subset["correct_v5"].mean() * 100
    v6_acc = subset["correct_v6"].mean() * 100

    both_wrong_count = (
        (~subset["correct_v5"])
        & (~subset["correct_v6"])
    ).sum()

    high_conf_both_wrong = (
        (~subset["correct_v5"])
        & (~subset["correct_v6"])
        & (
            subset["prediction_confidence_v5"]
            >= HIGH_CONF_THRESHOLD
        )
        & (
            subset["prediction_confidence_v6"]
            >= HIGH_CONF_THRESHOLD
        )
    ).sum()

    quality_rows.append({
        "label": label,
        "support": support,
        "v5_accuracy": v5_acc,
        "v6_accuracy": v6_acc,
        "both_wrong": int(both_wrong_count),
        "both_wrong_percentage": pct(
            both_wrong_count,
            support
        ),
        "high_conf_both_wrong": int(
            high_conf_both_wrong
        ),
        "high_conf_both_wrong_percentage": pct(
            high_conf_both_wrong,
            support
        ),
    })

label_quality = pd.DataFrame(
    quality_rows
)

label_quality["audit_priority"] = "LOW"

label_quality.loc[
    (
        label_quality["both_wrong_percentage"] >= 30
    ),
    "audit_priority"
] = "MEDIUM"

label_quality.loc[
    (
        label_quality["both_wrong_percentage"] >= 40
    )
    |
    (
        label_quality["high_conf_both_wrong_percentage"]
        >= 20
    ),
    "audit_priority"
] = "HIGH"

print(
    label_quality.to_string(index=False)
)

save_csv(
    label_quality,
    "19_label_quality_indicator.csv"
)


# =============================================================================
# 21. FINAL MANUAL REVIEW LIST
# =============================================================================

section("21. FINAL MANUAL REVIEW LIST")

manual = merged[
    (
        merged["label_v5"].isin(
            ["ripe", "old"]
        )
    )
    &
    (
        (
            merged["correct_v5"]
            & (~merged["correct_v6"])
        )
        |
        (
            (~merged["correct_v5"])
            & (~merged["correct_v6"])
        )
    )
].copy()

manual["review_priority"] = "MEDIUM"

#
# Highest priority:
# V5 correct -> V6 wrong
#

manual.loc[
    manual["correct_v5"]
    & (~manual["correct_v6"]),
    "review_priority"
] = "HIGH"

#
# Very high confidence both wrong
#

manual.loc[
    (
        (~manual["correct_v5"])
        & (~manual["correct_v6"])
        & (
            manual["prediction_confidence_v5"]
            >= VERY_HIGH_CONF_THRESHOLD
        )
        & (
            manual["prediction_confidence_v6"]
            >= VERY_HIGH_CONF_THRESHOLD
        )
    ),
    "review_priority"
] = "VERY_HIGH"

manual = manual.sort_values(
    [
        "review_priority",
        "label_v5",
        "prediction_confidence_v6",
    ],
    ascending=[
        True,
        True,
        False,
    ]
)

manual_output = manual[
    [
        "path",
        "filename_v5",
        "label_v5",
        "source_v5",
        "sequence_group_v5",
        "prediction_label_v5",
        "prediction_confidence_v5",
        "prediction_label_v6",
        "prediction_confidence_v6",
        "correct_v5",
        "correct_v6",
        "review_priority",
    ]
].copy()

print(
    f"Manual review candidates: "
    f"{len(manual_output)}"
)

save_csv(
    manual_output,
    "20_manual_review_candidates.csv"
)


# =============================================================================
# 22. SUMMARY
# =============================================================================

section("22. FINAL AUDIT SUMMARY")

total = len(merged)

v5_correct = int(
    merged["correct_v5"].sum()
)

v6_correct = int(
    merged["correct_v6"].sum()
)

lost_count = int(
    (
        merged["correct_v5"]
        & (~merged["correct_v6"])
    ).sum()
)

gained_count = int(
    (
        (~merged["correct_v5"])
        & merged["correct_v6"]
    ).sum()
)

both_wrong_count = int(
    (
        (~merged["correct_v5"])
        & (~merged["correct_v6"])
    ).sum()
)

ripe_subset = merged[
    merged["label_v5"] == "ripe"
]

old_subset = merged[
    merged["label_v5"] == "old"
]

print(f"Matched images       : {total}")

print()
print(f"V5 correct           : {v5_correct}")
print(f"V6 correct           : {v6_correct}")

print()
print(f"V5 accuracy          : {pct(v5_correct, total):.2f}%")
print(f"V6 accuracy          : {pct(v6_correct, total):.2f}%")

print()
print(f"V5 -> V6 LOST        : {lost_count}")
print(f"V5 -> V6 GAINED      : {gained_count}")
print(f"Both wrong           : {both_wrong_count}")

print()
print(
    f"Ripe support         : {len(ripe_subset)}"
)

print(
    f"Ripe V5 recall       : "
    f"{pct(ripe_subset['correct_v5'].sum(), len(ripe_subset)):.2f}%"
)

print(
    f"Ripe V6 recall       : "
    f"{pct(ripe_subset['correct_v6'].sum(), len(ripe_subset)):.2f}%"
)

print()
print(
    f"Old support          : {len(old_subset)}"
)

print(
    f"Old V5 recall        : "
    f"{pct(old_subset['correct_v5'].sum(), len(old_subset)):.2f}%"
)

print(
    f"Old V6 recall        : "
    f"{pct(old_subset['correct_v6'].sum(), len(old_subset)):.2f}%"
)

print()
print("Focus:")
print("  1. ripe")
print("  2. old")
print("  3. source differences")
print("  4. high-confidence both-wrong samples")
print("  5. sequence-group consistency")

print()
print("Recommended next step:")
print("  DO NOT TRAIN V7 YET.")
print("  Review the generated CSV files first.")

print()
print("=" * 90)
print("CIRCULENS QUALITY — LABEL AUDIT COMPLETED")
print("=" * 90)

print()
print("Audit output directory:")
print(OUTPUT_DIR)