import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

V5_PREDICTIONS = Path(
    "reports/quality_finetune_v5/test_predictions.csv"
)

V6_PREDICTIONS = Path(
    "reports/quality_finetune_v6/test_predictions.csv"
)

OUT_DIR = Path(
    "reports/quality_finetune_v6/v5_v6_error_analysis"
)

CLASS_NAMES = [
    "damaged",
    "dried",
    "old",
    "ripe",
    "unripe",
]

HIGH_CONFIDENCE_THRESHOLD = 0.80


# ============================================================================
# UTILITIES
# ============================================================================

def banner(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_csv(df, filename):
    path = OUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def normalize_path(value):
    """
    Normalize Windows / Git Bash / relative path representations
    so V5 and V6 can be matched reliably.
    """
    value = str(value).strip()

    value = value.replace("\\", "/")

    # Remove leading ./ if present.
    while value.startswith("./"):
        value = value[2:]

    return value.lower()


def find_common_key(v5_df, v6_df):
    """
    Prefer path as the unique image identifier.
    Fall back to filename if necessary.
    """

    if "path" in v5_df.columns and "path" in v6_df.columns:
        v5_paths = set(v5_df["path"].map(normalize_path))
        v6_paths = set(v6_df["path"].map(normalize_path))

        common = v5_paths & v6_paths

        if len(common) == len(v5_df) == len(v6_df):
            return "path"

        if len(common) > 0:
            print(
                f"\nPath overlap: {len(common)} "
                f"/ V5={len(v5_df)} "
                f"/ V6={len(v6_df)}"
            )

            return "path"

    if "filename" in v5_df.columns and "filename" in v6_df.columns:
        v5_files = set(
            v5_df["filename"].astype(str).str.lower()
        )
        v6_files = set(
            v6_df["filename"].astype(str).str.lower()
        )

        common = v5_files & v6_files

        if len(common) > 0:
            print(
                f"\nFilename overlap: {len(common)} "
                f"/ V5={len(v5_df)} "
                f"/ V6={len(v6_df)}"
            )

            return "filename"

    raise ValueError(
        "Unable to find a common image identifier between V5 and V6."
    )


# ============================================================================
# LOAD
# ============================================================================

def load_predictions():
    banner("1. LOADING V5 / V6 PREDICTIONS")

    if not V5_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"V5 predictions not found:\n{V5_PREDICTIONS}"
        )

    if not V6_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"V6 predictions not found:\n{V6_PREDICTIONS}"
        )

    v5 = pd.read_csv(V5_PREDICTIONS)
    v6 = pd.read_csv(V6_PREDICTIONS)

    print(f"\nV5 rows: {len(v5)}")
    print(f"V6 rows: {len(v6)}")

    print("\nV5 columns:")
    print(list(v5.columns))

    print("\nV6 columns:")
    print(list(v6.columns))

    return v5, v6


# ============================================================================
# VALIDATION
# ============================================================================

def validate_predictions(v5, v6):
    banner("2. VALIDATING PREDICTIONS")

    required = [
        "path",
        "filename",
        "label",
        "source",
        "sequence_group",
        "timestamp",
        "prediction_label",
        "prediction_confidence",
    ]

    for version, df in [
        ("V5", v5),
        ("V6", v6),
    ]:
        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{version} predictions are missing columns: {missing}"
            )

        print(f"\n{version}: PASS")

        print(
            f"  Missing path       : "
            f"{df['path'].isna().sum()}"
        )

        print(
            f"  Duplicate path     : "
            f"{df['path'].duplicated().sum()}"
        )

        print(
            f"  Missing true label : "
            f"{df['label'].isna().sum()}"
        )

        print(
            f"  Missing prediction : "
            f"{df['prediction_label'].isna().sum()}"
        )

        invalid_true = sorted(
            set(df["label"].astype(str))
            - set(CLASS_NAMES)
        )

        invalid_pred = sorted(
            set(df["prediction_label"].astype(str))
            - set(CLASS_NAMES)
        )

        if invalid_true:
            raise ValueError(
                f"{version}: invalid true labels: {invalid_true}"
            )

        if invalid_pred:
            raise ValueError(
                f"{version}: invalid prediction labels: {invalid_pred}"
            )


# ============================================================================
# MERGE
# ============================================================================

def merge_predictions(v5, v6):
    banner("3. MATCHING V5 AND V6")

    key = find_common_key(v5, v6)

    print(f"\nMatching key: {key}")

    v5_work = v5.copy()
    v6_work = v6.copy()

    if key == "path":
        v5_work["_match_key"] = v5_work["path"].map(
            normalize_path
        )
        v6_work["_match_key"] = v6_work["path"].map(
            normalize_path
        )
    else:
        v5_work["_match_key"] = (
            v5_work["filename"]
            .astype(str)
            .str.lower()
        )
        v6_work["_match_key"] = (
            v6_work["filename"]
            .astype(str)
            .str.lower()
        )

    v5_work = v5_work.rename(
        columns={
            "prediction_label": "v5_prediction",
            "prediction_confidence": "v5_confidence",
        }
    )

    v6_work = v6_work.rename(
        columns={
            "prediction_label": "v6_prediction",
            "prediction_confidence": "v6_confidence",
        }
    )

    # Columns from V6 that should not overwrite V5 metadata.
    v6_columns = [
        "_match_key",
        "v6_prediction",
        "v6_confidence",
    ]

    # Include V6 probability columns if present.
    v6_prob_columns = [
        column
        for column in v6_work.columns
        if column.startswith("prob_")
    ]

    v6_columns.extend(v6_prob_columns)

    merged = v5_work.merge(
        v6_work[v6_columns],
        on="_match_key",
        how="inner",
        validate="one_to_one",
    )

    print(f"\nMatched images: {len(merged)}")

    if len(merged) != len(v5) or len(merged) != len(v6):
        print(
            "\nWARNING:"
            "\nNot all V5/V6 images were matched."
        )

        print(
            f"V5 unmatched: {len(v5) - len(merged)}"
        )

        print(
            f"V6 unmatched: {len(v6) - len(merged)}"
        )

    if len(merged) == 0:
        raise ValueError(
            "No V5/V6 predictions could be matched."
        )

    # Make sure true labels agree.
    if "label" in merged.columns:
        mismatch = (
            merged["label"].astype(str)
            != merged.get(
                "label_v6",
                merged["label"],
            ).astype(str)
        )

        # In the current merge V6 label is not included,
        # so this simply documents that V5 is used as the
        # canonical true label.
        if mismatch.any():
            print(
                "WARNING: True-label mismatch detected."
            )

    merged["v5_correct"] = (
        merged["label"]
        == merged["v5_prediction"]
    )

    merged["v6_correct"] = (
        merged["label"]
        == merged["v6_prediction"]
    )

    merged["v5_error"] = ~merged["v5_correct"]
    merged["v6_error"] = ~merged["v6_correct"]

    # Confidence movement.
    merged["confidence_delta_v6_minus_v5"] = (
        merged["v6_confidence"]
        - merged["v5_confidence"]
    )

    return merged


# ============================================================================
# OVERALL COMPARISON
# ============================================================================

def create_overall_comparison(merged):
    banner("4. OVERALL V5 VS V6 COMPARISON")

    total = len(merged)

    v5_correct = int(merged["v5_correct"].sum())
    v6_correct = int(merged["v6_correct"].sum())

    v5_accuracy = v5_correct / total
    v6_accuracy = v6_correct / total

    v5_wrong_v6_correct = int(
        (
            merged["v5_error"]
            & merged["v6_correct"]
        ).sum()
    )

    v5_correct_v6_wrong = int(
        (
            merged["v5_correct"]
            & merged["v6_error"]
        ).sum()
    )

    both_correct = int(
        (
            merged["v5_correct"]
            & merged["v6_correct"]
        ).sum()
    )

    both_wrong = int(
        (
            merged["v5_error"]
            & merged["v6_error"]
        ).sum()
    )

    rows = [
        {
            "category": "V5 correct / V6 correct",
            "count": both_correct,
            "percentage": both_correct / total * 100,
        },
        {
            "category": "V5 correct / V6 wrong",
            "count": v5_correct_v6_wrong,
            "percentage": v5_correct_v6_wrong / total * 100,
        },
        {
            "category": "V5 wrong / V6 correct",
            "count": v5_wrong_v6_correct,
            "percentage": v5_wrong_v6_correct / total * 100,
        },
        {
            "category": "V5 wrong / V6 wrong",
            "count": both_wrong,
            "percentage": both_wrong / total * 100,
        },
        {
            "category": "V5 total correct",
            "count": v5_correct,
            "percentage": v5_accuracy * 100,
        },
        {
            "category": "V6 total correct",
            "count": v6_correct,
            "percentage": v6_accuracy * 100,
        },
    ]

    comparison = pd.DataFrame(rows)

    print("\n" + comparison.to_string(index=False))

    save_csv(
        comparison,
        "v5_v6_error_summary.csv",
    )

    print("\nImportant counts:")
    print(
        f"  V5 correct -> V6 wrong : "
        f"{v5_correct_v6_wrong}"
    )
    print(
        f"  V5 wrong   -> V6 correct: "
        f"{v5_wrong_v6_correct}"
    )
    print(
        f"  Both correct            : "
        f"{both_correct}"
    )
    print(
        f"  Both wrong              : "
        f"{both_wrong}"
    )

    return comparison


# ============================================================================
# FOUR-WAY GROUPS
# ============================================================================

def create_four_way_groups(merged):
    banner("5. FOUR-WAY ERROR GROUPS")

    v5_correct_v6_wrong = merged[
        merged["v5_correct"]
        & merged["v6_error"]
    ].copy()

    v5_wrong_v6_correct = merged[
        merged["v5_error"]
        & merged["v6_correct"]
    ].copy()

    both_correct = merged[
        merged["v5_correct"]
        & merged["v6_correct"]
    ].copy()

    both_wrong = merged[
        merged["v5_error"]
        & merged["v6_error"]
    ].copy()

    save_csv(
        v5_correct_v6_wrong,
        "v5_correct_v6_wrong.csv",
    )

    save_csv(
        v5_wrong_v6_correct,
        "v5_wrong_v6_correct.csv",
    )

    save_csv(
        both_correct,
        "both_correct.csv",
    )

    save_csv(
        both_wrong,
        "both_wrong.csv",
    )

    print(
        f"""
V5 correct -> V6 wrong : {len(v5_correct_v6_wrong)}
V5 wrong   -> V6 correct: {len(v5_wrong_v6_correct)}
Both correct            : {len(both_correct)}
Both wrong              : {len(both_wrong)}
"""
    )


# ============================================================================
# ERROR PAIRS
# ============================================================================

def create_error_pair_analysis(merged):
    banner("6. ERROR PAIR ANALYSIS")

    # V5 errors.
    v5_errors = merged[
        merged["v5_error"]
    ].copy()

    v5_error_pairs = (
        v5_errors
        .groupby(
            [
                "label",
                "v5_prediction",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "errors",
                "v5_prediction": "prediction",
            }
        )
        .sort_values(
            "errors",
            ascending=False,
        )
    )

    v5_error_pairs["percentage_of_errors"] = (
        v5_error_pairs["errors"]
        / max(len(v5_errors), 1)
        * 100
    )

    save_csv(
        v5_error_pairs,
        "v5_error_pairs.csv",
    )

    # V6 errors.
    v6_errors = merged[
        merged["v6_error"]
    ].copy()

    v6_error_pairs = (
        v6_errors
        .groupby(
            [
                "label",
                "v6_prediction",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "errors",
                "v6_prediction": "prediction",
            }
        )
        .sort_values(
            "errors",
            ascending=False,
        )
    )

    v6_error_pairs["percentage_of_errors"] = (
        v6_error_pairs["errors"]
        / max(len(v6_errors), 1)
        * 100
    )

    save_csv(
        v6_error_pairs,
        "v6_error_pairs.csv",
    )

    print("\nV5 top error pairs:")
    print(
        v5_error_pairs.head(15).to_string(
            index=False
        )
    )

    print("\nV6 top error pairs:")
    print(
        v6_error_pairs.head(15).to_string(
            index=False
        )
    )


# ============================================================================
# CLASS-LEVEL ANALYSIS
# ============================================================================

def create_class_analysis(merged):
    banner("7. CLASS-LEVEL V5 VS V6")

    rows = []

    for class_name in CLASS_NAMES:
        subset = merged[
            merged["label"] == class_name
        ]

        total = len(subset)

        if total == 0:
            continue

        v5_correct = int(
            subset["v5_correct"].sum()
        )

        v6_correct = int(
            subset["v6_correct"].sum()
        )

        rows.append(
            {
                "label": class_name,
                "support": total,
                "v5_correct": v5_correct,
                "v5_recall": v5_correct / total,
                "v6_correct": v6_correct,
                "v6_recall": v6_correct / total,
                "recall_delta_v6_minus_v5": (
                    v6_correct / total
                    - v5_correct / total
                ),
            }
        )

    class_df = pd.DataFrame(rows)

    print(
        class_df.to_string(index=False)
    )

    save_csv(
        class_df,
        "v5_v6_class_analysis.csv",
    )

    return class_df


# ============================================================================
# TARGETED RIPE ANALYSIS
# ============================================================================

def create_ripe_analysis(merged):
    banner("8. RIPE ERROR ANALYSIS")

    ripe = merged[
        merged["label"] == "ripe"
    ].copy()

    if ripe.empty:
        print("No ripe samples found.")
        return

    ripe["v5_status"] = np.where(
        ripe["v5_correct"],
        "correct",
        "wrong",
    )

    ripe["v6_status"] = np.where(
        ripe["v6_correct"],
        "correct",
        "wrong",
    )

    ripe["transition"] = (
        ripe["v5_status"]
        + " -> "
        + ripe["v6_status"]
    )

    transition = (
        ripe
        .groupby(
            [
                "v5_status",
                "v6_status",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
    )

    transition["percentage"] = (
        transition["samples"]
        / len(ripe)
        * 100
    )

    print("\nRipe V5 -> V6 status:")
    print(
        transition.to_string(index=False)
    )

    save_csv(
        transition,
        "ripe_status_transition.csv",
    )

    # V5 prediction distribution.
    v5_pred = (
        ripe
        .groupby(
            "v5_prediction",
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(
            "samples",
            ascending=False,
        )
    )

    v5_pred["percentage"] = (
        v5_pred["samples"]
        / len(ripe)
        * 100
    )

    # V6 prediction distribution.
    v6_pred = (
        ripe
        .groupby(
            "v6_prediction",
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(
            "samples",
            ascending=False,
        )
    )

    v6_pred["percentage"] = (
        v6_pred["samples"]
        / len(ripe)
        * 100
    )

    print("\nRipe V5 predictions:")
    print(
        v5_pred.to_string(index=False)
    )

    print("\nRipe V6 predictions:")
    print(
        v6_pred.to_string(index=False)
    )

    save_csv(
        v5_pred,
        "ripe_v5_prediction_distribution.csv",
    )

    save_csv(
        v6_pred,
        "ripe_v6_prediction_distribution.csv",
    )

    # Most important subset:
    # ripe samples that V5 got right but V6 lost.
    ripe_v5_right_v6_wrong = ripe[
        ripe["v5_correct"]
        & ripe["v6_error"]
    ].copy()

    # V5 wrong -> V6 right.
    ripe_v5_wrong_v6_right = ripe[
        ripe["v5_error"]
        & ripe["v6_correct"]
    ].copy()

    # Both wrong.
    ripe_both_wrong = ripe[
        ripe["v5_error"]
        & ripe["v6_error"]
    ].copy()

    save_csv(
        ripe_v5_right_v6_wrong,
        "ripe_v5_correct_v6_wrong.csv",
    )

    save_csv(
        ripe_v5_wrong_v6_right,
        "ripe_v5_wrong_v6_correct.csv",
    )

    save_csv(
        ripe_both_wrong,
        "ripe_both_wrong.csv",
    )

    print(
        f"""
Ripe total                  : {len(ripe)}
V5 correct                  : {ripe['v5_correct'].sum()}
V6 correct                  : {ripe['v6_correct'].sum()}
V5 correct -> V6 wrong      : {len(ripe_v5_right_v6_wrong)}
V5 wrong -> V6 correct      : {len(ripe_v5_wrong_v6_right)}
Both wrong                  : {len(ripe_both_wrong)}
"""
    )


# ============================================================================
# TARGETED OLD ANALYSIS
# ============================================================================

def create_old_analysis(merged):
    banner("9. OLD ERROR ANALYSIS")

    old = merged[
        merged["label"] == "old"
    ].copy()

    if old.empty:
        print("No old samples found.")
        return

    old["v5_status"] = np.where(
        old["v5_correct"],
        "correct",
        "wrong",
    )

    old["v6_status"] = np.where(
        old["v6_correct"],
        "correct",
        "wrong",
    )

    transition = (
        old
        .groupby(
            [
                "v5_status",
                "v6_status",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
    )

    transition["percentage"] = (
        transition["samples"]
        / len(old)
        * 100
    )

    print("\nOld V5 -> V6 status:")
    print(
        transition.to_string(index=False)
    )

    save_csv(
        transition,
        "old_status_transition.csv",
    )

    # Prediction distributions.
    v5_pred = (
        old
        .groupby(
            "v5_prediction",
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(
            "samples",
            ascending=False,
        )
    )

    v5_pred["percentage"] = (
        v5_pred["samples"]
        / len(old)
        * 100
    )

    v6_pred = (
        old
        .groupby(
            "v6_prediction",
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(
            "samples",
            ascending=False,
        )
    )

    v6_pred["percentage"] = (
        v6_pred["samples"]
        / len(old)
        * 100
    )

    print("\nOld V5 predictions:")
    print(
        v5_pred.to_string(index=False)
    )

    print("\nOld V6 predictions:")
    print(
        v6_pred.to_string(index=False)
    )

    save_csv(
        v5_pred,
        "old_v5_prediction_distribution.csv",
    )

    save_csv(
        v6_pred,
        "old_v6_prediction_distribution.csv",
    )

    old_v5_right_v6_wrong = old[
        old["v5_correct"]
        & old["v6_error"]
    ].copy()

    old_v5_wrong_v6_right = old[
        old["v5_error"]
        & old["v6_correct"]
    ].copy()

    old_both_wrong = old[
        old["v5_error"]
        & old["v6_error"]
    ].copy()

    save_csv(
        old_v5_right_v6_wrong,
        "old_v5_correct_v6_wrong.csv",
    )

    save_csv(
        old_v5_wrong_v6_right,
        "old_v5_wrong_v6_correct.csv",
    )

    save_csv(
        old_both_wrong,
        "old_both_wrong.csv",
    )

    print(
        f"""
Old total                   : {len(old)}
V5 correct                  : {old['v5_correct'].sum()}
V6 correct                  : {old['v6_correct'].sum()}
V5 correct -> V6 wrong      : {len(old_v5_right_v6_wrong)}
V5 wrong -> V6 correct      : {len(old_v5_wrong_v6_right)}
Both wrong                  : {len(old_both_wrong)}
"""
    )


# ============================================================================
# SOURCE ANALYSIS
# ============================================================================

def create_source_analysis(merged):
    banner("10. SOURCE ANALYSIS")

    rows = []

    for source, subset in merged.groupby("source"):
        total = len(subset)

        v5_correct = int(
            subset["v5_correct"].sum()
        )

        v6_correct = int(
            subset["v6_correct"].sum()
        )

        rows.append(
            {
                "source": source,
                "support": total,
                "v5_accuracy": v5_correct / total,
                "v6_accuracy": v6_correct / total,
                "delta_v6_minus_v5": (
                    v6_correct / total
                    - v5_correct / total
                ),
                "v5_errors": total - v5_correct,
                "v6_errors": total - v6_correct,
            }
        )

    source_df = pd.DataFrame(rows)

    print(
        source_df.to_string(index=False)
    )

    save_csv(
        source_df,
        "v5_v6_source_analysis.csv",
    )


# ============================================================================
# ERROR PAIR TRANSITIONS
# ============================================================================

def create_transition_analysis(merged):
    banner("11. PREDICTION TRANSITION ANALYSIS")

    transition = (
        merged
        .groupby(
            [
                "label",
                "v5_prediction",
                "v6_prediction",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(
            "samples",
            ascending=False,
        )
    )

    transition["percentage_of_test"] = (
        transition["samples"]
        / len(merged)
        * 100
    )

    save_csv(
        transition,
        "v5_v6_prediction_transitions.csv",
    )

    print(
        "\nLargest V5 -> V6 transitions:"
    )

    print(
        transition.head(30).to_string(
            index=False
        )
    )


# ============================================================================
# HIGH CONFIDENCE ERRORS
# ============================================================================

def create_high_confidence_analysis(merged):
    banner("12. HIGH-CONFIDENCE ERROR ANALYSIS")

    v5_high_conf_errors = merged[
        merged["v5_error"]
        & (
            merged["v5_confidence"]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    ].copy()

    v6_high_conf_errors = merged[
        merged["v6_error"]
        & (
            merged["v6_confidence"]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    ].copy()

    print(
        f"\nV5 high-confidence errors: "
        f"{len(v5_high_conf_errors)}"
    )

    print(
        f"V6 high-confidence errors: "
        f"{len(v6_high_conf_errors)}"
    )

    save_csv(
        v5_high_conf_errors,
        "v5_high_confidence_errors.csv",
    )

    save_csv(
        v6_high_conf_errors,
        "v6_high_confidence_errors.csv",
    )

    # Errors that became high-confidence only in V6.
    v6_new_high_conf = merged[
        merged["v6_error"]
        & (
            merged["v6_confidence"]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
        & (
            (
                ~merged["v5_error"]
            )
            | (
                merged["v5_confidence"]
                < HIGH_CONFIDENCE_THRESHOLD
            )
        )
    ].copy()

    save_csv(
        v6_new_high_conf,
        "v6_new_high_confidence_errors.csv",
    )

    print(
        f"\nNew V6 high-confidence errors: "
        f"{len(v6_new_high_conf)}"
    )


# ============================================================================
# CONFIDENCE MOVEMENT
# ============================================================================

def create_confidence_analysis(merged):
    banner("13. CONFIDENCE ANALYSIS")

    rows = []

    for class_name in CLASS_NAMES:
        subset = merged[
            merged["label"] == class_name
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "label": class_name,
                "samples": len(subset),
                "v5_mean_confidence": subset[
                    "v5_confidence"
                ].mean(),
                "v6_mean_confidence": subset[
                    "v6_confidence"
                ].mean(),
                "confidence_delta": (
                    subset["v6_confidence"].mean()
                    - subset["v5_confidence"].mean()
                ),
                "v5_correct_confidence": subset.loc[
                    subset["v5_correct"],
                    "v5_confidence",
                ].mean(),
                "v6_correct_confidence": subset.loc[
                    subset["v6_correct"],
                    "v6_confidence",
                ].mean(),
            }
        )

    confidence_df = pd.DataFrame(rows)

    print(
        confidence_df.to_string(index=False)
    )

    save_csv(
        confidence_df,
        "v5_v6_confidence_by_class.csv",
    )

    # Confidence comparison for the four groups.
    groups = {
        "both_correct": (
            merged["v5_correct"]
            & merged["v6_correct"]
        ),
        "v5_correct_v6_wrong": (
            merged["v5_correct"]
            & merged["v6_error"]
        ),
        "v5_wrong_v6_correct": (
            merged["v5_error"]
            & merged["v6_correct"]
        ),
        "both_wrong": (
            merged["v5_error"]
            & merged["v6_error"]
        ),
    }

    rows = []

    for name, mask in groups.items():
        subset = merged[mask]

        if subset.empty:
            continue

        rows.append(
            {
                "group": name,
                "samples": len(subset),
                "v5_mean_confidence": subset[
                    "v5_confidence"
                ].mean(),
                "v6_mean_confidence": subset[
                    "v6_confidence"
                ].mean(),
            }
        )

    group_confidence = pd.DataFrame(rows)

    print(
        "\nConfidence by V5/V6 outcome:"
    )

    print(
        group_confidence.to_string(
            index=False
        )
    )

    save_csv(
        group_confidence,
        "v5_v6_confidence_by_outcome.csv",
    )


# ============================================================================
# FINAL MASTER TABLE
# ============================================================================

def create_master_table(merged):
    banner("14. SAVING MASTER COMPARISON")

    columns = [
        "_match_key",
        "path",
        "filename",
        "label",
        "source",
        "sequence_group",
        "timestamp",
        "v5_prediction",
        "v5_confidence",
        "v6_prediction",
        "v6_confidence",
        "v5_correct",
        "v6_correct",
        "v5_error",
        "v6_error",
        "confidence_delta_v6_minus_v5",
    ]

    columns = [
        column
        for column in columns
        if column in merged.columns
    ]

    master = merged[columns].copy()

    save_csv(
        master,
        "v5_v6_comparison.csv",
    )

    return master


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def final_summary(merged):
    banner("15. FINAL ANALYSIS SUMMARY")

    total = len(merged)

    v5_correct = int(
        merged["v5_correct"].sum()
    )

    v6_correct = int(
        merged["v6_correct"].sum()
    )

    lost = int(
        (
            merged["v5_correct"]
            & merged["v6_error"]
        ).sum()
    )

    gained = int(
        (
            merged["v5_error"]
            & merged["v6_correct"]
        ).sum()
    )

    both_wrong = int(
        (
            merged["v5_error"]
            & merged["v6_error"]
        ).sum()
    )

    print(
        f"""
Matched test images : {total}

V5 correct          : {v5_correct}
V6 correct          : {v6_correct}

V5 -> V6 LOST       : {lost}
V5 -> V6 GAINED     : {gained}
Both wrong          : {both_wrong}

Net change in correct predictions:
{gained - lost:+d}

V5 accuracy:
{v5_correct / total * 100:.2f}%

V6 accuracy:
{v6_correct / total * 100:.2f}%

Accuracy delta:
{(v6_correct - v5_correct) / total * 100:+.2f} percentage points
"""
    )

    print(
        """
Interpretation:

1. V5 correct -> V6 wrong
   These are the most important samples to inspect.
   V6 lost information that V5 successfully captured.

2. V5 wrong -> V6 correct
   These show where the V6 changes actually helped.

3. Both wrong
   These are likely harder samples or potential
   dataset/label ambiguity.

4. Ripe and old are analyzed separately because
   they were the target of the V6 intervention.

5. Do NOT change the model again until these patterns
   have been reviewed.
"""
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    banner(
        "CIRCULENS QUALITY — V5 VS V6 ERROR ANALYSIS"
    )

    print(
        f"""
V5 predictions:
{V5_PREDICTIONS}

V6 predictions:
{V6_PREDICTIONS}

Output:
{OUT_DIR}
"""
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1.
    v5, v6 = load_predictions()

    # 2.
    validate_predictions(v5, v6)

    # 3.
    merged = merge_predictions(v5, v6)

    # 4.
    create_overall_comparison(merged)

    # 5.
    create_four_way_groups(merged)

    # 6.
    create_error_pair_analysis(merged)

    # 7.
    create_class_analysis(merged)

    # 8.
    create_ripe_analysis(merged)

    # 9.
    create_old_analysis(merged)

    # 10.
    create_source_analysis(merged)

    # 11.
    create_transition_analysis(merged)

    # 12.
    create_high_confidence_analysis(merged)

    # 13.
    create_confidence_analysis(merged)

    # 14.
    create_master_table(merged)

    # 15.
    final_summary(merged)

    banner(
        "CIRCULENS QUALITY — V5 VS V6 ANALYSIS COMPLETED"
    )

    print(
        f"""
All analysis files are available in:

{OUT_DIR}
"""
    )


if __name__ == "__main__":
    main()
