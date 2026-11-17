# analyze_sequence_0125_post_review.py

import os
import sys
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

AUDIT_CSV = (
    r"C:\Projects\circulens\reports\quality_finetune_v6"
    r"\v5_v6_error_analysis\label_audit\audit_master_v5_v6.csv"
)

REVIEW_CSV = (
    r"C:\Projects\circulens\reports\quality_finetune_v6"
    r"\v5_v6_error_analysis\label_audit\manual_review_report"
    r"\sequence_0125_review\sequence_0125_manual_decision_reviewed.csv"
)

SEQUENCE = "sequence_0125"

OUTPUT_DIR = (
    r"C:\Projects\circulens\reports\quality_finetune_v6"
    r"\v5_v6_error_analysis\label_audit\manual_review_report"
    r"\sequence_0125_review\post_review_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def normalize_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    mapping = {
        "ripe": "ripe",
        "unripe": "unripe",
        "old": "old",
        "damaged": "damaged",
        "dried": "dried",
        "ambiguous": "ambiguous",
        "r": "ripe",
        "u": "unripe",
        "o": "old",
        "d": "damaged",
    }

    return mapping.get(value, value)


def normalize_sequence(value):
    if pd.isna(value):
        return None

    return str(value).strip()


def normalize_filename(value):
    if pd.isna(value):
        return None

    return os.path.basename(str(value).strip())


def resolve_column(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            print(f"[OK] {description:<25}: {col}")
            return col

    print(f"[ERROR] Tidak menemukan kolom untuk: {description}")
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")

    sys.exit(1)


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")
    return path


def percentage(value, total):
    if total == 0:
        return 0.0

    return round((value / total) * 100, 2)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("CIRCULENS QUALITY — SEQUENCE 0125 POST-MANUAL-REVIEW ANALYSIS")
print("=" * 100)

print(f"Audit CSV  : {AUDIT_CSV}")
print(f"Review CSV : {REVIEW_CSV}")
print(f"Sequence   : {SEQUENCE}")

print()


# =============================================================================
# 1. CHECK INPUT
# =============================================================================

print("=" * 100)
print("1. CHECKING INPUT")
print("=" * 100)

if not os.path.exists(AUDIT_CSV):
    print("[ERROR] Audit CSV tidak ditemukan.")
    sys.exit(1)

if not os.path.exists(REVIEW_CSV):
    print("[ERROR] Review CSV tidak ditemukan.")
    sys.exit(1)

print("Audit CSV  : OK")
print("Review CSV : OK")
print()


# =============================================================================
# 2. LOADING DATA
# =============================================================================

print("=" * 100)
print("2. LOADING DATA")
print("=" * 100)

audit = pd.read_csv(AUDIT_CSV)
review = pd.read_csv(REVIEW_CSV)

print(f"Audit rows  : {len(audit)}")
print(f"Review rows : {len(review)}")
print()


# =============================================================================
# 3. RESOLVE AUDIT COLUMNS
# =============================================================================

print("=" * 100)
print("3. RESOLVING AUDIT COLUMNS")
print("=" * 100)

audit_filename_col = resolve_column(
    audit,
    ["filename_v5", "filename"],
    "Audit filename"
)

audit_sequence_col = resolve_column(
    audit,
    ["sequence_group_v5", "sequence_group", "sequence"],
    "Audit sequence"
)

audit_label_col = resolve_column(
    audit,
    ["label_v5", "label"],
    "Dataset label"
)

audit_v5_pred_col = resolve_column(
    audit,
    ["prediction_label_v5", "v5_prediction"],
    "V5 prediction"
)

audit_v6_pred_col = resolve_column(
    audit,
    ["prediction_label_v6", "v6_prediction"],
    "V6 prediction"
)

audit_v5_conf_col = resolve_column(
    audit,
    [
        "prediction_confidence_v5",
        "v5_confidence",
    ],
    "V5 confidence"
)

audit_v6_conf_col = resolve_column(
    audit,
    [
        "prediction_confidence_v6",
        "v6_confidence",
    ],
    "V6 confidence"
)

print()


# =============================================================================
# 4. RESOLVE REVIEW COLUMNS
# =============================================================================

print("=" * 100)
print("4. RESOLVING REVIEW COLUMNS")
print("=" * 100)

review_filename_col = resolve_column(
    review,
    ["filename_v5", "filename"],
    "Review filename"
)

review_decision_col = resolve_column(
    review,
    ["manual_decision", "decision"],
    "Manual decision"
)

review_label_col = resolve_column(
    review,
    ["manual_label", "reviewed_label", "label_manual"],
    "Manual label"
)

review_notes_col = None

for col in ["manual_notes", "review_notes", "notes"]:
    if col in review.columns:
        review_notes_col = col
        print(f"[OK] Review notes    : {col}")
        break

if review_notes_col is None:
    print("[INFO] Review notes tidak tersedia.")

print()


# =============================================================================
# 5. NORMALIZATION
# =============================================================================

print("=" * 100)
print("5. NORMALIZING DATA")
print("=" * 100)

audit["_filename_norm"] = audit[audit_filename_col].apply(
    normalize_filename
)

audit["_sequence_norm"] = audit[audit_sequence_col].apply(
    normalize_sequence
)

audit["_dataset_label_norm"] = audit[audit_label_col].apply(
    normalize_label
)

audit["_v5_norm"] = audit[audit_v5_pred_col].apply(
    normalize_label
)

audit["_v6_norm"] = audit[audit_v6_pred_col].apply(
    normalize_label
)

review["_filename_norm"] = review[review_filename_col].apply(
    normalize_filename
)

review["_manual_label_norm"] = review[review_label_col].apply(
    normalize_label
)

review["_manual_decision_norm"] = (
    review[review_decision_col]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("[OK] Normalization completed.")
print()


# =============================================================================
# 6. FILTER SEQUENCE
# =============================================================================

print("=" * 100)
print(f"6. FILTERING {SEQUENCE}")
print("=" * 100)

audit_seq = audit[
    audit["_sequence_norm"] == SEQUENCE
].copy()

review_seq = review[
    review["_filename_norm"].isin(
        audit_seq["_filename_norm"]
    )
].copy()

print(f"Sequence rows       : {len(audit_seq)}")
print(f"Manual review rows  : {len(review_seq)}")
print()


# =============================================================================
# 7. DUPLICATE CHECK
# =============================================================================

print("=" * 100)
print("7. CHECKING DUPLICATE FILENAMES")
print("=" * 100)

audit_duplicates = (
    audit_seq["_filename_norm"]
    .duplicated()
    .sum()
)

review_duplicates = (
    review_seq["_filename_norm"]
    .duplicated()
    .sum()
)

print(f"Audit duplicate filenames  : {audit_duplicates}")
print(f"Review duplicate filenames : {review_duplicates}")

if audit_duplicates == 0 and review_duplicates == 0:
    print("[OK] Tidak ada duplicate filename.")
else:
    print("[WARNING] Duplicate filename ditemukan.")

print()


# =============================================================================
# 8. MERGE AUDIT + MANUAL REVIEW
# =============================================================================

print("=" * 100)
print("8. MERGING AUDIT + MANUAL REVIEW")
print("=" * 100)

review_columns = [
    "_filename_norm",
    "_manual_label_norm",
    "_manual_decision_norm",
]

if review_notes_col:
    review_columns.append(review_notes_col)

review_merge = review_seq[review_columns].copy()

merged = audit_seq.merge(
    review_merge,
    on="_filename_norm",
    how="left",
    indicator=True
)

print(f"Merged rows : {len(merged)}")

missing_review = (
    merged["_merge"] != "both"
).sum()

if missing_review == 0:
    print("[OK] Semua sample memiliki manual review.")
else:
    print(
        f"[WARNING] Sample tanpa manual review : "
        f"{missing_review}"
    )

print()


# =============================================================================
# 9. MANUAL REVIEW VALIDATION
# =============================================================================

print("=" * 100)
print("9. MANUAL REVIEW VALIDATION")
print("=" * 100)

print()

print("Manual decisions:")
print(
    merged["_manual_decision_norm"]
    .value_counts(dropna=False)
)

print()

print("Manual labels:")
print(
    merged["_manual_label_norm"]
    .value_counts(dropna=False)
)

print()


# =============================================================================
# 10. DEFINE CLEAR VS AMBIGUOUS
# =============================================================================

print("=" * 100)
print("10. DEFINING CLEAR VS AMBIGUOUS SAMPLES")
print("=" * 100)

ambiguous_mask = (
    merged["_manual_label_norm"].isin(
        ["ambiguous", None]
    )
    |
    merged["_manual_decision_norm"].isin(
        ["A", "AMBIGUOUS"]
    )
)

merged["is_ambiguous"] = ambiguous_mask

clear = merged[
    ~merged["is_ambiguous"]
].copy()

ambiguous = merged[
    merged["is_ambiguous"]
].copy()

print(f"Total samples : {len(merged)}")
print(f"Clear samples : {len(clear)}")
print(f"Ambiguous     : {len(ambiguous)}")
print()


# =============================================================================
# 11. CORRECTNESS
# =============================================================================

print("=" * 100)
print("11. POST-REVIEW ACCURACY")
print("=" * 100)

clear["_v5_correct"] = (
    clear["_v5_norm"]
    == clear["_manual_label_norm"]
)

clear["_v6_correct"] = (
    clear["_v6_norm"]
    == clear["_manual_label_norm"]
)

clear["_dataset_correct"] = (
    clear["_dataset_label_norm"]
    == clear["_manual_label_norm"]
)

v5_correct = int(clear["_v5_correct"].sum())
v6_correct = int(clear["_v6_correct"].sum())
dataset_correct = int(clear["_dataset_correct"].sum())

clear_count = len(clear)

print(
    f"V5 correct vs manual     : "
    f"{v5_correct}/{clear_count} "
    f"({percentage(v5_correct, clear_count):.2f}%)"
)

print(
    f"V6 correct vs manual     : "
    f"{v6_correct}/{clear_count} "
    f"({percentage(v6_correct, clear_count):.2f}%)"
)

print(
    f"Dataset label correct    : "
    f"{dataset_correct}/{clear_count} "
    f"({percentage(dataset_correct, clear_count):.2f}%)"
)

print()


# =============================================================================
# 12. V5 → V6 CHANGE
# =============================================================================

print("=" * 100)
print("12. V5 → V6 CHANGE")
print("=" * 100)

clear["_changed"] = (
    clear["_v5_norm"]
    != clear["_v6_norm"]
)

clear["_improved"] = (
    (~clear["_v5_correct"])
    & clear["_v6_correct"]
)

clear["_regressed"] = (
    clear["_v5_correct"]
    & (~clear["_v6_correct"])
)

clear["_still_wrong"] = (
    (~clear["_v5_correct"])
    & (~clear["_v6_correct"])
)

both_correct = int(
    (
        clear["_v5_correct"]
        & clear["_v6_correct"]
    ).sum()
)

improved = int(clear["_improved"].sum())
regressed = int(clear["_regressed"].sum())
still_wrong = int(clear["_still_wrong"].sum())
changed = int(clear["_changed"].sum())

print(
    f"Changed V5→V6 : {changed}"
)

print(
    f"Correct in both : {both_correct}"
)

print(
    f"Improved V5→V6 : {improved}"
)

print(
    f"Regressed      : {regressed}"
)

print(
    f"Still wrong    : {still_wrong}"
)

print()


# =============================================================================
# 13. DATASET LABEL AUDIT
# =============================================================================

print("=" * 100)
print("13. DATASET LABEL AUDIT")
print("=" * 100)

dataset_mismatch = clear[
    ~clear["_dataset_correct"]
].copy()

print(
    f"Dataset-label mismatches : "
    f"{len(dataset_mismatch)}"
)

if len(dataset_mismatch) == 0:
    print(
        "PASS — all dataset labels agree "
        "with manual review."
    )
else:
    print(
        "WARNING — dataset labels disagree "
        "with manual review."
    )

print()


# =============================================================================
# 14. V6 ERROR ANALYSIS
# =============================================================================

print("=" * 100)
print("14. MODEL ERROR ANALYSIS")
print("=" * 100)

v6_errors = clear[
    ~clear["_v6_correct"]
].copy()

print(
    f"V6 errors vs manual : "
    f"{len(v6_errors)}"
)

if len(v6_errors) > 0:

    display_columns = [
        audit_filename_col,
        audit_label_col,
        audit_v5_pred_col,
        audit_v6_pred_col,
        audit_v5_conf_col,
        audit_v6_conf_col,
        "_manual_label_norm",
        "_manual_decision_norm",
    ]

    print()
    print("Remaining V6 errors:")

    print(
        v6_errors[
            display_columns
        ].to_string(index=False)
    )

print()


# =============================================================================
# 15. V6 ERROR TRANSITIONS
# =============================================================================

print("=" * 100)
print("15. V6 ERROR TYPE SUMMARY")
print("=" * 100)

if len(v6_errors) > 0:

    v6_errors["_error_type"] = (
        v6_errors["_manual_label_norm"]
        .astype(str)
        + " -> "
        + v6_errors["_v6_norm"].astype(str)
    )

    error_summary = (
        v6_errors["_error_type"]
        .value_counts()
        .reset_index()
    )

    error_summary.columns = [
        "error_type",
        "count",
    ]

    error_summary["percentage"] = (
        error_summary["count"]
        / len(v6_errors)
        * 100
    ).round(2)

    print(
        error_summary.to_string(index=False)
    )

else:

    error_summary = pd.DataFrame(
        columns=[
            "error_type",
            "count",
            "percentage",
        ]
    )

    print("No V6 errors.")

print()


# =============================================================================
# 16. V5 ERROR TRANSITIONS
# =============================================================================

print("=" * 100)
print("16. V5 ERROR TYPE SUMMARY")
print("=" * 100)

v5_errors = clear[
    ~clear["_v5_correct"]
].copy()

if len(v5_errors) > 0:

    v5_errors["_error_type"] = (
        v5_errors["_manual_label_norm"]
        .astype(str)
        + " -> "
        + v5_errors["_v5_norm"].astype(str)
    )

    v5_error_summary = (
        v5_errors["_error_type"]
        .value_counts()
        .reset_index()
    )

    v5_error_summary.columns = [
        "error_type",
        "count",
    ]

    v5_error_summary["percentage"] = (
        v5_error_summary["count"]
        / len(v5_errors)
        * 100
    ).round(2)

    print(
        v5_error_summary.to_string(index=False)
    )

else:

    v5_error_summary = pd.DataFrame(
        columns=[
            "error_type",
            "count",
            "percentage",
        ]
    )

    print("No V5 errors.")

print()


# =============================================================================
# 17. V5 → V6 TRANSITION MATRIX
# =============================================================================

print("=" * 100)
print("17. V5 → V6 TRANSITION MATRIX")
print("=" * 100)

transition_matrix = pd.crosstab(
    clear["_v5_norm"],
    clear["_v6_norm"]
)

print()
print("Rows = V5")
print("Columns = V6")
print()
print(transition_matrix)

save_csv(
    transition_matrix.reset_index(),
    "sequence_0125_v5_to_v6_transition_matrix.csv"
)

print()


# =============================================================================
# 18. V6 CONFUSION MATRIX VS MANUAL
# =============================================================================

print("=" * 100)
print("18. V6 CONFUSION MATRIX VS MANUAL")
print("=" * 100)

v6_confusion = pd.crosstab(
    clear["_manual_label_norm"],
    clear["_v6_norm"]
)

print()
print("Rows = MANUAL")
print("Columns = V6")
print()
print(v6_confusion)

save_csv(
    v6_confusion.reset_index(),
    "sequence_0125_v6_confusion_vs_manual.csv"
)

print()


# =============================================================================
# 19. V5 CONFUSION MATRIX VS MANUAL
# =============================================================================

print("=" * 100)
print("19. V5 CONFUSION MATRIX VS MANUAL")
print("=" * 100)

v5_confusion = pd.crosstab(
    clear["_manual_label_norm"],
    clear["_v5_norm"]
)

print()
print("Rows = MANUAL")
print("Columns = V5")
print()
print(v5_confusion)

save_csv(
    v5_confusion.reset_index(),
    "sequence_0125_v5_confusion_vs_manual.csv"
)

print()


# =============================================================================
# 20. DATASET VS MANUAL CONFUSION
# =============================================================================

print("=" * 100)
print("20. DATASET LABEL VS MANUAL CONFUSION")
print("=" * 100)

dataset_confusion = pd.crosstab(
    clear["_manual_label_norm"],
    clear["_dataset_label_norm"]
)

print()
print("Rows = MANUAL")
print("Columns = DATASET")
print()
print(dataset_confusion)

save_csv(
    dataset_confusion.reset_index(),
    "sequence_0125_dataset_vs_manual_confusion.csv"
)

print()


# =============================================================================
# 21. HIGH-CONFIDENCE V6 ERRORS
# =============================================================================

print("=" * 100)
print("21. HIGH-CONFIDENCE V6 ERRORS")
print("=" * 100)

v6_errors["_v6_confidence_numeric"] = pd.to_numeric(
    v6_errors[audit_v6_conf_col],
    errors="coerce"
)

high_conf_errors = v6_errors[
    v6_errors["_v6_confidence_numeric"] >= 0.80
].copy()

print(
    f"Total V6 errors              : "
    f"{len(v6_errors)}"
)

print(
    f"High-confidence V6 errors >=80% : "
    f"{len(high_conf_errors)}"
)

print(
    f"Percentage of V6 errors       : "
    f"{percentage(len(high_conf_errors), len(v6_errors)):.2f}%"
)

print()

save_csv(
    high_conf_errors,
    "sequence_0125_high_confidence_v6_errors.csv"
)


# =============================================================================
# 22. CONFIDENCE STATISTICS
# =============================================================================

print("=" * 100)
print("22. V6 ERROR CONFIDENCE ANALYSIS")
print("=" * 100)

if len(v6_errors) > 0:

    mean_conf = (
        v6_errors["_v6_confidence_numeric"]
        .mean()
    )

    median_conf = (
        v6_errors["_v6_confidence_numeric"]
        .median()
    )

    max_conf = (
        v6_errors["_v6_confidence_numeric"]
        .max()
    )

    min_conf = (
        v6_errors["_v6_confidence_numeric"]
        .min()
    )

    print(
        f"Mean V6 error confidence   : "
        f"{mean_conf:.4f}"
    )

    print(
        f"Median V6 error confidence : "
        f"{median_conf:.4f}"
    )

    print(
        f"Minimum V6 error confidence : "
        f"{min_conf:.4f}"
    )

    print(
        f"Maximum V6 error confidence : "
        f"{max_conf:.4f}"
    )

print()


# =============================================================================
# 23. MANUAL LABEL DISTRIBUTION
# =============================================================================

print("=" * 100)
print("23. MANUAL REVIEW DISTRIBUTION")
print("=" * 100)

manual_distribution = (
    clear["_manual_label_norm"]
    .value_counts()
    .reindex(
        [
            "ripe",
            "unripe",
            "old",
            "damaged",
            "dried",
        ],
        fill_value=0,
    )
)

for label, count in manual_distribution.items():

    print(
        f"{label.upper():<10}: "
        f"{count:>3} "
        f"({percentage(count, len(clear)):.2f}%)"
    )

print()


# =============================================================================
# 24. DATASET LABEL DISTRIBUTION
# =============================================================================

print("=" * 100)
print("24. DATASET LABEL DISTRIBUTION")
print("=" * 100)

dataset_distribution = (
    clear["_dataset_label_norm"]
    .value_counts()
)

for label, count in dataset_distribution.items():

    print(
        f"{str(label).upper():<10}: "
        f"{count:>3} "
        f"({percentage(count, len(clear)):.2f}%)"
    )

print()


# =============================================================================
# 25. V5 PREDICTION DISTRIBUTION
# =============================================================================

print("=" * 100)
print("25. V5 PREDICTION DISTRIBUTION")
print("=" * 100)

v5_distribution = (
    clear["_v5_norm"]
    .value_counts()
)

for label, count in v5_distribution.items():

    print(
        f"{str(label).upper():<10}: "
        f"{count:>3} "
        f"({percentage(count, len(clear)):.2f}%)"
    )

print()


# =============================================================================
# 26. V6 PREDICTION DISTRIBUTION
# =============================================================================

print("=" * 100)
print("26. V6 PREDICTION DISTRIBUTION")
print("=" * 100)

v6_distribution = (
    clear["_v6_norm"]
    .value_counts()
)

for label, count in v6_distribution.items():

    print(
        f"{str(label).upper():<10}: "
        f"{count:>3} "
        f"({percentage(count, len(clear)):.2f}%)"
    )

print()


# =============================================================================
# 27. CONFIRMED V6 DISAGREEMENTS
# =============================================================================

print("=" * 100)
print("27. CONFIRMED V6 DISAGREEMENTS")
print("=" * 100)

print(
    f"Confirmed V6 disagreements : "
    f"{len(v6_errors)}"
)

if len(v6_errors) > 0:

    for _, row in v6_errors.iterrows():

        print(
            f"{row[audit_filename_col]} | "
            f"manual={row['_manual_label_norm']} | "
            f"v6={row['_v6_norm']} | "
            f"confidence="
            f"{float(row['_v6_confidence_numeric']):.4f}"
        )

print()


# =============================================================================
# 28. CONFIRMED DATASET DISAGREEMENTS
# =============================================================================

print("=" * 100)
print("28. CONFIRMED DATASET LABEL DISAGREEMENTS")
print("=" * 100)

print(
    f"Confirmed dataset disagreements : "
    f"{len(dataset_mismatch)}"
)

if len(dataset_mismatch) > 0:

    for _, row in dataset_mismatch.iterrows():

        print(
            f"{row[audit_filename_col]} | "
            f"dataset={row['_dataset_label_norm']} | "
            f"manual={row['_manual_label_norm']}"
        )

print()


# =============================================================================
# 29. AMBIGUOUS SAMPLES
# =============================================================================

print("=" * 100)
print("29. AMBIGUOUS SAMPLES")
print("=" * 100)

print(
    f"Ambiguous samples : "
    f"{len(ambiguous)}"
)

if len(ambiguous) > 0:

    for _, row in ambiguous.iterrows():

        print(
            f"{row[audit_filename_col]} | "
            f"dataset={row['_dataset_label_norm']} | "
            f"v5={row['_v5_norm']} | "
            f"v6={row['_v6_norm']} | "
            f"manual={row['_manual_label_norm']}"
        )

print()


# =============================================================================
# 30. PRELIMINARY ROOT-CAUSE CLASSIFICATION
# =============================================================================

print("=" * 100)
print("30. PRELIMINARY ROOT-CAUSE CLASSIFICATION")
print("=" * 100)

dataset_disagreement_rate = percentage(
    len(dataset_mismatch),
    len(clear)
)

v6_error_rate = percentage(
    len(v6_errors),
    len(clear)
)

if len(clear) == 0:

    classification = "INSUFFICIENT_REVIEW"

elif dataset_disagreement_rate >= 30:

    classification = "LIKELY_DATASET_ISSUE"

elif (
    dataset_disagreement_rate < 10
    and v6_error_rate >= 30
):

    classification = "LIKELY_MODEL_ERROR"

else:

    classification = "MIXED_OR_UNCERTAIN"

print(
    f"Preliminary classification : "
    f"{classification}"
)

print()
print(
    "IMPORTANT:"
)
print(
    "Classification ini hanya preliminary."
)
print(
    "Jangan mengubah dataset hanya berdasarkan "
    f"{SEQUENCE}."
)
print(
    "Bandingkan dengan sequence_0048, sequence_0136, "
    "dan sequence lain terlebih dahulu."
)

print()


# =============================================================================
# 31. EXPORT MERGED DATA
# =============================================================================

print("=" * 100)
print("31. EXPORTING ANALYSIS FILES")
print("=" * 100)

export_columns = [
    audit_filename_col,
    audit_sequence_col,
    audit_label_col,
    audit_v5_pred_col,
    audit_v6_pred_col,
    audit_v5_conf_col,
    audit_v6_conf_col,
    "_manual_label_norm",
    "_manual_decision_norm",
    "is_ambiguous",
]

if review_notes_col:
    export_columns.append(review_notes_col)

merged_export = merged[
    export_columns
].copy()

merged_export = merged_export.rename(
    columns={
        "_manual_label_norm": "manual_label_normalized",
        "_manual_decision_norm": "manual_decision_normalized",
    }
)

save_csv(
    merged_export,
    "sequence_0125_post_review_merged.csv"
)

save_csv(
    v6_errors,
    "sequence_0125_v6_confirmed_errors.csv"
)

save_csv(
    dataset_mismatch,
    "sequence_0125_dataset_label_disagreements.csv"
)

save_csv(
    ambiguous,
    "sequence_0125_ambiguous_samples.csv"
)

print()


# =============================================================================
# 32. SUMMARY CSV
# =============================================================================

print("=" * 100)
print("32. CREATING SUMMARY CSV")
print("=" * 100)

summary = pd.DataFrame(
    [
        {
            "sequence": SEQUENCE,
            "total_samples": len(merged),
            "clear_samples": len(clear),
            "ambiguous_samples": len(ambiguous),
            "v5_correct": v5_correct,
            "v5_errors": len(clear) - v5_correct,
            "v5_accuracy_pct": percentage(
                v5_correct,
                len(clear)
            ),
            "v6_correct": v6_correct,
            "v6_errors": len(v6_errors),
            "v6_accuracy_pct": percentage(
                v6_correct,
                len(clear)
            ),
            "v6_error_rate_pct": percentage(
                len(v6_errors),
                len(clear)
            ),
            "dataset_correct": dataset_correct,
            "dataset_mismatches": len(dataset_mismatch),
            "dataset_agreement_pct": percentage(
                dataset_correct,
                len(clear)
            ),
            "v5_v6_changed": changed,
            "correct_in_both": both_correct,
            "improvements": improved,
            "regressions": regressed,
            "still_wrong": still_wrong,
            "high_confidence_v6_errors_ge_080": len(
                high_conf_errors
            ),
            "preliminary_classification": classification,
        }
    ]
)

save_csv(
    summary,
    "sequence_0125_post_review_summary.csv"
)

print()


# =============================================================================
# 33. FINAL SUMMARY
# =============================================================================

print("=" * 100)
print("33. SEQUENCE 0125 POST-REVIEW ANALYSIS COMPLETED")
print("=" * 100)

print()
print(f"Sequence              : {SEQUENCE}")
print(f"Total samples         : {len(merged)}")
print(f"Clear samples         : {len(clear)}")
print(f"Ambiguous samples     : {len(ambiguous)}")

print(
    f"Dataset disagreements : "
    f"{len(dataset_mismatch)}"
)

print(
    f"V5 disagreements      : "
    f"{len(clear) - v5_correct}"
)

print(
    f"V6 disagreements      : "
    f"{len(v6_errors)}"
)

print(
    f"V6 error rate         : "
    f"{percentage(len(v6_errors), len(clear)):.2f}%"
)

print(
    f"V5 → V6 changed       : "
    f"{changed}"
)

print(
    f"V5 → V6 regressions   : "
    f"{regressed}"
)

print(
    f"V5 → V6 improvements  : "
    f"{improved}"
)

print(
    f"High-conf V6 errors   : "
    f"{len(high_conf_errors)}"
)

print(
    f"Classification        : "
    f"{classification}"
)

print()
print("OUTPUT FILES:")

print(
    f" - {OUTPUT_DIR}\\sequence_0125_post_review_merged.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_v6_confirmed_errors.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_dataset_label_disagreements.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_ambiguous_samples.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_post_review_summary.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_v5_to_v6_transition_matrix.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_v6_confusion_vs_manual.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_v5_confusion_vs_manual.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_dataset_vs_manual_confusion.csv"
)

print(
    f" - {OUTPUT_DIR}\\sequence_0125_high_confidence_v6_errors.csv"
)

print()
print("NEXT STEP:")
print("1. Inspect hasil sequence_0125.")
print("2. Bandingkan dengan sequence_0048 dan sequence_0136.")
print("3. Fokus pada high-confidence V6 errors.")
print("4. Cari apakah pola visual/source sama.")
print("5. Jangan train V7 dulu.")
print()
print("=" * 100)
print("SEQUENCE 0125 POST-REVIEW ANALYSIS COMPLETED")
print("=" * 100)