from pathlib import Path
import pandas as pd


# =============================================================================
# CIRCULENS QUALITY — SEQUENCE 0048 POST-REVIEW ANALYSIS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIT_CSV = (
    BASE_DIR
    / "reports"
    / "quality_finetune_v6"
    / "v5_v6_error_analysis"
    / "label_audit"
    / "audit_master_v5_v6.csv"
)

REVIEWED_CSV = (
    BASE_DIR
    / "reports"
    / "quality_finetune_v6"
    / "v5_v6_error_analysis"
    / "label_audit"
    / "manual_review_report"
    / "sequence_0048_review"
    / "sequence_0048_manual_decision_reviewed.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "quality_finetune_v6"
    / "v5_v6_error_analysis"
    / "label_audit"
    / "manual_review_report"
    / "sequence_0048_review"
    / "post_review_analysis"
)

TARGET_SEQUENCE = "sequence_0048"

WIDTH = 90


# =============================================================================
# HELPERS
# =============================================================================

def title(text):
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def section(number, text):
    print()
    print("=" * WIDTH)
    print(f"{number}. {text}")
    print("=" * WIDTH)


def pct(value, total):
    if total == 0:
        return 0.0
    return value / total * 100.0


def normalize_label(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    mapping = {
        "r": "ripe",
        "ripe": "ripe",

        "u": "unripe",
        "unripe": "unripe",

        "o": "old",
        "old": "old",

        "d": "damaged",
        "damaged": "damaged",

        "k": "ambiguous",
        "ambiguous": "ambiguous",
        "ambiguous / not sure": "ambiguous",
    }

    return mapping.get(value, value)


def find_column(df, candidates, required=True):
    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lower_map:
            return lower_map[key]

    if required:
        raise KeyError(
            f"Column tidak ditemukan.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# =============================================================================
# HEADER
# =============================================================================

title(
    "CIRCULENS QUALITY — SEQUENCE 0048 POST-REVIEW ANALYSIS"
)

print(f"Audit CSV    : {AUDIT_CSV}")
print(f"Reviewed CSV : {REVIEWED_CSV}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"Target       : {TARGET_SEQUENCE}")


# =============================================================================
# 1. INPUT CHECK
# =============================================================================

section(1, "CHECKING INPUT FILES")

if not AUDIT_CSV.exists():
    raise FileNotFoundError(
        f"Audit CSV tidak ditemukan:\n{AUDIT_CSV}"
    )

if not REVIEWED_CSV.exists():
    raise FileNotFoundError(
        f"Reviewed CSV tidak ditemukan:\n{REVIEWED_CSV}"
    )

print("[OK] Audit CSV ditemukan.")
print("[OK] Reviewed CSV ditemukan.")


# =============================================================================
# 2. LOAD DATA
# =============================================================================

section(2, "LOADING DATA")

audit = pd.read_csv(AUDIT_CSV)
review = pd.read_csv(REVIEWED_CSV)

print(f"Audit rows    : {len(audit)}")
print(f"Reviewed rows : {len(review)}")


# =============================================================================
# 3. RESOLVE AUDIT COLUMNS
# =============================================================================

section(3, "RESOLVING AUDIT COLUMNS")

# IMPORTANT:
# audit_master_v5_v6.csv menggunakan:
#
# filename_v5
# sequence_group_v5
# label_v5
# prediction_label_v5
#
# dan pasangan V6.
#
# Untuk sequence_0048, kita menggunakan struktur V5 karena
# sequence tersebut berasal dari audit sequence V5.

audit_filename_col = find_column(
    audit,
    [
        "filename_v5",
        "filename",
        "file_name",
        "image_filename",
        "image_name",
        "basename",
    ],
)

audit_sequence_col = find_column(
    audit,
    [
        "sequence_group_v5",
        "sequence",
        "sequence_id",
        "sequence_name",
    ],
)

audit_label_col = find_column(
    audit,
    [
        "label_v5",
        "dataset_label",
        "label",
    ],
)

audit_v5_pred_col = find_column(
    audit,
    [
        "prediction_label_v5",
        "v5_prediction",
        "predicted_label_v5",
        "prediction_v5",
    ],
)

audit_v6_pred_col = find_column(
    audit,
    [
        "prediction_label_v6",
        "v6_prediction",
        "predicted_label_v6",
        "prediction_v6",
    ],
)

print(f"[OK] Audit filename   : {audit_filename_col}")
print(f"[OK] Audit sequence   : {audit_sequence_col}")
print(f"[OK] Dataset label    : {audit_label_col}")
print(f"[OK] V5 prediction    : {audit_v5_pred_col}")
print(f"[OK] V6 prediction    : {audit_v6_pred_col}")


# =============================================================================
# 4. RESOLVE REVIEW COLUMNS
# =============================================================================

section(4, "RESOLVING REVIEW COLUMNS")

review_filename_col = find_column(
    review,
    [
        "filename",
        "filename_v5",
        "file_name",
        "image_filename",
        "image_name",
        "basename",
    ],
)

review_sequence_col = find_column(
    review,
    [
        "sequence",
        "sequence_group",
        "sequence_id",
        "sequence_name",
    ],
    required=False,
)

review_decision_col = find_column(
    review,
    [
        "manual_decision",
        "decision",
    ],
)

review_label_col = find_column(
    review,
    [
        "manual_label",
        "actual_label",
    ],
)

review_notes_col = find_column(
    review,
    [
        "manual_notes",
        "notes",
        "review_notes",
    ],
    required=False,
)

print(f"[OK] Review filename : {review_filename_col}")
print(f"[OK] Manual decision : {review_decision_col}")
print(f"[OK] Manual label    : {review_label_col}")

if review_notes_col:
    print(f"[OK] Manual notes    : {review_notes_col}")


# =============================================================================
# 5. FILTER TARGET SEQUENCE
# =============================================================================

section(5, f"FILTERING {TARGET_SEQUENCE}")

audit_seq = audit[
    audit[audit_sequence_col]
    .astype(str)
    .str.strip()
    == TARGET_SEQUENCE
].copy()

print(f"Audit sequence rows : {len(audit_seq)}")

if len(audit_seq) == 0:
    raise ValueError(
        f"Tidak ada data untuk {TARGET_SEQUENCE} di audit."
    )

# Review CSV sudah khusus sequence_0048.
# Jika sequence column tersedia, filter juga.
review_seq = review.copy()

if review_sequence_col is not None:

    review_seq = review_seq[
        review_seq[review_sequence_col]
        .astype(str)
        .str.strip()
        == TARGET_SEQUENCE
    ].copy()

print(f"Review sequence rows : {len(review_seq)}")


# =============================================================================
# 6. NORMALIZE DATA
# =============================================================================

section(6, "PREPARING NORMALIZED DATA")

audit_seq["_filename"] = (
    audit_seq[audit_filename_col]
    .astype(str)
    .str.strip()
)

audit_seq["_dataset_label"] = (
    audit_seq[audit_label_col]
    .apply(normalize_label)
)

audit_seq["_v5"] = (
    audit_seq[audit_v5_pred_col]
    .apply(normalize_label)
)

audit_seq["_v6"] = (
    audit_seq[audit_v6_pred_col]
    .apply(normalize_label)
)

review_seq["_filename"] = (
    review_seq[review_filename_col]
    .astype(str)
    .str.strip()
)

review_seq["_manual_decision"] = (
    review_seq[review_decision_col]
    .apply(normalize_label)
)

review_seq["_manual_label"] = (
    review_seq[review_label_col]
    .apply(normalize_label)
)

print("[OK] Normalization completed.")


# =============================================================================
# 7. CHECK DUPLICATES
# =============================================================================

section(7, "CHECKING DUPLICATE FILENAMES")

audit_duplicates = (
    audit_seq["_filename"]
    .duplicated()
    .sum()
)

review_duplicates = (
    review_seq["_filename"]
    .duplicated()
    .sum()
)

print(f"Audit duplicate filenames  : {audit_duplicates}")
print(f"Review duplicate filenames : {review_duplicates}")

if audit_duplicates > 0:
    raise ValueError(
        "Audit memiliki duplicate filename."
    )

if review_duplicates > 0:
    raise ValueError(
        "Reviewed CSV memiliki duplicate filename."
    )

print("[OK] Tidak ada duplicate filename.")


# =============================================================================
# 8. MERGE AUDIT + MANUAL REVIEW
# =============================================================================

section(8, "MERGING AUDIT WITH MANUAL REVIEW")

audit_merge = audit_seq[
    [
        "_filename",
        "_dataset_label",
        "_v5",
        "_v6",
    ]
].copy()

review_merge_cols = [
    "_filename",
    "_manual_decision",
    "_manual_label",
]

if review_notes_col is not None:
    review_seq["_manual_notes"] = (
        review[review_notes_col]
        if review_notes_col in review.columns
        else ""
    )

merged = audit_merge.merge(
    review_seq[review_merge_cols],
    on="_filename",
    how="left",
    validate="one_to_one",
)

print(f"Merged rows : {len(merged)}")

missing_manual = (
    merged["_manual_label"]
    .isna()
    .sum()
)

if missing_manual > 0:
    raise ValueError(
        f"{missing_manual} sample tidak memiliki manual review."
    )

print("[OK] Semua sample memiliki manual review.")


# =============================================================================
# 9. DEFINE CLEAR / AMBIGUOUS
# =============================================================================

section(9, "DEFINING CLEAR VS AMBIGUOUS SAMPLES")

merged["is_ambiguous"] = (
    merged["_manual_label"]
    == "ambiguous"
)

merged["is_clear"] = ~merged["is_ambiguous"]

clear = merged[
    merged["is_clear"]
].copy()

ambiguous = merged[
    merged["is_ambiguous"]
].copy()

print(f"Total samples : {len(merged)}")
print(f"Clear samples : {len(clear)}")
print(f"Ambiguous     : {len(ambiguous)}")


# =============================================================================
# 10. DATASET LABEL VS MANUAL
# =============================================================================

section(10, "DATASET LABEL VS MANUAL REVIEW")

dataset_match = (
    clear["_dataset_label"]
    == clear["_manual_label"]
)

dataset_correct = int(
    dataset_match.sum()
)

dataset_wrong = int(
    (~dataset_match).sum()
)

print(
    f"Dataset label matches manual : "
    f"{dataset_correct}"
)

print(
    f"Dataset label disagrees      : "
    f"{dataset_wrong}"
)

print(
    f"Dataset agreement            : "
    f"{pct(dataset_correct, len(clear)):.2f}%"
)

print(
    f"Dataset disagreement         : "
    f"{pct(dataset_wrong, len(clear)):.2f}%"
)


# =============================================================================
# 11. V5 VS MANUAL
# =============================================================================

section(11, "V5 VS MANUAL REVIEW")

v5_match = (
    clear["_v5"]
    == clear["_manual_label"]
)

v5_correct = int(
    v5_match.sum()
)

v5_wrong = int(
    (~v5_match).sum()
)

print(f"V5 matches manual : {v5_correct}")
print(f"V5 disagrees      : {v5_wrong}")

print(
    f"V5 accuracy       : "
    f"{pct(v5_correct, len(clear)):.2f}%"
)


# =============================================================================
# 12. V6 VS MANUAL
# =============================================================================

section(12, "V6 VS MANUAL REVIEW")

v6_match = (
    clear["_v6"]
    == clear["_manual_label"]
)

v6_correct = int(
    v6_match.sum()
)

v6_wrong = int(
    (~v6_match).sum()
)

print(f"V6 matches manual : {v6_correct}")
print(f"V6 disagrees      : {v6_wrong}")

print(
    f"V6 accuracy       : "
    f"{pct(v6_correct, len(clear)):.2f}%"
)

print(
    f"V6 error rate     : "
    f"{pct(v6_wrong, len(clear)):.2f}%"
)


# =============================================================================
# 13. MANUAL DISTRIBUTION
# =============================================================================

section(13, "MANUAL REVIEW DISTRIBUTION")

manual_distribution = (
    merged["_manual_label"]
    .value_counts()
    .reindex(
        [
            "ripe",
            "unripe",
            "old",
            "damaged",
            "ambiguous",
        ],
        fill_value=0,
    )
)

for label, count in manual_distribution.items():

    print(
        f"{label.upper():10s}: "
        f"{count:3d} "
        f"({pct(count, len(merged)):.2f}%)"
    )


# =============================================================================
# 14. DATASET DISTRIBUTION
# =============================================================================

section(14, "DATASET LABEL DISTRIBUTION")

dataset_distribution = (
    merged["_dataset_label"]
    .value_counts()
)

for label, count in dataset_distribution.items():

    print(
        f"{label.upper():10s}: "
        f"{count:3d}"
    )


# =============================================================================
# 15. V5 DISTRIBUTION
# =============================================================================

section(15, "V5 PREDICTION DISTRIBUTION")

v5_distribution = (
    merged["_v5"]
    .value_counts()
)

for label, count in v5_distribution.items():

    print(
        f"{label.upper():10s}: "
        f"{count:3d}"
    )


# =============================================================================
# 16. V6 DISTRIBUTION
# =============================================================================

section(16, "V6 PREDICTION DISTRIBUTION")

v6_distribution = (
    merged["_v6"]
    .value_counts()
)

for label, count in v6_distribution.items():

    print(
        f"{label.upper():10s}: "
        f"{count:3d}"
    )


# =============================================================================
# 17. CONFUSION MATRIX V6
# =============================================================================

section(17, "V6 CONFUSION MATRIX VS MANUAL")

labels = [
    "ripe",
    "unripe",
    "old",
]

v6_clear = clear[
    clear["_manual_label"].isin(labels)
    & clear["_v6"].isin(labels)
].copy()

confusion_v6 = pd.crosstab(
    v6_clear["_manual_label"],
    v6_clear["_v6"],
)

confusion_v6 = confusion_v6.reindex(
    index=labels,
    columns=labels,
    fill_value=0,
)

print()
print("Rows    = MANUAL")
print("Columns = V6")
print()

print(confusion_v6.to_string())


# =============================================================================
# 18. CONFUSION MATRIX V5
# =============================================================================

section(18, "V5 CONFUSION MATRIX VS MANUAL")

v5_clear = clear[
    clear["_manual_label"].isin(labels)
    & clear["_v5"].isin(labels)
].copy()

confusion_v5 = pd.crosstab(
    v5_clear["_manual_label"],
    v5_clear["_v5"],
)

confusion_v5 = confusion_v5.reindex(
    index=labels,
    columns=labels,
    fill_value=0,
)

print()
print("Rows    = MANUAL")
print("Columns = V5")
print()

print(confusion_v5.to_string())


# =============================================================================
# 19. DATASET CONFUSION
# =============================================================================

section(19, "DATASET LABEL VS MANUAL CONFUSION")

dataset_clear = clear[
    clear["_manual_label"].isin(labels)
    & clear["_dataset_label"].isin(labels)
].copy()

confusion_dataset = pd.crosstab(
    dataset_clear["_manual_label"],
    dataset_clear["_dataset_label"],
)

confusion_dataset = confusion_dataset.reindex(
    index=labels,
    columns=labels,
    fill_value=0,
)

print()
print("Rows    = MANUAL")
print("Columns = DATASET")
print()

print(confusion_dataset.to_string())


# =============================================================================
# 20. V6 ERROR TYPES
# =============================================================================

section(20, "V6 ERROR TYPE SUMMARY")

v6_errors = clear[
    clear["_v6"]
    != clear["_manual_label"]
].copy()

if len(v6_errors) == 0:

    print("Tidak ada V6 disagreement.")

else:

    v6_errors["error_type"] = (
        v6_errors["_manual_label"]
        + " -> "
        + v6_errors["_v6"]
    )

    error_summary = (
        v6_errors["error_type"]
        .value_counts()
    )

    for error_type, count in error_summary.items():

        print(
            f"{error_type:25s}: "
            f"{count}"
        )


# =============================================================================
# 21. DATASET ERROR TYPES
# =============================================================================

section(21, "DATASET LABEL ERROR TYPE")

dataset_errors = clear[
    clear["_dataset_label"]
    != clear["_manual_label"]
].copy()

if len(dataset_errors) == 0:

    print(
        "Tidak ada dataset-label disagreement."
    )

else:

    dataset_errors["error_type"] = (
        dataset_errors["_dataset_label"]
        + " -> "
        + dataset_errors["_manual_label"]
    )

    dataset_error_summary = (
        dataset_errors["error_type"]
        .value_counts()
    )

    for error_type, count in dataset_error_summary.items():

        print(
            f"{error_type:25s}: "
            f"{count}"
        )


# =============================================================================
# 22. CONFIRMED V6 DISAGREEMENTS
# =============================================================================

section(22, "CONFIRMED V6 DISAGREEMENTS")

if len(v6_errors) == 0:

    print(
        "Tidak ada confirmed V6 disagreement."
    )

else:

    print(
        f"Confirmed V6 disagreements : "
        f"{len(v6_errors)}"
    )

    print()

    for _, row in v6_errors.iterrows():

        print(
            f"{row['_filename']} | "
            f"manual={row['_manual_label']} | "
            f"v6={row['_v6']}"
        )


# =============================================================================
# 23. DATASET DISAGREEMENTS
# =============================================================================

section(23, "CONFIRMED DATASET LABEL DISAGREEMENTS")

if len(dataset_errors) == 0:

    print(
        "Tidak ada confirmed dataset disagreement."
    )

else:

    print(
        f"Confirmed dataset disagreements : "
        f"{len(dataset_errors)}"
    )

    print()

    for _, row in dataset_errors.iterrows():

        print(
            f"{row['_filename']} | "
            f"dataset={row['_dataset_label']} | "
            f"manual={row['_manual_label']}"
        )


# =============================================================================
# 24. AMBIGUOUS SAMPLES
# =============================================================================

section(24, "AMBIGUOUS SAMPLES")

if len(ambiguous) == 0:

    print("Tidak ada ambiguous sample.")

else:

    print(
        f"Ambiguous samples : "
        f"{len(ambiguous)}"
    )

    print()

    for _, row in ambiguous.iterrows():

        print(
            f"{row['_filename']} | "
            f"dataset={row['_dataset_label']} | "
            f"v5={row['_v5']} | "
            f"v6={row['_v6']}"
        )


# =============================================================================
# 25. RIPE ANALYSIS
# =============================================================================

section(25, "MANUAL RIPE ERROR ANALYSIS")

manual_ripe = clear[
    clear["_manual_label"]
    == "ripe"
].copy()

print(
    f"Manual RIPE samples : "
    f"{len(manual_ripe)}"
)

if len(manual_ripe) > 0:

    v6_on_ripe = (
        manual_ripe["_v6"]
        .value_counts()
        .reindex(
            [
                "ripe",
                "unripe",
                "old",
            ],
            fill_value=0,
        )
    )

    print()
    print(
        "V6 predictions on manually confirmed RIPE:"
    )

    for label, count in v6_on_ripe.items():

        print(
            f"  RIPE -> {label.upper():7s}: "
            f"{count}"
        )


# =============================================================================
# 26. UNRIPE ANALYSIS
# =============================================================================

section(26, "MANUAL UNRIPE ERROR ANALYSIS")

manual_unripe = clear[
    clear["_manual_label"]
    == "unripe"
].copy()

print(
    f"Manual UNRIPE samples : "
    f"{len(manual_unripe)}"
)

if len(manual_unripe) > 0:

    v6_on_unripe = (
        manual_unripe["_v6"]
        .value_counts()
        .reindex(
            [
                "ripe",
                "unripe",
                "old",
            ],
            fill_value=0,
        )
    )

    print()
    print(
        "V6 predictions on manually confirmed UNRIPE:"
    )

    for label, count in v6_on_unripe.items():

        print(
            f"  UNRIPE -> {label.upper():7s}: "
            f"{count}"
        )


# =============================================================================
# 27. KEY FINDINGS
# =============================================================================

section(27, "KEY FINDINGS")

clear_total = len(clear)
ambiguous_total = len(ambiguous)

dataset_error_rate = pct(
    dataset_wrong,
    clear_total,
)

v5_error_rate = pct(
    v5_wrong,
    clear_total,
)

v6_error_rate = pct(
    v6_wrong,
    clear_total,
)

print()

print("Manual review:")
print(
    f" - Clear samples     : "
    f"{clear_total}"
)

print(
    f" - Ambiguous samples : "
    f"{ambiguous_total}"
)

print()

print("Dataset:")
print(
    f" - Disagreement : "
    f"{dataset_wrong}/{clear_total} "
    f"({dataset_error_rate:.2f}%)"
)

print()

print("V5:")
print(
    f" - Disagreement : "
    f"{v5_wrong}/{clear_total} "
    f"({v5_error_rate:.2f}%)"
)

print()

print("V6:")
print(
    f" - Disagreement : "
    f"{v6_wrong}/{clear_total} "
    f"({v6_error_rate:.2f}%)"
)

if len(v6_errors) > 0:

    top_v6_errors = (
        v6_errors["error_type"]
        .value_counts()
    )

    top_v6_error = (
        top_v6_errors.index[0]
    )

    top_v6_error_count = (
        top_v6_errors.iloc[0]
    )

    print()

    print(
        f"Dominant V6 error : "
        f"{top_v6_error} "
        f"({top_v6_error_count})"
    )

if len(dataset_errors) > 0:

    top_dataset_errors = (
        dataset_errors["error_type"]
        .value_counts()
    )

    top_dataset_error = (
        top_dataset_errors.index[0]
    )

    top_dataset_error_count = (
        top_dataset_errors.iloc[0]
    )

    print()

    print(
        f"Dominant dataset disagreement : "
        f"{top_dataset_error} "
        f"({top_dataset_error_count})"
    )


# =============================================================================
# 28. ROOT CAUSE
# =============================================================================

section(28, "PRELIMINARY ROOT-CAUSE CLASSIFICATION")

dataset_disagreement_ratio = (
    dataset_wrong / clear_total
    if clear_total > 0
    else 0
)

v6_disagreement_ratio = (
    v6_wrong / clear_total
    if clear_total > 0
    else 0
)

if (
    dataset_disagreement_ratio >= 0.50
    and v6_disagreement_ratio >= 0.50
):

    classification = (
        "MIXED / DATASET + MODEL ERROR"
    )

elif dataset_disagreement_ratio >= 0.50:

    classification = (
        "LIKELY DATASET LABEL ERROR"
    )

elif v6_disagreement_ratio >= 0.50:

    classification = (
        "LIKELY MODEL ERROR"
    )

else:

    classification = (
        "NO STRONG ROOT-CAUSE CONCLUSION YET"
    )

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
    "sequence_0048."
)

print(
    "Bandingkan dengan sequence_0136 dan "
    "sequence lain terlebih dahulu."
)


# =============================================================================
# 29. EXPORT
# =============================================================================

section(29, "EXPORTING ANALYSIS FILES")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

merged_output = (
    OUTPUT_DIR
    / "sequence_0048_post_review_merged.csv"
)

v6_error_output = (
    OUTPUT_DIR
    / "sequence_0048_v6_confirmed_errors.csv"
)

dataset_error_output = (
    OUTPUT_DIR
    / "sequence_0048_dataset_label_disagreements.csv"
)

ambiguous_output = (
    OUTPUT_DIR
    / "sequence_0048_ambiguous_samples.csv"
)

summary_output = (
    OUTPUT_DIR
    / "sequence_0048_post_review_summary.csv"
)


merged.to_csv(
    merged_output,
    index=False,
)

print(
    f"[SAVED] {merged_output}"
)


v6_errors.to_csv(
    v6_error_output,
    index=False,
)

print(
    f"[SAVED] {v6_error_output}"
)


dataset_errors.to_csv(
    dataset_error_output,
    index=False,
)

print(
    f"[SAVED] {dataset_error_output}"
)


ambiguous.to_csv(
    ambiguous_output,
    index=False,
)

print(
    f"[SAVED] {ambiguous_output}"
)


# =============================================================================
# 30. SUMMARY CSV
# =============================================================================

section(30, "CREATING SUMMARY CSV")

summary_rows = [
    {
        "sequence": TARGET_SEQUENCE,

        "total_samples": len(merged),

        "clear_samples": clear_total,

        "ambiguous_samples": ambiguous_total,

        "manual_ripe": int(
            (
                merged["_manual_label"]
                == "ripe"
            ).sum()
        ),

        "manual_unripe": int(
            (
                merged["_manual_label"]
                == "unripe"
            ).sum()
        ),

        "manual_old": int(
            (
                merged["_manual_label"]
                == "old"
            ).sum()
        ),

        "manual_damaged": int(
            (
                merged["_manual_label"]
                == "damaged"
            ).sum()
        ),

        "dataset_disagreements": dataset_wrong,

        "dataset_agreement_pct": round(
            pct(
                dataset_correct,
                clear_total,
            ),
            2,
        ),

        "v5_disagreements": v5_wrong,

        "v5_accuracy_pct": round(
            pct(
                v5_correct,
                clear_total,
            ),
            2,
        ),

        "v6_disagreements": v6_wrong,

        "v6_accuracy_pct": round(
            pct(
                v6_correct,
                clear_total,
            ),
            2,
        ),

        "v6_error_rate_pct": round(
            pct(
                v6_wrong,
                clear_total,
            ),
            2,
        ),

        "preliminary_root_cause": classification,
    }
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    summary_output,
    index=False,
)

print(
    f"[SAVED] {summary_output}"
)


# =============================================================================
# 31. FINAL
# =============================================================================

section(
    31,
    "SEQUENCE 0048 POST-REVIEW ANALYSIS COMPLETED"
)

print()

print(
    f"Sequence              : "
    f"{TARGET_SEQUENCE}"
)

print(
    f"Total samples         : "
    f"{len(merged)}"
)

print(
    f"Clear samples         : "
    f"{clear_total}"
)

print(
    f"Ambiguous samples     : "
    f"{ambiguous_total}"
)

print(
    f"Dataset disagreements : "
    f"{dataset_wrong}"
)

print(
    f"V5 disagreements      : "
    f"{v5_wrong}"
)

print(
    f"V6 disagreements      : "
    f"{v6_wrong}"
)

print(
    f"V6 error rate         : "
    f"{v6_error_rate:.2f}%"
)

print(
    f"Classification        : "
    f"{classification}"
)

print()

print("OUTPUT FILES:")
print(
    f" - {merged_output}"
)
print(
    f" - {v6_error_output}"
)
print(
    f" - {dataset_error_output}"
)
print(
    f" - {ambiguous_output}"
)
print(
    f" - {summary_output}"
)

print()

print("NEXT STEP:")
print(
    "1. Inspect hasil sequence_0048."
)
print(
    "2. Compare dengan sequence_0136."
)
print(
    "3. Cari pola error yang sama."
)
print(
    "4. Consolidate hasil manual review."
)
print(
    "5. Jangan train V7 dulu."
)

print()

print("=" * WIDTH)
print(
    "SEQUENCE 0048 POST-REVIEW ANALYSIS COMPLETED"
)
print("=" * WIDTH)