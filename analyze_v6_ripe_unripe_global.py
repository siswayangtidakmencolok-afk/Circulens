from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

AUDIT_CSV = Path(
    "reports/quality_finetune_v6/"
    "v5_v6_error_analysis/label_audit/"
    "audit_master_v5_v6.csv"
)

REVIEW_DIR = Path(
    "reports/quality_finetune_v6/"
    "v5_v6_error_analysis/label_audit/"
    "manual_review_report"
)

OUTPUT_DIR = Path(
    "reports/quality_finetune_v6/"
    "v5_v6_error_analysis/label_audit/"
    "global_error_mining"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("CIRCULENS QUALITY — GLOBAL V6 RIPE → UNRIPE ERROR MINING")
print("=" * 100)

print(f"Audit CSV   : {AUDIT_CSV}")
print(f"Review dir  : {REVIEW_DIR}")
print(f"Output      : {OUTPUT_DIR}")


# ============================================================
# 1. CHECK INPUT
# ============================================================

print("\n" + "=" * 100)
print("1. CHECKING INPUT")
print("=" * 100)

if not AUDIT_CSV.exists():
    raise FileNotFoundError(
        f"Audit CSV tidak ditemukan:\n{AUDIT_CSV}"
    )

if not REVIEW_DIR.exists():
    raise FileNotFoundError(
        f"Review directory tidak ditemukan:\n{REVIEW_DIR}"
    )

print("Audit CSV  : OK")
print("Review dir : OK")


# ============================================================
# 2. LOAD AUDIT
# ============================================================

print("\n" + "=" * 100)
print("2. LOADING AUDIT")
print("=" * 100)

audit = pd.read_csv(AUDIT_CSV)

print(f"Audit rows : {len(audit)}")


# ============================================================
# 3. LOAD ALL MANUAL REVIEW FILES
# ============================================================

print("\n" + "=" * 100)
print("3. LOADING MANUAL REVIEWS")
print("=" * 100)

review_files = list(
    REVIEW_DIR.rglob("*_manual_decision_reviewed.csv")
)

print(f"Review files found : {len(review_files)}")

if not review_files:
    raise FileNotFoundError(
        "Tidak ditemukan file *_manual_decision_reviewed.csv"
    )

review_frames = []

for review_file in review_files:

    try:
        df = pd.read_csv(review_file)

        print(
            f"[LOAD] {review_file} "
            f"({len(df)} rows)"
        )

        review_frames.append(df)

    except Exception as e:

        print(
            f"[ERROR] Failed to read {review_file}: {e}"
        )


if not review_frames:
    raise RuntimeError(
        "Tidak ada manual review yang berhasil dibaca."
    )

reviews = pd.concat(
    review_frames,
    ignore_index=True
)

print(f"\nTotal review rows loaded : {len(reviews)}")


# ============================================================
# 4. NORMALIZE
# ============================================================

print("\n" + "=" * 100)
print("4. NORMALIZING DATA")
print("=" * 100)


def normalize_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


audit["filename_norm"] = normalize_text(
    audit["filename_v5"]
)

audit["label_norm"] = normalize_text(
    audit["label_v5"]
)

audit["v5_prediction_norm"] = normalize_text(
    audit["prediction_label_v5"]
)

audit["v6_prediction_norm"] = normalize_text(
    audit["prediction_label_v6"]
)

audit["sequence_norm"] = normalize_text(
    audit["sequence_group_v5"]
)

audit["source_norm"] = normalize_text(
    audit["source_v5"]
)


reviews["filename_norm"] = normalize_text(
    reviews["filename"]
)

reviews["manual_label_norm"] = normalize_text(
    reviews["manual_label"]
)

reviews["manual_decision_norm"] = normalize_text(
    reviews["manual_decision"]
)

print("Normalization completed.")


# ============================================================
# 5. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 100)
print("5. CHECKING REVIEW DUPLICATES")
print("=" * 100)

duplicate_reviews = reviews[
    reviews["filename_norm"].duplicated(
        keep=False
    )
].copy()

duplicate_count = (
    duplicate_reviews["filename_norm"]
    .nunique()
)

print(
    f"Duplicate reviewed filenames : "
    f"{duplicate_count}"
)

if duplicate_count > 0:

    print("\nWARNING: duplicate review records detected.")

    print(
        duplicate_reviews[
            [
                "filename",
                "manual_decision",
                "manual_label"
            ]
        ]
        .sort_values("filename")
        .to_string(index=False)
    )

    reviews = (
        reviews
        .drop_duplicates(
            subset=["filename_norm"],
            keep="last"
        )
        .copy()
    )

    print(
        "\nLatest review record per filename will be used."
    )

else:

    print("No duplicate reviewed filenames.")


# ============================================================
# 6. MERGE AUDIT + REVIEW
# ============================================================

print("\n" + "=" * 100)
print("6. MERGING AUDIT + MANUAL REVIEW")
print("=" * 100)

review_columns = [
    "filename_norm",
    "manual_label",
    "manual_label_norm",
    "manual_decision",
    "manual_decision_norm",
    "manual_notes",
]

merged = audit.merge(
    reviews[review_columns],
    on="filename_norm",
    how="left",
    suffixes=("", "_review")
)

print(f"Merged rows : {len(merged)}")


# ============================================================
# 7. REVIEW COVERAGE
# ============================================================

print("\n" + "=" * 100)
print("7. MANUAL REVIEW COVERAGE")
print("=" * 100)

reviewed_mask = (
    merged["manual_label_norm"]
    .fillna("")
    .ne("")
)

reviewed_count = reviewed_mask.sum()
unreviewed_count = len(merged) - reviewed_count

print(f"Audit samples         : {len(merged)}")
print(f"Manually reviewed     : {reviewed_count}")
print(f"Not manually reviewed : {unreviewed_count}")


# ============================================================
# 8. VERIFIED LABEL
# ============================================================

print("\n" + "=" * 100)
print("8. BUILDING VERIFIED LABEL")
print("=" * 100)

merged["manual_label_norm"] = (
    merged["manual_label_norm"]
    .fillna("")
)

merged["label_norm"] = (
    merged["label_norm"]
    .fillna("")
)

merged["verified_label_norm"] = (
    merged["manual_label_norm"]
    .where(
        merged["manual_label_norm"].ne(""),
        merged["label_norm"]
    )
)

print("Verified label created.")


# ============================================================
# 9. DATASET LABEL AGREEMENT
# ============================================================

print("\n" + "=" * 100)
print("9. DATASET LABEL AGREEMENT")
print("=" * 100)

agreement_mask = (
    reviewed_mask
    & (
        merged["label_norm"]
        == merged["manual_label_norm"]
    )
)

agreement_count = agreement_mask.sum()

agreement_pct = (
    agreement_count / reviewed_count * 100
    if reviewed_count
    else 0
)

print(
    f"Dataset/manual agreement : "
    f"{agreement_count}/{reviewed_count} "
    f"({agreement_pct:.2f}%)"
)

dataset_mismatches = merged[
    reviewed_mask
    & (
        merged["label_norm"]
        != merged["manual_label_norm"]
    )
].copy()

if len(dataset_mismatches) > 0:

    print("\nDataset-label mismatches:")

    print(
        dataset_mismatches[
            [
                "filename_v5",
                "label_v5",
                "manual_label",
                "sequence_group_v5"
            ]
        ]
        .to_string(index=False)
    )

else:

    print("No dataset/manual mismatches.")


# ============================================================
# 10. GLOBAL RIPE → UNRIPE
# ============================================================

print("\n" + "=" * 100)
print("10. IDENTIFYING GLOBAL RIPE → UNRIPE ERRORS")
print("=" * 100)

ripe_to_unripe_mask = (
    merged["verified_label_norm"].eq("ripe")
    & merged["v6_prediction_norm"].eq("unripe")
)

ripe_to_unripe = merged[
    ripe_to_unripe_mask
].copy()

print(
    f"Global RIPE → UNRIPE errors : "
    f"{len(ripe_to_unripe)}"
)


# ============================================================
# 11. GLOBAL RIPE ERROR RATE
# ============================================================

print("\n" + "=" * 100)
print("11. GLOBAL RIPE ERROR RATE")
print("=" * 100)

verified_ripe = merged[
    merged["verified_label_norm"].eq("ripe")
].copy()

total_ripe = len(verified_ripe)
error_count = len(ripe_to_unripe)

error_rate = (
    error_count / total_ripe * 100
    if total_ripe
    else 0
)

print(f"Verified RIPE samples    : {total_ripe}")
print(f"RIPE → UNRIPE errors     : {error_count}")
print(f"RIPE → UNRIPE error rate : {error_rate:.2f}%")


# ============================================================
# 12. CONFIDENCE
# ============================================================

print("\n" + "=" * 100)
print("12. HIGH-CONFIDENCE RIPE → UNRIPE ERRORS")
print("=" * 100)

ripe_to_unripe[
    "prediction_confidence_v6"
] = pd.to_numeric(
    ripe_to_unripe[
        "prediction_confidence_v6"
    ],
    errors="coerce"
)

high_confidence = ripe_to_unripe[
    ripe_to_unripe[
        "prediction_confidence_v6"
    ] >= CONFIDENCE_THRESHOLD
].copy()

high_confidence = high_confidence.sort_values(
    "prediction_confidence_v6",
    ascending=False
)

high_conf_pct = (
    len(high_confidence)
    / len(ripe_to_unripe)
    * 100
    if len(ripe_to_unripe)
    else 0
)

print(
    f"Threshold : >= {CONFIDENCE_THRESHOLD:.2f}"
)

print(
    f"High-confidence errors : "
    f"{len(high_confidence)}/{len(ripe_to_unripe)} "
    f"({high_conf_pct:.2f}%)"
)

if len(high_confidence) > 0:

    print()

    print(
        high_confidence[
            [
                "filename_v5",
                "sequence_group_v5",
                "source_v5",
                "prediction_label_v5",
                "prediction_label_v6",
                "prediction_confidence_v5",
                "prediction_confidence_v6",
                "manual_label",
                "manual_decision"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# 13. ERROR BY SEQUENCE
# ============================================================

print("\n" + "=" * 100)
print("13. RIPE → UNRIPE ERRORS BY SEQUENCE")
print("=" * 100)

sequence_summary = (
    ripe_to_unripe
    .groupby(
        "sequence_group_v5",
        dropna=False
    )
    .agg(
        errors=("filename_v5", "count"),
        mean_v6_confidence=(
            "prediction_confidence_v6",
            "mean"
        ),
        max_v6_confidence=(
            "prediction_confidence_v6",
            "max"
        )
    )
    .reset_index()
    .sort_values(
        ["errors", "mean_v6_confidence"],
        ascending=[False, False]
    )
)

print(
    sequence_summary.to_string(
        index=False
    )
)


# ============================================================
# 14. ERROR BY SOURCE
# ============================================================

print("\n" + "=" * 100)
print("14. RIPE → UNRIPE ERRORS BY SOURCE")
print("=" * 100)

source_summary = (
    ripe_to_unripe
    .groupby(
        "source_v5",
        dropna=False
    )
    .agg(
        errors=("filename_v5", "count"),
        mean_v6_confidence=(
            "prediction_confidence_v6",
            "mean"
        ),
        max_v6_confidence=(
            "prediction_confidence_v6",
            "max"
        )
    )
    .reset_index()
    .sort_values(
        ["errors", "mean_v6_confidence"],
        ascending=[False, False]
    )
)

print(
    source_summary.to_string(
        index=False
    )
)


# ============================================================
# 15. V5 → V6 TRANSITION
# ============================================================

print("\n" + "=" * 100)
print("15. V5 → V6 BEHAVIOR")
print("=" * 100)

transition_summary = (
    ripe_to_unripe
    .groupby(
        [
            "prediction_label_v5",
            "prediction_label_v6"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="count"
    )
)

transition_summary["percentage"] = (
    transition_summary["count"]
    / len(ripe_to_unripe)
    * 100
    if len(ripe_to_unripe)
    else 0
)

print(
    transition_summary.to_string(
        index=False
    )
)


# ============================================================
# 16. CONFIDENCE STATISTICS
# ============================================================

print("\n" + "=" * 100)
print("16. CONFIDENCE STATISTICS")
print("=" * 100)

if len(ripe_to_unripe) > 0:

    confidence_stats = (
        ripe_to_unripe[
            "prediction_confidence_v6"
        ]
        .describe()
    )

    print(
        confidence_stats.to_string()
    )

else:

    print("No RIPE → UNRIPE errors.")


# ============================================================
# 17. TOP HARD ERRORS
# ============================================================

print("\n" + "=" * 100)
print("17. TOP HARD ERRORS")
print("=" * 100)

top_hard_errors = ripe_to_unripe.sort_values(
    "prediction_confidence_v6",
    ascending=False
)

display_columns = [
    "filename_v5",
    "sequence_group_v5",
    "source_v5",
    "prediction_label_v5",
    "prediction_label_v6",
    "prediction_confidence_v5",
    "prediction_confidence_v6",
    "verified_label_norm",
    "manual_decision",
    "manual_notes",
]

if len(top_hard_errors) > 0:

    print(
        top_hard_errors[
            display_columns
        ]
        .head(30)
        .to_string(index=False)
    )

else:

    print("No hard errors found.")


# ============================================================
# 18. SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 100)
print("18. SAVING OUTPUTS")
print("=" * 100)

ERROR_CSV = (
    OUTPUT_DIR
    / "global_v6_ripe_to_unripe_errors.csv"
)

HIGH_CONF_CSV = (
    OUTPUT_DIR
    / "global_v6_ripe_to_unripe_high_confidence.csv"
)

SEQUENCE_CSV = (
    OUTPUT_DIR
    / "global_v6_ripe_to_unripe_by_sequence.csv"
)

SOURCE_CSV = (
    OUTPUT_DIR
    / "global_v6_ripe_to_unripe_by_source.csv"
)

TRANSITION_CSV = (
    OUTPUT_DIR
    / "global_v6_ripe_to_unripe_transition.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "global_v6_ripe_to_unripe_summary.csv"
)


ripe_to_unripe.to_csv(
    ERROR_CSV,
    index=False
)

high_confidence.to_csv(
    HIGH_CONF_CSV,
    index=False
)

sequence_summary.to_csv(
    SEQUENCE_CSV,
    index=False
)

source_summary.to_csv(
    SOURCE_CSV,
    index=False
)

transition_summary.to_csv(
    TRANSITION_CSV,
    index=False
)


# ============================================================
# 19. SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame([
    {
        "metric": "Total audit samples",
        "value": len(audit)
    },
    {
        "metric": "Total manually reviewed samples",
        "value": reviewed_count
    },
    {
        "metric": "Verified RIPE samples",
        "value": total_ripe
    },
    {
        "metric": "Global RIPE -> UNRIPE errors",
        "value": error_count
    },
    {
        "metric": "RIPE -> UNRIPE error rate (%)",
        "value": error_rate
    },
    {
        "metric": "High-confidence errors >= 0.80",
        "value": len(high_confidence)
    },
    {
        "metric": "High-confidence error percentage (%)",
        "value": high_conf_pct
    },
    {
        "metric": "Dataset/manual agreement (%)",
        "value": agreement_pct
    },
    {
        "metric": "Unique sequences containing error",
        "value": ripe_to_unripe[
            "sequence_group_v5"
        ].nunique()
    },
    {
        "metric": "Unique sources containing error",
        "value": ripe_to_unripe[
            "source_v5"
        ].nunique()
    },
])

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# 20. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 100)
print("20. INITIAL INTERPRETATION")
print("=" * 100)

print(
    f"""
Global verified RIPE samples :
    {total_ripe}

Global RIPE → UNRIPE errors :
    {error_count}

RIPE → UNRIPE error rate :
    {error_rate:.2f}%

High-confidence errors >= {CONFIDENCE_THRESHOLD:.2f} :
    {len(high_confidence)}

Unique sequences containing this error :
    {ripe_to_unripe["sequence_group_v5"].nunique()}

Unique sources containing this error :
    {ripe_to_unripe["source_v5"].nunique()}
"""
)

if len(ripe_to_unripe) == 0:

    print(
        "NO RIPE → UNRIPE errors found."
    )

elif ripe_to_unripe[
    "sequence_group_v5"
].nunique() >= 3:

    print(
        """
IMPORTANT:

The RIPE → UNRIPE error appears across
multiple sequences.

This strengthens the hypothesis that the
failure is systematic rather than isolated.

DO NOT immediately modify labels.

Next:
    inspect the visual characteristics of the
    highest-confidence errors before V7.
"""
    )

else:

    print(
        """
The RIPE → UNRIPE error is concentrated in
a small number of sequences.

Do NOT conclude that the problem is global yet.

Next:
    inspect affected sequences and sources
    before designing V7 corrective training.
"""
    )


# ============================================================
# 21. OUTPUT FILES
# ============================================================

print("\n" + "=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print(f"Main errors:")
print(f"  {ERROR_CSV}")

print(f"\nHigh-confidence errors:")
print(f"  {HIGH_CONF_CSV}")

print(f"\nErrors by sequence:")
print(f"  {SEQUENCE_CSV}")

print(f"\nErrors by source:")
print(f"  {SOURCE_CSV}")

print(f"\nV5 → V6 transition:")
print(f"  {TRANSITION_CSV}")

print(f"\nSummary:")
print(f"  {SUMMARY_CSV}")


print("\n" + "=" * 100)
print("GLOBAL V6 RIPE → UNRIPE ERROR MINING COMPLETED")
print("=" * 100)