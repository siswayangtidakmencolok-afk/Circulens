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

REVIEW_CSV = Path(
    "reports/quality_finetune_v6/"
    "v5_v6_error_analysis/label_audit/"
    "manual_review_report/sequence_0136_review/"
    "sequence_0136_manual_decision_reviewed.csv"
)

OUTPUT_DIR = Path(
    "reports/quality_finetune_v6/"
    "v5_v6_error_analysis/label_audit/"
    "manual_review_report/sequence_0136_review/"
    "error_pattern_analysis"
)

TARGET_SEQUENCE = "sequence_0136"


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("CIRCULENS QUALITY — SEQUENCE 0136 ERROR PATTERN ANALYSIS")
print("=" * 100)

print(f"Audit CSV  : {AUDIT_CSV}")
print(f"Review CSV : {REVIEW_CSV}")
print(f"Output     : {OUTPUT_DIR}")
print(f"Sequence   : {TARGET_SEQUENCE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

if not REVIEW_CSV.exists():
    raise FileNotFoundError(
        f"Review CSV tidak ditemukan:\n{REVIEW_CSV}"
    )

print("Audit CSV  : OK")
print("Review CSV : OK")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 100)
print("2. LOADING DATA")
print("=" * 100)

audit = pd.read_csv(AUDIT_CSV)
review = pd.read_csv(REVIEW_CSV)

print(f"Audit rows  : {len(audit)}")
print(f"Review rows : {len(review)}")


# ============================================================
# 3. FILTER SEQUENCE
# ============================================================

print("\n" + "=" * 100)
print("3. FILTERING SEQUENCE")
print("=" * 100)

seq = audit[
    audit["sequence_group_v5"]
    .astype(str)
    .str.strip()
    .eq(TARGET_SEQUENCE)
].copy()

print(f"Sequence rows : {len(seq)}")

if len(seq) != 37:
    print(
        f"WARNING: expected 37 rows, found {len(seq)}"
    )


# ============================================================
# 4. NORMALIZE LABELS
# ============================================================

def normalize_label(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


for col in [
    "label_v5",
    "prediction_label_v5",
    "prediction_label_v6",
]:
    if col in seq.columns:
        seq[f"{col}_norm"] = seq[col].apply(
            normalize_label
        )


review["manual_label_norm"] = review[
    "manual_label"
].apply(normalize_label)


# ============================================================
# 5. MERGE AUDIT + MANUAL REVIEW
# ============================================================

print("\n" + "=" * 100)
print("5. MERGING AUDIT + MANUAL REVIEW")
print("=" * 100)

merged = seq.merge(
    review[
        [
            "filename",
            "manual_decision",
            "manual_label",
            "manual_label_norm",
            "manual_notes",
        ]
    ],
    left_on="filename_v5",
    right_on="filename",
    how="left",
    validate="one_to_one",
)

print(f"Merged rows : {len(merged)}")


# ============================================================
# 6. VALIDATE MANUAL REVIEW
# ============================================================

print("\n" + "=" * 100)
print("6. VALIDATING MANUAL REVIEW")
print("=" * 100)

missing_manual = merged[
    merged["manual_label_norm"].eq("")
]

print(
    f"Missing manual labels : {len(missing_manual)}"
)

if len(missing_manual) > 0:
    print("\nWARNING — missing manual review:")
    print(
        missing_manual[
            ["filename_v5", "manual_label"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Manual review belum lengkap."
    )

dataset_vs_manual = (
    merged["label_v5_norm"]
    == merged["manual_label_norm"]
)

print(
    "Dataset label agreement : "
    f"{dataset_vs_manual.sum()}/{len(merged)} "
    f"({dataset_vs_manual.mean() * 100:.2f}%)"
)


# ============================================================
# 7. IDENTIFY V6 ERRORS
# ============================================================

print("\n" + "=" * 100)
print("7. IDENTIFYING V6 ERRORS")
print("=" * 100)

merged["v6_correct"] = (
    merged["prediction_label_v6_norm"]
    == merged["manual_label_norm"]
)

errors = merged[
    ~merged["v6_correct"]
].copy()

print(f"Total V6 errors : {len(errors)}")

if len(errors) == 0:
    print("No V6 errors found.")
    raise SystemExit(0)


# ============================================================
# 8. ERROR TRANSITION PATTERN
# ============================================================

print("\n" + "=" * 100)
print("8. V5 → V6 ERROR TRANSITION PATTERN")
print("=" * 100)

errors["v5_to_v6"] = (
    errors["prediction_label_v5_norm"]
    + " -> "
    + errors["prediction_label_v6_norm"]
)

transition = (
    errors["v5_to_v6"]
    .value_counts()
    .rename_axis("transition")
    .reset_index(name="count")
)

transition["percentage"] = (
    transition["count"]
    / len(errors)
    * 100
)

print(
    transition.to_string(index=False)
)


# ============================================================
# 9. V6 PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("9. V6 ERROR PREDICTION DISTRIBUTION")
print("=" * 100)

v6_error_dist = (
    errors["prediction_label_v6_norm"]
    .value_counts()
    .rename_axis("v6_prediction")
    .reset_index(name="count")
)

v6_error_dist["percentage"] = (
    v6_error_dist["count"]
    / len(errors)
    * 100
)

print(
    v6_error_dist.to_string(index=False)
)


# ============================================================
# 10. MANUAL / TRUE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("10. TRUE LABEL DISTRIBUTION OF V6 ERRORS")
print("=" * 100)

true_dist = (
    errors["manual_label_norm"]
    .value_counts()
    .rename_axis("manual_label")
    .reset_index(name="count")
)

true_dist["percentage"] = (
    true_dist["count"]
    / len(errors)
    * 100
)

print(
    true_dist.to_string(index=False)
)


# ============================================================
# 11. CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 100)
print("11. V6 CONFIDENCE ANALYSIS")
print("=" * 100)

errors["v6_confidence_num"] = pd.to_numeric(
    errors["prediction_confidence_v6"],
    errors="coerce"
)

confidence_stats = errors[
    "v6_confidence_num"
].describe()

print(
    confidence_stats.to_string()
)


# ============================================================
# 12. HIGH-CONFIDENCE WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 100)
print("12. HIGH-CONFIDENCE WRONG PREDICTIONS")
print("=" * 100)

HIGH_CONF_THRESHOLD = 0.80

high_conf = errors[
    errors["v6_confidence_num"]
    >= HIGH_CONF_THRESHOLD
].copy()

print(
    f"Threshold : >= {HIGH_CONF_THRESHOLD:.2f}"
)

print(
    f"High-confidence errors : {len(high_conf)}/{len(errors)} "
    f"({len(high_conf) / len(errors) * 100:.2f}%)"
)

if len(high_conf) > 0:

    high_conf_display = high_conf[
        [
            "filename_v5",
            "manual_label_norm",
            "prediction_label_v5_norm",
            "prediction_label_v6_norm",
            "prediction_confidence_v5",
            "prediction_confidence_v6",
        ]
    ].copy()

    print(
        high_conf_display.to_string(
            index=False
        )
    )


# ============================================================
# 13. CONFIDENCE BY ERROR TRANSITION
# ============================================================

print("\n" + "=" * 100)
print("13. CONFIDENCE BY ERROR TRANSITION")
print("=" * 100)

transition_conf = (
    errors
    .groupby("v5_to_v6")["v6_confidence_num"]
    .agg(
        count="count",
        mean="mean",
        min="min",
        max="max",
    )
    .reset_index()
)

print(
    transition_conf.to_string(index=False)
)


# ============================================================
# 14. RIPE → UNRIPE DEEP CHECK
# ============================================================

print("\n" + "=" * 100)
print("14. RIPE → UNRIPE DEEP CHECK")
print("=" * 100)

ripe_unripe = errors[
    (
        errors["manual_label_norm"] == "ripe"
    )
    &
    (
        errors["prediction_label_v6_norm"]
        == "unripe"
    )
].copy()

print(
    f"RIPE → UNRIPE errors : {len(ripe_unripe)}"
)

if len(ripe_unripe) > 0:

    ripe_unripe["confidence"] = (
        errors.loc[
            ripe_unripe.index,
            "v6_confidence_num"
        ]
    )

    print(
        ripe_unripe[
            [
                "filename_v5",
                "prediction_label_v5_norm",
                "prediction_label_v6_norm",
                "prediction_confidence_v5",
                "prediction_confidence_v6",
                "manual_label_norm",
            ]
        ].to_string(index=False)
    )

    print("\nRIPE → UNRIPE confidence statistics:")

    print(
        ripe_unripe["confidence"].describe().to_string()
    )


# ============================================================
# 15. V5 → V6 REGRESSION
# ============================================================

print("\n" + "=" * 100)
print("15. V5 → V6 REGRESSION CASES")
print("=" * 100)

merged["v5_correct"] = (
    merged["prediction_label_v5_norm"]
    == merged["manual_label_norm"]
)

merged["regressed"] = (
    merged["v5_correct"]
    &
    ~merged["v6_correct"]
)

regressions = merged[
    merged["regressed"]
].copy()

print(
    f"Regression count : {len(regressions)}"
)

if len(regressions) > 0:

    print(
        regressions[
            [
                "filename_v5",
                "manual_label_norm",
                "prediction_label_v5_norm",
                "prediction_label_v6_norm",
                "prediction_confidence_v5",
                "prediction_confidence_v6",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 16. SAMPLE ORDER / SEQUENCE PATTERN
# ============================================================

print("\n" + "=" * 100)
print("16. SAMPLE ORDER / SEQUENCE PATTERN")
print("=" * 100)

sequence_pattern = merged[
    [
        "filename_v5",
        "manual_label_norm",
        "prediction_label_v5_norm",
        "prediction_label_v6_norm",
        "prediction_confidence_v6",
        "v6_correct",
    ]
].copy()

print(
    sequence_pattern.to_string(index=False)
)


# ============================================================
# 17. ERROR SUMMARY TABLE
# ============================================================

print("\n" + "=" * 100)
print("17. ERROR SUMMARY TABLE")
print("=" * 100)

summary = pd.DataFrame({
    "metric": [
        "Sequence samples",
        "V6 errors",
        "V6 accuracy",
        "RIPE → UNRIPE errors",
        "RIPE → OLD errors",
        "High-confidence V6 errors >= 0.80",
        "V5 → V6 regressions",
        "Dataset/manual agreement",
    ],
    "value": [
        len(merged),
        len(errors),
        (
            merged["v6_correct"].mean() * 100
        ),
        len(
            errors[
                (
                    errors["manual_label_norm"]
                    == "ripe"
                )
                &
                (
                    errors["prediction_label_v6_norm"]
                    == "unripe"
                )
            ]
        ),
        len(
            errors[
                (
                    errors["manual_label_norm"]
                    == "ripe"
                )
                &
                (
                    errors["prediction_label_v6_norm"]
                    == "old"
                )
            ]
        ),
        len(high_conf),
        len(regressions),
        (
            dataset_vs_manual.mean()
            * 100
        ),
    ],
})

print(
    summary.to_string(index=False)
)


# ============================================================
# 18. SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 100)
print("18. SAVING OUTPUTS")
print("=" * 100)

ERROR_CSV = OUTPUT_DIR / "sequence_0136_v6_errors.csv"
TRANSITION_CSV = OUTPUT_DIR / "sequence_0136_transition_analysis.csv"
SUMMARY_CSV = OUTPUT_DIR / "sequence_0136_error_summary.csv"
HIGH_CONF_CSV = OUTPUT_DIR / "sequence_0136_high_confidence_errors.csv"
RIPE_UNRIPE_CSV = OUTPUT_DIR / "sequence_0136_ripe_to_unripe_errors.csv"

errors.to_csv(
    ERROR_CSV,
    index=False
)

transition.to_csv(
    TRANSITION_CSV,
    index=False
)

summary.to_csv(
    SUMMARY_CSV,
    index=False
)

high_conf.to_csv(
    HIGH_CONF_CSV,
    index=False
)

ripe_unripe.to_csv(
    RIPE_UNRIPE_CSV,
    index=False
)

print(f"Errors CSV          : {ERROR_CSV}")
print(f"Transition CSV      : {TRANSITION_CSV}")
print(f"Summary CSV         : {SUMMARY_CSV}")
print(f"High-confidence CSV : {HIGH_CONF_CSV}")
print(f"Ripe→Unripe CSV     : {RIPE_UNRIPE_CSV}")


# ============================================================
# 19. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 100)
print("19. INITIAL INTERPRETATION")
print("=" * 100)

ripe_unripe_count = len(ripe_unripe)
ripe_old_count = len(
    errors[
        (
            errors["manual_label_norm"]
            == "ripe"
        )
        &
        (
            errors["prediction_label_v6_norm"]
            == "old"
        )
    ]
)

high_conf_count = len(high_conf)

print(
    f"""
Sequence status : MODEL_ERROR

Dataset/manual agreement :
    {dataset_vs_manual.sum()}/{len(merged)}
    ({dataset_vs_manual.mean() * 100:.2f}%)

V6 errors :
    {len(errors)}/{len(merged)}

Dominant error :
    RIPE → UNRIPE = {ripe_unripe_count}

RIPE → OLD :
    {ripe_old_count}

High-confidence wrong predictions :
    {high_conf_count}/{len(errors)}
    ({high_conf_count / len(errors) * 100:.2f}%)

Interpretation:
    The sequence does not currently provide evidence
    of a dataset-label problem.

    The dominant failure mode should be investigated
    as a MODEL / VISUAL FEATURE problem.

    In particular, inspect whether the model is using
    green coloration / green shoulder as a shortcut
    for the UNRIPE class.

IMPORTANT:
    This script does NOT modify the dataset.
    This script does NOT retrain V7.
    This script does NOT relabel any sample.

    The output is diagnostic evidence for deciding
    the correct V7 corrective-training strategy.
"""
)


print("\n" + "=" * 100)
print("SEQUENCE 0136 ERROR PATTERN ANALYSIS COMPLETED")
print("=" * 100)