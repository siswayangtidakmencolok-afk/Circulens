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

TARGET_SEQUENCE = "sequence_0136"


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("CIRCULENS QUALITY — SEQUENCE 0136 POST-MANUAL-REVIEW ANALYSIS")
print("=" * 100)

print(f"Audit CSV  : {AUDIT_CSV}")
print(f"Review CSV : {REVIEW_CSV}")
print(f"Sequence   : {TARGET_SEQUENCE}")


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n" + "=" * 100)
print("1. CHECKING INPUT")
print("=" * 100)

if not AUDIT_CSV.exists():
    raise FileNotFoundError(f"Audit CSV not found: {AUDIT_CSV}")

if not REVIEW_CSV.exists():
    raise FileNotFoundError(f"Review CSV not found: {REVIEW_CSV}")

print("Audit CSV  : OK")
print("Review CSV : OK")


# ============================================================
# 2. LOAD
# ============================================================

audit = pd.read_csv(AUDIT_CSV)
review = pd.read_csv(REVIEW_CSV)

print("\n" + "=" * 100)
print("2. LOADING DATA")
print("=" * 100)

print(f"Audit rows  : {len(audit)}")
print(f"Review rows : {len(review)}")


# ============================================================
# 3. FILTER SEQUENCE
# ============================================================

seq = audit[
    audit["sequence_group_v5"]
    .astype(str)
    .str.strip()
    .eq(TARGET_SEQUENCE)
].copy()

print("\n" + "=" * 100)
print("3. FILTERING SEQUENCE")
print("=" * 100)

print(f"Sequence rows : {len(seq)}")

if len(seq) != 37:
    raise ValueError(
        f"Expected 37 sequence samples, found {len(seq)}"
    )


# ============================================================
# 4. PREPARE REVIEW
# ============================================================

review_seq = review[
    review["filename"]
    .astype(str)
    .str.strip()
    .isin(
        set(seq["filename_v5"].astype(str).str.strip())
    )
].copy()

print(f"Manual review rows : {len(review_seq)}")

if len(review_seq) != 37:
    raise ValueError(
        f"Expected 37 manual review rows, found {len(review_seq)}"
    )


# ============================================================
# 5. MERGE
# ============================================================

print("\n" + "=" * 100)
print("5. MERGING AUDIT + MANUAL REVIEW")
print("=" * 100)

seq["filename_key"] = (
    seq["filename_v5"]
    .astype(str)
    .str.strip()
)

review_seq["filename_key"] = (
    review_seq["filename"]
    .astype(str)
    .str.strip()
)

merged = seq.merge(
    review_seq[
        [
            "filename_key",
            "manual_decision",
            "manual_label",
            "manual_notes",
        ]
    ],
    on="filename_key",
    how="left",
    validate="one_to_one",
)

print(f"Merged rows : {len(merged)}")


# ============================================================
# 6. MANUAL REVIEW VALIDATION
# ============================================================

print("\n" + "=" * 100)
print("6. MANUAL REVIEW VALIDATION")
print("=" * 100)

print(
    "Manual decisions:"
)

print(
    merged["manual_decision"]
    .value_counts(dropna=False)
    .to_string()
)

print(
    "\nManual labels:"
)

print(
    merged["manual_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 7. NORMALIZE LABELS
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


merged["manual_label_norm"] = merged["manual_label"].map(norm)
merged["label_v5_norm"] = merged["label_v5"].map(norm)
merged["prediction_v5_norm"] = merged["prediction_label_v5"].map(norm)
merged["prediction_v6_norm"] = merged["prediction_label_v6"].map(norm)


# ============================================================
# 8. COMPARE AGAINST MANUAL GROUND TRUTH
# ============================================================

merged["v5_vs_manual"] = (
    merged["prediction_v5_norm"]
    == merged["manual_label_norm"]
)

merged["v6_vs_manual"] = (
    merged["prediction_v6_norm"]
    == merged["manual_label_norm"]
)

merged["dataset_label_vs_manual"] = (
    merged["label_v5_norm"]
    == merged["manual_label_norm"]
)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("9. POST-REVIEW ACCURACY")
print("=" * 100)

total = len(merged)

v5_correct = int(merged["v5_vs_manual"].sum())
v6_correct = int(merged["v6_vs_manual"].sum())
dataset_correct = int(
    merged["dataset_label_vs_manual"].sum()
)

print(
    f"V5 correct vs manual     : "
    f"{v5_correct}/{total} "
    f"({v5_correct / total * 100:.2f}%)"
)

print(
    f"V6 correct vs manual     : "
    f"{v6_correct}/{total} "
    f"({v6_correct / total * 100:.2f}%)"
)

print(
    f"Dataset label correct    : "
    f"{dataset_correct}/{total} "
    f"({dataset_correct / total * 100:.2f}%)"
)


# ============================================================
# 10. V5 → V6 CHANGE
# ============================================================

print("\n" + "=" * 100)
print("10. V5 → V6 CHANGE")
print("=" * 100)

v5_to_v6 = pd.crosstab(
    merged["prediction_v5_norm"],
    merged["prediction_v6_norm"],
)

print(v5_to_v6.to_string())


improved = (
    (~merged["v5_vs_manual"])
    & merged["v6_vs_manual"]
).sum()

regressed = (
    merged["v5_vs_manual"]
    & (~merged["v6_vs_manual"])
).sum()

still_wrong = (
    (~merged["v5_vs_manual"])
    & (~merged["v6_vs_manual"])
).sum()

correct_both = (
    merged["v5_vs_manual"]
    & merged["v6_vs_manual"]
).sum()

print()
print(f"Correct in both : {correct_both}")
print(f"Improved V5→V6 : {improved}")
print(f"Regressed      : {regressed}")
print(f"Still wrong    : {still_wrong}")


# ============================================================
# 11. DATASET LABEL AUDIT
# ============================================================

print("\n" + "=" * 100)
print("11. DATASET LABEL AUDIT")
print("=" * 100)

label_mismatch = merged[
    ~merged["dataset_label_vs_manual"]
].copy()

print(
    f"Dataset-label mismatches : {len(label_mismatch)}"
)

if len(label_mismatch) > 0:

    print("\nPotential mislabeled samples:")

    print(
        label_mismatch[
            [
                "filename_v5",
                "label_v5",
                "manual_label",
                "manual_decision",
                "manual_notes",
            ]
        ].to_string(index=False)
    )

else:

    print(
        "PASS — all dataset labels agree with manual review."
    )


# ============================================================
# 12. MODEL ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 100)
print("12. MODEL ERROR ANALYSIS")
print("=" * 100)

model_errors = merged[
    (~merged["v6_vs_manual"])
].copy()

print(
    f"V6 errors vs manual : {len(model_errors)}"
)

if len(model_errors) > 0:

    print(
        "\nRemaining V6 errors:"
    )

    print(
        model_errors[
            [
                "filename_v5",
                "label_v5",
                "prediction_label_v5",
                "prediction_label_v6",
                "prediction_confidence_v5",
                "prediction_confidence_v6",
                "manual_label",
                "manual_decision",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 13. SEQUENCE DECISION
# ============================================================

print("\n" + "=" * 100)
print("13. SEQUENCE DECISION")
print("=" * 100)

if len(label_mismatch) > 0:

    decision = "RELABEL"

elif len(model_errors) > 0:

    decision = "MODEL_ERROR"

else:

    decision = "PASS"

print(f"SEQUENCE STATUS : {decision}")

if decision == "RELABEL":

    print(
        "Action: investigate dataset labels before V7."
    )

elif decision == "MODEL_ERROR":

    print(
        "Action: labels appear valid; investigate model failure."
    )

else:

    print(
        "Action: sequence is fully consistent with manual review."
    )


# ============================================================
# 14. PRINT FULL TABLE
# ============================================================

print("\n" + "=" * 100)
print("14. FULL POST-REVIEW TABLE")
print("=" * 100)

display_columns = [
    "filename_v5",
    "label_v5",
    "prediction_label_v5",
    "prediction_label_v6",
    "manual_label",
    "manual_decision",
    "v5_vs_manual",
    "v6_vs_manual",
    "dataset_label_vs_manual",
]

print(
    merged[display_columns].to_string(index=False)
)


# ============================================================
# 15. FINAL
# ============================================================

print("\n" + "=" * 100)
print("FINAL")
print("=" * 100)

print(f"Sequence                : {TARGET_SEQUENCE}")
print(f"Samples                 : {total}")
print(f"V5 accuracy vs manual   : {v5_correct / total * 100:.2f}%")
print(f"V6 accuracy vs manual   : {v6_correct / total * 100:.2f}%")
print(f"Dataset-label agreement : {dataset_correct / total * 100:.2f}%")
print(f"Improved V5→V6          : {improved}")
print(f"Regressed V5→V6         : {regressed}")
print(f"Still wrong V6          : {still_wrong}")
print(f"Sequence status         : {decision}")

print("\n" + "=" * 100)
print("SEQUENCE 0136 POST-REVIEW ANALYSIS COMPLETED")
print("=" * 100)