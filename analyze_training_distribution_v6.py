from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys


# ============================================================
# CIRCULENS QUALITY — V6 TRAINING DISTRIBUTION ANALYSIS
# ============================================================

print("=" * 100)
print("CIRCULENS QUALITY — V6 TRAINING DISTRIBUTION ANALYSIS")
print("=" * 100)


# ============================================================
# 0. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Projects\circulens")

AUDIT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "quality_finetune_v6"
    / "v5_v6_error_analysis"
    / "label_audit"
    / "audit_master_v5_v6.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "quality_finetune_v6"
    / "v5_v6_error_analysis"
    / "label_audit"
    / "training_distribution_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Candidate dataset/training metadata locations
# ------------------------------------------------------------

DATASET_CANDIDATES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "data" / "quality",
    PROJECT_ROOT / "datasets" / "quality",
]

TRAIN_CANDIDATES = [
    PROJECT_ROOT / "data" / "train",
    PROJECT_ROOT / "dataset" / "train",
    PROJECT_ROOT / "datasets" / "train",
]

VAL_CANDIDATES = [
    PROJECT_ROOT / "data" / "val",
    PROJECT_ROOT / "data" / "validation",
    PROJECT_ROOT / "dataset" / "val",
    PROJECT_ROOT / "dataset" / "validation",
    PROJECT_ROOT / "datasets" / "val",
    PROJECT_ROOT / "datasets" / "validation",
]

TEST_CANDIDATES = [
    PROJECT_ROOT / "data" / "test",
    PROJECT_ROOT / "dataset" / "test",
    PROJECT_ROOT / "datasets" / "test",
]


# ============================================================
# 1. HELPERS
# ============================================================

def normalize_label(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    aliases = {
        "ripe": "ripe",
        "mature": "ripe",

        "unripe": "unripe",
        "immature": "unripe",

        "old": "old",
        "aged": "old",

        "dried": "dried",
        "dry": "dried",

        "damaged": "damaged",
        "damage": "damaged",
    }

    return aliases.get(value, value)


def resolve_column(df, candidates, required=True):
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # fuzzy fallback
    for c in df.columns:
        c_norm = str(c).strip().lower()

        for candidate in candidates:
            if candidate.lower() in c_norm:
                return c

    if required:
        raise KeyError(
            f"Tidak menemukan column. Candidates={candidates}. "
            f"Available={list(df.columns)}"
        )

    return None


def safe_pct(numerator, denominator):
    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def infer_source(filename):
    """
    Attempt to infer dataset/source from filename.

    Examples:
        new_mexico_green_chile_ripe_...
        chile_pepper_ripe_...
    """

    if pd.isna(filename):
        return "unknown"

    name = str(filename).lower()

    if "new_mexico" in name:
        return "new_mexico"

    if "chile_pepper" in name:
        return "chile_pepper"

    if "green_chile" in name:
        return "green_chile"

    if "pepper" in name:
        return "pepper"

    return "unknown"


def infer_class_from_filename(filename):
    """
    Infer class from filename when possible.
    """

    if pd.isna(filename):
        return np.nan

    name = str(filename).lower()

    patterns = [
        ("ripe", r"(?:^|_)ripe(?:_|\.|$)"),
        ("unripe", r"(?:^|_)unripe(?:_|\.|$)"),
        ("old", r"(?:^|_)old(?:_|\.|$)"),
        ("dried", r"(?:^|_)dried(?:_|\.|$)"),
        ("damaged", r"(?:^|_)damaged(?:_|\.|$)"),
    ]

    for label, pattern in patterns:
        if re.search(pattern, name):
            return label

    return np.nan


def infer_split_from_path(path):
    """
    Infer train/validation/test from path.
    """

    path_str = str(path).lower()

    if re.search(r"(^|[\\/])train([\\/]|$)", path_str):
        return "train"

    if re.search(r"(^|[\\/])(val|valid|validation)([\\/]|$)", path_str):
        return "validation"

    if re.search(r"(^|[\\/])test([\\/]|$)", path_str):
        return "test"

    return "unknown"


def find_image_files(root):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    results = []

    if not root.exists():
        return results

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            results.append(path)

    return results


# ============================================================
# 2. CHECK INPUT
# ============================================================

print()
print("=" * 100)
print("1. CHECKING INPUT")
print("=" * 100)

print(f"Audit CSV : {AUDIT_CSV}")

if not AUDIT_CSV.exists():
    print("[ERROR] Audit CSV tidak ditemukan.")
    sys.exit(1)

print("[OK] Audit CSV ditemukan.")


# ============================================================
# 3. LOAD AUDIT
# ============================================================

print()
print("=" * 100)
print("2. LOADING AUDIT DATA")
print("=" * 100)

audit = pd.read_csv(AUDIT_CSV)

print(f"Rows    : {len(audit)}")
print(f"Columns : {len(audit.columns)}")


# ============================================================
# 4. RESOLVE COLUMNS
# ============================================================

print()
print("=" * 100)
print("3. RESOLVING REQUIRED COLUMNS")
print("=" * 100)

filename_col = resolve_column(
    audit,
    [
        "filename_v5",
        "filename",
        "image_filename",
        "file_name",
    ],
)

label_col = resolve_column(
    audit,
    [
        "label_v5",
        "label",
        "dataset_label",
        "class",
    ],
)

sequence_col = resolve_column(
    audit,
    [
        "sequence_group_v5",
        "sequence",
        "sequence_group",
    ],
    required=False,
)

v5_pred_col = resolve_column(
    audit,
    [
        "prediction_label_v5",
        "prediction_v5",
        "v5_prediction",
    ],
    required=False,
)

v6_pred_col = resolve_column(
    audit,
    [
        "prediction_label_v6",
        "prediction_v6",
        "v6_prediction",
    ],
    required=False,
)

print(f"[OK] Filename       : {filename_col}")
print(f"[OK] Dataset label  : {label_col}")

if sequence_col:
    print(f"[OK] Sequence       : {sequence_col}")
else:
    print("[INFO] Sequence column tidak ditemukan.")

if v5_pred_col:
    print(f"[OK] V5 prediction  : {v5_pred_col}")

if v6_pred_col:
    print(f"[OK] V6 prediction  : {v6_pred_col}")


# ============================================================
# 5. NORMALIZE
# ============================================================

print()
print("=" * 100)
print("4. NORMALIZING DATA")
print("=" * 100)

audit["_filename"] = audit[filename_col].astype(str)
audit["_label"] = audit[label_col].apply(normalize_label)

if sequence_col:
    audit["_sequence"] = audit[sequence_col].astype(str)

if v5_pred_col:
    audit["_v5_prediction"] = audit[v5_pred_col].apply(normalize_label)

if v6_pred_col:
    audit["_v6_prediction"] = audit[v6_pred_col].apply(normalize_label)

audit["_inferred_source"] = audit["_filename"].apply(infer_source)
audit["_filename_class"] = audit["_filename"].apply(
    infer_class_from_filename
)

print("[OK] Normalization completed.")


# ============================================================
# 6. BASIC DATA QUALITY
# ============================================================

print()
print("=" * 100)
print("5. BASIC DATA QUALITY")
print("=" * 100)

duplicate_count = audit["_filename"].duplicated().sum()
missing_labels = audit["_label"].isna().sum()

print(f"Unique filenames     : {audit['_filename'].nunique()}")
print(f"Duplicate filenames  : {duplicate_count}")
print(f"Missing labels       : {missing_labels}")

if duplicate_count == 0:
    print("[OK] Tidak ada duplicate filename.")
else:
    print("[WARNING] Duplicate filename ditemukan.")

if missing_labels == 0:
    print("[OK] Semua sample memiliki label.")
else:
    print("[WARNING] Ada sample tanpa label.")


# ============================================================
# 7. GLOBAL CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("6. GLOBAL CLASS DISTRIBUTION")
print("=" * 100)

class_counts = (
    audit["_label"]
    .value_counts(dropna=False)
    .rename_axis("label")
    .reset_index(name="count")
)

class_counts["percentage"] = (
    class_counts["count"]
    / len(audit)
    * 100
).round(2)

print(class_counts.to_string(index=False))

class_counts.to_csv(
    OUTPUT_DIR / "global_class_distribution.csv",
    index=False
)

print(
    f"[SAVED] {OUTPUT_DIR / 'global_class_distribution.csv'}"
)


# ============================================================
# 8. CLASS BALANCE
# ============================================================

print()
print("=" * 100)
print("7. CLASS BALANCE ANALYSIS")
print("=" * 100)

valid_class_counts = (
    audit["_label"]
    .value_counts()
    .sort_values(ascending=False)
)

if len(valid_class_counts) > 0:
    majority_class = valid_class_counts.idxmax()
    minority_class = valid_class_counts.idxmin()

    majority_count = valid_class_counts.max()
    minority_count = valid_class_counts.min()

    imbalance_ratio = (
        majority_count / minority_count
        if minority_count > 0
        else np.inf
    )

    print(f"Majority class      : {majority_class}")
    print(f"Majority count      : {majority_count}")
    print(f"Minority class      : {minority_class}")
    print(f"Minority count      : {minority_count}")
    print(f"Imbalance ratio     : {imbalance_ratio:.2f}x")

    balance_rows = []

    for label, count in valid_class_counts.items():
        balance_rows.append(
            {
                "label": label,
                "count": count,
                "percentage": safe_pct(count, len(audit)),
                "relative_to_majority_pct": safe_pct(
                    count,
                    majority_count
                ),
            }
        )

    balance_df = pd.DataFrame(balance_rows)

    balance_df.to_csv(
        OUTPUT_DIR / "class_balance_analysis.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'class_balance_analysis.csv'}"
    )


# ============================================================
# 9. SOURCE DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("8. SOURCE DISTRIBUTION")
print("=" * 100)

source_distribution = (
    audit["_inferred_source"]
    .value_counts()
    .rename_axis("source")
    .reset_index(name="count")
)

source_distribution["percentage"] = (
    source_distribution["count"]
    / len(audit)
    * 100
).round(2)

print(source_distribution.to_string(index=False))

source_distribution.to_csv(
    OUTPUT_DIR / "source_distribution.csv",
    index=False
)

print(
    f"[SAVED] {OUTPUT_DIR / 'source_distribution.csv'}"
)


# ============================================================
# 10. CLASS × SOURCE
# ============================================================

print()
print("=" * 100)
print("9. CLASS × SOURCE DISTRIBUTION")
print("=" * 100)

class_source = pd.crosstab(
    audit["_label"],
    audit["_inferred_source"]
)

print(class_source.to_string())

class_source.to_csv(
    OUTPUT_DIR / "class_by_source.csv"
)

print(
    f"[SAVED] {OUTPUT_DIR / 'class_by_source.csv'}"
)


# ============================================================
# 11. CLASS × SOURCE PERCENTAGE
# ============================================================

print()
print("=" * 100)
print("10. CLASS × SOURCE PERCENTAGE")
print("=" * 100)

class_source_pct = pd.crosstab(
    audit["_label"],
    audit["_inferred_source"],
    normalize="index"
) * 100

class_source_pct = class_source_pct.round(2)

print(class_source_pct.to_string())

class_source_pct.to_csv(
    OUTPUT_DIR / "class_by_source_percentage.csv"
)

print(
    f"[SAVED] {OUTPUT_DIR / 'class_by_source_percentage.csv'}"
)


# ============================================================
# 12. SOURCE × CLASS PERCENTAGE
# ============================================================

print()
print("=" * 100)
print("11. SOURCE → CLASS COMPOSITION")
print("=" * 100)

source_class_pct = pd.crosstab(
    audit["_inferred_source"],
    audit["_label"],
    normalize="index"
) * 100

source_class_pct = source_class_pct.round(2)

print(source_class_pct.to_string())

source_class_pct.to_csv(
    OUTPUT_DIR / "source_class_composition.csv"
)

print(
    f"[SAVED] {OUTPUT_DIR / 'source_class_composition.csv'}"
)


# ============================================================
# 13. SEQUENCE DISTRIBUTION
# ============================================================

if sequence_col:

    print()
    print("=" * 100)
    print("12. SEQUENCE DISTRIBUTION")
    print("=" * 100)

    sequence_distribution = (
        audit["_sequence"]
        .value_counts()
        .rename_axis("sequence")
        .reset_index(name="count")
    )

    sequence_distribution["percentage"] = (
        sequence_distribution["count"]
        / len(audit)
        * 100
    ).round(2)

    print(sequence_distribution.to_string(index=False))

    sequence_distribution.to_csv(
        OUTPUT_DIR / "sequence_distribution.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'sequence_distribution.csv'}"
    )


# ============================================================
# 14. CLASS × SEQUENCE
# ============================================================

if sequence_col:

    print()
    print("=" * 100)
    print("13. CLASS × SEQUENCE DISTRIBUTION")
    print("=" * 100)

    class_sequence = pd.crosstab(
        audit["_sequence"],
        audit["_label"]
    )

    print(class_sequence.to_string())

    class_sequence.to_csv(
        OUTPUT_DIR / "class_by_sequence.csv"
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'class_by_sequence.csv'}"
    )


# ============================================================
# 15. SOURCE × SEQUENCE
# ============================================================

if sequence_col:

    print()
    print("=" * 100)
    print("14. SOURCE × SEQUENCE DISTRIBUTION")
    print("=" * 100)

    source_sequence = pd.crosstab(
        audit["_sequence"],
        audit["_inferred_source"]
    )

    print(source_sequence.to_string())

    source_sequence.to_csv(
        OUTPUT_DIR / "source_by_sequence.csv"
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'source_by_sequence.csv'}"
    )


# ============================================================
# 16. FILENAME CLASS VS DATASET LABEL
# ============================================================

print()
print("=" * 100)
print("15. FILENAME-IMPLIED CLASS VS DATASET LABEL")
print("=" * 100)

filename_class_check = audit[
    audit["_filename_class"].notna()
].copy()

if len(filename_class_check) > 0:

    filename_class_check["_match"] = (
        filename_class_check["_filename_class"]
        == filename_class_check["_label"]
    )

    filename_class_summary = (
        filename_class_check["_match"]
        .value_counts()
        .rename_axis("match")
        .reset_index(name="count")
    )

    filename_class_summary["percentage"] = (
        filename_class_summary["count"]
        / len(filename_class_check)
        * 100
    ).round(2)

    print(
        filename_class_summary.to_string(index=False)
    )

    mismatches = filename_class_check[
        ~filename_class_check["_match"]
    ].copy()

    print()
    print(
        f"Filename class mismatches : {len(mismatches)}"
    )

    if len(mismatches) > 0:
        print(
            mismatches[
                [
                    "_filename",
                    "_filename_class",
                    "_label",
                ]
            ].to_string(index=False)
        )

    mismatches[
        [
            "_filename",
            "_filename_class",
            "_label",
        ]
    ].to_csv(
        OUTPUT_DIR / "filename_class_label_mismatches.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'filename_class_label_mismatches.csv'}"
    )

else:
    print(
        "[INFO] Tidak cukup informasi class dari filename."
    )


# ============================================================
# 17. V5 → V6 TRANSITION DISTRIBUTION
# ============================================================

if v5_pred_col and v6_pred_col:

    print()
    print("=" * 100)
    print("16. V5 → V6 PREDICTION DISTRIBUTION")
    print("=" * 100)

    transition = pd.crosstab(
        audit["_v5_prediction"],
        audit["_v6_prediction"]
    )

    print(transition.to_string())

    transition.to_csv(
        OUTPUT_DIR / "v5_to_v6_transition_matrix.csv"
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'v5_to_v6_transition_matrix.csv'}"
    )


# ============================================================
# 18. V5/V6 PERFORMANCE BY CLASS
# ============================================================

if v5_pred_col and v6_pred_col:

    print()
    print("=" * 100)
    print("17. V5/V6 PERFORMANCE BY DATASET CLASS")
    print("=" * 100)

    rows = []

    for label in sorted(
        audit["_label"].dropna().unique()
    ):

        subset = audit[
            audit["_label"] == label
        ]

        v5_correct = (
            subset["_v5_prediction"] == label
        ).sum()

        v6_correct = (
            subset["_v6_prediction"] == label
        ).sum()

        rows.append(
            {
                "label": label,
                "samples": len(subset),
                "v5_correct": v5_correct,
                "v5_accuracy_pct": safe_pct(
                    v5_correct,
                    len(subset)
                ),
                "v6_correct": v6_correct,
                "v6_accuracy_pct": safe_pct(
                    v6_correct,
                    len(subset)
                ),
                "v6_errors": len(subset) - v6_correct,
            }
        )

    class_performance = pd.DataFrame(rows)

    print(
        class_performance.to_string(index=False)
    )

    class_performance.to_csv(
        OUTPUT_DIR / "v5_v6_performance_by_class.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'v5_v6_performance_by_class.csv'}"
    )


# ============================================================
# 19. V6 ERROR × SOURCE
# ============================================================

if v6_pred_col:

    print()
    print("=" * 100)
    print("18. V6 ERRORS × SOURCE")
    print("=" * 100)

    audit["_v6_correct"] = (
        audit["_v6_prediction"]
        == audit["_label"]
    )

    error_data = audit[
        ~audit["_v6_correct"]
    ].copy()

    if len(error_data) > 0:

        error_source = (
            error_data["_inferred_source"]
            .value_counts()
            .rename_axis("source")
            .reset_index(name="v6_errors")
        )

        error_source["percentage_of_v6_errors"] = (
            error_source["v6_errors"]
            / len(error_data)
            * 100
        ).round(2)

        print(
            error_source.to_string(index=False)
        )

        error_source.to_csv(
            OUTPUT_DIR / "v6_errors_by_source.csv",
            index=False
        )

        print(
            f"[SAVED] {OUTPUT_DIR / 'v6_errors_by_source.csv'}"
        )


# ============================================================
# 20. V6 ERROR × CLASS × SOURCE
# ============================================================

if v6_pred_col:

    print()
    print("=" * 100)
    print("19. V6 ERRORS — CLASS × SOURCE")
    print("=" * 100)

    error_class_source = pd.crosstab(
        error_data["_label"],
        error_data["_inferred_source"]
    )

    print(
        error_class_source.to_string()
    )

    error_class_source.to_csv(
        OUTPUT_DIR / "v6_errors_class_by_source.csv"
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'v6_errors_class_by_source.csv'}"
    )


# ============================================================
# 21. FOCUS: OLD → DRIED
# ============================================================

if v6_pred_col:

    print()
    print("=" * 100)
    print("20. FOCUS ANALYSIS — OLD → DRIED")
    print("=" * 100)

    old_dried = audit[
        (audit["_label"] == "old")
        & (audit["_v6_prediction"] == "dried")
    ].copy()

    print(
        f"OLD samples          : "
        f"{(audit['_label'] == 'old').sum()}"
    )

    print(
        f"OLD → DRIED errors    : {len(old_dried)}"
    )

    if len(old_dried) > 0:

        old_dried_source = (
            old_dried["_inferred_source"]
            .value_counts()
            .rename_axis("source")
            .reset_index(name="count")
        )

        old_dried_source["percentage"] = (
            old_dried_source["count"]
            / len(old_dried)
            * 100
        ).round(2)

        print()
        print(
            old_dried_source.to_string(index=False)
        )

        old_dried_source.to_csv(
            OUTPUT_DIR / "old_to_dried_by_source.csv",
            index=False
        )

        old_dried[
            [
                "_filename",
                "_label",
                "_v6_prediction",
                "_inferred_source",
            ]
        ].to_csv(
            OUTPUT_DIR / "old_to_dried_samples.csv",
            index=False
        )

        print(
            f"[SAVED] {OUTPUT_DIR / 'old_to_dried_samples.csv'}"
        )


# ============================================================
# 22. FOCUS: RIPE → UNRIPE / OLD
# ============================================================

if v6_pred_col:

    print()
    print("=" * 100)
    print("21. FOCUS ANALYSIS — RIPE FAILURE PATTERNS")
    print("=" * 100)

    ripe_subset = audit[
        audit["_label"] == "ripe"
    ].copy()

    ripe_errors = ripe_subset[
        ripe_subset["_v6_prediction"] != "ripe"
    ].copy()

    print(
        f"Total RIPE samples   : {len(ripe_subset)}"
    )

    print(
        f"Correct RIPE         : "
        f"{(ripe_subset['_v6_prediction'] == 'ripe').sum()}"
    )

    print(
        f"RIPE errors          : {len(ripe_errors)}"
    )

    ripe_transition = (
        ripe_errors["_v6_prediction"]
        .value_counts()
        .rename_axis("prediction")
        .reset_index(name="count")
    )

    if len(ripe_errors) > 0:
        ripe_transition["percentage"] = (
            ripe_transition["count"]
            / len(ripe_errors)
            * 100
        ).round(2)

    print()
    print(
        ripe_transition.to_string(index=False)
    )

    ripe_transition.to_csv(
        OUTPUT_DIR / "ripe_failure_predictions.csv",
        index=False
    )

    ripe_source = pd.crosstab(
        ripe_errors["_inferred_source"],
        ripe_errors["_v6_prediction"]
    )

    print()
    print("RIPE ERROR × SOURCE")
    print(
        ripe_source.to_string()
    )

    ripe_source.to_csv(
        OUTPUT_DIR / "ripe_error_by_source.csv"
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'ripe_error_by_source.csv'}"
    )


# ============================================================
# 23. CLASS-SOURCE DOMINANCE DETECTION
# ============================================================

print()
print("=" * 100)
print("22. CLASS-SOURCE DOMINANCE CHECK")
print("=" * 100)

dominance_rows = []

for label in sorted(
    audit["_label"].dropna().unique()
):

    subset = audit[
        audit["_label"] == label
    ]

    source_counts = (
        subset["_inferred_source"]
        .value_counts()
    )

    if len(source_counts) == 0:
        continue

    dominant_source = source_counts.index[0]
    dominant_count = source_counts.iloc[0]

    dominance_rows.append(
        {
            "label": label,
            "total_samples": len(subset),
            "dominant_source": dominant_source,
            "dominant_source_count": dominant_count,
            "dominant_source_pct": safe_pct(
                dominant_count,
                len(subset)
            ),
            "source_count": len(source_counts),
        }
    )

dominance_df = pd.DataFrame(dominance_rows)

if len(dominance_df) > 0:
    print(
        dominance_df.to_string(index=False)
    )

    dominance_df.to_csv(
        OUTPUT_DIR / "class_source_dominance.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'class_source_dominance.csv'}"
    )


# ============================================================
# 24. POTENTIAL DOMAIN SHIFT SIGNAL
# ============================================================

print()
print("=" * 100)
print("23. POTENTIAL DOMAIN SHIFT SIGNAL")
print("=" * 100)

domain_rows = []

for source in audit["_inferred_source"].unique():

    source_subset = audit[
        audit["_inferred_source"] == source
    ]

    total = len(source_subset)

    class_counts = (
        source_subset["_label"]
        .value_counts()
    )

    dominant_class = (
        class_counts.index[0]
        if len(class_counts) > 0
        else "unknown"
    )

    dominant_count = (
        class_counts.iloc[0]
        if len(class_counts) > 0
        else 0
    )

    domain_rows.append(
        {
            "source": source,
            "samples": total,
            "dominant_class": dominant_class,
            "dominant_class_count": dominant_count,
            "dominant_class_pct": safe_pct(
                dominant_count,
                total
            ),
            "unique_classes": len(class_counts),
        }
    )

domain_df = pd.DataFrame(domain_rows)

print(
    domain_df.to_string(index=False)
)

domain_df.to_csv(
    OUTPUT_DIR / "potential_domain_shift_analysis.csv",
    index=False
)

print(
    f"[SAVED] {OUTPUT_DIR / 'potential_domain_shift_analysis.csv'}"
)


# ============================================================
# 25. IMAGE DATASET DISCOVERY
# ============================================================

print()
print("=" * 100)
print("24. SEARCHING DATASET IMAGE DIRECTORIES")
print("=" * 100)

existing_roots = []

for root in DATASET_CANDIDATES:
    if root.exists() and root.is_dir():
        existing_roots.append(root)

for root in TRAIN_CANDIDATES + VAL_CANDIDATES + TEST_CANDIDATES:
    if root.exists() and root.is_dir():
        if root not in existing_roots:
            existing_roots.append(root)

if len(existing_roots) == 0:

    print(
        "[INFO] Tidak menemukan directory dataset standard."
    )

    print(
        "       Analysis dilanjutkan menggunakan audit CSV."
    )

else:

    for root in existing_roots:
        print(
            f"[FOUND] {root}"
        )


# ============================================================
# 26. SPLIT DISTRIBUTION FROM FILESYSTEM
# ============================================================

print()
print("=" * 100)
print("25. TRAIN / VALIDATION / TEST DISTRIBUTION")
print("=" * 100)

filesystem_rows = []

for root in existing_roots:

    files = find_image_files(root)

    for image_path in files:

        split = infer_split_from_path(image_path)

        filename = image_path.name

        inferred_class = infer_class_from_filename(
            filename
        )

        filesystem_rows.append(
            {
                "path": str(image_path),
                "filename": filename,
                "split": split,
                "filename_class": inferred_class,
            }
        )

if len(filesystem_rows) > 0:

    filesystem_df = pd.DataFrame(
        filesystem_rows
    )

    split_counts = (
        filesystem_df["split"]
        .value_counts()
        .rename_axis("split")
        .reset_index(name="count")
    )

    split_counts["percentage"] = (
        split_counts["count"]
        / len(filesystem_df)
        * 100
    ).round(2)

    print(
        split_counts.to_string(index=False)
    )

    split_counts.to_csv(
        OUTPUT_DIR / "filesystem_split_distribution.csv",
        index=False
    )

    filesystem_df.to_csv(
        OUTPUT_DIR / "filesystem_image_inventory.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'filesystem_image_inventory.csv'}"
    )

else:

    print(
        "[INFO] Tidak ada image filesystem yang ditemukan "
        "dari candidate directories."
    )


# ============================================================
# 27. POTENTIAL SPLIT LEAKAGE
# ============================================================

if len(filesystem_rows) > 0:

    print()
    print("=" * 100)
    print("26. POTENTIAL TRAIN/VAL/TEST LEAKAGE")
    print("=" * 100)

    split_df = filesystem_df[
        filesystem_df["split"] != "unknown"
    ].copy()

    if len(split_df) > 0:

        filename_split_counts = (
            split_df
            .groupby("filename")["split"]
            .nunique()
        )

        leaked_filenames = (
            filename_split_counts[
                filename_split_counts > 1
            ]
            .index
            .tolist()
        )

        print(
            f"Potential duplicated filenames "
            f"across splits : {len(leaked_filenames)}"
        )

        if len(leaked_filenames) > 0:

            leakage_df = split_df[
                split_df["filename"].isin(
                    leaked_filenames
                )
            ].sort_values("filename")

            print(
                leakage_df.to_string(index=False)
            )

            leakage_df.to_csv(
                OUTPUT_DIR / "potential_split_leakage.csv",
                index=False
            )

            print(
                f"[SAVED] {OUTPUT_DIR / 'potential_split_leakage.csv'}"
            )

        else:

            print(
                "[OK] Tidak ditemukan filename duplicate "
                "antar split."
            )


# ============================================================
# 28. AUDIT LABEL VS PREDICTION DISTRIBUTION
# ============================================================

if v5_pred_col and v6_pred_col:

    print()
    print("=" * 100)
    print("27. LABEL vs V5 vs V6 DISTRIBUTION")
    print("=" * 100)

    distribution_rows = []

    classes = sorted(
        set(
            audit["_label"].dropna().unique()
        )
        |
        set(
            audit["_v5_prediction"].dropna().unique()
        )
        |
        set(
            audit["_v6_prediction"].dropna().unique()
        )
    )

    for label in classes:

        dataset_count = (
            audit["_label"] == label
        ).sum()

        v5_count = (
            audit["_v5_prediction"] == label
        ).sum()

        v6_count = (
            audit["_v6_prediction"] == label
        ).sum()

        distribution_rows.append(
            {
                "class": label,
                "dataset_label_count": dataset_count,
                "v5_prediction_count": v5_count,
                "v6_prediction_count": v6_count,
                "v5_minus_dataset":
                    v5_count - dataset_count,
                "v6_minus_dataset":
                    v6_count - dataset_count,
            }
        )

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    print(
        distribution_df.to_string(index=False)
    )

    distribution_df.to_csv(
        OUTPUT_DIR / "label_v5_v6_distribution_comparison.csv",
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_DIR / 'label_v5_v6_distribution_comparison.csv'}"
    )


# ============================================================
# 29. ROOT-CAUSE SIGNAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("28. PRELIMINARY ROOT-CAUSE SIGNAL SUMMARY")
print("=" * 100)

signals = []

# Class imbalance
if len(valid_class_counts) > 0:

    max_count = valid_class_counts.max()
    min_count = valid_class_counts.min()

    if min_count > 0:

        ratio = max_count / min_count

        if ratio >= 5:
            signals.append(
                (
                    "CLASS_IMBALANCE",
                    "STRONG",
                    f"Majority/minority ratio = {ratio:.2f}x"
                )
            )

        elif ratio >= 2:
            signals.append(
                (
                    "CLASS_IMBALANCE",
                    "MODERATE",
                    f"Majority/minority ratio = {ratio:.2f}x"
                )
            )

        else:
            signals.append(
                (
                    "CLASS_IMBALANCE",
                    "LOW",
                    f"Majority/minority ratio = {ratio:.2f}x"
                )
            )


# Filename-label mismatch
if len(filename_class_check) > 0:

    mismatch_rate = safe_pct(
        (~filename_class_check["_match"]).sum(),
        len(filename_class_check)
    )

    if mismatch_rate > 10:

        signals.append(
            (
                "FILENAME_LABEL_CONSISTENCY",
                "STRONG",
                f"Mismatch = {mismatch_rate:.2f}%"
            )
        )

    elif mismatch_rate > 0:

        signals.append(
            (
                "FILENAME_LABEL_CONSISTENCY",
                "MODERATE",
                f"Mismatch = {mismatch_rate:.2f}%"
            )
        )

    else:

        signals.append(
            (
                "FILENAME_LABEL_CONSISTENCY",
                "LOW",
                "No filename-label mismatch detected"
            )
        )


# Source dominance
if len(dominance_df) > 0:

    strong_dominance = dominance_df[
        dominance_df["dominant_source_pct"] >= 80
    ]

    if len(strong_dominance) > 0:

        labels = ", ".join(
            strong_dominance["label"].astype(str)
        )

        signals.append(
            (
                "SOURCE_DOMINANCE",
                "STRONG",
                f"Class dominated >=80% by one source: {labels}"
            )
        )

    else:

        signals.append(
            (
                "SOURCE_DOMINANCE",
                "LOW",
                "No class dominated >=80% by one inferred source"
            )
        )


# OLD → DRIED
if v6_pred_col:

    old_count = (
        audit["_label"] == "old"
    ).sum()

    old_dried_count = len(
        audit[
            (audit["_label"] == "old")
            & (audit["_v6_prediction"] == "dried")
        ]
    )

    if old_count > 0:

        old_dried_rate = safe_pct(
            old_dried_count,
            old_count
        )

        if old_dried_rate >= 80:

            signals.append(
                (
                    "OLD_TO_DRIED",
                    "STRONG",
                    f"{old_dried_count}/{old_count} "
                    f"OLD predicted as DRIED "
                    f"({old_dried_rate:.2f}%)"
                )
            )

        elif old_dried_rate >= 50:

            signals.append(
                (
                    "OLD_TO_DRIED",
                    "MODERATE",
                    f"{old_dried_count}/{old_count} "
                    f"OLD predicted as DRIED "
                    f"({old_dried_rate:.2f}%)"
                )
            )

        else:

            signals.append(
                (
                    "OLD_TO_DRIED",
                    "LOW",
                    f"{old_dried_count}/{old_count} "
                    f"OLD predicted as DRIED "
                    f"({old_dried_rate:.2f}%)"
                )
            )


signal_df = pd.DataFrame(
    signals,
    columns=[
        "signal",
        "severity",
        "interpretation",
    ]
)

print(
    signal_df.to_string(index=False)
)

signal_df.to_csv(
    OUTPUT_DIR / "preliminary_root_cause_signals.csv",
    index=False
)

print(
    f"[SAVED] {OUTPUT_DIR / 'preliminary_root_cause_signals.csv'}"
)


# ============================================================
# 30. CONSOLIDATED AUDIT DATA
# ============================================================

print()
print("=" * 100)
print("29. EXPORTING CONSOLIDATED AUDIT DATA")
print("=" * 100)

export_columns = [
    filename_col,
    label_col,
]

if sequence_col:
    export_columns.append(sequence_col)

if v5_pred_col:
    export_columns.append(v5_pred_col)

if v6_pred_col:
    export_columns.append(v6_pred_col)

export_columns = list(
    dict.fromkeys(export_columns)
)

consolidated = audit[export_columns].copy()

consolidated["inferred_source"] = (
    audit["_inferred_source"]
)

consolidated["filename_class"] = (
    audit["_filename_class"]
)

if v5_pred_col:
    consolidated["v5_correct"] = (
        audit["_v5_prediction"]
        == audit["_label"]
    )

if v6_pred_col:
    consolidated["v6_correct"] = (
        audit["_v6_prediction"]
        == audit["_label"]
    )

consolidated.to_csv(
    OUTPUT_DIR / "training_distribution_audit_consolidated.csv",
    index=False
)

print(
    f"[SAVED] "
    f"{OUTPUT_DIR / 'training_distribution_audit_consolidated.csv'}"
)


# ============================================================
# 31. FINAL DIAGNOSTIC
# ============================================================

print()
print("=" * 100)
print("30. FINAL DIAGNOSTIC")
print("=" * 100)

print()
print("GLOBAL DATASET")
print(
    f" - Samples                  : {len(audit)}"
)
print(
    f" - Classes                  : "
    f"{audit['_label'].nunique()}"
)
print(
    f" - Sources inferred        : "
    f"{audit['_inferred_source'].nunique()}"
)

if v6_pred_col:

    v6_errors_count = (
        audit["_v6_prediction"]
        != audit["_label"]
    ).sum()

    v6_accuracy = safe_pct(
        len(audit) - v6_errors_count,
        len(audit)
    )

    print(
        f" - V6 accuracy vs dataset  : "
        f"{v6_accuracy:.2f}%"
    )

    print(
        f" - V6 errors               : "
        f"{v6_errors_count}"
    )


print()
print("IMPORTANT INTERPRETATION:")

print(
    "1. Script ini TIDAK melakukan training."
)

print(
    "2. Script ini TIDAK mengubah dataset."
)

print(
    "3. Dataset label tetap dianggap sebagai audit reference,"
    " bukan manual ground truth."
)

print(
    "4. Inferred source berasal dari filename dan"
    " harus diverifikasi terhadap struktur dataset asli."
)

print(
    "5. Jika satu class sangat didominasi satu source,"
    " kemungkinan domain/source bias perlu diperiksa."
)

print(
    "6. Jika OLD → DRIED sangat tinggi,"
    " periksa representasi visual OLD vs DRIED,"
    " bukan hanya jumlah sample."
)

print(
    "7. Jangan train V7 hanya berdasarkan output script ini."
)

print(
    "8. Hasil ini harus digabungkan dengan manual-review"
    " evidence dan training configuration V6."
)


# ============================================================
# 32. OUTPUT FILES
# ============================================================

print()
print("=" * 100)
print("31. OUTPUT FILES")
print("=" * 100)

for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f" - {path}")


print()
print("=" * 100)
print("V6 TRAINING DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 100)