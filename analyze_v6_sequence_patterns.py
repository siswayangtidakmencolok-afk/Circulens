from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# CIRCULENS QUALITY — GLOBAL V5/V6 SEQUENCE PATTERN ANALYSIS
# =============================================================================
#
# PURPOSE
# -------
# Menganalisis seluruh sequence yang tersedia di audit_master_v5_v6.csv
# untuk menemukan pola error V5 -> V6.
#
# SCRIPT INI TIDAK:
#   - mengubah dataset
#   - mengubah label
#   - melakukan training
#   - melakukan manual review
#
# SCRIPT INI HANYA UNTUK DIAGNOSTIC / ANALYSIS.
#
# OUTPUT UTAMA:
#   1. sequence-level performance
#   2. V5 -> V6 transition matrix
#   3. error type distribution
#   4. ranking sequence berdasarkan V6 error
#   5. ranking sequence berdasarkan V5 -> V6 regression
#   6. sequence dengan pola RIPE -> UNRIPE / OLD
#   7. kandidat sequence untuk manual review berikutnya
#
# =============================================================================


# =============================================================================
# PATH CONFIGURATION
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

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "quality_finetune_v6"
    / "v5_v6_error_analysis"
    / "label_audit"
    / "global_sequence_analysis"
)


# =============================================================================
# CONFIG
# =============================================================================

WIDTH = 100

LABELS = [
    "ripe",
    "unripe",
    "old",
    "damaged",
]


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


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# =============================================================================
# HEADER
# =============================================================================

title(
    "CIRCULENS QUALITY — GLOBAL V5/V6 SEQUENCE PATTERN ANALYSIS"
)

print(f"Audit CSV : {AUDIT_CSV}")
print(f"Output dir: {OUTPUT_DIR}")


# =============================================================================
# 1. CHECK INPUT
# =============================================================================

section(
    1,
    "CHECKING INPUT FILE"
)

if not AUDIT_CSV.exists():

    raise FileNotFoundError(
        f"Audit CSV tidak ditemukan:\n{AUDIT_CSV}"
    )

print("[OK] Audit CSV ditemukan.")


# =============================================================================
# 2. LOAD DATA
# =============================================================================

section(
    2,
    "LOADING AUDIT DATA"
)

audit = pd.read_csv(
    AUDIT_CSV
)

print(
    f"Rows    : {len(audit)}"
)

print(
    f"Columns : {len(audit.columns)}"
)


# =============================================================================
# 3. RESOLVE COLUMNS
# =============================================================================

section(
    3,
    "RESOLVING REQUIRED COLUMNS"
)


sequence_col = find_column(
    audit,
    [
        "sequence_group_v5",
        "sequence",
        "sequence_id",
        "sequence_name",
    ],
)


filename_col = find_column(
    audit,
    [
        "filename_v5",
        "filename",
        "file_name",
        "image_filename",
    ],
)


v5_label_col = find_column(
    audit,
    [
        "label_v5",
        "dataset_label",
        "label",
    ],
)


v5_pred_col = find_column(
    audit,
    [
        "prediction_label_v5",
        "v5_prediction",
        "predicted_label_v5",
    ],
)


v6_pred_col = find_column(
    audit,
    [
        "prediction_label_v6",
        "v6_prediction",
        "predicted_label_v6",
    ],
)


v5_conf_col = find_column(
    audit,
    [
        "prediction_confidence_v5",
        "v5_confidence",
        "confidence_v5",
    ],
    required=False,
)


v6_conf_col = find_column(
    audit,
    [
        "prediction_confidence_v6",
        "v6_confidence",
        "confidence_v6",
    ],
    required=False,
)


print(
    f"[OK] Sequence       : {sequence_col}"
)

print(
    f"[OK] Filename       : {filename_col}"
)

print(
    f"[OK] Dataset label  : {v5_label_col}"
)

print(
    f"[OK] V5 prediction  : {v5_pred_col}"
)

print(
    f"[OK] V6 prediction  : {v6_pred_col}"
)

print(
    f"[OK] V5 confidence  : {v5_conf_col}"
)

print(
    f"[OK] V6 confidence  : {v6_conf_col}"
)


# =============================================================================
# 4. NORMALIZATION
# =============================================================================

section(
    4,
    "NORMALIZING DATA"
)


audit["_sequence"] = (
    audit[sequence_col]
    .astype(str)
    .str.strip()
)


audit["_filename"] = (
    audit[filename_col]
    .astype(str)
    .str.strip()
)


audit["_dataset_label"] = (
    audit[v5_label_col]
    .apply(normalize_label)
)


audit["_v5"] = (
    audit[v5_pred_col]
    .apply(normalize_label)
)


audit["_v6"] = (
    audit[v6_pred_col]
    .apply(normalize_label)
)


if v5_conf_col is not None:

    audit["_v5_conf"] = safe_numeric(
        audit[v5_conf_col]
    )

else:

    audit["_v5_conf"] = np.nan


if v6_conf_col is not None:

    audit["_v6_conf"] = safe_numeric(
        audit[v6_conf_col]
    )

else:

    audit["_v6_conf"] = np.nan


print(
    "[OK] Normalization completed."
)


# =============================================================================
# 5. BASIC DATA QUALITY CHECK
# =============================================================================

section(
    5,
    "BASIC DATA QUALITY CHECK"
)


print(
    f"Unique sequences : "
    f"{audit['_sequence'].nunique()}"
)

print(
    f"Unique filenames : "
    f"{audit['_filename'].nunique()}"
)


duplicate_count = int(
    audit["_filename"]
    .duplicated()
    .sum()
)


print(
    f"Duplicate filenames : {duplicate_count}"
)


if duplicate_count > 0:

    print(
        "[WARNING] Duplicate filename ditemukan."
    )

else:

    print(
        "[OK] Tidak ada duplicate filename."
    )


# =============================================================================
# 6. DEFINE CORRECTNESS
# =============================================================================

section(
    6,
    "DEFINING V5/V6 CORRECTNESS"
)


audit["v5_correct"] = (
    audit["_v5"]
    == audit["_dataset_label"]
)


audit["v6_correct"] = (
    audit["_v6"]
    == audit["_dataset_label"]
)


audit["v5_to_v6_changed"] = (
    audit["_v5"]
    != audit["_v6"]
)


# NOTE:
# Pada tahap global ini kita belum memiliki manual review untuk seluruh
# dataset. Karena itu "correct" di sini berarti cocok dengan dataset label,
# BUKAN manual ground truth.
#
# Sequence 0048 dan 0136 manual review nanti dipakai sebagai external
# validation evidence.


# =============================================================================
# 7. GLOBAL SUMMARY
# =============================================================================

section(
    7,
    "GLOBAL V5/V6 SUMMARY"
)


total_samples = len(audit)

v5_correct_total = int(
    audit["v5_correct"].sum()
)

v6_correct_total = int(
    audit["v6_correct"].sum()
)

v5_error_total = (
    total_samples
    - v5_correct_total
)

v6_error_total = (
    total_samples
    - v6_correct_total
)

changed_total = int(
    audit["v5_to_v6_changed"].sum()
)


print(
    f"Total samples       : {total_samples}"
)

print(
    f"V5 correct vs label : "
    f"{v5_correct_total} "
    f"({pct(v5_correct_total, total_samples):.2f}%)"
)

print(
    f"V5 errors vs label  : "
    f"{v5_error_total} "
    f"({pct(v5_error_total, total_samples):.2f}%)"
)

print(
    f"V6 correct vs label : "
    f"{v6_correct_total} "
    f"({pct(v6_correct_total, total_samples):.2f}%)"
)

print(
    f"V6 errors vs label  : "
    f"{v6_error_total} "
    f"({pct(v6_error_total, total_samples):.2f}%)"
)

print(
    f"V5 → V6 changed     : "
    f"{changed_total} "
    f"({pct(changed_total, total_samples):.2f}%)"
)


# =============================================================================
# 8. GLOBAL V5 -> V6 TRANSITION MATRIX
# =============================================================================

section(
    8,
    "GLOBAL V5 → V6 TRANSITION MATRIX"
)


transition = pd.crosstab(
    audit["_v5"],
    audit["_v6"],
)


transition_labels = [
    "ripe",
    "unripe",
    "old",
    "damaged",
]


transition = transition.reindex(
    index=transition_labels,
    columns=transition_labels,
    fill_value=0,
)


print()
print(
    "Rows    = V5"
)

print(
    "Columns = V6"
)

print()

print(
    transition.to_string()
)


# =============================================================================
# 9. GLOBAL TRANSITION EXPORT
# =============================================================================

transition_output = (
    OUTPUT_DIR
    / "global_v5_to_v6_transition_matrix.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


transition.to_csv(
    transition_output
)


print()
print(
    f"[SAVED] {transition_output}"
)


# =============================================================================
# 10. BUILD SEQUENCE-LEVEL SUMMARY
# =============================================================================

section(
    10,
    "BUILDING SEQUENCE-LEVEL PERFORMANCE"
)


sequence_rows = []


for sequence, group in audit.groupby(
    "_sequence",
    sort=True,
):

    n = len(group)

    v5_correct = int(
        group["v5_correct"].sum()
    )

    v6_correct = int(
        group["v6_correct"].sum()
    )

    v5_errors = (
        n
        - v5_correct
    )

    v6_errors = (
        n
        - v6_correct
    )

    changed = int(
        group["v5_to_v6_changed"].sum()
    )


    # V5 correct -> V6 wrong
    regressions = int(
        (
            group["v5_correct"]
            & ~group["v6_correct"]
        ).sum()
    )


    # V5 wrong -> V6 correct
    improvements = int(
        (
            ~group["v5_correct"]
            & group["v6_correct"]
        ).sum()
    )


    # Both wrong
    still_wrong = int(
        (
            ~group["v5_correct"]
            & ~group["v6_correct"]
        ).sum()
    )


    # Both correct
    correct_both = int(
        (
            group["v5_correct"]
            & group["v6_correct"]
        ).sum()
    )


    row = {

        "sequence": sequence,

        "samples": n,

        "v5_correct": v5_correct,

        "v5_errors": v5_errors,

        "v5_accuracy_pct": round(
            pct(v5_correct, n),
            2,
        ),

        "v6_correct": v6_correct,

        "v6_errors": v6_errors,

        "v6_accuracy_pct": round(
            pct(v6_correct, n),
            2,
        ),

        "v6_error_rate_pct": round(
            pct(v6_errors, n),
            2,
        ),

        "v5_v6_changed": changed,

        "v5_v6_changed_pct": round(
            pct(changed, n),
            2,
        ),

        "regressions": regressions,

        "regression_pct": round(
            pct(regressions, n),
            2,
        ),

        "improvements": improvements,

        "improvement_pct": round(
            pct(improvements, n),
            2,
        ),

        "still_wrong": still_wrong,

        "correct_in_both": correct_both,

        "net_accuracy_change_pct": round(
            pct(v6_correct, n)
            - pct(v5_correct, n),
            2,
        ),
    }

    sequence_rows.append(
        row
    )


sequence_summary = pd.DataFrame(
    sequence_rows
)


# =============================================================================
# 11. SEQUENCE SUMMARY DISPLAY
# =============================================================================

section(
    11,
    "SEQUENCE-LEVEL PERFORMANCE"
)


display_columns = [
    "sequence",
    "samples",
    "v5_accuracy_pct",
    "v6_accuracy_pct",
    "v6_error_rate_pct",
    "regressions",
    "improvements",
    "still_wrong",
    "net_accuracy_change_pct",
]


print()

print(
    sequence_summary[
        display_columns
    ]
    .sort_values(
        "v6_accuracy_pct",
        ascending=True,
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# 12. EXPORT SEQUENCE SUMMARY
# =============================================================================

sequence_summary_output = (
    OUTPUT_DIR
    / "sequence_level_v5_v6_summary.csv"
)


sequence_summary.to_csv(
    sequence_summary_output,
    index=False,
)


print()
print(
    f"[SAVED] {sequence_summary_output}"
)


# =============================================================================
# 13. WORST V6 SEQUENCES
# =============================================================================

section(
    13,
    "WORST SEQUENCES BY V6 ACCURACY"
)


worst_v6 = (
    sequence_summary
    .sort_values(
        [
            "v6_accuracy_pct",
            "samples",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .head(20)
)


print()

print(
    worst_v6[
        [
            "sequence",
            "samples",
            "v5_accuracy_pct",
            "v6_accuracy_pct",
            "v6_error_rate_pct",
            "regressions",
            "improvements",
        ]
    ]
    .to_string(
        index=False
    )
)


worst_v6_output = (
    OUTPUT_DIR
    / "worst_sequences_by_v6_accuracy.csv"
)


worst_v6.to_csv(
    worst_v6_output,
    index=False,
)


print()
print(
    f"[SAVED] {worst_v6_output}"
)


# =============================================================================
# 14. WORST REGRESSION SEQUENCES
# =============================================================================

section(
    14,
    "SEQUENCES WITH STRONGEST V5 → V6 REGRESSION"
)


worst_regression = (
    sequence_summary
    .sort_values(
        [
            "regression_pct",
            "regressions",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(20)
)


print()

print(
    worst_regression[
        [
            "sequence",
            "samples",
            "v5_accuracy_pct",
            "v6_accuracy_pct",
            "regressions",
            "improvements",
            "net_accuracy_change_pct",
        ]
    ]
    .to_string(
        index=False
    )
)


worst_regression_output = (
    OUTPUT_DIR
    / "sequences_with_v5_v6_regression.csv"
)


worst_regression.to_csv(
    worst_regression_output,
    index=False,
)


print()
print(
    f"[SAVED] {worst_regression_output}"
)


# =============================================================================
# 15. GLOBAL ERROR TYPES
# =============================================================================

section(
    15,
    "GLOBAL V5 → V6 ERROR TRANSITIONS"
)


changed = audit[
    audit["_v5"]
    != audit["_v6"]
].copy()


if len(changed) == 0:

    print(
        "Tidak ada V5 → V6 perubahan."
    )

    global_error_summary = pd.DataFrame(
        columns=[
            "error_type",
            "count",
            "percentage_of_changed",
        ]
    )

else:

    changed["transition"] = (
        changed["_v5"]
        + " -> "
        + changed["_v6"]
    )

    global_error_summary = (
        changed["transition"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(
            name="count"
        )
    )

    global_error_summary[
        "percentage_of_changed"
    ] = (
        global_error_summary["count"]
        / len(changed)
        * 100
    ).round(2)


    print()

    print(
        global_error_summary.to_string(
            index=False
        )
    )


global_error_output = (
    OUTPUT_DIR
    / "global_v5_v6_error_transitions.csv"
)


global_error_summary.to_csv(
    global_error_output,
    index=False,
)


print()
print(
    f"[SAVED] {global_error_output}"
)


# =============================================================================
# 16. RIPE → UNRIPE / OLD ANALYSIS
# =============================================================================

section(
    16,
    "SPECIAL ANALYSIS — RIPE → UNRIPE / OLD"
)


ripe_to_unripe = audit[
    (
        audit["_v5"] == "ripe"
    )
    &
    (
        audit["_v6"] == "unripe"
    )
].copy()


ripe_to_old = audit[
    (
        audit["_v5"] == "ripe"
    )
    &
    (
        audit["_v6"] == "old"
    )
].copy()


print(
    f"RIPE → UNRIPE : "
    f"{len(ripe_to_unripe)}"
)


print(
    f"RIPE → OLD    : "
    f"{len(ripe_to_old)}"
)


# =============================================================================
# 17. RIPE TRANSITION BY SEQUENCE
# =============================================================================

section(
    17,
    "RIPE → UNRIPE / OLD BY SEQUENCE"
)


ripe_transition_rows = []


for sequence, group in audit.groupby(
    "_sequence",
    sort=True,
):

    ripe_v5 = group[
        group["_v5"] == "ripe"
    ]


    if len(ripe_v5) == 0:
        continue


    ripe_to_unripe_count = int(
        (
            ripe_v5["_v6"]
            == "unripe"
        ).sum()
    )


    ripe_to_old_count = int(
        (
            ripe_v5["_v6"]
            == "old"
        ).sum()
    )


    ripe_stayed_ripe = int(
        (
            ripe_v5["_v6"]
            == "ripe"
        ).sum()
    )


    ripe_transition_rows.append(
        {
            "sequence": sequence,
            "v5_ripe_samples": len(ripe_v5),
            "ripe_to_ripe": ripe_stayed_ripe,
            "ripe_to_unripe": ripe_to_unripe_count,
            "ripe_to_old": ripe_to_old_count,
            "ripe_transition_errors": (
                ripe_to_unripe_count
                + ripe_to_old_count
            ),
            "ripe_error_rate_pct": round(
                pct(
                    ripe_to_unripe_count
                    + ripe_to_old_count,
                    len(ripe_v5),
                ),
                2,
            ),
        }
    )


ripe_transition_summary = pd.DataFrame(
    ripe_transition_rows
)


ripe_transition_summary = (
    ripe_transition_summary
    .sort_values(
        [
            "ripe_error_rate_pct",
            "v5_ripe_samples",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


print()

print(
    ripe_transition_summary.to_string(
        index=False
    )
)


ripe_transition_output = (
    OUTPUT_DIR
    / "ripe_transition_by_sequence.csv"
)


ripe_transition_summary.to_csv(
    ripe_transition_output,
    index=False,
)


print()
print(
    f"[SAVED] {ripe_transition_output}"
)


# =============================================================================
# 18. SEQUENCE CANDIDATES FOR MANUAL REVIEW
# =============================================================================

section(
    18,
    "CANDIDATES FOR NEXT MANUAL REVIEW"
)


# Ranking logic:
#
# Priority 1:
#   Banyak V5 -> V6 regressions
#
# Priority 2:
#   V6 accuracy rendah
#
# Priority 3:
#   Banyak RIPE -> UNRIPE / OLD
#
# Kita tidak mengklaim bahwa sequence tersebut memiliki label salah.
# Ini hanya PRIORITY CANDIDATE untuk investigasi.


candidate = sequence_summary.copy()


ripe_lookup = (
    ripe_transition_summary[
        [
            "sequence",
            "v5_ripe_samples",
            "ripe_to_unripe",
            "ripe_to_old",
            "ripe_error_rate_pct",
        ]
    ]
)


candidate = candidate.merge(
    ripe_lookup,
    on="sequence",
    how="left",
)


candidate[
    [
        "v5_ripe_samples",
        "ripe_to_unripe",
        "ripe_to_old",
        "ripe_error_rate_pct",
    ]
] = candidate[
    [
        "v5_ripe_samples",
        "ripe_to_unripe",
        "ripe_to_old",
        "ripe_error_rate_pct",
    ]
].fillna(0)


# Composite diagnostic score.
#
# Ini bukan statistical model.
# Hanya ranking eksploratif untuk menentukan sequence mana
# yang paling informatif untuk manual review berikutnya.


candidate["priority_score"] = (
    candidate["regressions"] * 3
    + candidate["ripe_to_unripe"] * 2
    + candidate["ripe_to_old"] * 2
    + candidate["v6_errors"]
)


candidate = candidate.sort_values(
    [
        "priority_score",
        "v6_errors",
        "regressions",
    ],
    ascending=[
        False,
        False,
        False,
    ],
)


candidate["review_priority"] = range(
    1,
    len(candidate) + 1,
)


candidate_columns = [
    "review_priority",
    "sequence",
    "samples",
    "v5_accuracy_pct",
    "v6_accuracy_pct",
    "v6_errors",
    "regressions",
    "improvements",
    "ripe_to_unripe",
    "ripe_to_old",
    "ripe_error_rate_pct",
    "priority_score",
]


print()

print(
    candidate[
        candidate_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


candidate_output = (
    OUTPUT_DIR
    / "manual_review_priority_candidates.csv"
)


candidate.to_csv(
    candidate_output,
    index=False,
)


print()
print(
    f"[SAVED] {candidate_output}"
)


# =============================================================================
# 19. SEQUENCE 0048 / 0136 CROSS-CHECK
# =============================================================================

section(
    19,
    "CROSS-CHECK — SEQUENCE 0048 / 0136"
)


target_sequences = [
    "sequence_0048",
    "sequence_0136",
]


cross_check = sequence_summary[
    sequence_summary["sequence"]
    .isin(target_sequences)
].copy()


cross_check = cross_check[
    [
        "sequence",
        "samples",
        "v5_accuracy_pct",
        "v6_accuracy_pct",
        "v6_error_rate_pct",
        "v5_v6_changed",
        "regressions",
        "improvements",
        "still_wrong",
        "net_accuracy_change_pct",
    ]
]


print()

print(
    cross_check.to_string(
        index=False
    )
)


cross_check_output = (
    OUTPUT_DIR
    / "sequence_0048_0136_global_cross_check.csv"
)


cross_check.to_csv(
    cross_check_output,
    index=False,
)


print()
print(
    f"[SAVED] {cross_check_output}"
)


# =============================================================================
# 20. CONFIDENCE ANALYSIS
# =============================================================================

section(
    20,
    "V6 CONFIDENCE ANALYSIS ON V6 ERRORS"
)


confidence_rows = []


for sequence, group in audit.groupby(
    "_sequence",
    sort=True,
):

    errors = group[
        ~group["v6_correct"]
    ]


    if len(errors) == 0:
        continue


    valid_conf = errors[
        "_v6_conf"
    ].dropna()


    if len(valid_conf) == 0:

        mean_conf = np.nan
        median_conf = np.nan
        high_conf_errors = np.nan

    else:

        mean_conf = valid_conf.mean()

        median_conf = valid_conf.median()

        high_conf_errors = int(
            (
                valid_conf >= 0.80
            ).sum()
        )


    confidence_rows.append(
        {
            "sequence": sequence,

            "v6_errors": len(errors),

            "mean_v6_error_confidence": (
                round(
                    mean_conf,
                    4,
                )
                if not pd.isna(mean_conf)
                else np.nan
            ),

            "median_v6_error_confidence": (
                round(
                    median_conf,
                    4,
                )
                if not pd.isna(median_conf)
                else np.nan
            ),

            "high_confidence_errors_ge_080": (
                high_conf_errors
            ),
        }
    )


confidence_summary = pd.DataFrame(
    confidence_rows
)


confidence_summary = (
    confidence_summary
    .sort_values(
        [
            "mean_v6_error_confidence",
            "v6_errors",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


print()

print(
    confidence_summary
    .head(20)
    .to_string(
        index=False
    )
)


confidence_output = (
    OUTPUT_DIR
    / "v6_error_confidence_by_sequence.csv"
)


confidence_summary.to_csv(
    confidence_output,
    index=False,
)


print()
print(
    f"[SAVED] {confidence_output}"
)


# =============================================================================
# 21. HIGH-CONFIDENCE V6 ERRORS
# =============================================================================

section(
    21,
    "HIGH-CONFIDENCE V6 ERRORS"
)


high_conf_errors = audit[
    (
        ~audit["v6_correct"]
    )
    &
    (
        audit["_v6_conf"]
        >= 0.80
    )
].copy()


print(
    f"Total V6 errors              : "
    f"{v6_error_total}"
)


print(
    f"High-confidence V6 errors >=80% : "
    f"{len(high_conf_errors)}"
)


print(
    f"Percentage of V6 errors      : "
    f"{pct(len(high_conf_errors), v6_error_total):.2f}%"
)


high_conf_output = (
    OUTPUT_DIR
    / "high_confidence_v6_errors.csv"
)


high_conf_errors[
    [
        "_sequence",
        "_filename",
        "_dataset_label",
        "_v5",
        "_v6",
        "_v6_conf",
    ]
].to_csv(
    high_conf_output,
    index=False,
)


print()

print(
    f"[SAVED] {high_conf_output}"
)


# =============================================================================
# 22. DIAGNOSTIC SUMMARY
# =============================================================================

section(
    22,
    "DIAGNOSTIC SUMMARY"
)


worst_sequence = (
    sequence_summary
    .sort_values(
        "v6_accuracy_pct",
        ascending=True,
    )
    .iloc[0]
)


highest_regression_sequence = (
    sequence_summary
    .sort_values(
        [
            "regression_pct",
            "regressions",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .iloc[0]
)


highest_ripe_error_sequence = (
    ripe_transition_summary
    .sort_values(
        [
            "ripe_error_rate_pct",
            "v5_ripe_samples",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .iloc[0]
)


print()

print(
    "Worst V6 accuracy sequence:"
)

print(
    f" - {worst_sequence['sequence']}"
)

print(
    f" - Samples : "
    f"{int(worst_sequence['samples'])}"
)

print(
    f" - V5 accuracy : "
    f"{worst_sequence['v5_accuracy_pct']:.2f}%"
)

print(
    f" - V6 accuracy : "
    f"{worst_sequence['v6_accuracy_pct']:.2f}%"
)


print()

print(
    "Strongest V5 → V6 regression sequence:"
)

print(
    f" - {highest_regression_sequence['sequence']}"
)

print(
    f" - Regressions : "
    f"{int(highest_regression_sequence['regressions'])}"
)

print(
    f" - Improvements : "
    f"{int(highest_regression_sequence['improvements'])}"
)

print(
    f" - Net accuracy change : "
    f"{highest_regression_sequence['net_accuracy_change_pct']:.2f} pp"
)


print()

print(
    "Strongest RIPE transition error:"
)

print(
    f" - {highest_ripe_error_sequence['sequence']}"
)

print(
    f" - RIPE samples : "
    f"{int(highest_ripe_error_sequence['v5_ripe_samples'])}"
)

print(
    f" - RIPE → UNRIPE : "
    f"{int(highest_ripe_error_sequence['ripe_to_unripe'])}"
)

print(
    f" - RIPE → OLD : "
    f"{int(highest_ripe_error_sequence['ripe_to_old'])}"
)


# =============================================================================
# 23. IMPORTANT INTERPRETATION
# =============================================================================

section(
    23,
    "INTERPRETATION RULES"
)


print()

print(
    "IMPORTANT:"
)

print(
    "1. Global sequence accuracy di sini masih dihitung terhadap"
)

print(
    "   dataset label, BUKAN manual ground truth."
)

print()

print(
    "2. Sequence 0048 menunjukkan bahwa dataset label sendiri"
)

print(
    "   memiliki disagreement terhadap manual review."
)

print()

print(
    "3. Sequence 0136 menunjukkan bahwa dataset label dapat valid"
)

print(
    "   tetapi V6 tetap gagal."
)

print()

print(
    "4. Karena itu ranking ini TIDAK boleh dianggap sebagai bukti"
)

print(
    "   bahwa dataset sequence tersebut salah."
)

print()

print(
    "5. Ranking hanya digunakan untuk memilih sequence yang"
)

print(
    "   paling informatif untuk manual review berikutnya."
)

print()

print(
    "6. Jangan mengubah dataset dan jangan train V7 berdasarkan"
)

print(
    "   output script ini saja."
)


# =============================================================================
# 24. FINAL
# =============================================================================

section(
    24,
    "GLOBAL SEQUENCE ANALYSIS COMPLETED"
)


print()

print(
    f"Total samples       : {total_samples}"
)

print(
    f"Unique sequences    : "
    f"{audit['_sequence'].nunique()}"
)

print(
    f"V5 accuracy/label   : "
    f"{pct(v5_correct_total, total_samples):.2f}%"
)

print(
    f"V6 accuracy/label   : "
    f"{pct(v6_correct_total, total_samples):.2f}%"
)

print(
    f"V5 → V6 changed     : "
    f"{changed_total}"
)

print(
    f"V5 → V6 regressions : "
    f"{int(sequence_summary['regressions'].sum())}"
)

print(
    f"V5 → V6 improvements: "
    f"{int(sequence_summary['improvements'].sum())}"
)

print(
    f"RIPE → UNRIPE       : "
    f"{len(ripe_to_unripe)}"
)

print(
    f"RIPE → OLD          : "
    f"{len(ripe_to_old)}"
)

print(
    f"High-conf V6 errors : "
    f"{len(high_conf_errors)}"
)

print()

print(
    "OUTPUT FILES:"
)

print(
    f" - {sequence_summary_output}"
)

print(
    f" - {transition_output}"
)

print(
    f" - {global_error_output}"
)

print(
    f" - {worst_v6_output}"
)

print(
    f" - {worst_regression_output}"
)

print(
    f" - {ripe_transition_output}"
)

print(
    f" - {candidate_output}"
)

print(
    f" - {confidence_output}"
)

print(
    f" - {high_conf_output}"
)

print(
    f" - {cross_check_output}"
)

print()

print(
    "NEXT STEP:"
)

print(
    "1. Review global sequence ranking."
)

print(
    "2. Identify sequences with repeated V5 → V6 failure patterns."
)

print(
    "3. Prioritize the most informative sequences for manual review."
)

print(
    "4. Compare their visual/source characteristics."
)

print(
    "5. Only after sufficient evidence decide whether dataset"
)

print(
    "   correction or V7 model changes are justified."
)

print()

print(
    "DO NOT TRAIN V7 YET."
)

print()

print("=" * WIDTH)
print(
    "GLOBAL V5/V6 SEQUENCE PATTERN ANALYSIS COMPLETED"
)
print("=" * WIDTH)