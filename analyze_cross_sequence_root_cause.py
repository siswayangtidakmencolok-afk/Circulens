from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    r"C:\Projects\circulens\reports\quality_finetune_v6"
    r"\\v5_v6_error_analysis\label_audit\manual_review_report"
)

OUTPUT_DIR = BASE_DIR.parent / "global_sequence_analysis"

SEQUENCES = [
    "sequence_0048",
    "sequence_0136",
    "sequence_0125",
]

CONFIDENCE_THRESHOLDS = [0.80, 0.90, 0.95]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

WIDTH = 100


def banner(title):
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def section(number, title):
    print("\n" + "=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def saved(path):
    print(f"[SAVED] {path}")


def normalize_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace(
            {
                "nan": pd.NA,
                "none": pd.NA,
                "": pd.NA,
            }
        )
    )


def normalize_columns(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = normalize_text(df[col])

    return df


def find_column(df, candidates, required=True):
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Column tidak ditemukan. Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# =============================================================================
# REVIEW FILE DISCOVERY
# =============================================================================

def locate_review_csv(sequence):
    review_dir = BASE_DIR / f"{sequence}_review"

    candidates = [
        review_dir / f"{sequence}_manual_decision_reviewed.csv",
        review_dir / f"{sequence}_manual_decision.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fallback: cari CSV yang mengandung manual_decision
    csv_files = list(review_dir.glob("*.csv"))

    for path in csv_files:
        if "manual" in path.name.lower() and "review" in path.name.lower():
            return path

    return None


# =============================================================================
# LOAD AUDIT MASTER
# =============================================================================

def load_audit_master():
    audit_path = (
        BASE_DIR.parent / "audit_master_v5_v6.csv"
    )

    if not audit_path.exists():
        raise FileNotFoundError(
            f"Audit CSV tidak ditemukan:\n{audit_path}"
        )

    print(f"Audit CSV : {audit_path}")

    df = pd.read_csv(audit_path)

    print(f"Audit rows: {len(df)}")

    return df, audit_path


# =============================================================================
# RESOLVE AUDIT COLUMNS
# =============================================================================

def resolve_audit_columns(df):

    mapping = {}

    mapping["filename"] = find_column(
        df,
        [
            "filename_v5",
            "filename",
            "file_name",
        ],
    )

    mapping["sequence"] = find_column(
        df,
        [
            "sequence_group_v5",
            "sequence",
            "sequence_group",
        ],
    )

    mapping["dataset_label"] = find_column(
        df,
        [
            "label_v5",
            "dataset_label",
            "label",
        ],
    )

    mapping["v5_prediction"] = find_column(
        df,
        [
            "prediction_label_v5",
            "v5_prediction",
            "prediction_v5",
        ],
    )

    mapping["v6_prediction"] = find_column(
        df,
        [
            "prediction_label_v6",
            "v6_prediction",
            "prediction_v6",
        ],
    )

    mapping["v5_confidence"] = find_column(
        df,
        [
            "prediction_confidence_v5",
            "v5_confidence",
            "confidence_v5",
        ],
        required=False,
    )

    mapping["v6_confidence"] = find_column(
        df,
        [
            "prediction_confidence_v6",
            "v6_confidence",
            "confidence_v6",
        ],
        required=False,
    )

    return mapping


# =============================================================================
# RESOLVE REVIEW COLUMNS
# =============================================================================

def resolve_review_columns(df):

    mapping = {}

    mapping["filename"] = find_column(
        df,
        [
            "filename",
            "filename_v5",
            "file_name",
        ],
    )

    mapping["manual_decision"] = find_column(
        df,
        [
            "manual_decision",
            "decision",
        ],
    )

    mapping["manual_label"] = find_column(
        df,
        [
            "manual_label",
            "label_manual",
            "review_label",
        ],
    )

    mapping["manual_notes"] = find_column(
        df,
        [
            "manual_notes",
            "notes",
            "review_notes",
        ],
        required=False,
    )

    return mapping


# =============================================================================
# NORMALIZE MANUAL DECISION
# =============================================================================

def normalize_manual_decision(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    mapping = {
        "r": "reviewed",
        "review": "reviewed",
        "reviewed": "reviewed",
        "o": "reviewed",
        "ok": "reviewed",
        "clear": "reviewed",
        "a": "ambiguous",
        "ambiguous": "ambiguous",
        "uncertain": "ambiguous",
        "u": "ambiguous",
    }

    return mapping.get(value, value)


# =============================================================================
# PREPARE ONE SEQUENCE
# =============================================================================

def prepare_sequence(audit_df, sequence):

    review_path = locate_review_csv(sequence)

    if review_path is None:
        raise FileNotFoundError(
            f"Review CSV untuk {sequence} tidak ditemukan."
        )

    print(f"\n[{sequence}]")
    print(f"Review CSV: {review_path}")

    review_df = pd.read_csv(review_path)

    audit_cols = resolve_audit_columns(audit_df)
    review_cols = resolve_review_columns(review_df)

    audit = audit_df.copy()

    audit = audit[
        audit[audit_cols["sequence"]]
        .astype(str)
        .str.strip()
        .str.lower()
        == sequence.lower()
    ].copy()

    audit_small = pd.DataFrame(
        {
            "filename": audit[audit_cols["filename"]].astype(str).str.strip(),
            "sequence": sequence,
            "dataset_label": normalize_text(
                audit[audit_cols["dataset_label"]]
            ),
            "v5_prediction": normalize_text(
                audit[audit_cols["v5_prediction"]]
            ),
            "v6_prediction": normalize_text(
                audit[audit_cols["v6_prediction"]]
            ),
        }
    )

    if audit_cols["v5_confidence"]:
        audit_small["v5_confidence"] = pd.to_numeric(
            audit[audit_cols["v5_confidence"]],
            errors="coerce",
        )
    else:
        audit_small["v5_confidence"] = np.nan

    if audit_cols["v6_confidence"]:
        audit_small["v6_confidence"] = pd.to_numeric(
            audit[audit_cols["v6_confidence"]],
            errors="coerce",
        )
    else:
        audit_small["v6_confidence"] = np.nan

    review_small = pd.DataFrame(
        {
            "filename": review_df[review_cols["filename"]]
            .astype(str)
            .str.strip(),
            "manual_decision": review_df[
                review_cols["manual_decision"]
            ].apply(normalize_manual_decision),
            "manual_label": normalize_text(
                review_df[review_cols["manual_label"]]
            ),
        }
    )

    if review_cols["manual_notes"]:
        review_small["manual_notes"] = review_df[
            review_cols["manual_notes"]
        ]
    else:
        review_small["manual_notes"] = pd.NA

    # Remove accidental duplicate review rows
    review_small = review_small.drop_duplicates(
        subset=["filename"],
        keep="last",
    )

    merged = audit_small.merge(
        review_small,
        on="filename",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Manual review status
    # -------------------------------------------------------------------------

    merged["is_reviewed"] = (
        merged["manual_label"].notna()
        & merged["manual_decision"].notna()
        & (
            merged["manual_decision"]
            .astype("string")
            .isin(["reviewed"])
        )
    )

    # Ambiguous samples should NOT be used as ground truth.
    merged["is_ambiguous"] = (
        merged["manual_decision"]
        .astype("string")
        .isin(["ambiguous"])
    )

    # Clear manually reviewed sample
    merged["is_clear_manual"] = (
        merged["manual_label"].notna()
        & ~merged["is_ambiguous"]
    )

    # -------------------------------------------------------------------------
    # Correctness against MANUAL ground truth
    # -------------------------------------------------------------------------

    merged["v5_correct_manual"] = (
        merged["is_clear_manual"]
        & (
            merged["v5_prediction"]
            == merged["manual_label"]
        )
    )

    merged["v6_correct_manual"] = (
        merged["is_clear_manual"]
        & (
            merged["v6_prediction"]
            == merged["manual_label"]
        )
    )

    merged["dataset_correct_manual"] = (
        merged["is_clear_manual"]
        & (
            merged["dataset_label"]
            == merged["manual_label"]
        )
    )

    # -------------------------------------------------------------------------
    # V5 -> V6 transitions
    # -------------------------------------------------------------------------

    merged["v5_v6_changed"] = (
        merged["v5_prediction"]
        != merged["v6_prediction"]
    )

    merged["v5_to_v6_transition"] = (
        merged["v5_prediction"]
        .fillna("unknown")
        + " -> "
        + merged["v6_prediction"].fillna("unknown")
    )

    # -------------------------------------------------------------------------
    # V6 error transition against manual
    # -------------------------------------------------------------------------

    merged["v6_error_type"] = np.where(
        merged["is_clear_manual"] & ~merged["v6_correct_manual"],
        merged["manual_label"].fillna("unknown")
        + " -> "
        + merged["v6_prediction"].fillna("unknown"),
        pd.NA,
    )

    # -------------------------------------------------------------------------
    # V5 error transition against manual
    # -------------------------------------------------------------------------

    merged["v5_error_type"] = np.where(
        merged["is_clear_manual"] & ~merged["v5_correct_manual"],
        merged["manual_label"].fillna("unknown")
        + " -> "
        + merged["v5_prediction"].fillna("unknown"),
        pd.NA,
    )

    # -------------------------------------------------------------------------
    # Regression / improvement
    # -------------------------------------------------------------------------

    merged["regression"] = (
        merged["is_clear_manual"]
        & merged["v5_correct_manual"]
        & ~merged["v6_correct_manual"]
    )

    merged["improvement"] = (
        merged["is_clear_manual"]
        & ~merged["v5_correct_manual"]
        & merged["v6_correct_manual"]
    )

    merged["still_wrong"] = (
        merged["is_clear_manual"]
        & ~merged["v5_correct_manual"]
        & ~merged["v6_correct_manual"]
    )

    return merged


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "CIRCULENS QUALITY — CROSS-SEQUENCE ROOT-CAUSE ANALYSIS"
    )

    print("Sequences:")
    for seq in SEQUENCES:
        print(f" - {seq}")

    print("\nGround truth:")
    print(" - Manual review only")
    print(" - Ambiguous samples excluded")
    print(" - Dataset labels NOT treated as ground truth")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # 1. INPUT
    # =========================================================================

    section(
        1,
        "CHECKING INPUT FILES",
    )

    audit_df, audit_path = load_audit_master()

    print("[OK] Audit CSV ditemukan.")

    # =========================================================================
    # 2. PREPARE SEQUENCES
    # =========================================================================

    section(
        2,
        "LOADING MANUAL REVIEW DATA",
    )

    sequence_frames = []

    for sequence in SEQUENCES:

        try:

            df = prepare_sequence(
                audit_df,
                sequence,
            )

            sequence_frames.append(df)

            print(
                f"[OK] {sequence}: "
                f"{len(df)} audit samples"
            )

            print(
                f"     Clear manual: "
                f"{df['is_clear_manual'].sum()}"
            )

            print(
                f"     Ambiguous/unreviewed: "
                f"{len(df) - df['is_clear_manual'].sum()}"
            )

        except Exception as exc:

            print(
                f"[ERROR] {sequence}: {exc}"
            )

    if not sequence_frames:
        raise RuntimeError(
            "Tidak ada sequence yang berhasil dimuat."
        )

    combined = pd.concat(
        sequence_frames,
        ignore_index=True,
    )

    # =========================================================================
    # 3. BASIC SUMMARY
    # =========================================================================

    section(
        3,
        "COMBINED DATA SUMMARY",
    )

    print(
        f"Total audit samples        : {len(combined)}"
    )

    print(
        f"Clear manually reviewed    : "
        f"{combined['is_clear_manual'].sum()}"
    )

    print(
        f"Ambiguous/unreviewed       : "
        f"{(~combined['is_clear_manual']).sum()}"
    )

    print(
        f"Unique filenames           : "
        f"{combined['filename'].nunique()}"
    )

    print(
        f"Duplicate filenames        : "
        f"{combined['filename'].duplicated().sum()}"
    )

    # =========================================================================
    # 4. SEQUENCE PERFORMANCE
    # =========================================================================

    section(
        4,
        "CROSS-SEQUENCE MANUAL-GROUND-TRUTH PERFORMANCE",
    )

    rows = []

    for sequence, group in combined.groupby(
        "sequence",
        sort=False,
    ):

        clear = group[group["is_clear_manual"]].copy()

        n = len(clear)

        if n == 0:
            continue

        v5_correct = int(clear["v5_correct_manual"].sum())
        v6_correct = int(clear["v6_correct_manual"].sum())
        dataset_correct = int(
            clear["dataset_correct_manual"].sum()
        )

        regressions = int(
            clear["regression"].sum()
        )

        improvements = int(
            clear["improvement"].sum()
        )

        still_wrong = int(
            clear["still_wrong"].sum()
        )

        changed = int(
            clear["v5_v6_changed"].sum()
        )

        rows.append(
            {
                "sequence": sequence,
                "clear_samples": n,
                "ambiguous_or_unreviewed": len(group) - n,
                "v5_correct": v5_correct,
                "v5_accuracy_pct": round(
                    100 * v5_correct / n,
                    2,
                ),
                "v6_correct": v6_correct,
                "v6_accuracy_pct": round(
                    100 * v6_correct / n,
                    2,
                ),
                "dataset_correct": dataset_correct,
                "dataset_agreement_pct": round(
                    100 * dataset_correct / n,
                    2,
                ),
                "v5_v6_changed": changed,
                "regressions": regressions,
                "improvements": improvements,
                "still_wrong": still_wrong,
                "net_accuracy_change_pct": round(
                    100
                    * (v6_correct - v5_correct)
                    / n,
                    2,
                ),
            }
        )

    sequence_summary = pd.DataFrame(rows)

    print(
        sequence_summary.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_manual_performance.csv"
    )

    sequence_summary.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 5. GLOBAL MANUAL PERFORMANCE
    # =========================================================================

    section(
        5,
        "GLOBAL MANUAL-GROUND-TRUTH PERFORMANCE",
    )

    clear_all = combined[
        combined["is_clear_manual"]
    ].copy()

    total_clear = len(clear_all)

    v5_correct = int(
        clear_all["v5_correct_manual"].sum()
    )

    v6_correct = int(
        clear_all["v6_correct_manual"].sum()
    )

    dataset_correct = int(
        clear_all["dataset_correct_manual"].sum()
    )

    regressions = int(
        clear_all["regression"].sum()
    )

    improvements = int(
        clear_all["improvement"].sum()
    )

    still_wrong = int(
        clear_all["still_wrong"].sum()
    )

    changed = int(
        clear_all["v5_v6_changed"].sum()
    )

    print(
        f"Clear manual samples : {total_clear}"
    )

    print(
        f"V5 correct           : "
        f"{v5_correct}/{total_clear} "
        f"({100*v5_correct/total_clear:.2f}%)"
    )

    print(
        f"V6 correct           : "
        f"{v6_correct}/{total_clear} "
        f"({100*v6_correct/total_clear:.2f}%)"
    )

    print(
        f"Dataset correct      : "
        f"{dataset_correct}/{total_clear} "
        f"({100*dataset_correct/total_clear:.2f}%)"
    )

    print(
        f"V5 → V6 changed      : {changed}"
    )

    print(
        f"Regressions          : {regressions}"
    )

    print(
        f"Improvements         : {improvements}"
    )

    print(
        f"Still wrong          : {still_wrong}"
    )

    # =========================================================================
    # 6. MANUAL CLASS DISTRIBUTION
    # =========================================================================

    section(
        6,
        "MANUAL GROUND-TRUTH CLASS DISTRIBUTION",
    )

    class_distribution = (
        clear_all["manual_label"]
        .value_counts()
        .rename_axis("manual_label")
        .reset_index(name="count")
    )

    class_distribution["percentage"] = (
        100
        * class_distribution["count"]
        / total_clear
    ).round(2)

    print(
        class_distribution.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_manual_class_distribution.csv"
    )

    class_distribution.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 7. V6 ERROR TRANSITIONS
    # =========================================================================

    section(
        7,
        "V6 ERROR TRANSITIONS VS MANUAL GROUND TRUTH",
    )

    v6_errors = clear_all[
        ~clear_all["v6_correct_manual"]
    ].copy()

    v6_transition = (
        v6_errors["v6_error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    if len(v6_errors) > 0:
        v6_transition["percentage"] = (
            100
            * v6_transition["count"]
            / len(v6_errors)
        ).round(2)
    else:
        v6_transition["percentage"] = 0

    print(
        v6_transition.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v6_manual_error_transitions.csv"
    )

    v6_transition.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 8. V5 ERROR TRANSITIONS
    # =========================================================================

    section(
        8,
        "V5 ERROR TRANSITIONS VS MANUAL GROUND TRUTH",
    )

    v5_errors = clear_all[
        ~clear_all["v5_correct_manual"]
    ].copy()

    v5_transition = (
        v5_errors["v5_error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    if len(v5_errors) > 0:
        v5_transition["percentage"] = (
            100
            * v5_transition["count"]
            / len(v5_errors)
        ).round(2)
    else:
        v5_transition["percentage"] = 0

    print(
        v5_transition.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v5_manual_error_transitions.csv"
    )

    v5_transition.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 9. V5 -> V6 TRANSITION
    # =========================================================================

    section(
        9,
        "V5 → V6 TRANSITION PATTERNS",
    )

    changed_df = clear_all[
        clear_all["v5_v6_changed"]
    ].copy()

    transition = (
        changed_df["v5_to_v6_transition"]
        .value_counts()
        .rename_axis("transition")
        .reset_index(name="count")
    )

    if len(changed_df) > 0:
        transition["percentage"] = (
            100
            * transition["count"]
            / len(changed_df)
        ).round(2)
    else:
        transition["percentage"] = 0

    print(
        transition.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v5_to_v6_transitions_manual_subset.csv"
    )

    transition.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 10. CLASS-SPECIFIC V6 PERFORMANCE
    # =========================================================================

    section(
        10,
        "V6 PERFORMANCE BY MANUAL CLASS",
    )

    class_rows = []

    for manual_label, group in clear_all.groupby(
        "manual_label",
        sort=False,
    ):

        n = len(group)

        correct = int(
            group["v6_correct_manual"].sum()
        )

        errors = n - correct

        class_rows.append(
            {
                "manual_class": manual_label,
                "samples": n,
                "v6_correct": correct,
                "v6_errors": errors,
                "v6_accuracy_pct": round(
                    100 * correct / n,
                    2,
                ),
                "v6_error_rate_pct": round(
                    100 * errors / n,
                    2,
                ),
            }
        )

    class_performance = pd.DataFrame(
        class_rows
    ).sort_values(
        "v6_error_rate_pct",
        ascending=False,
    )

    print(
        class_performance.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v6_performance_by_manual_class.csv"
    )

    class_performance.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 11. V6 CONFUSION MATRIX
    # =========================================================================

    section(
        11,
        "GLOBAL V6 CONFUSION MATRIX VS MANUAL",
    )

    v6_confusion = pd.crosstab(
        clear_all["manual_label"],
        clear_all["v6_prediction"],
    )

    print(
        "\nRows = MANUAL GROUND TRUTH"
    )

    print(
        "Columns = V6 PREDICTION\n"
    )

    print(
        v6_confusion.to_string()
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v6_confusion_vs_manual.csv"
    )

    v6_confusion.to_csv(path)

    saved(path)

    # =========================================================================
    # 12. V5 CONFUSION MATRIX
    # =========================================================================

    section(
        12,
        "GLOBAL V5 CONFUSION MATRIX VS MANUAL",
    )

    v5_confusion = pd.crosstab(
        clear_all["manual_label"],
        clear_all["v5_prediction"],
    )

    print(
        "\nRows = MANUAL GROUND TRUTH"
    )

    print(
        "Columns = V5 PREDICTION\n"
    )

    print(
        v5_confusion.to_string()
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v5_confusion_vs_manual.csv"
    )

    v5_confusion.to_csv(path)

    saved(path)

    # =========================================================================
    # 13. ERROR PATTERN BY SEQUENCE
    # =========================================================================

    section(
        13,
        "ERROR TRANSITION BY SEQUENCE",
    )

    sequence_error = (
        clear_all[
            ~clear_all["v6_correct_manual"]
        ]
        .groupby(
            ["sequence", "v6_error_type"]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            ["sequence", "count"],
            ascending=[True, False],
        )
    )

    print(
        sequence_error.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_error_patterns_by_sequence.csv"
    )

    sequence_error.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 14. CLASS → PREDICTION ERROR MATRIX
    # =========================================================================

    section(
        14,
        "MANUAL CLASS → V6 PREDICTION ERROR MATRIX",
    )

    class_prediction_matrix = pd.crosstab(
        clear_all["manual_label"],
        clear_all["v6_prediction"],
    )

    print(
        class_prediction_matrix.to_string()
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_manual_class_vs_v6_prediction.csv"
    )

    class_prediction_matrix.to_csv(path)

    saved(path)

    # =========================================================================
    # 15. CONFIDENCE ANALYSIS
    # =========================================================================

    section(
        15,
        "V6 ERROR CONFIDENCE ANALYSIS",
    )

    v6_errors = clear_all[
        ~clear_all["v6_correct_manual"]
    ].copy()

    if len(v6_errors) > 0:

        confidence_summary = {
            "total_v6_errors": len(v6_errors),
            "mean_confidence": v6_errors[
                "v6_confidence"
            ].mean(),
            "median_confidence": v6_errors[
                "v6_confidence"
            ].median(),
            "min_confidence": v6_errors[
                "v6_confidence"
            ].min(),
            "max_confidence": v6_errors[
                "v6_confidence"
            ].max(),
        }

        print(
            f"Total V6 errors       : "
            f"{confidence_summary['total_v6_errors']}"
        )

        print(
            f"Mean confidence      : "
            f"{confidence_summary['mean_confidence']:.4f}"
        )

        print(
            f"Median confidence    : "
            f"{confidence_summary['median_confidence']:.4f}"
        )

        print(
            f"Minimum confidence   : "
            f"{confidence_summary['min_confidence']:.4f}"
        )

        print(
            f"Maximum confidence   : "
            f"{confidence_summary['max_confidence']:.4f}"
        )

    else:

        confidence_summary = {
            "total_v6_errors": 0,
            "mean_confidence": np.nan,
            "median_confidence": np.nan,
            "min_confidence": np.nan,
            "max_confidence": np.nan,
        }

        print("Tidak ada V6 errors.")

    # =========================================================================
    # 16. CONFIDENCE THRESHOLD BREAKDOWN
    # =========================================================================

    section(
        16,
        "HIGH-CONFIDENCE V6 ERRORS",
    )

    threshold_rows = []

    total_errors = len(v6_errors)

    for threshold in CONFIDENCE_THRESHOLDS:

        count = int(
            (
                v6_errors["v6_confidence"]
                >= threshold
            ).sum()
        )

        percentage = (
            100 * count / total_errors
            if total_errors > 0
            else 0
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "error_count": count,
                "percentage_of_v6_errors": round(
                    percentage,
                    2,
                ),
            }
        )

        print(
            f">= {threshold:.2f} : "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    confidence_thresholds = pd.DataFrame(
        threshold_rows
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_v6_error_confidence_thresholds.csv"
    )

    confidence_thresholds.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 17. HIGH-CONFIDENCE ERROR TYPES
    # =========================================================================

    section(
        17,
        "HIGH-CONFIDENCE ERROR TYPES",
    )

    high_conf = v6_errors[
        v6_errors["v6_confidence"] >= 0.80
    ].copy()

    high_conf_error_types = (
        high_conf["v6_error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    if len(high_conf) > 0:
        high_conf_error_types["percentage"] = (
            100
            * high_conf_error_types["count"]
            / len(high_conf)
        ).round(2)
    else:
        high_conf_error_types["percentage"] = 0

    print(
        high_conf_error_types.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_high_confidence_v6_error_types.csv"
    )

    high_conf_error_types.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 18. HIGH-CONFIDENCE ERROR DETAILS
    # =========================================================================

    section(
        18,
        "HIGH-CONFIDENCE V6 ERROR SAMPLES",
    )

    high_conf_details = high_conf[
        [
            "sequence",
            "filename",
            "manual_label",
            "v5_prediction",
            "v6_prediction",
            "v5_confidence",
            "v6_confidence",
            "v6_error_type",
            "regression",
        ]
    ].copy()

    high_conf_details = high_conf_details.sort_values(
        [
            "v6_confidence",
        ],
        ascending=False,
    )

    print(
        high_conf_details.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_high_confidence_v6_errors.csv"
    )

    high_conf_details.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 19. RIPE FAILURE ANALYSIS
    # =========================================================================

    section(
        19,
        "RIPE CLASS FAILURE ANALYSIS",
    )

    ripe = clear_all[
        clear_all["manual_label"] == "ripe"
    ].copy()

    if len(ripe) > 0:

        ripe_rows = []

        for sequence, group in ripe.groupby(
            "sequence",
            sort=False,
        ):

            n = len(group)

            correct = int(
                group["v6_correct_manual"].sum()
            )

            ripe_to_unripe = int(
                (
                    group["v6_prediction"]
                    == "unripe"
                ).sum()
            )

            ripe_to_old = int(
                (
                    group["v6_prediction"]
                    == "old"
                ).sum()
            )

            ripe_rows.append(
                {
                    "sequence": sequence,
                    "manual_ripe_samples": n,
                    "ripe_correct": correct,
                    "ripe_to_unripe": ripe_to_unripe,
                    "ripe_to_old": ripe_to_old,
                    "ripe_other_errors": n
                    - correct
                    - ripe_to_unripe
                    - ripe_to_old,
                    "ripe_error_rate_pct": round(
                        100 * (n - correct) / n,
                        2,
                    ),
                }
            )

        ripe_analysis = pd.DataFrame(
            ripe_rows
        ).sort_values(
            "ripe_error_rate_pct",
            ascending=False,
        )

        print(
            ripe_analysis.to_string(
                index=False
            )
        )

    else:

        ripe_analysis = pd.DataFrame()

        print(
            "Tidak ada manual RIPE samples."
        )

    path = (
        OUTPUT_DIR
        / "cross_sequence_ripe_failure_analysis.csv"
    )

    ripe_analysis.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 20. OLD FAILURE ANALYSIS
    # =========================================================================

    section(
        20,
        "OLD CLASS FAILURE ANALYSIS",
    )

    old = clear_all[
        clear_all["manual_label"] == "old"
    ].copy()

    if len(old) > 0:

        old_rows = []

        for sequence, group in old.groupby(
            "sequence",
            sort=False,
        ):

            n = len(group)

            correct = int(
                group["v6_correct_manual"].sum()
            )

            old_to_dried = int(
                (
                    group["v6_prediction"]
                    == "dried"
                ).sum()
            )

            old_to_ripe = int(
                (
                    group["v6_prediction"]
                    == "ripe"
                ).sum()
            )

            old_to_unripe = int(
                (
                    group["v6_prediction"]
                    == "unripe"
                ).sum()
            )

            old_rows.append(
                {
                    "sequence": sequence,
                    "manual_old_samples": n,
                    "old_correct": correct,
                    "old_to_dried": old_to_dried,
                    "old_to_ripe": old_to_ripe,
                    "old_to_unripe": old_to_unripe,
                    "old_error_rate_pct": round(
                        100 * (n - correct) / n,
                        2,
                    ),
                }
            )

        old_analysis = pd.DataFrame(
            old_rows
        ).sort_values(
            "old_error_rate_pct",
            ascending=False,
        )

        print(
            old_analysis.to_string(
                index=False
            )
        )

    else:

        old_analysis = pd.DataFrame()

        print(
            "Tidak ada manual OLD samples."
        )

    path = (
        OUTPUT_DIR
        / "cross_sequence_old_failure_analysis.csv"
    )

    old_analysis.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 21. DATASET AGREEMENT ANALYSIS
    # =========================================================================

    section(
        21,
        "DATASET LABEL AGREEMENT VS MANUAL",
    )

    dataset_rows = []

    for sequence, group in clear_all.groupby(
        "sequence",
        sort=False,
    ):

        n = len(group)

        correct = int(
            group["dataset_correct_manual"].sum()
        )

        disagreements = n - correct

        dataset_rows.append(
            {
                "sequence": sequence,
                "clear_samples": n,
                "dataset_agreement": correct,
                "dataset_disagreement": disagreements,
                "agreement_pct": round(
                    100 * correct / n,
                    2,
                ),
            }
        )

    dataset_agreement = pd.DataFrame(
        dataset_rows
    ).sort_values(
        "agreement_pct"
    )

    print(
        dataset_agreement.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_dataset_agreement_vs_manual.csv"
    )

    dataset_agreement.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 22. ROOT CAUSE CLASSIFICATION
    # =========================================================================

    section(
        22,
        "PRELIMINARY ROOT-CAUSE CLASSIFICATION",
    )

    # -------------------------------------------------------------------------
    # Evidence calculations
    # -------------------------------------------------------------------------

    global_dataset_disagreement = int(
        (~clear_all["dataset_correct_manual"]).sum()
    )

    global_v6_errors = int(
        (~clear_all["v6_correct_manual"]).sum()
    )

    global_v6_error_rate = (
        100 * global_v6_errors / total_clear
        if total_clear > 0
        else 0
    )

    high_conf_count = int(
        (
            v6_errors["v6_confidence"]
            >= 0.80
        ).sum()
    )

    high_conf_pct = (
        100 * high_conf_count / global_v6_errors
        if global_v6_errors > 0
        else 0
    )

    # -------------------------------------------------------------------------
    # Dominant error
    # -------------------------------------------------------------------------

    dominant_error = None
    dominant_error_count = 0

    if len(v6_transition) > 0:

        dominant_error = v6_transition.iloc[0][
            "error_type"
        ]

        dominant_error_count = int(
            v6_transition.iloc[0]["count"]
        )

    # -------------------------------------------------------------------------
    # Sequence-level severe failures
    # -------------------------------------------------------------------------

    severe_sequences = sequence_summary[
        sequence_summary["v6_accuracy_pct"] < 50
    ]

    # -------------------------------------------------------------------------
    # Dataset problem indicator
    # -------------------------------------------------------------------------

    dataset_problem_sequences = (
        dataset_agreement[
            dataset_agreement["agreement_pct"] < 90
        ]
    )

    # -------------------------------------------------------------------------
    # Model problem indicator
    # -------------------------------------------------------------------------

    model_failure_sequences = (
        sequence_summary[
            sequence_summary["v6_accuracy_pct"] < 70
        ]
    )

    # -------------------------------------------------------------------------
    # Classification logic
    # -------------------------------------------------------------------------

    if (
        global_v6_errors > 0
        and high_conf_pct >= 40
        and len(model_failure_sequences) >= 2
    ):

        classification = "STRONG_MODEL_ERROR_SIGNAL"

    elif (
        global_v6_errors > 0
        and high_conf_pct >= 25
    ):

        classification = "LIKELY_MODEL_ERROR"

    elif (
        global_dataset_disagreement > 0
        and global_dataset_disagreement
        >= global_v6_errors
    ):

        classification = "POSSIBLE_DATASET_PROBLEM"

    else:

        classification = "MIXED_OR_INCONCLUSIVE"

    print(
        f"Classification : {classification}"
    )

    print(
        f"\nClear samples : {total_clear}"
    )

    print(
        f"Dataset disagreements : "
        f"{global_dataset_disagreement}"
    )

    print(
        f"V6 errors : "
        f"{global_v6_errors}"
    )

    print(
        f"V6 error rate : "
        f"{global_v6_error_rate:.2f}%"
    )

    print(
        f"High-confidence V6 errors >=80% : "
        f"{high_conf_count}"
    )

    print(
        f"High-confidence error percentage : "
        f"{high_conf_pct:.2f}%"
    )

    if dominant_error:
        print(
            f"Dominant V6 error : "
            f"{dominant_error} "
            f"({dominant_error_count})"
        )

    # =========================================================================
    # 23. EVIDENCE REPORT
    # =========================================================================

    section(
        23,
        "ROOT-CAUSE EVIDENCE REPORT",
    )

    evidence_rows = []

    evidence_rows.append(
        {
            "evidence": "manual_clear_samples",
            "value": total_clear,
            "interpretation": (
                "Samples yang benar-benar digunakan "
                "sebagai manual ground truth."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "dataset_disagreements",
            "value": global_dataset_disagreement,
            "interpretation": (
                "Jumlah sample dataset label "
                "yang tidak cocok dengan manual review."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "v6_errors",
            "value": global_v6_errors,
            "interpretation": (
                "Jumlah V6 prediction yang salah "
                "terhadap manual ground truth."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "v6_error_rate_pct",
            "value": round(
                global_v6_error_rate,
                2,
            ),
            "interpretation": (
                "Error rate V6 pada subset manual review."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "high_confidence_errors_ge_80_pct",
            "value": round(
                high_conf_pct,
                2,
            ),
            "interpretation": (
                "Proporsi error V6 yang terjadi "
                "dengan confidence minimal 80%."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "dominant_error",
            "value": dominant_error,
            "interpretation": (
                "Error transition V6 yang paling sering."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "severe_model_failure_sequences",
            "value": len(severe_sequences),
            "interpretation": (
                "Sequence dengan V6 accuracy <50%."
            ),
        }
    )

    evidence_rows.append(
        {
            "evidence": "dataset_problem_sequences",
            "value": len(
                dataset_problem_sequences
            ),
            "interpretation": (
                "Sequence dengan dataset agreement <90%."
            ),
        }
    )

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    print(
        evidence_df.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_root_cause_evidence.csv"
    )

    evidence_df.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 24. DECISION MATRIX
    # =========================================================================

    section(
        24,
        "DATASET VS MODEL DECISION MATRIX",
    )

    decision_rows = []

    for sequence, group in clear_all.groupby(
        "sequence",
        sort=False,
    ):

        n = len(group)

        dataset_disagreement = int(
            (~group["dataset_correct_manual"]).sum()
        )

        v6_errors_seq = int(
            (~group["v6_correct_manual"]).sum()
        )

        v6_error_rate_seq = (
            100 * v6_errors_seq / n
            if n > 0
            else 0
        )

        high_conf_seq = int(
            (
                group.loc[
                    ~group["v6_correct_manual"],
                    "v6_confidence",
                ]
                >= 0.80
            ).sum()
        )

        if v6_errors_seq > 0:

            high_conf_seq_pct = (
                100 * high_conf_seq / v6_errors_seq
            )

        else:

            high_conf_seq_pct = 0

        if (
            dataset_disagreement > 0
            and v6_error_rate_seq < 50
        ):

            interpretation = (
                "DATASET_REVIEW_SIGNAL"
            )

        elif (
            v6_error_rate_seq >= 70
            and high_conf_seq_pct >= 40
            and dataset_disagreement == 0
        ):

            interpretation = (
                "STRONG_MODEL_FAILURE_SIGNAL"
            )

        elif (
            v6_error_rate_seq >= 50
            and dataset_disagreement == 0
        ):

            interpretation = (
                "MODEL_FAILURE_SIGNAL"
            )

        else:

            interpretation = (
                "LOWER_PRIORITY"
            )

        decision_rows.append(
            {
                "sequence": sequence,
                "clear_samples": n,
                "dataset_disagreements": dataset_disagreement,
                "v6_errors": v6_errors_seq,
                "v6_error_rate_pct": round(
                    v6_error_rate_seq,
                    2,
                ),
                "high_conf_v6_errors": high_conf_seq,
                "high_conf_error_pct": round(
                    high_conf_seq_pct,
                    2,
                ),
                "interpretation": interpretation,
            }
        )

    decision_matrix = pd.DataFrame(
        decision_rows
    ).sort_values(
        [
            "v6_error_rate_pct",
            "high_conf_error_pct",
        ],
        ascending=False,
    )

    print(
        decision_matrix.to_string(
            index=False
        )
    )

    path = (
        OUTPUT_DIR
        / "cross_sequence_dataset_vs_model_decision_matrix.csv"
    )

    decision_matrix.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 25. CONSOLIDATED REVIEW DATA
    # =========================================================================

    section(
        25,
        "EXPORTING CONSOLIDATED MANUAL REVIEW DATA",
    )

    export_columns = [
        "sequence",
        "filename",
        "dataset_label",
        "v5_prediction",
        "v6_prediction",
        "v5_confidence",
        "v6_confidence",
        "manual_label",
        "manual_decision",
        "manual_notes",
        "is_clear_manual",
        "v5_correct_manual",
        "v6_correct_manual",
        "dataset_correct_manual",
        "v5_v6_changed",
        "v5_to_v6_transition",
        "v5_error_type",
        "v6_error_type",
        "regression",
        "improvement",
        "still_wrong",
    ]

    consolidated = combined[
        [
            c
            for c in export_columns
            if c in combined.columns
        ]
    ].copy()

    path = (
        OUTPUT_DIR
        / "cross_sequence_manual_review_consolidated.csv"
    )

    consolidated.to_csv(
        path,
        index=False,
    )

    saved(path)

    # =========================================================================
    # 26. FINAL DIAGNOSTIC
    # =========================================================================

    section(
        26,
        "FINAL DIAGNOSTIC",
    )

    print(
        "SEQUENCES ANALYZED:"
    )

    for sequence in SEQUENCES:

        if sequence in sequence_summary[
            "sequence"
        ].values:

            row = sequence_summary[
                sequence_summary["sequence"]
                == sequence
            ].iloc[0]

            print(
                f" - {sequence}: "
                f"{int(row['clear_samples'])} clear | "
                f"V5={row['v5_accuracy_pct']:.2f}% | "
                f"V6={row['v6_accuracy_pct']:.2f}% | "
                f"dataset={row['dataset_agreement_pct']:.2f}%"
            )

    print(
        "\nGLOBAL:"
    )

    print(
        f" - Clear manual samples : {total_clear}"
    )

    print(
        f" - V5 accuracy          : "
        f"{100*v5_correct/total_clear:.2f}%"
    )

    print(
        f" - V6 accuracy          : "
        f"{100*v6_correct/total_clear:.2f}%"
    )

    print(
        f" - Dataset agreement    : "
        f"{100*dataset_correct/total_clear:.2f}%"
    )

    print(
        f" - V6 errors            : {global_v6_errors}"
    )

    print(
        f" - High-conf V6 errors  : "
        f"{high_conf_count}"
    )

    print(
        f" - Dominant V6 error    : "
        f"{dominant_error}"
    )

    print(
        f" - Classification       : "
        f"{classification}"
    )

    # =========================================================================
    # 27. IMPORTANT INTERPRETATION
    # =========================================================================

    section(
        27,
        "INTERPRETATION RULES",
    )

    print(
        """
IMPORTANT:

1. Analisis ini menggunakan MANUAL REVIEW sebagai ground truth.
2. Sample ambiguous/unreviewed TIDAK digunakan untuk menghitung accuracy.
3. Dataset label hanya digunakan untuk mengecek apakah label dataset
   konsisten dengan manual review.
4. Jangan mengubah dataset hanya karena satu sequence bermasalah.
5. High-confidence error menunjukkan model sangat yakin terhadap
   prediction yang salah, tetapi BELUM otomatis berarti architecture
   model harus diganti.
6. Pola error yang berulang pada beberapa sequence lebih penting
   daripada error isolated pada satu sequence.
7. Sequence 0048 harus diperlakukan berbeda karena ditemukan
   dataset-label disagreement.
8. Sequence 0136 memberikan evidence yang lebih kuat untuk
   model failure karena dataset label 100% agreement dengan manual.
9. Sequence 0125 memberikan evidence kuat untuk systematic
   OLD → DRIED confusion karena seluruh 34 reviewed samples
   salah dengan pola yang sama.
10. Jangan train V7 berdasarkan script ini saja.
11. Sebelum V7, tentukan apakah masalah utama berada pada:
      - dataset distribution,
      - class representation,
      - source/domain shift,
      - visual ambiguity,
      - label semantics,
      - preprocessing/augmentation,
      - atau model decision boundary.
"""
    )

    # =========================================================================
    # 28. FINAL OUTPUT LIST
    # =========================================================================

    section(
        28,
        "CROSS-SEQUENCE ROOT-CAUSE ANALYSIS COMPLETED",
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print(
        "\nGenerated files:"
    )

    output_files = sorted(
        OUTPUT_DIR.glob(
            "cross_sequence_*.csv"
        )
    )

    for path in output_files:
        print(f" - {path}")

    print(
        "\nNEXT STEP:"
    )

    print(
        """
1. Inspect cross_sequence_v6_manual_error_transitions.csv
2. Inspect cross_sequence_v6_performance_by_manual_class.csv
3. Inspect cross_sequence_high_confidence_v6_error_types.csv
4. Inspect cross_sequence_dataset_vs_model_decision_matrix.csv
5. Fokus utama:
      RIPE → OLD
      RIPE → UNRIPE
      OLD → DRIED
6. Bandingkan source/visual characteristics dari sample error tersebut.
7. Setelah itu baru putuskan strategi V7.

DO NOT TRAIN V7 YET.
"""
    )

    banner(
        "CROSS-SEQUENCE ROOT-CAUSE ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()