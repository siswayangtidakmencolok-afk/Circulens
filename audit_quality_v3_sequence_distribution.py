from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

MANIFEST_PATH = Path(
    "dataset/circulens_quality_clean_v3/quality_clean_v3_manifest.csv"
)

REPORT_ROOT = Path(
    "reports/quality_v3_sequence_audit"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("CIRCULENS QUALITY V3 SEQUENCE DISTRIBUTION AUDIT")
print("=" * 90)


# ============================================================
# LOAD MANIFEST
# ============================================================

print("\n1. LOADING V3 MANIFEST")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )

df = pd.read_csv(MANIFEST_PATH)

print(f"Total images: {len(df)}")

required_columns = [
    "task",
    "split",
    "label",
    "source",
    "sequence_group",
    "timestamp",
    "path",
    "filename",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("PASS: Required columns exist.")


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

print("\n2. BASIC DISTRIBUTION")


print("\nImages per split:")

images_per_split = (
    df["split"]
    .value_counts()
    .sort_index()
)

print(images_per_split)


print("\nSequences per split:")

sequences_per_split = (
    df.groupby("split")["sequence_group"]
    .nunique()
    .sort_index()
)

print(sequences_per_split)


print("\nImages per class:")

images_per_class = (
    df["label"]
    .value_counts()
    .sort_index()
)

print(images_per_class)


print("\nSequences per class:")

sequences_per_class = (
    df.groupby("label")["sequence_group"]
    .nunique()
    .sort_index()
)

print(sequences_per_class)


# ============================================================
# BUILD SEQUENCE TABLE
# ============================================================

print("\n3. BUILDING SEQUENCE TABLE")

sequence_table = (
    df.groupby("sequence_group")
    .agg(
        source=("source", "first"),
        label=("label", "first"),
        split=("split", "first"),
        image_count=("filename", "count"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
    .reset_index()
)

print(
    f"Total sequences: {len(sequence_table)}"
)


# ============================================================
# VERIFY SEQUENCE CONSISTENCY
# ============================================================

print("\n4. VERIFYING SEQUENCE CONSISTENCY")

sequence_split_check = (
    df.groupby("sequence_group")["split"]
    .nunique()
)

leaking_sequences = (
    sequence_split_check[
        sequence_split_check > 1
    ]
)

print(
    "Sequences appearing in multiple splits:",
    len(leaking_sequences)
)

if len(leaking_sequences) > 0:

    print(
        leaking_sequences
    )

    raise ValueError(
        "Sequence leakage detected."
    )

print(
    "PASS: Every sequence belongs to exactly one split."
)


# ============================================================
# SOURCE × CLASS SEQUENCE DISTRIBUTION
# ============================================================

print("\n5. SOURCE × CLASS SEQUENCE DISTRIBUTION")

source_class_sequences = (
    sequence_table
    .groupby(
        [
            "source",
            "label"
        ]
    )
    .agg(
        sequence_count=(
            "sequence_group",
            "count"
        ),
        total_images=(
            "image_count",
            "sum"
        ),
        min_images=(
            "image_count",
            "min"
        ),
        median_images=(
            "image_count",
            "median"
        ),
        max_images=(
            "image_count",
            "max"
        ),
    )
    .reset_index()
)

print(
    source_class_sequences.to_string(
        index=False
    )
)


# ============================================================
# SOURCE × CLASS × SPLIT SEQUENCE COUNTS
# ============================================================

print(
    "\n6. SOURCE × CLASS × SPLIT SEQUENCE COUNTS"
)

source_class_split_sequences = (
    sequence_table
    .groupby(
        [
            "source",
            "label",
            "split"
        ]
    )
    .agg(
        sequence_count=(
            "sequence_group",
            "count"
        ),
        image_count=(
            "image_count",
            "sum"
        ),
    )
    .reset_index()
)

print(
    source_class_split_sequences.to_string(
        index=False
    )
)


# ============================================================
# COMPLETE SOURCE × CLASS × SPLIT MATRIX
# ============================================================

print(
    "\n7. BUILDING COMPLETE SOURCE × CLASS × SPLIT MATRIX"
)

sources = sorted(
    df["source"].unique()
)

classes = sorted(
    df["label"].unique()
)

splits = [
    "train",
    "val",
    "test"
]

matrix_rows = []

for source in sources:

    for label in classes:

        for split in splits:

            subset = sequence_table[
                (sequence_table["source"] == source)
                &
                (sequence_table["label"] == label)
                &
                (sequence_table["split"] == split)
            ]

            matrix_rows.append(
                {
                    "source": source,
                    "label": label,
                    "split": split,
                    "images": int(
                        subset["image_count"].sum()
                    ),
                    "sequences": int(
                        len(subset)
                    ),
                }
            )

complete_matrix = pd.DataFrame(
    matrix_rows
)

print(
    complete_matrix.to_string(
        index=False
    )
)


# ============================================================
# ZERO-SPLIT CHECK
# ============================================================

print("\n8. ZERO-SPLIT CHECK")

zero_cells = complete_matrix[
    (complete_matrix["images"] == 0)
    |
    (complete_matrix["sequences"] == 0)
].copy()

if zero_cells.empty:

    print(
        "PASS: Every source × class "
        "exists in train/val/test."
    )

else:

    print(
        f"WARNING: {len(zero_cells)} zero cells detected."
    )

    print(
        zero_cells.to_string(
            index=False
        )
    )


# ============================================================
# SMALL SEQUENCE CELLS
# ============================================================

print(
    "\n9. SMALL SOURCE × CLASS × SPLIT CELLS"
)

small_cells = complete_matrix[
    complete_matrix["images"] < 5
].copy()

if small_cells.empty:

    print(
        "PASS: No source × class × split "
        "cell has fewer than 5 images."
    )

else:

    print(
        f"WARNING: {len(small_cells)} cells "
        f"have fewer than 5 images."
    )

    print(
        small_cells.to_string(
            index=False
        )
    )


# ============================================================
# SEQUENCE SIZE ANALYSIS
# ============================================================

print("\n10. SEQUENCE SIZE ANALYSIS")

sequence_size_stats = (
    sequence_table
    .groupby(
        [
            "source",
            "label"
        ]
    )
    .agg(
        sequence_count=(
            "sequence_group",
            "count"
        ),
        min_images=(
            "image_count",
            "min"
        ),
        median_images=(
            "image_count",
            "median"
        ),
        mean_images=(
            "image_count",
            "mean"
        ),
        max_images=(
            "image_count",
            "max"
        ),
        total_images=(
            "image_count",
            "sum"
        ),
    )
    .reset_index()
)

print(
    sequence_size_stats.to_string(
        index=False
    )
)


# ============================================================
# GLOBAL SEQUENCE SIZE STATISTICS
# ============================================================

print(
    "\n11. GLOBAL SEQUENCE SIZE STATISTICS"
)

global_sequence_stats = (
    sequence_table["image_count"]
    .describe()
)

print(
    global_sequence_stats
)


# ============================================================
# SEQUENCE SIZE BY SPLIT
# ============================================================

print(
    "\n12. SEQUENCE SIZE BY SPLIT"
)

sequence_split_stats = (
    sequence_table
    .groupby("split")
    .agg(
        sequence_count=(
            "sequence_group",
            "count"
        ),
        total_images=(
            "image_count",
            "sum"
        ),
        min_images=(
            "image_count",
            "min"
        ),
        median_images=(
            "image_count",
            "median"
        ),
        mean_images=(
            "image_count",
            "mean"
        ),
        max_images=(
            "image_count",
            "max"
        ),
    )
    .reset_index()
)

print(
    sequence_split_stats.to_string(
        index=False
    )
)


# ============================================================
# SMALL SEQUENCES
# ============================================================

print(
    "\n13. SEQUENCES WITH VERY FEW IMAGES"
)

small_sequences = (
    sequence_table[
        sequence_table["image_count"] <= 3
    ]
    .sort_values(
        [
            "source",
            "label",
            "split",
            "image_count"
        ]
    )
)

print(
    f"Sequences with <= 3 images: "
    f"{len(small_sequences)}"
)

if not small_sequences.empty:

    print(
        small_sequences[
            [
                "sequence_group",
                "source",
                "label",
                "split",
                "image_count",
                "first_timestamp",
                "last_timestamp",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# LARGE SEQUENCES
# ============================================================

print(
    "\n14. LARGEST SEQUENCES"
)

largest_sequences = (
    sequence_table
    .sort_values(
        "image_count",
        ascending=False
    )
    .head(20)
)

print(
    largest_sequences[
        [
            "sequence_group",
            "source",
            "label",
            "split",
            "image_count",
            "first_timestamp",
            "last_timestamp",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# CLASS SPLIT RATIOS
# ============================================================

print(
    "\n15. CLASS SPLIT RATIOS"
)

class_split = (
    df.groupby(
        [
            "label",
            "split"
        ]
    )
    .size()
    .reset_index(
        name="images"
    )
)

class_totals = (
    df.groupby("label")
    .size()
    .rename("total_images")
    .reset_index()
)

class_split = class_split.merge(
    class_totals,
    on="label"
)

class_split["percentage"] = (
    class_split["images"]
    /
    class_split["total_images"]
    * 100
)

print(
    class_split.to_string(
        index=False
    )
)


# ============================================================
# SOURCE SPLIT RATIOS
# ============================================================

print(
    "\n16. SOURCE SPLIT RATIOS"
)

source_split = (
    df.groupby(
        [
            "source",
            "split"
        ]
    )
    .size()
    .reset_index(
        name="images"
    )
)

source_totals = (
    df.groupby("source")
    .size()
    .rename("total_images")
    .reset_index()
)

source_split = source_split.merge(
    source_totals,
    on="source"
)

source_split["percentage"] = (
    source_split["images"]
    /
    source_split["total_images"]
    * 100
)

print(
    source_split.to_string(
        index=False
    )
)


# ============================================================
# SAVE REPORTS
# ============================================================

print(
    "\n17. SAVING AUDIT REPORTS"
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


sequence_table.to_csv(
    REPORT_ROOT / "sequence_table.csv",
    index=False
)

source_class_sequences.to_csv(
    REPORT_ROOT
    / "source_class_sequence_distribution.csv",
    index=False
)

source_class_split_sequences.to_csv(
    REPORT_ROOT
    / "source_class_split_sequence_distribution.csv",
    index=False
)

complete_matrix.to_csv(
    REPORT_ROOT
    / "complete_source_class_split_matrix.csv",
    index=False
)

zero_cells.to_csv(
    REPORT_ROOT
    / "zero_split_cells.csv",
    index=False
)

small_cells.to_csv(
    REPORT_ROOT
    / "small_evaluation_cells.csv",
    index=False
)

sequence_size_stats.to_csv(
    REPORT_ROOT
    / "sequence_size_statistics.csv",
    index=False
)

sequence_split_stats.to_csv(
    REPORT_ROOT
    / "sequence_size_by_split.csv",
    index=False
)

small_sequences.to_csv(
    REPORT_ROOT
    / "small_sequences.csv",
    index=False
)

largest_sequences.to_csv(
    REPORT_ROOT
    / "largest_sequences.csv",
    index=False
)

class_split.to_csv(
    REPORT_ROOT
    / "class_split_ratios.csv",
    index=False
)

source_split.to_csv(
    REPORT_ROOT
    / "source_split_ratios.csv",
    index=False
)


# ============================================================
# FINAL DIAGNOSTIC
# ============================================================

print("\n")
print("=" * 90)
print("FINAL V3 SEQUENCE DISTRIBUTION DIAGNOSTIC")
print("=" * 90)

print(
    f"\nTotal images       : {len(df)}"
)

print(
    f"Total sequences    : "
    f"{df['sequence_group'].nunique()}"
)

print(
    f"Sources            : "
    f"{df['source'].nunique()}"
)

print(
    f"Classes            : "
    f"{df['label'].nunique()}"
)

print(
    f"Zero split cells   : "
    f"{len(zero_cells)}"
)

print(
    f"Small eval cells   : "
    f"{len(small_cells)}"
)

print(
    f"Small sequences    : "
    f"{len(small_sequences)}"
)

print(
    "\nOutput directory:"
)

print(
    REPORT_ROOT
)

print("\n")

if len(zero_cells) == 0:

    print(
        "STRUCTURAL COVERAGE: PASS"
    )

else:

    print(
        "STRUCTURAL COVERAGE: WARNING"
    )


if len(small_cells) == 0:

    print(
        "EVALUATION CELL SIZE: PASS"
    )

else:

    print(
        "EVALUATION CELL SIZE: WARNING"
    )


print("=" * 90)
print(
    "V3 SEQUENCE DISTRIBUTION AUDIT COMPLETED"
)
print("=" * 90)