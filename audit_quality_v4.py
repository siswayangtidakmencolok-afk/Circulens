from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

MANIFEST_PATH = Path(
    "dataset/circulens_quality_clean_v4/quality_clean_v4_manifest_rebalanced.csv"
)

OUTPUT_ROOT = Path(
    "reports/quality_baseline_v4"
)

SMALL_CELL_THRESHOLD = 5


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("CIRCULENS QUALITY V4 FINAL STATISTICAL AUDIT")
print("=" * 90)


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING MANIFEST")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
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
    "filename"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("PASS: All required columns exist.")


# ============================================================
# 2. BASIC DATA INTEGRITY
# ============================================================

print("\n2. BASIC DATA INTEGRITY")

missing_images = []

for path in df["path"]:

    if not Path(path).exists():
        missing_images.append(path)

print(
    f"Missing images: {len(missing_images)}"
)

if missing_images:

    print("\nExample missing paths:")

    print(
        "\n".join(
            missing_images[:10]
        )
    )

else:

    print(
        "PASS: All image paths exist."
    )


# ============================================================
# 3. GLOBAL SPLIT DISTRIBUTION
# ============================================================

print("\n3. GLOBAL SPLIT DISTRIBUTION")

global_split = (
    df["split"]
    .value_counts()
    .sort_index()
)

global_split_pct = (
    global_split
    / len(df)
    * 100
).round(2)

global_distribution = pd.DataFrame(
    {
        "images": global_split,
        "percentage": global_split_pct
    }
)

print(
    global_distribution
)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n4. CLASS DISTRIBUTION")

class_distribution = (
    df["label"]
    .value_counts()
    .sort_index()
)

class_percentage = (
    class_distribution
    / len(df)
    * 100
).round(2)

class_distribution_table = pd.DataFrame(
    {
        "images": class_distribution,
        "percentage": class_percentage
    }
)

print(
    class_distribution_table
)


# ============================================================
# 5. CLASS × SPLIT
# ============================================================

print("\n5. CLASS × SPLIT")

class_split = (
    df.groupby(
        ["label", "split"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

for split in ["train", "val", "test"]:

    if split not in class_split.columns:
        class_split[split] = 0

class_split = class_split[
    ["train", "val", "test"]
]

class_split["total"] = (
    class_split.sum(axis=1)
)

class_split["train_pct"] = (
    class_split["train"]
    / class_split["total"]
    * 100
).round(2)

class_split["val_pct"] = (
    class_split["val"]
    / class_split["total"]
    * 100
).round(2)

class_split["test_pct"] = (
    class_split["test"]
    / class_split["total"]
    * 100
).round(2)

print(
    class_split
)


# ============================================================
# 6. SOURCE DISTRIBUTION
# ============================================================

print("\n6. SOURCE DISTRIBUTION")

source_distribution = (
    df["source"]
    .value_counts()
    .sort_index()
)

source_percentage = (
    source_distribution
    / len(df)
    * 100
).round(2)

source_distribution_table = pd.DataFrame(
    {
        "images": source_distribution,
        "percentage": source_percentage
    }
)

print(
    source_distribution_table
)


# ============================================================
# 7. SOURCE × SPLIT
# ============================================================

print("\n7. SOURCE × SPLIT")

source_split = (
    df.groupby(
        ["source", "split"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

for split in ["train", "val", "test"]:

    if split not in source_split.columns:
        source_split[split] = 0

source_split = source_split[
    ["train", "val", "test"]
]

source_split["total"] = (
    source_split.sum(axis=1)
)

source_split["train_pct"] = (
    source_split["train"]
    / source_split["total"]
    * 100
).round(2)

source_split["val_pct"] = (
    source_split["val"]
    / source_split["total"]
    * 100
).round(2)

source_split["test_pct"] = (
    source_split["test"]
    / source_split["total"]
    * 100
).round(2)

print(
    source_split
)


# ============================================================
# 8. SOURCE × CLASS
# ============================================================

print("\n8. SOURCE × CLASS")

source_class = (
    df.groupby(
        ["source", "label"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    source_class
)


# ============================================================
# 9. SOURCE × CLASS × SPLIT
# ============================================================

print(
    "\n9. SOURCE × CLASS × SPLIT"
)

source_class_split = (
    df.groupby(
        [
            "source",
            "label",
            "split"
        ]
    )
    .size()
    .reset_index(
        name="images"
    )
)

print(
    source_class_split
)


# ============================================================
# 10. ZERO-SPLIT CHECK
# ============================================================

print("\n10. ZERO-SPLIT CHECK")

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

zero_cells = []

for source in sources:

    for label in classes:

        for split in splits:

            count = len(
                df[
                    (df["source"] == source)
                    &
                    (df["label"] == label)
                    &
                    (df["split"] == split)
                ]
            )

            if count == 0:

                zero_cells.append(
                    {
                        "source": source,
                        "label": label,
                        "split": split,
                        "images": count
                    }
                )

if zero_cells:

    zero_split_df = pd.DataFrame(
        zero_cells
    )

    print(
        "WARNING: Zero-split cells detected."
    )

    print(
        zero_split_df
    )

else:

    zero_split_df = pd.DataFrame(
        columns=[
            "source",
            "label",
            "split",
            "images"
        ]
    )

    print(
        "PASS: Every source × class "
        "exists in train/val/test."
    )


# ============================================================
# 11. SMALL EVALUATION CELLS
# ============================================================

print(
    "\n11. SMALL EVALUATION CELLS"
)

evaluation_df = source_class_split[
    source_class_split["split"].isin(
        ["val", "test"]
    )
].copy()

small_cells = evaluation_df[
    evaluation_df["images"]
    < SMALL_CELL_THRESHOLD
].copy()

print(
    f"Cells with fewer than "
    f"{SMALL_CELL_THRESHOLD} images: "
    f"{len(small_cells)}"
)

if len(small_cells) > 0:

    print(
        small_cells
    )

else:

    print(
        "PASS: No small evaluation cells."
    )


# ============================================================
# 12. SEQUENCE CONSISTENCY
# ============================================================

print(
    "\n12. SEQUENCE CONSISTENCY"
)

sequence_split_counts = (
    df.groupby(
        "sequence_group"
    )["split"]
    .nunique()
)

leaking_sequences = (
    sequence_split_counts[
        sequence_split_counts > 1
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

else:

    print(
        "PASS: No sequence leakage."
    )


# ============================================================
# 13. SEQUENCE DISTRIBUTION
# ============================================================

print(
    "\n13. SEQUENCE DISTRIBUTION"
)

sequence_table = (
    df.groupby(
        "sequence_group"
    )
    .agg(
        source=("source", "first"),
        label=("label", "first"),
        split=("split", "first"),
        image_count=("filename", "count")
    )
    .reset_index()
)

sequence_distribution = (
    sequence_table
    .groupby("split")
    .agg(
        sequences=(
            "sequence_group",
            "count"
        ),
        images=(
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
        )
    )
    .reset_index()
)

print(
    sequence_distribution
)


# ============================================================
# 14. GLOBAL SEQUENCE SIZE STATISTICS
# ============================================================

print(
    "\n14. GLOBAL SEQUENCE SIZE STATISTICS"
)

sequence_sizes = (
    sequence_table["image_count"]
)

print(
    sequence_sizes.describe()
)


# ============================================================
# 15. SOURCE / CLASS COVERAGE
# ============================================================

print(
    "\n15. SOURCE / CLASS COVERAGE"
)

coverage_rows = []

for source in sources:

    for label in classes:

        row = {
            "source": source,
            "label": label
        }

        for split in splits:

            row[split] = len(
                df[
                    (df["source"] == source)
                    &
                    (df["label"] == label)
                    &
                    (df["split"] == split)
                ]
            )

        row["all_splits_present"] = all(
            row[s] > 0
            for s in splits
        )

        coverage_rows.append(row)

coverage_df = pd.DataFrame(
    coverage_rows
)

print(
    coverage_df
)


# ============================================================
# 16. DATASET QUALITY FLAGS
# ============================================================

print(
    "\n16. DATASET QUALITY FLAGS"
)

quality_flags = []

quality_flags.append(
    {
        "check": "missing_images",
        "value": len(missing_images),
        "status":
            "PASS"
            if len(missing_images) == 0
            else "FAIL"
    }
)

quality_flags.append(
    {
        "check": "sequence_leakage",
        "value": len(leaking_sequences),
        "status":
            "PASS"
            if len(leaking_sequences) == 0
            else "FAIL"
    }
)

quality_flags.append(
    {
        "check": "zero_split_cells",
        "value": len(zero_split_df),
        "status":
            "PASS"
            if len(zero_split_df) == 0
            else "FAIL"
    }
)

quality_flags.append(
    {
        "check": "small_evaluation_cells",
        "value": len(small_cells),
        "status":
            "PASS"
            if len(small_cells) == 0
            else "WARNING"
    }
)

quality_flags_df = pd.DataFrame(
    quality_flags
)

print(
    quality_flags_df
)


# ============================================================
# 17. SAVE REPORTS
# ============================================================

print(
    "\n17. SAVING AUDIT REPORTS"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

global_distribution.to_csv(
    OUTPUT_ROOT /
    "global_split_distribution.csv"
)

class_distribution_table.to_csv(
    OUTPUT_ROOT /
    "class_distribution.csv"
)

class_split.to_csv(
    OUTPUT_ROOT /
    "class_split_distribution.csv"
)

source_distribution_table.to_csv(
    OUTPUT_ROOT /
    "source_distribution.csv"
)

source_split.to_csv(
    OUTPUT_ROOT /
    "source_split_distribution.csv"
)

source_class.to_csv(
    OUTPUT_ROOT /
    "source_class_distribution.csv"
)

source_class_split.to_csv(
    OUTPUT_ROOT /
    "source_class_split_distribution.csv",
    index=False
)

zero_split_df.to_csv(
    OUTPUT_ROOT /
    "zero_split_cells.csv",
    index=False
)

small_cells.to_csv(
    OUTPUT_ROOT /
    "small_evaluation_cells.csv",
    index=False
)

sequence_distribution.to_csv(
    OUTPUT_ROOT /
    "sequence_distribution.csv",
    index=False
)

sequence_table.to_csv(
    OUTPUT_ROOT /
    "sequence_table.csv",
    index=False
)

coverage_df.to_csv(
    OUTPUT_ROOT /
    "source_class_coverage.csv",
    index=False
)

quality_flags_df.to_csv(
    OUTPUT_ROOT /
    "quality_flags.csv",
    index=False
)


# ============================================================
# 18. FINAL DIAGNOSTIC
# ============================================================

print("\n")
print("=" * 90)
print("FINAL V4 DATASET DIAGNOSTIC")
print("=" * 90)

print(
    f"Total images       : {len(df)}"
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
    f"Missing images     : "
    f"{len(missing_images)}"
)

print(
    f"Sequence leakage   : "
    f"{len(leaking_sequences)}"
)

print(
    f"Zero coverage      : "
    f"{len(zero_split_df)}"
)

print(
    f"Small eval cells   : "
    f"{len(small_cells)}"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_ROOT
)


# ============================================================
# FINAL STATUS
# ============================================================

structural_pass = (
    len(missing_images) == 0
    and len(leaking_sequences) == 0
    and len(zero_split_df) == 0
)

evaluation_warning = (
    len(small_cells) > 0
)

print("\n")

if structural_pass:

    print(
        "STRUCTURAL CHECK: PASS"
    )

else:

    print(
        "STRUCTURAL CHECK: FAIL"
    )


if evaluation_warning:

    print(
        "EVALUATION CHECK: WARNING"
    )

    print(
        "Some source × class × split "
        "cells are very small."
    )

else:

    print(
        "EVALUATION CHECK: PASS"
    )


print("=" * 90)

if structural_pass:

    print(
        "\nV4 STRUCTURE IS VALID."
    )

if evaluation_warning:

    print(
        "IMPORTANT: Small evaluation cells "
        "must be considered when interpreting "
        "per-source/per-class metrics."
    )

print(
    "\nDO NOT TRAIN THE MODEL YET."
)

print(
    "Next step: review the final audit result "
    "before deciding whether V4 becomes the "
    "training dataset."
)

print("=" * 90)