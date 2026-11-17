from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = Path(
    "dataset/circulens_quality_clean_v3/quality_clean_v3_manifest.csv"
)

OUTPUT = Path(
    "reports/quality_clean_v3_final_audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CIRCULENS QUALITY V3 FINAL STATISTICAL AUDIT")
print("=" * 90)


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING MANIFEST")

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")

required = [
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
    col for col in required
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("PASS: All required columns exist.")


# ============================================================
# 2. GLOBAL SPLIT DISTRIBUTION
# ============================================================

print("\n2. GLOBAL SPLIT DISTRIBUTION")

split_counts = (
    df["split"]
    .value_counts()
    .reindex(["train", "val", "test"])
    .fillna(0)
    .astype(int)
)

split_percent = (
    split_counts / len(df) * 100
).round(2)

split_summary = pd.DataFrame({
    "images": split_counts,
    "percentage": split_percent,
})

print(split_summary)

split_summary.to_csv(
    OUTPUT / "global_split_distribution.csv"
)


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\n3. CLASS DISTRIBUTION")

class_counts = (
    df["label"]
    .value_counts()
)

class_percent = (
    class_counts / len(df) * 100
).round(2)

class_summary = pd.DataFrame({
    "images": class_counts,
    "percentage": class_percent,
})

print(class_summary)

class_summary.to_csv(
    OUTPUT / "class_distribution.csv"
)


# ============================================================
# 4. CLASS × SPLIT
# ============================================================

print("\n4. CLASS × SPLIT")

class_split = pd.crosstab(
    df["label"],
    df["split"]
)

for split in ["train", "val", "test"]:
    if split not in class_split.columns:
        class_split[split] = 0

class_split = class_split[
    ["train", "val", "test"]
]

class_split["total"] = class_split.sum(
    axis=1
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

print(class_split)

class_split.to_csv(
    OUTPUT / "class_split_distribution.csv"
)


# ============================================================
# 5. SOURCE DISTRIBUTION
# ============================================================

print("\n5. SOURCE DISTRIBUTION")

source_counts = (
    df["source"]
    .value_counts()
)

source_percent = (
    source_counts / len(df) * 100
).round(2)

source_summary = pd.DataFrame({
    "images": source_counts,
    "percentage": source_percent,
})

print(source_summary)

source_summary.to_csv(
    OUTPUT / "source_distribution.csv"
)


# ============================================================
# 6. SOURCE × SPLIT
# ============================================================

print("\n6. SOURCE × SPLIT")

source_split = pd.crosstab(
    df["source"],
    df["split"]
)

for split in ["train", "val", "test"]:
    if split not in source_split.columns:
        source_split[split] = 0

source_split = source_split[
    ["train", "val", "test"]
]

source_split["total"] = source_split.sum(
    axis=1
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

print(source_split)

source_split.to_csv(
    OUTPUT / "source_split_distribution.csv"
)


# ============================================================
# 7. SOURCE × CLASS
# ============================================================

print("\n7. SOURCE × CLASS")

source_class = pd.crosstab(
    df["source"],
    df["label"]
)

print(source_class)

source_class.to_csv(
    OUTPUT / "source_class_distribution.csv"
)


# ============================================================
# 8. SOURCE × CLASS × SPLIT
# ============================================================

print("\n8. SOURCE × CLASS × SPLIT")

source_class_split = (
    df.groupby(
        ["source", "label", "split"]
    )
    .size()
    .reset_index(
        name="images"
    )
)

print(
    source_class_split
    .to_string(index=False)
)

source_class_split.to_csv(
    OUTPUT / "source_class_split_distribution.csv",
    index=False
)


# ============================================================
# 9. SMALL EVALUATION CELLS
# ============================================================

print("\n9. SMALL EVALUATION CELLS")

matrix = []

for source in sorted(
    df["source"].unique()
):

    for label in sorted(
        df["label"].unique()
    ):

        for split in [
            "train",
            "val",
            "test"
        ]:

            count = len(
                df[
                    (df["source"] == source)
                    & (df["label"] == label)
                    & (df["split"] == split)
                ]
            )

            matrix.append({
                "source": source,
                "label": label,
                "split": split,
                "images": count,
            })

matrix = pd.DataFrame(matrix)

small_cells = matrix[
    matrix["images"] < 5
]

if len(small_cells) == 0:

    print(
        "PASS: No source × class × split "
        "cell has fewer than 5 images."
    )

else:

    print(
        f"WARNING: {len(small_cells)} cells "
        "have fewer than 5 images."
    )

    print(
        small_cells.to_string(
            index=False
        )
    )

small_cells.to_csv(
    OUTPUT / "small_evaluation_cells.csv",
    index=False
)


# ============================================================
# 10. VALIDATION / TEST SAMPLE COUNTS
# ============================================================

print("\n10. VALIDATION / TEST SAMPLE COUNTS")

evaluation = (
    df.groupby(
        ["label", "split"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

for split in ["val", "test"]:

    if split not in evaluation.columns:
        evaluation[split] = 0

evaluation = evaluation[
    ["val", "test"]
]

print(evaluation)

evaluation.to_csv(
    OUTPUT / "evaluation_class_counts.csv"
)


# ============================================================
# 11. SEQUENCE DISTRIBUTION
# ============================================================

print("\n11. SEQUENCE DISTRIBUTION")

sequence_counts = (
    df.groupby(
        "sequence_group"
    )
    .size()
)

print(
    f"Total sequences: "
    f"{len(sequence_counts)}"
)

print("\nSequence size statistics:")

print(
    sequence_counts.describe()
)

sequence_counts.describe().to_csv(
    OUTPUT / "sequence_size_statistics.csv"
)


# ============================================================
# 12. SEQUENCE SPLIT CONSISTENCY
# ============================================================

print("\n12. SEQUENCE SPLIT CONSISTENCY")

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
    f"Sequences appearing in multiple splits: "
    f"{len(leaking_sequences)}"
)

if len(leaking_sequences) == 0:

    print(
        "PASS: No sequence appears "
        "in multiple splits."
    )

else:

    print(
        "FAIL: Sequence leakage detected."
    )

leaking_sequences.to_csv(
    OUTPUT / "leaking_sequences.csv"
)


# ============================================================
# 13. SOURCE/CLASS COVERAGE
# ============================================================

print("\n13. SOURCE / CLASS COVERAGE")

sources = sorted(
    df["source"].unique()
)

classes = sorted(
    df["label"].unique()
)

coverage_rows = []

for source in sources:

    for label in classes:

        subset = df[
            (df["source"] == source)
            & (df["label"] == label)
        ]

        split_presence = {
            split: int(
                (subset["split"] == split).sum()
            )
            for split in [
                "train",
                "val",
                "test"
            ]
        }

        coverage_rows.append({
            "source": source,
            "label": label,
            "train": split_presence["train"],
            "val": split_presence["val"],
            "test": split_presence["test"],
            "all_splits_present": all(
                value > 0
                for value in split_presence.values()
            ),
        })

coverage = pd.DataFrame(
    coverage_rows
)

print(
    coverage.to_string(
        index=False
    )
)

coverage.to_csv(
    OUTPUT / "source_class_coverage.csv",
    index=False
)


# ============================================================
# 14. DATASET QUALITY FLAGS
# ============================================================

print("\n14. DATASET QUALITY FLAGS")

flags = []

# Missing images
missing_paths = sum(
    not Path(path).exists()
    for path in df["path"]
)

if missing_paths > 0:

    flags.append({
        "check": "missing_images",
        "value": missing_paths,
        "status": "FAIL",
    })

else:

    flags.append({
        "check": "missing_images",
        "value": 0,
        "status": "PASS",
    })


# Sequence leakage
if len(leaking_sequences) > 0:

    flags.append({
        "check": "sequence_leakage",
        "value": len(leaking_sequences),
        "status": "FAIL",
    })

else:

    flags.append({
        "check": "sequence_leakage",
        "value": 0,
        "status": "PASS",
    })


# Zero coverage
zero_coverage = coverage[
    ~coverage["all_splits_present"]
]

if len(zero_coverage) > 0:

    flags.append({
        "check": "source_class_zero_split",
        "value": len(zero_coverage),
        "status": "FAIL",
    })

else:

    flags.append({
        "check": "source_class_zero_split",
        "value": 0,
        "status": "PASS",
    })


# Small evaluation cells
if len(small_cells) > 0:

    flags.append({
        "check": "small_evaluation_cells",
        "value": len(small_cells),
        "status": "WARNING",
    })

else:

    flags.append({
        "check": "small_evaluation_cells",
        "value": 0,
        "status": "PASS",
    })


flags_df = pd.DataFrame(flags)

print(
    flags_df.to_string(
        index=False
    )
)

flags_df.to_csv(
    OUTPUT / "quality_flags.csv",
    index=False
)


# ============================================================
# 15. FINAL DIAGNOSTIC
# ============================================================

print("\n" + "=" * 90)
print("FINAL V3 DATASET DIAGNOSTIC")
print("=" * 90)

print(
    f"\nTotal images       : {len(df)}"
)

print(
    f"Total sequences    : {df['sequence_group'].nunique()}"
)

print(
    f"Sources            : {df['source'].nunique()}"
)

print(
    f"Classes            : {df['label'].nunique()}"
)

print(
    f"Missing images     : {missing_paths}"
)

print(
    f"Leaking sequences  : {len(leaking_sequences)}"
)

print(
    f"Zero coverage      : {len(zero_coverage)}"
)

print(
    f"Small eval cells   : {len(small_cells)}"
)

print("\nOutput directory:")
print(OUTPUT)

print("\nGenerated files:")

for file in sorted(
    OUTPUT.glob("*.csv")
):

    print(
        f"  {file.name}"
    )

print("\n" + "=" * 90)

if (
    missing_paths == 0
    and len(leaking_sequences) == 0
    and len(zero_coverage) == 0
):

    print(
        "STRUCTURAL CHECK: PASS"
    )

else:

    print(
        "STRUCTURAL CHECK: FAIL"
    )

if len(small_cells) > 0:

    print(
        "EVALUATION CHECK: WARNING"
    )

    print(
        "Some source × class × split cells "
        "are very small."
    )

else:

    print(
        "EVALUATION CHECK: PASS"
    )

print(
    "\nDO NOT TRAIN THE MODEL YET."
)

print(
    "Use this audit to decide whether V3 "
    "is ready for model training."
)