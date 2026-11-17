from pathlib import Path
import pandas as pd

# ============================================================
# CIRCULENS CLEAN SEQUENCE DISTRIBUTION AUDIT
# ============================================================

MANIFEST = Path(
    "dataset/circulens_quality_clean/quality_clean_manifest.csv"
)

OUTPUT = Path(
    "reports/quality_clean_sequence_audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CIRCULENS CLEAN SEQUENCE DISTRIBUTION AUDIT")
print("=" * 90)

# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING CLEAN MANIFEST")

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")

required_columns = [
    "label",
    "source",
    "sequence_group",
    "split",
    "filename",
    "path"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("PASS: Required columns exist.")

# ============================================================
# 2. BASIC DISTRIBUTION
# ============================================================

print("\n2. BASIC DISTRIBUTION")

print("\nImages per split:")
print(df["split"].value_counts())

print("\nSequences per split:")
print(
    df.groupby("split")["sequence_group"]
      .nunique()
)

print("\nImages per class:")
print(df["label"].value_counts())

print("\nSequences per class:")
print(
    df.groupby("label")["sequence_group"]
      .nunique()
)

# ============================================================
# 3. SOURCE × CLASS SEQUENCE DISTRIBUTION
# ============================================================

print("\n3. SOURCE × CLASS SEQUENCE DISTRIBUTION")

source_class_sequences = (
    df.groupby(
        ["source", "label"]
    )["sequence_group"]
    .nunique()
    .reset_index(
        name="sequence_count"
    )
)

print(source_class_sequences.to_string(index=False))

source_class_sequences.to_csv(
    OUTPUT / "source_class_sequence_counts.csv",
    index=False
)

# ============================================================
# 4. SOURCE × CLASS × SPLIT IMAGE COUNTS
# ============================================================

print("\n4. SOURCE × CLASS × SPLIT IMAGE COUNTS")

image_counts = (
    df.groupby(
        ["source", "label", "split"]
    )
    .size()
    .reset_index(name="image_count")
)

print(image_counts.to_string(index=False))

image_counts.to_csv(
    OUTPUT / "source_class_split_image_counts.csv",
    index=False
)

# ============================================================
# 5. SOURCE × CLASS × SPLIT SEQUENCE COUNTS
# ============================================================

print("\n5. SOURCE × CLASS × SPLIT SEQUENCE COUNTS")

sequence_counts = (
    df.groupby(
        ["source", "label", "split"]
    )["sequence_group"]
    .nunique()
    .reset_index(
        name="sequence_count"
    )
)

print(sequence_counts.to_string(index=False))

sequence_counts.to_csv(
    OUTPUT / "source_class_split_sequence_counts.csv",
    index=False
)

# ============================================================
# 6. BUILD COMPLETE MATRIX
# ============================================================

print("\n6. BUILDING COMPLETE SOURCE × CLASS × SPLIT MATRIX")

sources = sorted(df["source"].unique())
classes = sorted(df["label"].unique())
splits = ["train", "val", "test"]

records = []

for source in sources:

    for label in classes:

        for split in splits:

            subset = df[
                (df["source"] == source)
                &
                (df["label"] == label)
                &
                (df["split"] == split)
            ]

            records.append({
                "source": source,
                "label": label,
                "split": split,
                "images": len(subset),
                "sequences": subset["sequence_group"].nunique()
            })

matrix = pd.DataFrame(records)

print(matrix.to_string(index=False))

matrix.to_csv(
    OUTPUT / "complete_source_class_split_matrix.csv",
    index=False
)

# ============================================================
# 7. FIND ZERO-SPLIT COMBINATIONS
# ============================================================

print("\n7. ZERO-SPLIT CHECK")

zero_split_records = []

for source in sources:

    for label in classes:

        subset = matrix[
            (matrix["source"] == source)
            &
            (matrix["label"] == label)
        ]

        for split in splits:

            row = subset[
                subset["split"] == split
            ]

            images = int(row["images"].iloc[0])
            sequences = int(row["sequences"].iloc[0])

            if images == 0 or sequences == 0:

                zero_split_records.append({
                    "source": source,
                    "label": label,
                    "split": split,
                    "images": images,
                    "sequences": sequences
                })

zero_df = pd.DataFrame(
    zero_split_records
)

if len(zero_df) == 0:

    print("PASS: Every source × class combination exists in all splits.")

else:

    print(
        f"WARNING: {len(zero_df)} source × class × split combinations "
        "have zero images or zero sequences."
    )

    print(
        zero_df.to_string(index=False)
    )

zero_df.to_csv(
    OUTPUT / "zero_split_combinations.csv",
    index=False
)

# ============================================================
# 8. SEQUENCE-LEVEL SPLIT CONSISTENCY
# ============================================================

print("\n8. VERIFYING SEQUENCE SPLIT CONSISTENCY")

sequence_split_counts = (
    df.groupby("sequence_group")["split"]
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
    print("PASS: No sequence group appears in multiple splits.")
else:
    print("CRITICAL: Sequence leakage detected.")

# ============================================================
# 9. SEQUENCE SIZE BY SOURCE / CLASS
# ============================================================

print("\n9. SEQUENCE SIZE ANALYSIS")

sequence_sizes = (
    df.groupby(
        ["source", "label", "sequence_group"]
    )
    .size()
    .reset_index(
        name="image_count"
    )
)

print("\nSequence size statistics:")

stats = (
    sequence_sizes
    .groupby(["source", "label"])["image_count"]
    .agg(
        sequence_count="count",
        min_images="min",
        median_images="median",
        max_images="max",
        total_images="sum"
    )
    .reset_index()
)

print(
    stats.to_string(index=False)
)

stats.to_csv(
    OUTPUT / "sequence_size_statistics.csv",
    index=False
)

# ============================================================
# 10. CLASS SPLIT RATIO
# ============================================================

print("\n10. CLASS SPLIT RATIO")

class_split = (
    df.groupby(
        ["label", "split"]
    )
    .size()
    .reset_index(
        name="images"
    )
)

class_totals = (
    df.groupby("label")
      .size()
      .reset_index(
          name="total_images"
      )
)

class_split = class_split.merge(
    class_totals,
    on="label"
)

class_split["percentage"] = (
    class_split["images"]
    /
    class_split["total_images"]
    *
    100
)

print(
    class_split.to_string(index=False)
)

class_split.to_csv(
    OUTPUT / "class_split_percentages.csv",
    index=False
)

# ============================================================
# 11. SOURCE SPLIT RATIO
# ============================================================

print("\n11. SOURCE SPLIT RATIO")

source_split = (
    df.groupby(
        ["source", "split"]
    )
    .size()
    .reset_index(
        name="images"
    )
)

source_totals = (
    df.groupby("source")
      .size()
      .reset_index(
          name="total_images"
      )
)

source_split = source_split.merge(
    source_totals,
    on="source"
)

source_split["percentage"] = (
    source_split["images"]
    /
    source_split["total_images"]
    *
    100
)

print(
    source_split.to_string(index=False)
)

source_split.to_csv(
    OUTPUT / "source_split_percentages.csv",
    index=False
)

# ============================================================
# 12. FINAL DIAGNOSTIC
# ============================================================

print("\n" + "=" * 90)
print("FINAL DIAGNOSTIC")
print("=" * 90)

print(f"\nTotal images              : {len(df)}")
print(
    f"Total sequences           : "
    f"{df['sequence_group'].nunique()}"
)
print(
    f"Sources                   : "
    f"{df['source'].nunique()}"
)
print(
    f"Classes                   : "
    f"{df['label'].nunique()}"
)

print(
    f"Leaking sequences         : "
    f"{len(leaking_sequences)}"
)

print(
    f"Zero split combinations   : "
    f"{len(zero_df)}"
)

if len(leaking_sequences) == 0:
    print("PASS: No sequence leakage.")

if len(zero_df) > 0:
    print(
        "WARNING: Some source × class combinations "
        "are missing from one or more splits."
    )

print("\nOutput directory:")
print(OUTPUT)

print("\nGenerated files:")

for file in sorted(OUTPUT.iterdir()):
    print(f"  {file.name}")

print("\n" + "=" * 90)
print("CLEAN SEQUENCE DISTRIBUTION AUDIT COMPLETED")
print("=" * 90)