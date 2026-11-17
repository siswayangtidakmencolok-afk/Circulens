from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MANIFEST = Path(
    "dataset/circulens_quality_clean_v2/quality_clean_v2_manifest.csv"
)

OUT_DIR = Path(
    "reports/quality_v2_sequence_audit"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 90)
print("CIRCULENS QUALITY V2 SEQUENCE DISTRIBUTION AUDIT")
print("=" * 90)


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING V2 MANIFEST")

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")

required = [
    "split",
    "label",
    "source",
    "sequence_group",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
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
# 3. BUILD SEQUENCE TABLE
# ============================================================

print("\n3. BUILDING SEQUENCE TABLE")

seq = (
    df.groupby("sequence_group")
    .agg(
        source=("source", "first"),
        label=("label", "first"),
        split=("split", "first"),
        image_count=("sequence_group", "size"),
    )
    .reset_index()
)

print(f"Total sequences: {len(seq)}")


# ============================================================
# 4. SOURCE × CLASS SEQUENCE DISTRIBUTION
# ============================================================

print("\n4. SOURCE × CLASS SEQUENCE DISTRIBUTION")

source_class_seq = (
    seq.groupby(["source", "label"])
    .size()
    .reset_index(name="sequence_count")
)

print(source_class_seq.to_string(index=False))

source_class_seq.to_csv(
    OUT_DIR / "source_class_sequence_counts.csv",
    index=False,
)


# ============================================================
# 5. SOURCE × CLASS × SPLIT SEQUENCES
# ============================================================

print("\n5. SOURCE × CLASS × SPLIT SEQUENCE COUNTS")

seq_matrix = (
    seq.groupby(["source", "label", "split"])
    .size()
    .reset_index(name="sequence_count")
)

print(seq_matrix.to_string(index=False))

seq_matrix.to_csv(
    OUT_DIR / "source_class_split_sequence_counts.csv",
    index=False,
)


# ============================================================
# 6. SOURCE × CLASS × SPLIT IMAGES
# ============================================================

print("\n6. SOURCE × CLASS × SPLIT IMAGE COUNTS")

image_matrix = (
    df.groupby(["source", "label", "split"])
    .size()
    .reset_index(name="image_count")
)

print(image_matrix.to_string(index=False))

image_matrix.to_csv(
    OUT_DIR / "source_class_split_image_counts.csv",
    index=False,
)


# ============================================================
# 7. COMPLETE MATRIX
# ============================================================

print("\n7. BUILDING COMPLETE SOURCE × CLASS × SPLIT MATRIX")

sources = sorted(df["source"].unique())
labels = sorted(df["label"].unique())
splits = ["train", "val", "test"]

rows = []

for source in sources:
    for label in labels:
        for split in splits:

            subset_images = df[
                (df["source"] == source)
                & (df["label"] == label)
                & (df["split"] == split)
            ]

            subset_sequences = seq[
                (seq["source"] == source)
                & (seq["label"] == label)
                & (seq["split"] == split)
            ]

            rows.append(
                {
                    "source": source,
                    "label": label,
                    "split": split,
                    "images": len(subset_images),
                    "sequences": len(subset_sequences),
                }
            )

complete = pd.DataFrame(rows)

print(complete.to_string(index=False))

complete.to_csv(
    OUT_DIR / "complete_source_class_split_matrix.csv",
    index=False,
)


# ============================================================
# 8. ZERO-SPLIT CHECK
# ============================================================

print("\n8. ZERO-SPLIT CHECK")

zero_cells = complete[
    (complete["images"] == 0)
    | (complete["sequences"] == 0)
]

if len(zero_cells) == 0:

    print("PASS: No zero source × class × split cells.")

else:

    print(
        f"WARNING: {len(zero_cells)} zero cells detected."
    )

    print(
        zero_cells.to_string(index=False)
    )

zero_cells.to_csv(
    OUT_DIR / "zero_split_combinations.csv",
    index=False,
)


# ============================================================
# 9. SEQUENCE SIZE ANALYSIS
# ============================================================

print("\n9. SEQUENCE SIZE ANALYSIS")

sequence_stats = (
    seq.groupby(["source", "label"])
    .agg(
        sequence_count=("sequence_group", "count"),
        min_images=("image_count", "min"),
        median_images=("image_count", "median"),
        max_images=("image_count", "max"),
        total_images=("image_count", "sum"),
    )
    .reset_index()
)

print(
    sequence_stats.to_string(index=False)
)

sequence_stats.to_csv(
    OUT_DIR / "sequence_size_statistics.csv",
    index=False,
)


# ============================================================
# 10. SEQUENCE SPLIT CONSISTENCY
# ============================================================

print("\n10. VERIFYING SEQUENCE SPLIT CONSISTENCY")

split_counts = (
    df.groupby("sequence_group")["split"]
    .nunique()
)

leaking_groups = split_counts[
    split_counts > 1
]

print(
    f"Sequences appearing in multiple splits: "
    f"{len(leaking_groups)}"
)

if len(leaking_groups) == 0:
    print("PASS: No sequence appears in multiple splits.")
else:
    print("WARNING: Sequence leakage detected!")


# ============================================================
# 11. CLASS SPLIT RATIOS
# ============================================================

print("\n11. CLASS SPLIT RATIOS")

class_counts = (
    df.groupby(["label", "split"])
    .size()
    .reset_index(name="images")
)

class_totals = (
    df.groupby("label")
    .size()
    .reset_index(name="total_images")
)

class_ratio = class_counts.merge(
    class_totals,
    on="label",
)

class_ratio["percentage"] = (
    class_ratio["images"]
    / class_ratio["total_images"]
    * 100
)

print(
    class_ratio.to_string(index=False)
)

class_ratio.to_csv(
    OUT_DIR / "class_split_percentages.csv",
    index=False,
)


# ============================================================
# 12. SOURCE SPLIT RATIOS
# ============================================================

print("\n12. SOURCE SPLIT RATIOS")

source_counts = (
    df.groupby(["source", "split"])
    .size()
    .reset_index(name="images")
)

source_totals = (
    df.groupby("source")
    .size()
    .reset_index(name="total_images")
)

source_ratio = source_counts.merge(
    source_totals,
    on="source",
)

source_ratio["percentage"] = (
    source_ratio["images"]
    / source_ratio["total_images"]
    * 100
)

print(
    source_ratio.to_string(index=False)
)

source_ratio.to_csv(
    OUT_DIR / "source_split_percentages.csv",
    index=False,
)


# ============================================================
# 13. FINAL DIAGNOSTIC
# ============================================================

print("\n" + "=" * 90)
print("FINAL V2 SEQUENCE DIAGNOSTIC")
print("=" * 90)

print(f"\nTotal images       : {len(df)}")
print(f"Total sequences    : {len(seq)}")
print(f"Sources            : {df['source'].nunique()}")
print(f"Classes            : {df['label'].nunique()}")
print(
    f"Leaking sequences  : {len(leaking_groups)}"
)
print(
    f"Zero split cells   : {len(zero_cells)}"
)

print("\nOutput directory:")
print(OUT_DIR)

print("\n" + "=" * 90)
print("V2 SEQUENCE DISTRIBUTION AUDIT COMPLETED")
print("=" * 90)