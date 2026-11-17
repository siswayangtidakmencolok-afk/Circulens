import os
import re
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CIRCULENS QUALITY DATASET
# SOURCE + SEQUENCE + NEAR-DUPLICATE AUDIT
# ============================================================

MANIFEST = Path("reports/final_dataset/quality_manifest.csv")
OUTPUT_DIR = Path("reports/quality_source_audit")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 90)
print("CIRCULENS QUALITY SOURCE + SEQUENCE AUDIT")
print("=" * 90)


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING MANIFEST")

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest tidak ditemukan: {MANIFEST}"
    )

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. VERIFY PATHS
# ============================================================

print("\n2. VERIFYING IMAGE PATHS")

df["path"] = df["path"].astype(str)

missing = []

for path in df["path"]:
    if not Path(path).exists():
        missing.append(path)

print(f"Missing images: {len(missing)}")

if missing:
    pd.DataFrame({"path": missing}).to_csv(
        OUTPUT_DIR / "missing_images.csv",
        index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'missing_images.csv'}")
else:
    print("PASS: All image paths exist.")


# ============================================================
# 3. DETECT SOURCE
# ============================================================

print("\n3. DETECTING DATA SOURCES")


def detect_source(filename):
    filename = filename.lower()

    if "new_mexico_green_chile" in filename:
        return "new_mexico_green_chile"

    if "chile_pepper" in filename:
        return "chile_pepper"

    return "unknown"


df["source"] = df["filename"].apply(detect_source)

print("\nSource distribution:")
print(df["source"].value_counts())


# Save source distribution
source_distribution = (
    df["source"]
    .value_counts()
    .rename_axis("source")
    .reset_index(name="count")
)

source_distribution.to_csv(
    OUTPUT_DIR / "source_distribution.csv",
    index=False
)


# ============================================================
# 4. SOURCE × CLASS
# ============================================================

print("\n4. SOURCE × CLASS")

source_class = pd.crosstab(
    df["source"],
    df["label"]
)

print(source_class)

source_class.to_csv(
    OUTPUT_DIR / "source_x_class.csv"
)


# ============================================================
# 5. SOURCE × SPLIT
# ============================================================

print("\n5. SOURCE × SPLIT")

source_split = pd.crosstab(
    df["source"],
    df["split"]
)

print(source_split)

source_split.to_csv(
    OUTPUT_DIR / "source_x_split.csv"
)


# ============================================================
# 6. SOURCE × CLASS × SPLIT
# ============================================================

print("\n6. SOURCE × CLASS × SPLIT")

source_class_split = (
    df.groupby(
        ["source", "label", "split"]
    )
    .size()
    .reset_index(name="count")
)

print(source_class_split)

source_class_split.to_csv(
    OUTPUT_DIR / "source_class_split.csv",
    index=False
)


# ============================================================
# 7. FILENAME PATTERN ANALYSIS
# ============================================================

print("\n7. FILENAME PATTERN ANALYSIS")


def extract_datetime(filename):
    """
    Extract timestamp from filenames such as:

    IMG_20220916_080334.jpg
    """

    match = re.search(
        r"IMG_(\d{8})_(\d{6})",
        filename
    )

    if not match:
        return pd.NaT

    date_part = match.group(1)
    time_part = match.group(2)

    value = f"{date_part}_{time_part}"

    return pd.to_datetime(
        value,
        format="%Y%m%d_%H%M%S",
        errors="coerce"
    )


df["image_datetime"] = df["filename"].apply(
    extract_datetime
)

print(
    f"Images with timestamp: "
    f"{df['image_datetime'].notna().sum()}"
)

print(
    f"Images without timestamp: "
    f"{df['image_datetime'].isna().sum()}"
)


# ============================================================
# 8. TIMESTAMP SEQUENCE ANALYSIS
# ============================================================

print("\n8. TIMESTAMP SEQUENCE ANALYSIS")

timestamp_df = df[
    df["image_datetime"].notna()
].copy()

timestamp_df = timestamp_df.sort_values(
    ["source", "image_datetime"]
)

timestamp_df["previous_datetime"] = (
    timestamp_df
    .groupby("source")["image_datetime"]
    .shift(1)
)

timestamp_df["seconds_from_previous"] = (
    timestamp_df["image_datetime"]
    - timestamp_df["previous_datetime"]
).dt.total_seconds()


# Show suspiciously close images
sequence_candidates = timestamp_df[
    timestamp_df["seconds_from_previous"].notna()
    &
    (timestamp_df["seconds_from_previous"] <= 10)
].copy()

print(
    f"Potential sequential images "
    f"(<=10 seconds): {len(sequence_candidates)}"
)


sequence_candidates[
    [
        "source",
        "label",
        "split",
        "filename",
        "image_datetime",
        "previous_datetime",
        "seconds_from_previous",
    ]
].to_csv(
    OUTPUT_DIR / "timestamp_sequences.csv",
    index=False
)


# ============================================================
# 9. CROSS-SPLIT SEQUENCE LEAKAGE
# ============================================================

print("\n9. CROSS-SPLIT SEQUENCE LEAKAGE")

cross_split_sequences = sequence_candidates[
    sequence_candidates["split"]
    !=
    sequence_candidates
    .groupby("source")["split"]
    .shift(1)
].copy()

# The above comparison can miss some cases because sorting is
# global by source. We therefore perform a direct neighboring
# comparison below.

cross_split_rows = []

for source, group in timestamp_df.groupby("source"):

    group = group.sort_values("image_datetime").reset_index(drop=True)

    for i in range(1, len(group)):

        current = group.iloc[i]
        previous = group.iloc[i - 1]

        if pd.isna(current["image_datetime"]):
            continue

        if pd.isna(previous["image_datetime"]):
            continue

        delta = (
            current["image_datetime"]
            - previous["image_datetime"]
        ).total_seconds()

        if delta <= 10 and current["split"] != previous["split"]:

            cross_split_rows.append({
                "source": source,
                "previous_filename": previous["filename"],
                "previous_split": previous["split"],
                "previous_label": previous["label"],
                "current_filename": current["filename"],
                "current_split": current["split"],
                "current_label": current["label"],
                "seconds_apart": delta,
            })


cross_split_df = pd.DataFrame(
    cross_split_rows
)

print(
    f"Cross-split sequential pairs: "
    f"{len(cross_split_df)}"
)

if len(cross_split_df) > 0:
    print("\nWARNING: Possible sequence leakage detected!")
    print(cross_split_df.head(20).to_string(index=False))
else:
    print("PASS: No cross-split sequential pairs <=10 seconds.")


cross_split_df.to_csv(
    OUTPUT_DIR / "cross_split_sequence_leakage.csv",
    index=False
)


# ============================================================
# 10. IMAGE HASHING
# ============================================================

print("\n10. CALCULATING IMAGE HASHES")

print("This may take a little while...")

def md5_hash(path):
    try:
        h = hashlib.md5()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return None


def dhash(path, hash_size=8):
    """
    Difference Hash.
    Produces a 64-bit perceptual hash.
    """

    try:
        image = Image.open(path).convert("L")

        image = image.resize(
            (hash_size + 1, hash_size),
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(image)

        difference = pixels[:, 1:] > pixels[:, :-1]

        bits = 0

        for bit in difference.flatten():
            bits = (bits << 1) | int(bit)

        return bits

    except Exception:
        return None


df["md5"] = df["path"].apply(md5_hash)

df["dhash"] = df["path"].apply(dhash)


print(
    f"MD5 failures: "
    f"{df['md5'].isna().sum()}"
)

print(
    f"dHash failures: "
    f"{df['dhash'].isna().sum()}"
)


# ============================================================
# 11. EXACT DUPLICATE CONTENT
# ============================================================

print("\n11. EXACT DUPLICATE CONTENT")

md5_counts = df["md5"].value_counts()

duplicate_md5 = md5_counts[
    md5_counts > 1
]

print(
    f"Exact duplicate groups: "
    f"{len(duplicate_md5)}"
)

exact_duplicate_rows = []

for md5_value in duplicate_md5.index:

    group = df[df["md5"] == md5_value]

    for _, row in group.iterrows():

        exact_duplicate_rows.append({
            "md5": md5_value,
            "split": row["split"],
            "label": row["label"],
            "source": row["source"],
            "filename": row["filename"],
            "path": row["path"],
        })


exact_duplicate_df = pd.DataFrame(
    exact_duplicate_rows
)

if len(exact_duplicate_df) > 0:

    print("\nWARNING: Exact duplicate images found!")

    print(
        exact_duplicate_df.head(20).to_string(
            index=False
        )
    )

else:

    print(
        "PASS: No exact duplicate image content."
    )


exact_duplicate_df.to_csv(
    OUTPUT_DIR / "exact_duplicate_images.csv",
    index=False
)


# ============================================================
# 12. CROSS-SPLIT EXACT DUPLICATES
# ============================================================

print("\n12. CROSS-SPLIT EXACT DUPLICATES")

cross_split_exact = []

for md5_value, group in df.groupby("md5"):

    if len(group) <= 1:
        continue

    splits = group["split"].unique()

    if len(splits) > 1:

        for _, row in group.iterrows():

            cross_split_exact.append({
                "md5": md5_value,
                "source": row["source"],
                "label": row["label"],
                "split": row["split"],
                "filename": row["filename"],
                "path": row["path"],
            })


cross_split_exact_df = pd.DataFrame(
    cross_split_exact
)

print(
    f"Cross-split exact duplicate images: "
    f"{len(cross_split_exact_df)}"
)

if len(cross_split_exact_df) > 0:
    print("\nWARNING: EXACT SPLIT LEAKAGE FOUND!")
else:
    print("PASS: No exact cross-split duplicate images.")


cross_split_exact_df.to_csv(
    OUTPUT_DIR / "cross_split_exact_duplicates.csv",
    index=False
)


# ============================================================
# 13. PERCEPTUAL HASH / NEAR DUPLICATE
# ============================================================

print("\n13. CHECKING NEAR-DUPLICATE IMAGES")

valid_hash_df = df[
    df["dhash"].notna()
].copy()

print(
    f"Images available for dHash comparison: "
    f"{len(valid_hash_df)}"
)


def hamming_distance(a, b):
    return int((int(a) ^ int(b)).bit_count())


near_duplicate_rows = []

# Compare only images from different splits.
#
# Threshold:
# <= 5 = highly similar
#
# We use a conservative threshold because many images
# have similar white backgrounds.

records = valid_hash_df[
    [
        "dhash",
        "split",
        "label",
        "source",
        "filename",
        "path",
    ]
].to_dict("records")


for i in range(len(records)):

    a = records[i]

    for j in range(i + 1, len(records)):

        b = records[j]

        # Same split is not interesting for leakage audit
        if a["split"] == b["split"]:
            continue

        distance = hamming_distance(
            a["dhash"],
            b["dhash"]
        )

        if distance <= 5:

            near_duplicate_rows.append({
                "distance": distance,

                "image_a_split": a["split"],
                "image_a_label": a["label"],
                "image_a_source": a["source"],
                "image_a_filename": a["filename"],
                "image_a_path": a["path"],

                "image_b_split": b["split"],
                "image_b_label": b["label"],
                "image_b_source": b["source"],
                "image_b_filename": b["filename"],
                "image_b_path": b["path"],
            })


near_duplicate_df = pd.DataFrame(
    near_duplicate_rows
)

print(
    f"Potential cross-split near-duplicate pairs: "
    f"{len(near_duplicate_df)}"
)


if len(near_duplicate_df) > 0:

    print(
        "\nWARNING: Potential near-duplicate "
        "cross-split images detected!"
    )

    print(
        near_duplicate_df
        .sort_values("distance")
        .head(30)
        .to_string(index=False)
    )

else:

    print(
        "PASS: No cross-split near-duplicate pairs "
        "under dHash threshold."
    )


near_duplicate_df.to_csv(
    OUTPUT_DIR / "cross_split_near_duplicates.csv",
    index=False
)


# ============================================================
# 14. SOURCE × SPLIT SUMMARY
# ============================================================

print("\n14. GENERATING FINAL SUMMARY")

summary_rows = []

for source in sorted(df["source"].unique()):

    source_df = df[
        df["source"] == source
    ]

    for split in ["train", "val", "test"]:

        split_df = source_df[
            source_df["split"] == split
        ]

        for label in sorted(df["label"].unique()):

            count = len(
                split_df[
                    split_df["label"] == label
                ]
            )

            summary_rows.append({
                "source": source,
                "split": split,
                "label": label,
                "count": count,
            })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_DIR / "final_source_split_summary.csv",
    index=False
)


# ============================================================
# 15. FINAL REPORT
# ============================================================

print("\n" + "=" * 90)
print("FINAL AUDIT SUMMARY")
print("=" * 90)

print(f"\nTotal images                 : {len(df)}")

print(
    f"Sources detected            : "
    f"{df['source'].nunique()}"
)

print(
    f"Missing images              : "
    f"{len(missing)}"
)

print(
    f"Exact duplicate groups      : "
    f"{len(duplicate_md5)}"
)

print(
    f"Cross-split exact duplicates: "
    f"{len(cross_split_exact_df)}"
)

print(
    f"Cross-split sequence pairs  : "
    f"{len(cross_split_df)}"
)

print(
    f"Cross-split near duplicates : "
    f"{len(near_duplicate_df)}"
)


print("\nDetected sources:")

for source, count in df["source"].value_counts().items():

    print(
        f"  {source:<30} : {count}"
    )


# ============================================================
# 16. SAVE MASTER AUDIT TABLE
# ============================================================

audit_columns = [
    "task",
    "split",
    "label",
    "source",
    "filename",
    "path",
    "image_datetime",
    "md5",
    "dhash",
]

df[audit_columns].to_csv(
    OUTPUT_DIR / "quality_master_audit.csv",
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 90)
print("QUALITY SOURCE + SEQUENCE AUDIT COMPLETED")
print("=" * 90)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):

    print(f"  {file.name}")

print("\nNEXT STEP:")
print(
    "Kirim seluruh output terminal dari script ini "
    "ke ChatGPT."
)

print(
    "\nJANGAN TRAIN MODEL BARU DULU."
)