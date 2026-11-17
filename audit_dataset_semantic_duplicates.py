import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV belum terinstall.")
    print("Install dengan:")
    print("pip install opencv-python")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("dataset")

OUTPUT_DIR = Path(
    "reports/dataset_semantic_duplicates"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

# Perceptual hash threshold.
#
# Semakin kecil = semakin strict.
#
# 0-4   : sangat mirip
# 5-8   : kemungkinan sama
# 9-12  : perlu review
# >12   : biasanya berbeda
#
# Kita gunakan 8 sebagai threshold awal.
PHASH_THRESHOLD = 8


# ============================================================
# UTILITY
# ============================================================

def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def is_image(path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def detect_dataset(relative_path):
    """
    Menentukan dataset berdasarkan struktur folder.
    """

    parts = list(relative_path.parts)

    if "circulens_quality_normalized" in parts:
        return "circulens_quality_normalized"

    if "circulens_quality_strict" in parts:
        return "circulens_quality_strict"

    if "circulens_quality" in parts:
        return "circulens_quality"

    if "dataset 1" in parts:
        return "dataset_1"

    if "dataset 3" in parts:
        return "dataset_3"

    return "unknown"


def detect_split(relative_path):
    parts = [p.lower() for p in relative_path.parts]

    if "train" in parts:
        return "train"

    if "val" in parts:
        return "val"

    if "validation" in parts:
        return "val"

    if "test" in parts:
        return "test"

    if "dataset 3" in parts:
        return "dataset3"

    return "unknown"


def detect_quality(relative_path):
    """
    Mengambil label kualitas dari folder.
    """

    quality_labels = {
        "damaged",
        "dried",
        "old",
        "ripe",
        "unripe"
    }

    parts = relative_path.parts

    for part in reversed(parts):

        normalized = part.lower()

        if normalized in quality_labels:
            return normalized

        # Dataset 1 menggunakan:
        # Tomato_Ripe
        # Chile Pepper_Old
        # etc.

        for quality in quality_labels:

            if normalized.endswith("_" + quality):
                return quality

    return "unknown"


def detect_commodity(relative_path):

    parts = relative_path.parts

    commodities = {
        "bell pepper": "Bell Pepper",
        "chile pepper": "Chile Pepper",
        "new mexico green chile": "New Mexico Green Chile",
        "tomato": "Tomato"
    }

    for part in parts:

        normalized = part.lower()

        for key, value in commodities.items():

            if normalized.startswith(key + "_"):
                return value

    return "unknown"


def detect_dataset3_class(relative_path):

    parts = [
        p.lower()
        for p in relative_path.parts
    ]

    if "healthy" in parts:
        return "Healthy"

    if "anthracnose" in parts:
        return "Anthracnose"

    return "unknown"


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def average_hash(image, size=32):

    image = image.convert("L")
    image = image.resize(
        (size, size),
        Image.Resampling.LANCZOS
    )

    arr = np.asarray(image)

    mean = arr.mean()

    bits = arr > mean

    return bits.flatten()


def phash(image, hash_size=16):

    image = image.convert("L")

    image = image.resize(
        (
            hash_size * 4,
            hash_size * 4
        ),
        Image.Resampling.LANCZOS
    )

    pixels = np.asarray(
        image,
        dtype=np.float32
    )

    dct = cv2.dct(pixels)

    low_freq = dct[
        :hash_size,
        :hash_size
    ]

    median = np.median(
        low_freq[1:]
    )

    return (
        low_freq > median
    ).flatten()


def hamming_distance(hash_a, hash_b):

    return int(
        np.count_nonzero(
            hash_a != hash_b
        )
    )


def hash_to_string(hash_array):

    return "".join(
        "1" if x else "0"
        for x in hash_array
    )


# ============================================================
# IMAGE INDEX
# ============================================================

def build_index():

    section(
        "1. SCANNING DATASET"
    )

    image_paths = []

    for path in DATASET_ROOT.rglob("*"):

        if is_image(path):

            image_paths.append(path)

    image_paths.sort()

    print(
        f"Images discovered: {len(image_paths)}"
    )

    rows = []

    for i, path in enumerate(
        image_paths,
        start=1
    ):

        relative = path.relative_to(
            DATASET_ROOT
        )

        dataset = detect_dataset(
            relative
        )

        split = detect_split(
            relative
        )

        quality = detect_quality(
            relative
        )

        commodity = detect_commodity(
            relative
        )

        dataset3_class = (
            detect_dataset3_class(
                relative
            )
        )

        rows.append({

            "path": str(path),

            "relative_path": str(
                relative
            ),

            "filename": path.name,

            "dataset": dataset,

            "split": split,

            "quality": quality,

            "commodity": commodity,

            "dataset3_class":
                dataset3_class

        })

        if i % 1000 == 0:

            print(
                f"Indexed {i}/{len(image_paths)}..."
            )

    return pd.DataFrame(rows)


# ============================================================
# COMPUTE PERCEPTUAL HASHES
# ============================================================

def calculate_hashes(df):

    section(
        "2. CALCULATING PERCEPTUAL HASHES"
    )

    phashes = []
    ahashes = []

    failed = []

    total = len(df)

    for i, row in df.iterrows():

        path = row["path"]

        try:

            with Image.open(path) as image:

                image.load()

                p_hash = phash(image)

                a_hash = average_hash(
                    image
                )

                phashes.append(
                    hash_to_string(
                        p_hash
                    )
                )

                ahashes.append(
                    hash_to_string(
                        a_hash
                    )
                )

        except Exception as e:

            phashes.append(None)

            ahashes.append(None)

            failed.append({

                "path": path,

                "error": str(e)

            })

        if (i + 1) % 500 == 0:

            print(
                f"Hashed {i + 1}/{total}..."
            )

    df = df.copy()

    df["phash"] = phashes

    df["ahash"] = ahashes

    if failed:

        failed_df = pd.DataFrame(
            failed
        )

        failed_df.to_csv(
            OUTPUT_DIR /
            "hash_failures.csv",
            index=False
        )

    return df


# ============================================================
# EXACT PERCEPTUAL HASH GROUPS
# ============================================================

def find_exact_phash_groups(df):

    section(
        "3. EXACT PERCEPTUAL HASH GROUPS"
    )

    groups = (
        df.dropna(
            subset=["phash"]
        )
        .groupby("phash")
    )

    rows = []

    group_id = 0

    for hash_value, group in groups:

        if len(group) < 2:
            continue

        group_id += 1

        for _, row in group.iterrows():

            rows.append({

                "group_id":
                    group_id,

                "phash":
                    hash_value,

                "path":
                    row["path"],

                "dataset":
                    row["dataset"],

                "split":
                    row["split"],

                "quality":
                    row["quality"],

                "commodity":
                    row["commodity"],

                "dataset3_class":
                    row["dataset3_class"]

            })

    result = pd.DataFrame(rows)

    output = (
        OUTPUT_DIR /
        "perceptual_duplicate_groups.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    if len(result) == 0:

        print(
            "No exact perceptual duplicate groups found."
        )

    else:

        print(
            "Duplicate groups:",
            result["group_id"].nunique()
        )

        print(
            "Images involved:",
            len(result)
        )

        print(
            "Saved:",
            output
        )

    return result


# ============================================================
# CROSS SPLIT PERCEPTUAL DUPLICATES
# ============================================================

def analyze_cross_split_duplicates(
    df
):

    section(
        "4. CROSS-SPLIT PERCEPTUAL DUPLICATES"
    )

    valid = df.dropna(
        subset=["phash"]
    )

    groups = (
        valid.groupby("phash")
    )

    rows = []

    for hash_value, group in groups:

        splits = set(
            group["split"]
        )

        real_splits = (
            splits
            & {
                "train",
                "val",
                "test"
            }
        )

        if len(real_splits) < 2:
            continue

        rows.append({

            "phash":
                hash_value,

            "splits":
                "|".join(
                    sorted(real_splits)
                ),

            "count":
                len(group),

            "paths":
                " || ".join(
                    group["path"]
                    .astype(str)
                )

        })

    result = pd.DataFrame(rows)

    output = (
        OUTPUT_DIR /
        "cross_split_perceptual_duplicates.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print(
        f"Cross-split groups: {len(result)}"
    )

    print(
        f"Saved: {output}"
    )

    return result


# ============================================================
# CROSS DATASET PERCEPTUAL DUPLICATES
# ============================================================

def analyze_cross_dataset_duplicates(
    df
):

    section(
        "5. CROSS-DATASET PERCEPTUAL DUPLICATES"
    )

    valid = df.dropna(
        subset=["phash"]
    )

    groups = (
        valid.groupby("phash")
    )

    rows = []

    for hash_value, group in groups:

        datasets = (
            group["dataset"]
            .dropna()
            .unique()
        )

        datasets = [
            x for x in datasets
            if x != "unknown"
        ]

        if len(datasets) < 2:
            continue

        rows.append({

            "phash":
                hash_value,

            "datasets":
                "|".join(
                    sorted(datasets)
                ),

            "count":
                len(group),

            "paths":
                " || ".join(
                    group["path"]
                    .astype(str)
                )

        })

    result = pd.DataFrame(rows)

    output = (
        OUTPUT_DIR /
        "cross_dataset_perceptual_duplicates.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print(
        f"Cross-dataset groups: {len(result)}"
    )

    print(
        f"Saved: {output}"
    )

    return result


# ============================================================
# CLASS CONFLICT ANALYSIS
# ============================================================

def analyze_label_conflicts(
    df
):

    section(
        "6. PERCEPTUAL DUPLICATE LABEL CONFLICTS"
    )

    valid = df.dropna(
        subset=["phash"]
    )

    groups = (
        valid.groupby("phash")
    )

    rows = []

    for hash_value, group in groups:

        qualities = (
            group["quality"]
            .dropna()
            .unique()
        )

        qualities = [
            x for x in qualities
            if x != "unknown"
        ]

        dataset3_classes = (
            group[
                "dataset3_class"
            ]
            .dropna()
            .unique()
        )

        dataset3_classes = [
            x
            for x in dataset3_classes
            if x != "unknown"
        ]

        if len(qualities) > 1:

            rows.append({

                "phash":
                    hash_value,

                "type":
                    "quality_conflict",

                "labels":
                    "|".join(
                        sorted(
                            qualities
                        )
                    ),

                "count":
                    len(group),

                "paths":
                    " || ".join(
                        group["path"]
                        .astype(str)
                    )

            })

        if len(dataset3_classes) > 1:

            rows.append({

                "phash":
                    hash_value,

                "type":
                    "dataset3_class_conflict",

                "labels":
                    "|".join(
                        sorted(
                            dataset3_classes
                        )
                    ),

                "count":
                    len(group),

                "paths":
                    " || ".join(
                        group["path"]
                        .astype(str)
                    )

            })

    result = pd.DataFrame(rows)

    output = (
        OUTPUT_DIR /
        "perceptual_label_conflicts.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print(
        f"Label conflict groups: {len(result)}"
    )

    print(
        f"Saved: {output}"
    )

    return result


# ============================================================
# SIMILAR IMAGE SEARCH
# ============================================================

def calculate_near_duplicates(
    df,
    threshold=PHASH_THRESHOLD
):

    section(
        "7. NEAR-DUPLICATE SEARCH"
    )

    valid = df.dropna(
        subset=["phash"]
    ).copy()

    hashes = []

    for _, row in valid.iterrows():

        hashes.append({

            "path":
                row["path"],

            "dataset":
                row["dataset"],

            "split":
                row["split"],

            "quality":
                row["quality"],

            "phash":
                np.array(
                    [
                        c == "1"
                        for c
                        in row["phash"]
                    ],
                    dtype=bool
                )

        })

    results = []

    total = len(hashes)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do not compare every image against every other image.
    #
    # Instead we compare only images from different splits
    # or different datasets.
    # --------------------------------------------------------

    for i in range(total):

        a = hashes[i]

        for j in range(i + 1, total):

            b = hashes[j]

            # Ignore same physical split
            if (
                a["dataset"]
                == b["dataset"]
                and
                a["split"]
                == b["split"]
            ):
                continue

            distance = hamming_distance(
                a["phash"],
                b["phash"]
            )

            if distance <= threshold:

                results.append({

                    "path_a":
                        a["path"],

                    "path_b":
                        b["path"],

                    "dataset_a":
                        a["dataset"],

                    "dataset_b":
                        b["dataset"],

                    "split_a":
                        a["split"],

                    "split_b":
                        b["split"],

                    "quality_a":
                        a["quality"],

                    "quality_b":
                        b["quality"],

                    "phash_distance":
                        distance

                })

        if (i + 1) % 500 == 0:

            print(
                f"Compared {i + 1}/{total}..."
            )

    result = pd.DataFrame(
        results
    )

    if len(result):

        result = result.sort_values(
            "phash_distance"
        )

    output = (
        OUTPUT_DIR /
        "near_duplicate_pairs.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print(
        f"Near-duplicate pairs: {len(result)}"
    )

    print(
        f"Saved: {output}"
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(
    df,
    perceptual_groups,
    cross_split,
    cross_dataset,
    conflicts,
    near_duplicates
):

    section(
        "8. FINAL DIAGNOSTIC"
    )

    print(
        f"Total images: {len(df)}"
    )

    print(
        "Perceptual duplicate groups:",
        (
            perceptual_groups["group_id"]
            .nunique()
            if len(perceptual_groups)
            else 0
        )
    )

    print(
        "Cross-split perceptual groups:",
        len(cross_split)
    )

    print(
        "Cross-dataset perceptual groups:",
        len(cross_dataset)
    )

    print(
        "Label conflict groups:",
        len(conflicts)
    )

    print(
        "Near-duplicate pairs:",
        len(near_duplicates)
    )

    print()

    print(
        "DIAGNOSTIC"
    )

    if len(cross_split) > 0:

        print(
            "CRITICAL: Perceptual duplicates exist across splits."
        )

    else:

        print(
            "PASS: No cross-split perceptual duplicates detected."
        )

    if len(cross_dataset) > 0:

        print(
            "WARNING: Perceptual duplicates exist across datasets."
        )

    else:

        print(
            "PASS: No cross-dataset perceptual duplicates detected."
        )

    if len(conflicts) > 0:

        print(
            "CRITICAL: Duplicate images have conflicting labels."
        )

    else:

        print(
            "PASS: No perceptual duplicate label conflicts detected."
        )

    if len(near_duplicates) > 0:

        print(
            "WARNING: Near-duplicate images require manual review."
        )

    else:

        print(
            "PASS: No near-duplicate pairs detected."
        )

    summary = pd.DataFrame([{

        "total_images":
            len(df),

        "perceptual_duplicate_groups":
            (
                perceptual_groups[
                    "group_id"
                ].nunique()
                if len(perceptual_groups)
                else 0
            ),

        "cross_split_groups":
            len(cross_split),

        "cross_dataset_groups":
            len(cross_dataset),

        "label_conflict_groups":
            len(conflicts),

        "near_duplicate_pairs":
            len(near_duplicates)

    }])

    summary.to_csv(
        OUTPUT_DIR /
        "semantic_duplicate_summary.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "CIRCULENS SEMANTIC / PERCEPTUAL DUPLICATE AUDIT"
    )
    print("=" * 100)

    print()
    print(
        "Dataset root:"
    )
    print(
        DATASET_ROOT
    )

    print()
    print(
        "Output directory:"
    )
    print(
        OUTPUT_DIR
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not DATASET_ROOT.exists():

        print(
            "ERROR: Dataset root tidak ditemukan."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    df = build_index()

    df.to_csv(
        OUTPUT_DIR /
        "image_index.csv",
        index=False
    )

    print(
        "Saved:",
        OUTPUT_DIR /
        "image_index.csv"
    )

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    df = calculate_hashes(
        df
    )

    df.to_csv(
        OUTPUT_DIR /
        "image_index_with_phash.csv",
        index=False
    )

    print(
        "Saved:",
        OUTPUT_DIR /
        "image_index_with_phash.csv"
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    perceptual_groups = (
        find_exact_phash_groups(
            df
        )
    )

    cross_split = (
        analyze_cross_split_duplicates(
            df
        )
    )

    cross_dataset = (
        analyze_cross_dataset_duplicates(
            df
        )
    )

    conflicts = (
        analyze_label_conflicts(
            df
        )
    )

    # --------------------------------------------------------
    # NEAR DUPLICATES
    # --------------------------------------------------------

    near_duplicates = (
        calculate_near_duplicates(
            df
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    generate_summary(
        df,
        perceptual_groups,
        cross_split,
        cross_dataset,
        conflicts,
        near_duplicates
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = (
        OUTPUT_DIR /
        "semantic_duplicate_audit_report.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CIRCULENS SEMANTIC / PERCEPTUAL "
            "DUPLICATE AUDIT\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Total images: {len(df)}\n"
        )

        f.write(
            "Perceptual duplicate groups: "
            f"{perceptual_groups['group_id'].nunique() if len(perceptual_groups) else 0}\n"
        )

        f.write(
            "Cross-split groups: "
            f"{len(cross_split)}\n"
        )

        f.write(
            "Cross-dataset groups: "
            f"{len(cross_dataset)}\n"
        )

        f.write(
            "Label conflict groups: "
            f"{len(conflicts)}\n"
        )

        f.write(
            "Near-duplicate pairs: "
            f"{len(near_duplicates)}\n"
        )

        f.write("\n")

        f.write(
            "THRESHOLD\n"
        )

        f.write(
            f"pHash Hamming threshold: "
            f"{PHASH_THRESHOLD}\n"
        )

    print()
    print("=" * 100)
    print(
        "SEMANTIC / PERCEPTUAL DUPLICATE AUDIT COMPLETED"
    )
    print("=" * 100)

    print()
    print(
        "Important output:"
    )

    print(
        OUTPUT_DIR /
        "perceptual_duplicate_groups.csv"
    )

    print(
        OUTPUT_DIR /
        "cross_split_perceptual_duplicates.csv"
    )

    print(
        OUTPUT_DIR /
        "cross_dataset_perceptual_duplicates.csv"
    )

    print(
        OUTPUT_DIR /
        "perceptual_label_conflicts.csv"
    )

    print(
        OUTPUT_DIR /
        "near_duplicate_pairs.csv"
    )

    print(
        OUTPUT_DIR /
        "semantic_duplicate_summary.csv"
    )

    print(
        OUTPUT_DIR /
        "semantic_duplicate_audit_report.txt"
    )

    print()


if __name__ == "__main__":
    main()