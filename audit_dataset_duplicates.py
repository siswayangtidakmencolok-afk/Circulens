import os
import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd


# ============================================================
# CIRCULENS DATASET DUPLICATE & LEAKAGE AUDIT
# ============================================================

DATASET_ROOT = Path("dataset")
OUTPUT_DIR = Path("reports/dataset_duplicate_audit")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# UTILITIES
# ============================================================

def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


def hash_file(path, chunk_size=1024 * 1024):
    """
    Calculate SHA256 hash of an image file.
    """
    sha256 = hashlib.sha256()

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception as e:
        print(f"WARNING: Failed hashing {path}: {e}")
        return None


def detect_dataset_info(path):
    """
    Determine dataset name, split, commodity, and quality
    from the directory structure.
    """

    parts = [p.lower() for p in path.parts]

    dataset = "unknown"
    split = "unknown"
    commodity = "unknown"
    quality = "unknown"

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if "dataset 1" in parts:
        dataset = "dataset_1"

    elif "circulens_quality_normalized" in parts:
        dataset = "circulens_quality_normalized"

    elif "circulens_quality_strict" in parts:
        dataset = "circulens_quality_strict"

    elif "circulens_quality" in parts:
        dataset = "circulens_quality"

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    for s in ["train", "val", "test"]:
        if s in parts:
            split = s
            break

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality_names = {
        "damaged",
        "dried",
        "old",
        "ripe",
        "unripe",
    }

    for p in path.parts:

        if p.lower() in quality_names:
            quality = p.lower()
            break

    # --------------------------------------------------------
    # Commodity
    # --------------------------------------------------------

    commodity_names = {
        "bell pepper": "Bell Pepper",
        "chile pepper": "Chile Pepper",
        "new mexico green chile": "New Mexico Green Chile",
        "tomato": "Tomato",
    }

    for p in path.parts:

        lower_p = p.lower()

        for key, value in commodity_names.items():

            if lower_p.startswith(key + "_"):
                commodity = value
                break

        if commodity != "unknown":
            break

    return dataset, split, commodity, quality


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "CIRCULENS DATASET DUPLICATE & LEAKAGE AUDIT"
    )

    print("Dataset root:")
    print(DATASET_ROOT)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SCAN IMAGES
    # ========================================================

    section("SCANNING DATASETS")

    image_paths = []

    for path in DATASET_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_paths.append(path)

    image_paths = sorted(image_paths)

    print(
        f"Images discovered: {len(image_paths)}"
    )

    if len(image_paths) == 0:

        print("ERROR: No images found.")

        return

    # ========================================================
    # CREATE INDEX
    # ========================================================

    section("CREATING IMAGE INDEX")

    rows = []

    for i, path in enumerate(image_paths, start=1):

        dataset, split, commodity, quality = (
            detect_dataset_info(path)
        )

        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "dataset": dataset,
                "split": split,
                "commodity": commodity,
                "quality": quality,
                "file_size": path.stat().st_size,
            }
        )

        if i % 1000 == 0:

            print(
                f"Indexed {i}/{len(image_paths)} images..."
            )

    df = pd.DataFrame(rows)

    index_path = (
        OUTPUT_DIR /
        "image_index.csv"
    )

    df.to_csv(
        index_path,
        index=False
    )

    print()
    print(
        f"Saved: {index_path}"
    )

    # ========================================================
    # HASH IMAGES
    # ========================================================

    section("CALCULATING IMAGE HASHES")

    hashes = []

    for i, path in enumerate(image_paths, start=1):

        file_hash = hash_file(path)

        hashes.append(file_hash)

        if i % 500 == 0:

            print(
                f"Hashed {i}/{len(image_paths)} images..."
            )

    df["sha256"] = hashes

    indexed_path = (
        OUTPUT_DIR /
        "image_index_with_hash.csv"
    )

    df.to_csv(
        indexed_path,
        index=False
    )

    print()
    print(
        f"Saved: {indexed_path}"
    )

    # ========================================================
    # EXACT DUPLICATES
    # ========================================================

    section("EXACT DUPLICATE ANALYSIS")

    hash_counts = (
        df["sha256"]
        .value_counts()
    )

    duplicate_hashes = (
        hash_counts[
            hash_counts > 1
        ]
    )

    duplicate_groups = []

    for file_hash, count in duplicate_hashes.items():

        group = df[
            df["sha256"] == file_hash
        ].copy()

        duplicate_groups.append(
            {
                "sha256": file_hash,
                "count": count,
                "datasets": "|".join(
                    sorted(
                        group[
                            "dataset"
                        ].unique()
                    )
                ),
                "splits": "|".join(
                    sorted(
                        group[
                            "split"
                        ].unique()
                    )
                ),
                "qualities": "|".join(
                    sorted(
                        group[
                            "quality"
                        ].unique()
                    )
                ),
                "paths": " || ".join(
                    group[
                        "path"
                    ].tolist()
                ),
            }
        )

    duplicate_df = pd.DataFrame(
        duplicate_groups
    )

    duplicate_path = (
        OUTPUT_DIR /
        "exact_duplicate_groups.csv"
    )

    duplicate_df.to_csv(
        duplicate_path,
        index=False
    )

    print(
        f"Duplicate hash groups: "
        f"{len(duplicate_df)}"
    )

    duplicate_images = (
        duplicate_df["count"].sum()
        if len(duplicate_df) > 0
        else 0
    )

    print(
        f"Images involved in duplicates: "
        f"{duplicate_images}"
    )

    # ========================================================
    # TRAIN TEST LEAKAGE
    # ========================================================

    section("TRAIN / VALIDATION / TEST LEAKAGE")

    leakage_rows = []

    grouped = df.groupby(
        "sha256"
    )

    for file_hash, group in grouped:

        if pd.isna(file_hash):
            continue

        splits = set(
            group["split"].tolist()
        )

        valid_splits = {
            "train",
            "val",
            "test",
        }

        actual_splits = (
            splits &
            valid_splits
        )

        if len(actual_splits) <= 1:
            continue

        leakage_rows.append(
            {
                "sha256": file_hash,
                "splits": "|".join(
                    sorted(actual_splits)
                ),
                "datasets": "|".join(
                    sorted(
                        group[
                            "dataset"
                        ].unique()
                    )
                ),
                "count": len(group),
                "paths": " || ".join(
                    group[
                        "path"
                    ].tolist()
                ),
            }
        )

    leakage_df = pd.DataFrame(
        leakage_rows
    )

    leakage_path = (
        OUTPUT_DIR /
        "split_leakage_groups.csv"
    )

    leakage_df.to_csv(
        leakage_path,
        index=False
    )

    print(
        f"Leakage groups: "
        f"{len(leakage_df)}"
    )

    # ========================================================
    # CROSS-DATASET DUPLICATES
    # ========================================================

    section("CROSS-DATASET DUPLICATE ANALYSIS")

    cross_dataset_rows = []

    for file_hash, group in grouped:

        if pd.isna(file_hash):
            continue

        datasets = set(
            group["dataset"].tolist()
        )

        if len(datasets) <= 1:
            continue

        cross_dataset_rows.append(
            {
                "sha256": file_hash,
                "datasets": "|".join(
                    sorted(datasets)
                ),
                "dataset_count": len(
                    datasets
                ),
                "count": len(group),
                "splits": "|".join(
                    sorted(
                        group[
                            "split"
                        ].unique()
                    )
                ),
                "paths": " || ".join(
                    group[
                        "path"
                    ].tolist()
                ),
            }
        )

    cross_dataset_df = pd.DataFrame(
        cross_dataset_rows
    )

    cross_dataset_path = (
        OUTPUT_DIR /
        "cross_dataset_duplicates.csv"
    )

    cross_dataset_df.to_csv(
        cross_dataset_path,
        index=False
    )

    print(
        f"Cross-dataset duplicate groups: "
        f"{len(cross_dataset_df)}"
    )

    # ========================================================
    # DATASET PAIR OVERLAP
    # ========================================================

    section("DATASET PAIR OVERLAP")

    pair_counter = defaultdict(int)

    for file_hash, group in grouped:

        if pd.isna(file_hash):
            continue

        datasets = sorted(
            set(
                group[
                    "dataset"
                ].tolist()
            )
        )

        if len(datasets) < 2:
            continue

        for i in range(len(datasets)):

            for j in range(
                i + 1,
                len(datasets)
            ):

                pair = (
                    datasets[i],
                    datasets[j]
                )

                pair_counter[pair] += 1

    pair_rows = []

    for pair, count in sorted(
        pair_counter.items()
    ):

        pair_rows.append(
            {
                "dataset_a": pair[0],
                "dataset_b": pair[1],
                "duplicate_hash_groups": count,
            }
        )

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_path = (
        OUTPUT_DIR /
        "dataset_pair_overlap.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False
    )

    print(
        pair_df.to_string(
            index=False
        )
        if len(pair_df) > 0
        else "No cross-dataset overlap detected."
    )

    # ========================================================
    # DATASET DUPLICATE SUMMARY
    # ========================================================

    section("DATASET DUPLICATE SUMMARY")

    summary_rows = []

    for dataset_name, group in df.groupby(
        "dataset"
    ):

        total = len(group)

        unique_hashes = (
            group["sha256"]
            .nunique()
        )

        duplicated_images = (
            total -
            unique_hashes
        )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "total_images": total,
                "unique_hashes": unique_hashes,
                "duplicated_images": duplicated_images,
                "duplicate_percentage": (
                    duplicated_images /
                    total *
                    100
                    if total > 0
                    else 0
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR /
        "dataset_duplicate_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # IDENTICAL DATASET CHECK
    # ========================================================

    section("IDENTICAL DATASET CHECK")

    dataset_hash_sets = {}

    for dataset_name, group in df.groupby(
        "dataset"
    ):

        dataset_hash_sets[
            dataset_name
        ] = set(
            group[
                "sha256"
            ].dropna()
        )

    identical_rows = []

    dataset_names = sorted(
        dataset_hash_sets.keys()
    )

    for i in range(
        len(dataset_names)
    ):

        for j in range(
            i + 1,
            len(dataset_names)
        ):

            a = dataset_names[i]
            b = dataset_names[j]

            hashes_a = dataset_hash_sets[a]
            hashes_b = dataset_hash_sets[b]

            if not hashes_a or not hashes_b:
                continue

            intersection = (
                hashes_a &
                hashes_b
            )

            union = (
                hashes_a |
                hashes_b
            )

            overlap_pct_a = (
                len(intersection) /
                len(hashes_a) *
                100
            )

            overlap_pct_b = (
                len(intersection) /
                len(hashes_b) *
                100
            )

            identical_rows.append(
                {
                    "dataset_a": a,
                    "dataset_b": b,
                    "unique_images_a": len(
                        hashes_a
                    ),
                    "unique_images_b": len(
                        hashes_b
                    ),
                    "shared_images": len(
                        intersection
                    ),
                    "overlap_pct_a": overlap_pct_a,
                    "overlap_pct_b": overlap_pct_b,
                    "identical_dataset": (
                        hashes_a == hashes_b
                    ),
                }
            )

    identical_df = pd.DataFrame(
        identical_rows
    )

    identical_path = (
        OUTPUT_DIR /
        "dataset_identity_check.csv"
    )

    identical_df.to_csv(
        identical_path,
        index=False
    )

    if len(identical_df) > 0:

        print(
            identical_df.to_string(
                index=False
            )
        )

    else:

        print(
            "No dataset pairs available."
        )

    # ========================================================
    # SPLIT HASH SUMMARY
    # ========================================================

    section("SPLIT HASH SUMMARY")

    split_rows = []

    for (
        dataset_name,
        split_name
    ), group in df.groupby(
        [
            "dataset",
            "split",
        ]
    ):

        valid_hashes = (
            group["sha256"]
            .dropna()
        )

        split_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "images": len(group),
                "unique_hashes": valid_hashes.nunique(),
            }
        )

    split_hash_df = pd.DataFrame(
        split_rows
    )

    split_hash_path = (
        OUTPUT_DIR /
        "split_hash_summary.csv"
    )

    split_hash_df.to_csv(
        split_hash_path,
        index=False
    )

    print(
        split_hash_df.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL DIAGNOSTIC
    # ========================================================

    section("AUTOMATIC DIAGNOSTIC")

    diagnostic_lines = []

    if len(leakage_df) > 0:

        diagnostic_lines.append(
            "CRITICAL: Exact image leakage exists "
            "between train/val/test."
        )

    else:

        diagnostic_lines.append(
            "PASS: No exact hash leakage detected "
            "between train/val/test."
        )

    if len(cross_dataset_df) > 0:

        diagnostic_lines.append(
            "WARNING: Exact duplicate images exist "
            "across datasets."
        )

    else:

        diagnostic_lines.append(
            "PASS: No exact cross-dataset duplicates detected."
        )

    if len(identical_df) > 0:

        for _, row in identical_df.iterrows():

            if row["identical_dataset"]:

                diagnostic_lines.append(
                    f"CRITICAL: {row['dataset_a']} "
                    f"and {row['dataset_b']} appear "
                    f"to contain the same image set."
                )

            elif (
                row["overlap_pct_a"] >= 90
                and
                row["overlap_pct_b"] >= 90
            ):

                diagnostic_lines.append(
                    f"WARNING: {row['dataset_a']} "
                    f"and {row['dataset_b']} have "
                    f"very high image overlap."
                )

    if len(duplicate_df) > 0:

        diagnostic_lines.append(
            f"INFO: {len(duplicate_df)} exact "
            f"duplicate groups detected."
        )

    else:

        diagnostic_lines.append(
            "INFO: No exact duplicate groups detected."
        )

    for line in diagnostic_lines:

        print(line)

    # ========================================================
    # REPORT
    # ========================================================

    section("GENERATING FINAL REPORT")

    report_path = (
        OUTPUT_DIR /
        "dataset_duplicate_audit_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CIRCULENS DATASET DUPLICATE & "
            "LEAKAGE AUDIT\n"
        )

        f.write(
            "=" * 80
        )

        f.write("\n\n")

        f.write(
            f"Total images scanned: "
            f"{len(df)}\n"
        )

        f.write(
            f"Exact duplicate groups: "
            f"{len(duplicate_df)}\n"
        )

        f.write(
            f"Split leakage groups: "
            f"{len(leakage_df)}\n"
        )

        f.write(
            f"Cross-dataset duplicate groups: "
            f"{len(cross_dataset_df)}\n"
        )

        f.write("\n")

        f.write(
            "AUTOMATIC DIAGNOSTIC\n"
        )

        f.write(
            "-" * 80
        )

        f.write("\n\n")

        for line in diagnostic_lines:

            f.write(
                line + "\n"
            )

        f.write("\n")

        f.write(
            "DATASET PAIR OVERLAP\n"
        )

        f.write(
            "-" * 80
        )

        f.write("\n\n")

        if len(identical_df) > 0:

            f.write(
                identical_df.to_string(
                    index=False
                )
            )

            f.write("\n")

    print(
        f"Saved report: {report_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "DATASET DUPLICATE & LEAKAGE AUDIT COMPLETED"
    )

    print(
        f"Total images : {len(df)}"
    )

    print(
        f"Duplicate groups : {len(duplicate_df)}"
    )

    print(
        f"Leakage groups : {len(leakage_df)}"
    )

    print(
        f"Cross-dataset groups : "
        f"{len(cross_dataset_df)}"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Important files:"
    )

    print(
        "  image_index_with_hash.csv"
    )

    print(
        "  exact_duplicate_groups.csv"
    )

    print(
        "  split_leakage_groups.csv"
    )

    print(
        "  cross_dataset_duplicates.csv"
    )

    print(
        "  dataset_pair_overlap.csv"
    )

    print(
        "  dataset_identity_check.csv"
    )

    print(
        "  dataset_duplicate_summary.csv"
    )

    print(
        "  split_hash_summary.csv"
    )

    print(
        "  dataset_duplicate_audit_report.txt"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()