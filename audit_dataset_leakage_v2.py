import os
import hashlib
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("dataset")
OUTPUT_DIR = Path("reports/dataset_leakage_v2")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# UTILITY
# ============================================================

def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def normalize_path(path):
    return str(path).replace("\\", "/")


def detect_dataset(relative_path):
    """
    Detect dataset from actual directory structure.
    """

    p = normalize_path(relative_path).lower()

    if "chili/dataset 1/" in p:
        return "dataset_1"

    if "chili/dataset 3/" in p:
        return "dataset_3"

    if "circulens_quality_normalized/" in p:
        return "circulens_quality_normalized"

    if "circulens_quality_strict/" in p:
        return "circulens_quality_strict"

    if "circulens_quality/" in p:
        return "circulens_quality"

    return "unknown"


def detect_split(relative_path, dataset_name):
    """
    Detect train / val / test.
    Dataset 3 does not contain normal train/val/test directories.
    """

    p = normalize_path(relative_path).lower()

    if dataset_name == "dataset_3":
        return "dataset3"

    if "/train/" in p:
        return "train"

    if "/val/" in p:
        return "val"

    if "/test/" in p:
        return "test"

    return "unknown"


def detect_quality(relative_path):
    """
    Detect quality class from folder names.
    """

    parts = normalize_path(relative_path).lower().split("/")

    quality_map = {
        "damaged": "damaged",
        "dried": "dried",
        "old": "old",
        "ripe": "ripe",
        "unripe": "unripe",
    }

    for part in reversed(parts):
        if part in quality_map:
            return quality_map[part]

        for quality in quality_map:
            if part.endswith("_" + quality):
                return quality

    return "unknown"


def detect_commodity(relative_path):
    """
    Detect commodity for dataset 1.
    """

    parts = normalize_path(relative_path).split("/")

    for part in reversed(parts):
        lower = part.lower()

        if lower.startswith("bell pepper_"):
            return "Bell Pepper"

        if lower.startswith("chile pepper_"):
            return "Chile Pepper"

        if lower.startswith("new mexico green chile_"):
            return "New Mexico Green Chile"

        if lower.startswith("tomato_"):
            return "Tomato"

    return "unknown"


def detect_dataset3_class(relative_path):
    """
    Detect Dataset 3 class.
    """

    parts = normalize_path(relative_path).split("/")

    classes = {
        "anthracnose",
        "healthy",
    }

    for part in reversed(parts):
        if part.lower() in classes:
            return part

    return "unknown"


def calculate_md5(path, chunk_size=1024 * 1024):
    """
    Calculate exact file hash.
    """

    md5 = hashlib.md5()

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:
        return None


# ============================================================
# SCAN DATASET
# ============================================================

def scan_images():

    section("SCANNING DATASETS")

    records = []

    image_paths = []

    for root, _, files in os.walk(DATASET_ROOT):

        for filename in files:

            path = Path(root) / filename

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_paths.append(path)

    image_paths = sorted(image_paths)

    print(f"Images discovered: {len(image_paths)}")

    for index, path in enumerate(image_paths, start=1):

        relative = path.relative_to(DATASET_ROOT)

        dataset_name = detect_dataset(relative)

        split = detect_split(relative, dataset_name)

        quality = detect_quality(relative)

        commodity = detect_commodity(relative)

        dataset3_class = detect_dataset3_class(relative)

        records.append({
            "path": str(path),
            "relative_path": normalize_path(relative),
            "filename": path.name,
            "dataset": dataset_name,
            "split": split,
            "quality": quality,
            "commodity": commodity,
            "dataset3_class": dataset3_class,
        })

        if index % 1000 == 0:
            print(
                f"Scanned {index}/{len(image_paths)} images..."
            )

    df = pd.DataFrame(records)

    print()
    print(f"Indexed images: {len(df)}")

    return df


# ============================================================
# HASHING
# ============================================================

def calculate_hashes(df):

    section("CALCULATING IMAGE HASHES")

    hashes = []

    total = len(df)

    for index, path in enumerate(
        df["path"],
        start=1
    ):

        hashes.append(
            calculate_md5(path)
        )

        if index % 500 == 0:
            print(
                f"Hashed {index}/{total} images..."
            )

    df = df.copy()

    df["md5"] = hashes

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def generate_dataset_summary(df):

    section("DATASET SUMMARY")

    rows = []

    for dataset_name, group in df.groupby(
        "dataset",
        dropna=False
    ):

        rows.append({
            "dataset": dataset_name,
            "total_images": len(group),
            "train": int(
                (group["split"] == "train").sum()
            ),
            "val": int(
                (group["split"] == "val").sum()
            ),
            "test": int(
                (group["split"] == "test").sum()
            ),
            "dataset3": int(
                (group["split"] == "dataset3").sum()
            ),
            "unknown_split": int(
                (group["split"] == "unknown").sum()
            ),
            "unique_hashes": group["md5"].nunique(),
            "unique_files": group["path"].nunique(),
            "quality_classes": group[
                "quality"
            ].nunique(),
            "dataset3_classes": group[
                "dataset3_class"
            ].nunique(),
        })

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )

    return result


# ============================================================
# DATASET 3 ANALYSIS
# ============================================================

def analyze_dataset3(df):

    section("DATASET 3 ANALYSIS")

    dataset3 = df[
        df["dataset"] == "dataset_3"
    ].copy()

    if dataset3.empty:

        print(
            "WARNING: Dataset 3 was not detected."
        )

        return pd.DataFrame()

    print(
        f"Dataset 3 images: {len(dataset3)}"
    )

    print()

    class_counts = (
        dataset3[
            "dataset3_class"
        ]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "dataset3_class",
        "count",
    ]

    print(
        class_counts.to_string(
            index=False
        )
    )

    class_counts.to_csv(
        OUTPUT_DIR / "dataset3_class_distribution.csv",
        index=False
    )

    return class_counts


# ============================================================
# SPLIT LEAKAGE
# ============================================================

def analyze_split_leakage(df):

    section("SPLIT LEAKAGE ANALYSIS")

    results = []

    leakage_groups = []

    datasets = [
        d for d in df["dataset"].unique()
        if d != "unknown"
    ]

    for dataset_name in datasets:

        subset = df[
            df["dataset"] == dataset_name
        ].copy()

        split_hashes = {}

        for split_name in [
            "train",
            "val",
            "test",
        ]:

            split_hashes[split_name] = set(
                subset[
                    subset["split"] == split_name
                ]["md5"]
                .dropna()
            )

        for split_a, split_b in combinations(
            ["train", "val", "test"],
            2
        ):

            shared = (
                split_hashes[split_a]
                &
                split_hashes[split_b]
            )

            results.append({
                "dataset": dataset_name,
                "comparison": (
                    f"{split_a}_vs_{split_b}"
                ),
                "shared_hash_groups": len(shared),
                "images_involved": sum(
                    (
                        subset["md5"].isin(shared)
                        &
                        subset["split"].isin(
                            [split_a, split_b]
                        )
                    )
                ),
            })

            for hash_value in shared:

                rows = subset[
                    subset["md5"] == hash_value
                ]

                leakage_groups.append(
                    rows.assign(
                        leakage_type=(
                            f"{split_a}_vs_{split_b}"
                        )
                    )
                )

    result = pd.DataFrame(results)

    if result.empty:

        result = pd.DataFrame(
            columns=[
                "dataset",
                "comparison",
                "shared_hash_groups",
                "images_involved",
            ]
        )

    print(
        result.to_string(index=False)
    )

    result.to_csv(
        OUTPUT_DIR / "split_leakage_summary.csv",
        index=False
    )

    if leakage_groups:

        leakage_df = pd.concat(
            leakage_groups,
            ignore_index=True
        )

    else:

        leakage_df = pd.DataFrame(
            columns=list(df.columns)
            + ["leakage_type"]
        )

    leakage_df.to_csv(
        OUTPUT_DIR / "split_leakage_groups.csv",
        index=False
    )

    return result, leakage_df


# ============================================================
# CROSS DATASET OVERLAP
# ============================================================

def analyze_cross_dataset_overlap(df):

    section("CROSS-DATASET OVERLAP")

    datasets = sorted(
        d for d in df["dataset"].unique()
        if d != "unknown"
    )

    results = []

    overlap_groups = []

    for dataset_a, dataset_b in combinations(
        datasets,
        2
    ):

        hashes_a = set(
            df[
                df["dataset"] == dataset_a
            ]["md5"]
            .dropna()
        )

        hashes_b = set(
            df[
                df["dataset"] == dataset_b
            ]["md5"]
            .dropna()
        )

        shared = hashes_a & hashes_b

        results.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "shared_hash_groups": len(shared),
            "images_dataset_a": len(hashes_a),
            "images_dataset_b": len(hashes_b),
        })

        for hash_value in shared:

            rows = df[
                df["md5"] == hash_value
            ].copy()

            rows["dataset_pair"] = (
                f"{dataset_a}__{dataset_b}"
            )

            overlap_groups.append(rows)

    result = pd.DataFrame(results)

    if result.empty:

        result = pd.DataFrame(
            columns=[
                "dataset_a",
                "dataset_b",
                "shared_hash_groups",
                "images_dataset_a",
                "images_dataset_b",
            ]
        )

    else:

        result = result.sort_values(
            "shared_hash_groups",
            ascending=False
        )

    print(
        result.to_string(index=False)
    )

    result.to_csv(
        OUTPUT_DIR / "cross_dataset_overlap.csv",
        index=False
    )

    if overlap_groups:

        overlap_df = pd.concat(
            overlap_groups,
            ignore_index=True
        )

    else:

        overlap_df = pd.DataFrame(
            columns=list(df.columns)
            + ["dataset_pair"]
        )

    overlap_df.to_csv(
        OUTPUT_DIR / "cross_dataset_duplicates.csv",
        index=False
    )

    return result


# ============================================================
# DATASET IDENTITY
# ============================================================

def analyze_dataset_identity(df):

    section("DATASET IDENTITY CHECK")

    datasets = sorted(
        d for d in df["dataset"].unique()
        if d != "unknown"
    )

    results = []

    for dataset_a, dataset_b in combinations(
        datasets,
        2
    ):

        hashes_a = set(
            df[
                df["dataset"] == dataset_a
            ]["md5"]
            .dropna()
        )

        hashes_b = set(
            df[
                df["dataset"] == dataset_b
            ]["md5"]
            .dropna()
        )

        shared = hashes_a & hashes_b

        overlap_a = (
            len(shared) / len(hashes_a) * 100
            if hashes_a
            else 0
        )

        overlap_b = (
            len(shared) / len(hashes_b) * 100
            if hashes_b
            else 0
        )

        identical = (
            len(hashes_a) > 0
            and hashes_a == hashes_b
        )

        results.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "unique_images_a": len(hashes_a),
            "unique_images_b": len(hashes_b),
            "shared_images": len(shared),
            "overlap_pct_a": round(
                overlap_a,
                4
            ),
            "overlap_pct_b": round(
                overlap_b,
                4
            ),
            "identical_dataset": identical,
        })

    result = pd.DataFrame(results)

    if result.empty:

        result = pd.DataFrame(
            columns=[
                "dataset_a",
                "dataset_b",
                "unique_images_a",
                "unique_images_b",
                "shared_images",
                "overlap_pct_a",
                "overlap_pct_b",
                "identical_dataset",
            ]
        )

    print(
        result.to_string(index=False)
    )

    result.to_csv(
        OUTPUT_DIR / "dataset_identity_check.csv",
        index=False
    )

    return result


# ============================================================
# HASH SUMMARY
# ============================================================

def generate_hash_summary(df):

    section("HASH SUMMARY")

    rows = []

    for (
        dataset_name,
        split_name
    ), group in df.groupby(
        ["dataset", "split"],
        dropna=False
    ):

        rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "images": len(group),
            "unique_hashes": group[
                "md5"
            ].nunique(),
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(index=False)
    )

    result.to_csv(
        OUTPUT_DIR / "split_hash_summary.csv",
        index=False
    )

    return result


# ============================================================
# DUPLICATE SUMMARY
# ============================================================

def generate_duplicate_summary(df):

    section("DUPLICATE SUMMARY")

    rows = []

    for dataset_name, group in df.groupby(
        "dataset"
    ):

        total = len(group)

        unique_hashes = group[
            "md5"
        ].nunique()

        duplicated_images = (
            total - unique_hashes
        )

        percentage = (
            duplicated_images / total * 100
            if total
            else 0
        )

        rows.append({
            "dataset": dataset_name,
            "total_images": total,
            "unique_hashes": unique_hashes,
            "duplicated_images": duplicated_images,
            "duplicate_percentage": round(
                percentage,
                4
            ),
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(index=False)
    )

    result.to_csv(
        OUTPUT_DIR / "dataset_duplicate_summary.csv",
        index=False
    )

    return result


# ============================================================
# DIAGNOSTIC
# ============================================================

def generate_diagnostic(
    df,
    split_result,
    cross_result,
    identity_result
):

    section("AUTOMATIC DIAGNOSTIC")

    diagnostics = []

    leakage_exists = False

    if not split_result.empty:

        leakage_exists = (
            split_result[
                "shared_hash_groups"
            ].fillna(0)
            > 0
        ).any()

    if leakage_exists:

        print(
            "CRITICAL: Exact image leakage "
            "exists between train/val/test."
        )

        diagnostics.append(
            "CRITICAL: Exact image leakage exists "
            "between train/val/test."
        )

    else:

        print(
            "PASS: No exact hash leakage detected "
            "between train/val/test."
        )

        diagnostics.append(
            "PASS: No exact hash leakage detected "
            "between train/val/test."
        )

    if not cross_result.empty:

        cross_overlap = (
            cross_result[
                "shared_hash_groups"
            ].fillna(0)
            > 0
        ).any()

        if cross_overlap:

            print(
                "WARNING: Exact duplicate images "
                "exist across datasets."
            )

            diagnostics.append(
                "WARNING: Exact duplicate images "
                "exist across datasets."
            )

    if not identity_result.empty:

        identical_rows = identity_result[
            identity_result[
                "identical_dataset"
            ] == True
        ]

        for _, row in identical_rows.iterrows():

            message = (
                f"CRITICAL: "
                f"{row['dataset_a']} and "
                f"{row['dataset_b']} appear "
                f"to contain identical image sets."
            )

            print(message)

            diagnostics.append(message)

    if (
        df["dataset"] == "dataset_3"
    ).any():

        print(
            "INFO: Dataset 3 detected successfully."
        )

        diagnostics.append(
            "INFO: Dataset 3 detected successfully."
        )

    else:

        print(
            "WARNING: Dataset 3 was not detected."
        )

        diagnostics.append(
            "WARNING: Dataset 3 was not detected."
        )

    return diagnostics


# ============================================================
# FINAL REPORT
# ============================================================

def generate_report(
    df,
    diagnostics
):

    section("GENERATING FINAL REPORT")

    report_path = (
        OUTPUT_DIR
        / "dataset_leakage_v2_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CIRCULENS DATASET LEAKAGE AUDIT V2\n"
        )

        f.write("=" * 80 + "\n\n")

        f.write(
            f"Total images: {len(df)}\n"
        )

        f.write(
            f"Unique hashes: {df['md5'].nunique()}\n"
        )

        f.write("\nDATASET COUNTS\n")
        f.write("-" * 80 + "\n")

        dataset_counts = (
            df["dataset"]
            .value_counts()
        )

        for dataset_name, count in (
            dataset_counts.items()
        ):

            f.write(
                f"{dataset_name}: {count}\n"
            )

        f.write("\nDIAGNOSTICS\n")
        f.write("-" * 80 + "\n")

        for diagnostic in diagnostics:

            f.write(
                diagnostic + "\n"
            )

    print(
        f"Saved report: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "CIRCULENS DATASET LEAKAGE AUDIT V2"
    )

    print()
    print(
        f"Dataset root:\n{DATASET_ROOT}"
    )

    print()
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    df = scan_images()

    df.to_csv(
        OUTPUT_DIR / "image_index.csv",
        index=False
    )

    print(
        f"Saved: {OUTPUT_DIR / 'image_index.csv'}"
    )

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    df = calculate_hashes(df)

    df.to_csv(
        OUTPUT_DIR / "image_index_with_hash.csv",
        index=False
    )

    print(
        f"Saved: "
        f"{OUTPUT_DIR / 'image_index_with_hash.csv'}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    generate_dataset_summary(df)

    # --------------------------------------------------------
    # DATASET 3
    # --------------------------------------------------------

    analyze_dataset3(df)

    # --------------------------------------------------------
    # SPLIT LEAKAGE
    # --------------------------------------------------------

    split_result, leakage_df = (
        analyze_split_leakage(df)
    )

    # --------------------------------------------------------
    # CROSS DATASET
    # --------------------------------------------------------

    cross_result = (
        analyze_cross_dataset_overlap(df)
    )

    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    identity_result = (
        analyze_dataset_identity(df)
    )

    # --------------------------------------------------------
    # HASH SUMMARY
    # --------------------------------------------------------

    generate_hash_summary(df)

    # --------------------------------------------------------
    # DUPLICATE SUMMARY
    # --------------------------------------------------------

    generate_duplicate_summary(df)

    # --------------------------------------------------------
    # DIAGNOSTIC
    # --------------------------------------------------------

    diagnostics = generate_diagnostic(
        df,
        split_result,
        cross_result,
        identity_result
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    generate_report(
        df,
        diagnostics
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    section(
        "DATASET LEAKAGE AUDIT V2 COMPLETED"
    )

    print(
        f"Total images : {len(df)}"
    )

    print(
        f"Unique hashes: {df['md5'].nunique()}"
    )

    print()
    print(
        "Dataset distribution:"
    )

    print(
        df["dataset"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "Important files:"
    )

    for filename in [
        "image_index.csv",
        "image_index_with_hash.csv",
        "dataset_summary.csv",
        "dataset3_class_distribution.csv",
        "split_leakage_summary.csv",
        "split_leakage_groups.csv",
        "cross_dataset_overlap.csv",
        "cross_dataset_duplicates.csv",
        "dataset_identity_check.csv",
        "split_hash_summary.csv",
        "dataset_duplicate_summary.csv",
        "dataset_leakage_v2_report.txt",
    ]:

        path = OUTPUT_DIR / filename

        if path.exists():
            print(f"  {filename}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()