import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = "dataset"

OUTPUT_DIR = "reports/dataset_distribution"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# DATASET DEFINITIONS
# ============================================================

DATASETS = {
    "dataset_1": {
        "root": os.path.join(
            DATASET_ROOT,
            "chili",
            "dataset 1",
        ),
        "type": "source_dataset",
    },

    "circulens_quality": {
        "root": os.path.join(
            DATASET_ROOT,
            "circulens_quality",
        ),
        "type": "normalized_dataset",
    },

    "circulens_quality_normalized": {
        "root": os.path.join(
            DATASET_ROOT,
            "circulens_quality_normalized",
        ),
        "type": "normalized_dataset",
    },

    "circulens_quality_strict": {
        "root": os.path.join(
            DATASET_ROOT,
            "circulens_quality_strict",
        ),
        "type": "normalized_dataset",
    },
}


# ============================================================
# KNOWN QUALITY CLASSES
# ============================================================

QUALITY_CLASSES = [
    "damaged",
    "dried",
    "old",
    "ripe",
    "unripe",
]


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)

pd.set_option(
    "display.max_rows",
    300,
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.4f}",
)


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


def is_image(filename):

    return (
        os.path.splitext(filename)[1].lower()
        in IMAGE_EXTENSIONS
    )


def normalize_quality(value):

    if value is None:
        return None

    value = str(value).strip().lower()

    for quality in QUALITY_CLASSES:

        if quality == value:
            return quality

    return None


def parse_dataset_1_folder(folder_name):

    """
    Dataset 1 format:

        Bell Pepper_Damaged
        Chile Pepper_Ripe
        New Mexico Green Chile_Old
        Tomato_Unripe

    Returns:

        commodity
        quality
    """

    if "_" not in folder_name:
        return None, None

    commodity, quality = (
        folder_name.rsplit(
            "_",
            1,
        )
    )

    quality = normalize_quality(
        quality
    )

    if quality is None:
        return None, None

    return commodity, quality


def scan_images(root, dataset_name):

    records = []

    if not os.path.exists(root):

        print(
            f"WARNING: Dataset not found: {root}"
        )

        return records

    for current_root, _, files in os.walk(root):

        for filename in files:

            if not is_image(filename):
                continue

            full_path = os.path.join(
                current_root,
                filename,
            )

            relative_path = os.path.relpath(
                full_path,
                root,
            )

            parts = relative_path.split(
                os.sep
            )

            split = None
            commodity = None
            quality = None

            # =================================================
            # DATASET 1
            # =================================================

            if dataset_name == "dataset_1":

                # Expected:
                #
                # Train/Class_Name/image.jpg
                # Test/Class_Name/image.jpg

                if len(parts) >= 3:

                    possible_split = (
                        parts[0]
                    )

                    if possible_split.lower() in {
                        "train",
                        "test",
                        "val",
                        "validation",
                    }:

                        split = (
                            possible_split.lower()
                        )

                        if split == "validation":
                            split = "val"

                        class_folder = (
                            parts[1]
                        )

                        commodity, quality = (
                            parse_dataset_1_folder(
                                class_folder
                            )
                        )

            # =================================================
            # NORMALIZED DATASETS
            # =================================================

            else:

                # Expected:
                #
                # train/damaged/image.jpg
                # val/ripe/image.jpg
                # test/old/image.jpg

                if len(parts) >= 3:

                    possible_split = (
                        parts[0]
                    )

                    if possible_split.lower() in {
                        "train",
                        "test",
                        "val",
                        "validation",
                    }:

                        split = (
                            possible_split.lower()
                        )

                        if split == "validation":
                            split = "val"

                        quality = normalize_quality(
                            parts[1]
                        )

            records.append(
                {
                    "dataset": dataset_name,
                    "path": full_path,
                    "relative_path": relative_path,
                    "filename": filename,
                    "split": split,
                    "commodity": commodity,
                    "quality": quality,
                }
            )

    return records


# ============================================================
# BASIC SUMMARY
# ============================================================

def create_dataset_summary(df):

    rows = []

    for dataset_name in sorted(
        df["dataset"].dropna().unique()
    ):

        subset = df[
            df["dataset"] == dataset_name
        ]

        total = len(subset)

        valid_quality = subset[
            subset["quality"].notna()
        ]

        rows.append(
            {
                "dataset": dataset_name,
                "total_images": total,
                "images_with_quality": len(
                    valid_quality
                ),
                "images_without_quality": (
                    total -
                    len(valid_quality)
                ),
                "train": int(
                    (
                        subset["split"]
                        == "train"
                    ).sum()
                ),
                "val": int(
                    (
                        subset["split"]
                        == "val"
                    ).sum()
                ),
                "test": int(
                    (
                        subset["split"]
                        == "test"
                    ).sum()
                ),
                "unknown_split": int(
                    subset["split"]
                    .isna()
                    .sum()
                ),
                "commodities": (
                    subset["commodity"]
                    .dropna()
                    .nunique()
                ),
                "quality_classes": (
                    subset["quality"]
                    .dropna()
                    .nunique()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def create_class_distribution(df):

    rows = []

    for dataset_name in sorted(
        df["dataset"].dropna().unique()
    ):

        subset = df[
            df["dataset"] == dataset_name
        ]

        total = len(subset)

        for quality in QUALITY_CLASSES:

            count = int(
                (
                    subset["quality"]
                    == quality
                ).sum()
            )

            percentage = (
                count /
                total *
                100
                if total > 0
                else 0
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "quality": quality,
                    "count": count,
                    "percentage": percentage,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

def create_split_distribution(df):

    rows = []

    split_order = [
        "train",
        "val",
        "test",
    ]

    for dataset_name in sorted(
        df["dataset"].dropna().unique()
    ):

        subset = df[
            df["dataset"] == dataset_name
        ]

        for split in split_order:

            split_subset = subset[
                subset["split"] == split
            ]

            total = len(
                split_subset
            )

            for quality in QUALITY_CLASSES:

                count = int(
                    (
                        split_subset[
                            "quality"
                        ]
                        == quality
                    ).sum()
                )

                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "quality": quality,
                        "count": count,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# COMMODITY DISTRIBUTION
# ============================================================

def create_commodity_distribution(df):

    rows = []

    dataset_1 = df[
        df["dataset"] == "dataset_1"
    ].copy()

    if len(dataset_1) == 0:
        return pd.DataFrame()

    commodity_counts = (
        dataset_1[
            "commodity"
        ]
        .value_counts(
            dropna=False
        )
    )

    total = len(
        dataset_1
    )

    for commodity, count in (
        commodity_counts.items()
    ):

        if pd.isna(commodity):
            commodity_name = (
                "UNKNOWN"
            )
        else:
            commodity_name = (
                str(commodity)
            )

        rows.append(
            {
                "commodity": commodity_name,
                "count": int(count),
                "percentage": (
                    count /
                    total *
                    100
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# COMMODITY × QUALITY
# ============================================================

def create_commodity_quality_distribution(
    df
):

    dataset_1 = df[
        df["dataset"] == "dataset_1"
    ].copy()

    dataset_1 = dataset_1[
        dataset_1["commodity"].notna()
        &
        dataset_1["quality"].notna()
    ]

    if len(dataset_1) == 0:
        return pd.DataFrame()

    table = pd.crosstab(
        dataset_1["commodity"],
        dataset_1["quality"],
    )

    table = table.reindex(
        columns=QUALITY_CLASSES,
        fill_value=0,
    )

    table["total"] = (
        table.sum(axis=1)
    )

    return table.reset_index()


# ============================================================
# SPLIT × QUALITY PERCENTAGE
# ============================================================

def create_split_quality_percentage(
    df
):

    rows = []

    for dataset_name in sorted(
        df["dataset"].dropna().unique()
    ):

        subset = df[
            df["dataset"] == dataset_name
        ]

        for split in [
            "train",
            "val",
            "test",
        ]:

            split_subset = subset[
                subset["split"] == split
            ]

            total = len(
                split_subset
            )

            if total == 0:
                continue

            for quality in QUALITY_CLASSES:

                count = int(
                    (
                        split_subset[
                            "quality"
                        ]
                        == quality
                    ).sum()
                )

                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "quality": quality,
                        "count": count,
                        "percentage": (
                            count /
                            total *
                            100
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

def create_data_quality_report(df):

    rows = []

    for dataset_name in sorted(
        df["dataset"].dropna().unique()
    ):

        subset = df[
            df["dataset"] == dataset_name
        ]

        total = len(
            subset
        )

        missing_split = int(
            subset["split"]
            .isna()
            .sum()
        )

        missing_quality = int(
            subset["quality"]
            .isna()
            .sum()
        )

        missing_commodity = int(
            subset["commodity"]
            .isna()
            .sum()
        )

        rows.append(
            {
                "dataset": dataset_name,
                "total_images": total,
                "missing_split": missing_split,
                "missing_quality": missing_quality,
                "missing_commodity": missing_commodity,
                "quality_coverage_pct": (
                    (
                        total -
                        missing_quality
                    )
                    /
                    total
                    *
                    100
                    if total > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOT — CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(
    class_distribution
):

    pivot = (
        class_distribution
        .pivot(
            index="quality",
            columns="dataset",
            values="count",
        )
        .fillna(0)
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Dataset Class Distribution"
    )

    ax.set_xlabel(
        "Quality Class"
    )

    ax.set_ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "class_distribution.png",
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PLOT — CLASS PERCENTAGE
# ============================================================

def plot_class_percentage(
    class_distribution
):

    pivot = (
        class_distribution
        .pivot(
            index="quality",
            columns="dataset",
            values="percentage",
        )
        .fillna(0)
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Dataset Class Distribution (%)"
    )

    ax.set_xlabel(
        "Quality Class"
    )

    ax.set_ylabel(
        "Percentage (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "class_distribution_percentage.png",
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PLOT — SPLIT DISTRIBUTION
# ============================================================

def plot_split_distribution(
    split_distribution
):

    if len(split_distribution) == 0:
        return

    pivot = (
        split_distribution
        .groupby(
            [
                "dataset",
                "split",
            ]
        )["count"]
        .sum()
        .unstack(
            fill_value=0
        )
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Dataset Split Distribution"
    )

    ax.set_xlabel(
        "Dataset"
    )

    ax.set_ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "split_distribution.png",
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PLOT — COMMODITY DISTRIBUTION
# ============================================================

def plot_commodity_distribution(
    commodity_distribution
):

    if len(commodity_distribution) == 0:
        return

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        commodity_distribution[
            "commodity"
        ],
        commodity_distribution[
            "count"
        ],
    )

    plt.title(
        "Dataset 1 Commodity Distribution"
    )

    plt.xlabel(
        "Commodity"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "commodity_distribution.png",
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PLOT — COMMODITY × QUALITY
# ============================================================

def plot_commodity_quality(
    commodity_quality
):

    if len(commodity_quality) == 0:
        return

    plot_df = (
        commodity_quality
        .set_index(
            "commodity"
        )
    )

    if "total" in plot_df.columns:

        plot_df = plot_df.drop(
            columns=["total"]
        )

    ax = plot_df.plot(
        kind="bar",
        figsize=(14, 8),
    )

    ax.set_title(
        "Dataset 1: Commodity × Quality"
    )

    ax.set_xlabel(
        "Commodity"
    )

    ax.set_ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "commodity_quality_distribution.png",
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# GENERATE TEXT REPORT
# ============================================================

def generate_report(
    df,
    dataset_summary,
    class_distribution,
    split_distribution,
    commodity_distribution,
    commodity_quality,
    data_quality,
):

    report = []

    report.append(
        "CIRCULENS DATASET DISTRIBUTION AUDIT"
    )

    report.append(
        "=" * 80
    )

    report.append("")

    report.append(
        "PURPOSE"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "This report audits dataset composition before "
        "V4 training. It focuses on class balance, "
        "train/validation/test distribution, commodity "
        "distribution, and dataset structure."
    )

    report.append("")

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    report.append(
        "DATASET SUMMARY"
    )

    report.append(
        "-" * 80
    )

    report.append(
        dataset_summary.to_string(
            index=False
        )
    )

    report.append("")

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    report.append(
        "CLASS DISTRIBUTION"
    )

    report.append(
        "-" * 80
    )

    report.append(
        class_distribution.to_string(
            index=False
        )
    )

    report.append("")

    # ========================================================
    # SPLIT DISTRIBUTION
    # ========================================================

    report.append(
        "SPLIT DISTRIBUTION"
    )

    report.append(
        "-" * 80
    )

    report.append(
        split_distribution.to_string(
            index=False
        )
    )

    report.append("")

    # ========================================================
    # COMMODITY
    # ========================================================

    report.append(
        "COMMODITY DISTRIBUTION"
    )

    report.append(
        "-" * 80
    )

    if len(commodity_distribution) > 0:

        report.append(
            commodity_distribution.to_string(
                index=False
            )
        )

    else:

        report.append(
            "No commodity information available."
        )

    report.append("")

    # ========================================================
    # COMMODITY × QUALITY
    # ========================================================

    report.append(
        "COMMODITY × QUALITY"
    )

    report.append(
        "-" * 80
    )

    if len(commodity_quality) > 0:

        report.append(
            commodity_quality.to_string(
                index=False
            )
        )

    else:

        report.append(
            "No commodity-quality information."
        )

    report.append("")

    # ========================================================
    # DATA QUALITY
    # ========================================================

    report.append(
        "DATA QUALITY"
    )

    report.append(
        "-" * 80
    )

    report.append(
        data_quality.to_string(
            index=False
        )
    )

    report.append("")

    # ========================================================
    # AUTOMATIC OBSERVATIONS
    # ========================================================

    report.append(
        "AUTOMATIC OBSERVATIONS"
    )

    report.append(
        "-" * 80
    )

    observations = []

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    for dataset_name in sorted(
        class_distribution[
            "dataset"
        ].unique()
    ):

        subset = class_distribution[
            class_distribution[
                "dataset"
            ]
            == dataset_name
        ]

        if len(subset) == 0:
            continue

        max_row = subset.loc[
            subset["count"].idxmax()
        ]

        min_positive = subset[
            subset["count"] > 0
        ]

        if len(min_positive) > 0:

            min_row = min_positive.loc[
                min_positive["count"].idxmin()
            ]

            ratio = (
                max_row["count"]
                /
                max(
                    min_row["count"],
                    1,
                )
            )

            observations.append(
                f"{dataset_name}: "
                f"largest class = "
                f"{max_row['quality']} "
                f"({int(max_row['count'])}), "
                f"smallest non-zero class = "
                f"{min_row['quality']} "
                f"({int(min_row['count'])}), "
                f"ratio = {ratio:.2f}x."
            )

    # --------------------------------------------------------
    # MISSING LABEL
    # --------------------------------------------------------

    for _, row in data_quality.iterrows():

        if row[
            "missing_quality"
        ] > 0:

            observations.append(
                f"WARNING: {row['dataset']} "
                f"contains "
                f"{int(row['missing_quality'])} "
                "images whose quality class "
                "could not be inferred."
            )

    # --------------------------------------------------------
    # SPLIT CHECK
    # --------------------------------------------------------

    for dataset_name in sorted(
        df["dataset"].unique()
    ):

        subset = df[
            df["dataset"] == dataset_name
        ]

        splits = set(
            subset[
                "split"
            ].dropna()
        )

        if not {
            "train",
            "val",
            "test",
        }.issubset(splits):

            observations.append(
                f"INFO: {dataset_name} "
                "does not contain all three "
                "train/val/test splits."
            )

    # --------------------------------------------------------
    # DATASET 1 COMMODITY
    # --------------------------------------------------------

    if len(
        commodity_distribution
    ) > 0:

        top_commodity = (
            commodity_distribution
            .iloc[0]
        )

        observations.append(
            "Dataset 1 has commodity-level "
            "variation. The model may therefore "
            "encounter commodity/source-specific "
            "visual patterns."
        )

        observations.append(
            f"Largest Dataset 1 commodity: "
            f"{top_commodity['commodity']} "
            f"({int(top_commodity['count'])} images, "
            f"{top_commodity['percentage']:.2f}%)."
        )

    # --------------------------------------------------------
    # DOMAIN DISJOINT CONTEXT
    # --------------------------------------------------------

    observations.append(
        "IMPORTANT: Dataset distribution alone "
        "cannot prove the cause of the V3 domain-disjoint "
        "failure. Visual domain analysis and embedding "
        "analysis should be considered together."
    )

    for index, observation in enumerate(
        observations,
        start=1,
    ):

        report.append(
            f"{index}. {observation}"
        )

    report.append("")

    report.append(
        "NEXT STEP"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "Use this audit together with the V3 "
        "domain-disjoint failure analysis and "
        "embedding/domain analysis before deciding "
        "the V4 training strategy."
    )

    return "\n".join(
        report
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # HEADER
    # ========================================================

    section(
        "CIRCULENS DATASET DISTRIBUTION AUDIT"
    )

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

    # ========================================================
    # SCAN DATASETS
    # ========================================================

    section(
        "SCANNING DATASETS"
    )

    all_records = []

    for dataset_name, config in (
        DATASETS.items()
    ):

        root = config["root"]

        print(
            f"{dataset_name}:"
        )

        print(
            f"  Root: {root}"
        )

        if not os.path.exists(root):

            print(
                "  STATUS: NOT FOUND"
            )

            print()

            continue

        records = scan_images(
            root,
            dataset_name,
        )

        print(
            f"  Images: {len(records)}"
        )

        all_records.extend(
            records
        )

        print()

    # ========================================================
    # DATAFRAME
    # ========================================================

    section(
        "CREATING DATAFRAME"
    )

    df = pd.DataFrame(
        all_records
    )

    if len(df) == 0:

        raise RuntimeError(
            "No images were found."
        )

    print(
        f"Total scanned images: {len(df)}"
    )

    print()

    print(
        "Dataset counts:"
    )

    print(
        df[
            "dataset"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # BASIC DATASET SUMMARY
    # ========================================================

    section(
        "DATASET SUMMARY"
    )

    dataset_summary = (
        create_dataset_summary(
            df
        )
    )

    print(
        dataset_summary.to_string(
            index=False
        )
    )

    dataset_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dataset_summary.csv",
        ),
        index=False,
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    section(
        "CLASS DISTRIBUTION"
    )

    class_distribution = (
        create_class_distribution(
            df
        )
    )

    print(
        class_distribution.to_string(
            index=False
        )
    )

    class_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "class_distribution.csv",
        ),
        index=False,
    )

    # ========================================================
    # SPLIT DISTRIBUTION
    # ========================================================

    section(
        "SPLIT DISTRIBUTION"
    )

    split_distribution = (
        create_split_distribution(
            df
        )
    )

    print(
        split_distribution.to_string(
            index=False
        )
    )

    split_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "split_distribution.csv",
        ),
        index=False,
    )

    # ========================================================
    # SPLIT PERCENTAGE
    # ========================================================

    section(
        "SPLIT × CLASS PERCENTAGE"
    )

    split_quality_percentage = (
        create_split_quality_percentage(
            df
        )
    )

    print(
        split_quality_percentage.to_string(
            index=False
        )
    )

    split_quality_percentage.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "split_quality_percentage.csv",
        ),
        index=False,
    )

    # ========================================================
    # COMMODITY
    # ========================================================

    section(
        "COMMODITY DISTRIBUTION"
    )

    commodity_distribution = (
        create_commodity_distribution(
            df
        )
    )

    if len(
        commodity_distribution
    ) > 0:

        print(
            commodity_distribution.to_string(
                index=False
            )
        )

        commodity_distribution.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "commodity_distribution.csv",
            ),
            index=False,
        )

    else:

        print(
            "No commodity information available."
        )

    # ========================================================
    # COMMODITY × QUALITY
    # ========================================================

    section(
        "COMMODITY × QUALITY"
    )

    commodity_quality = (
        create_commodity_quality_distribution(
            df
        )
    )

    if len(
        commodity_quality
    ) > 0:

        print(
            commodity_quality.to_string(
                index=False
            )
        )

        commodity_quality.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "commodity_quality_distribution.csv",
            ),
            index=False,
        )

    else:

        print(
            "No commodity-quality information."
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    section(
        "DATA QUALITY CHECK"
    )

    data_quality = (
        create_data_quality_report(
            df
        )
    )

    print(
        data_quality.to_string(
            index=False
        )
    )

    data_quality.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "data_quality.csv",
        ),
        index=False,
    )

    # ========================================================
    # SAVE RAW SCAN
    # ========================================================

    section(
        "SAVING IMAGE INDEX"
    )

    image_index_path = os.path.join(
        OUTPUT_DIR,
        "dataset_image_index.csv",
    )

    df.to_csv(
        image_index_path,
        index=False,
    )

    print(
        f"Saved: {image_index_path}"
    )

    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    section(
        "GENERATING VISUALIZATIONS"
    )

    plot_class_distribution(
        class_distribution
    )

    plot_class_percentage(
        class_distribution
    )

    plot_split_distribution(
        split_distribution
    )

    plot_commodity_distribution(
        commodity_distribution
    )

    plot_commodity_quality(
        commodity_quality
    )

    # ========================================================
    # REPORT
    # ========================================================

    section(
        "GENERATING REPORT"
    )

    report = generate_report(
        df=df,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        split_distribution=split_distribution,
        commodity_distribution=commodity_distribution,
        commodity_quality=commodity_quality,
        data_quality=data_quality,
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "dataset_distribution_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report
        )

    print(
        f"Saved report: {report_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "DATASET DISTRIBUTION AUDIT COMPLETED"
    )

    print(
        f"Total images scanned: {len(df)}"
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
        "Generated files:"
    )

    for filename in sorted(
        os.listdir(
            OUTPUT_DIR
        )
    ):

        print(
            f"  {filename}"
        )

    print()

    print(
        "=" * 100
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()