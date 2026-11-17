import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = (
    "reports/v3_domain_disjoint_analysis"
)

PREDICTIONS_FILE = (
    "reports/v3_domain_disjoint/"
    "v3_domain_disjoint_predictions.csv"
)

ENRICHED_FILE = (
    INPUT_DIR +
    "/v3_domain_disjoint_enriched.csv"
)

OUTPUT_DIR = (
    "reports/v3_domain_disjoint_visual"
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

pd.set_option(
    "display.max_rows",
    200
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}"
)


# ============================================================
# PLOT SETTINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=UserWarning
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


def save_figure(filename):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )

    return path


def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


def normalize_confidence(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    # --------------------------------------------------------
    # If confidence appears to be percentage values
    # such as 99.91 instead of 0.9991,
    # convert to 0-1.
    # --------------------------------------------------------

    if values.notna().any():

        maximum = values.max()

        if maximum > 1.0:

            values = values / 100.0

    return values


def label_value(value):

    return str(value)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # HEADER
    # ========================================================

    section(
        "V3 DOMAIN-DISJOINT VISUAL ANALYSIS"
    )

    print(
        f"Input directory:\n{INPUT_DIR}"
    )

    print()

    print(
        f"Prediction file:\n{PREDICTIONS_FILE}"
    )

    print()

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    section(
        "LOADING DATA"
    )

    if os.path.exists(
        ENRICHED_FILE
    ):

        print(
            "Using enriched analysis file:"
        )

        print(
            ENRICHED_FILE
        )

        df = pd.read_csv(
            ENRICHED_FILE
        )

    elif os.path.exists(
        PREDICTIONS_FILE
    ):

        print(
            "Enriched analysis file not found."
        )

        print(
            "Using prediction file instead:"
        )

        print(
            PREDICTIONS_FILE
        )

        df = pd.read_csv(
            PREDICTIONS_FILE
        )

    else:

        raise FileNotFoundError(
            "Neither enriched analysis file "
            "nor prediction file was found."
        )

    print()

    print(
        f"Rows   : {len(df)}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # ========================================================
    # IDENTIFY COLUMNS
    # ========================================================

    section(
        "IDENTIFYING COLUMNS"
    )

    actual_col = find_column(
        df,
        [
            "analysis_actual",
            "label",
            "actual",
            "actual_label",
            "true_label"
        ]
    )

    prediction_col = find_column(
        df,
        [
            "analysis_predicted",
            "prediction",
            "predicted",
            "predicted_label"
        ]
    )

    confidence_col = find_column(
        df,
        [
            "analysis_confidence",
            "confidence",
            "prediction_confidence"
        ]
    )

    correct_col = find_column(
        df,
        [
            "analysis_correct",
            "correct"
        ]
    )

    source_col = find_column(
        df,
        [
            "analysis_source",
            "source",
            "dataset",
            "source_name"
        ]
    )

    date_col = find_column(
        df,
        [
            "analysis_date",
            "date",
            "capture_date"
        ]
    )

    target_domain_col = find_column(
        df,
        [
            "target_domain"
        ]
    )

    path_col = find_column(
        df,
        [
            "path",
            "image_path",
            "filepath"
        ]
    )

    print(
        f"Actual column     : {actual_col}"
    )

    print(
        f"Prediction column : {prediction_col}"
    )

    print(
        f"Confidence column : {confidence_col}"
    )

    print(
        f"Correct column    : {correct_col}"
    )

    print(
        f"Source column     : {source_col}"
    )

    print(
        f"Date column       : {date_col}"
    )

    print(
        f"Target domain     : {target_domain_col}"
    )

    print(
        f"Path column       : {path_col}"
    )

    if actual_col is None:

        raise ValueError(
            "Actual-label column not found."
        )

    if prediction_col is None:

        raise ValueError(
            "Prediction column not found."
        )

    # ========================================================
    # PREPARE ANALYSIS DATA
    # ========================================================

    section(
        "PREPARING VISUAL ANALYSIS DATA"
    )

    analysis = df.copy()

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    analysis["visual_actual"] = (
        analysis[actual_col]
        .astype(str)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    analysis["visual_prediction"] = (
        analysis[prediction_col]
        .astype(str)
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if confidence_col is not None:

        analysis["visual_confidence"] = (
            normalize_confidence(
                analysis[confidence_col]
            )
        )

    else:

        analysis["visual_confidence"] = (
            np.nan
        )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    if correct_col is not None:

        correct_values = (
            analysis[correct_col]
            .astype(str)
            .str.lower()
        )

        analysis["visual_correct"] = (
            correct_values.isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        )

    else:

        analysis["visual_correct"] = (
            analysis["visual_actual"]
            ==
            analysis["visual_prediction"]
        )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if source_col is not None:

        analysis["visual_source"] = (
            analysis[source_col]
            .astype(str)
        )

    else:

        analysis["visual_source"] = (
            "unknown"
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if date_col is not None:

        analysis["visual_date"] = (
            analysis[date_col]
            .astype(str)
        )

    else:

        analysis["visual_date"] = (
            "unknown"
        )

    # --------------------------------------------------------
    # Target domain
    # --------------------------------------------------------

    if target_domain_col is not None:

        analysis["visual_target_domain"] = (
            analysis[target_domain_col]
            .astype(str)
        )

    else:

        analysis["visual_target_domain"] = (
            analysis[
                "visual_source"
            ]
            +
            " | " +
            analysis[
                "visual_date"
            ]
            +
            " | " +
            analysis[
                "visual_actual"
            ]
        )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    section(
        "VISUAL ANALYSIS SUMMARY"
    )

    total = len(
        analysis
    )

    correct = int(
        analysis[
            "visual_correct"
        ].sum()
    )

    errors = (
        total -
        correct
    )

    accuracy = (
        correct /
        total *
        100
        if total > 0
        else 0
    )

    print(
        f"Total samples : {total}"
    )

    print(
        f"Correct       : {correct}"
    )

    print(
        f"Errors        : {errors}"
    )

    print(
        f"Accuracy      : {accuracy:.2f}%"
    )

    # ========================================================
    # ERROR DATA
    # ========================================================

    errors_df = analysis[
        ~analysis[
            "visual_correct"
        ]
    ].copy()

    print()

    print(
        f"Error samples : {len(errors_df)}"
    )

    # ========================================================
    # 1. CONFUSION MATRIX
    # ========================================================

    section(
        "1. CONFUSION MATRIX VISUALIZATION"
    )

    confusion = pd.crosstab(
        analysis["visual_actual"],
        analysis["visual_prediction"]
    )

    actual_labels = list(
        confusion.index
    )

    predicted_labels = list(
        confusion.columns
    )

    matrix = (
        confusion
        .reindex(
            index=actual_labels,
            columns=predicted_labels,
            fill_value=0
        )
        .values
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto"
    )

    plt.title(
        "V3 Domain-Disjoint Confusion Matrix",
        fontsize=15
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xticks(
        range(len(predicted_labels)),
        predicted_labels,
        rotation=30,
        ha="right"
    )

    plt.yticks(
        range(len(actual_labels)),
        actual_labels
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=12
            )

    plt.colorbar(
        label="Number of Samples"
    )

    save_figure(
        "v3_domain_disjoint_confusion_matrix.png"
    )

    # ========================================================
    # 2. ACTUAL VS PREDICTED DISTRIBUTION
    # ========================================================

    section(
        "2. ACTUAL VS PREDICTED DISTRIBUTION"
    )

    actual_counts = (
        analysis[
            "visual_actual"
        ]
        .value_counts()
    )

    prediction_counts = (
        analysis[
            "visual_prediction"
        ]
        .value_counts()
    )

    all_classes = sorted(
        set(
            actual_counts.index
        )
        |
        set(
            prediction_counts.index
        )
    )

    actual_values = [
        actual_counts.get(
            label,
            0
        )
        for label in all_classes
    ]

    prediction_values = [
        prediction_counts.get(
            label,
            0
        )
        for label in all_classes
    ]

    x = np.arange(
        len(all_classes)
    )

    width = 0.38

    plt.figure(
        figsize=(11, 7)
    )

    plt.bar(
        x - width / 2,
        actual_values,
        width,
        label="Actual"
    )

    plt.bar(
        x + width / 2,
        prediction_values,
        width,
        label="Predicted"
    )

    plt.title(
        "V3 Domain-Disjoint Actual vs Predicted Distribution",
        fontsize=15
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.xticks(
        x,
        all_classes,
        rotation=30,
        ha="right"
    )

    plt.legend()

    save_figure(
        "v3_actual_vs_predicted_distribution.png"
    )

    # ========================================================
    # 3. CONFIDENCE DISTRIBUTION
    # ========================================================

    section(
        "3. CONFIDENCE DISTRIBUTION"
    )

    confidence_valid = analysis[
        analysis[
            "visual_confidence"
        ].notna()
    ].copy()

    if len(confidence_valid) > 0:

        confidence_percent = (
            confidence_valid[
                "visual_confidence"
            ]
            * 100
        )

        plt.figure(
            figsize=(11, 7)
        )

        plt.hist(
            confidence_percent,
            bins=[
                0,
                50,
                70,
                80,
                90,
                95,
                99,
                100
            ],
            edgecolor="black"
        )

        plt.axvline(
            90,
            linestyle="--",
            label="90% confidence"
        )

        plt.axvline(
            99,
            linestyle="--",
            label="99% confidence"
        )

        plt.title(
            "V3 Domain-Disjoint Prediction Confidence",
            fontsize=15
        )

        plt.xlabel(
            "Confidence (%)"
        )

        plt.ylabel(
            "Number of Samples"
        )

        plt.legend()

        save_figure(
            "v3_confidence_distribution.png"
        )

    else:

        print(
            "Confidence data unavailable."
        )

    # ========================================================
    # 4. CONFIDENCE BY ACTUAL CLASS
    # ========================================================

    section(
        "4. CONFIDENCE BY ACTUAL CLASS"
    )

    if len(confidence_valid) > 0:

        class_values = sorted(
            confidence_valid[
                "visual_actual"
            ].unique()
        )

        box_data = []

        box_labels = []

        for class_name in class_values:

            values = (
                confidence_valid[
                    confidence_valid[
                        "visual_actual"
                    ] == class_name
                ]["visual_confidence"]
                * 100
            )

            if len(values) > 0:

                box_data.append(
                    values.values
                )

                box_labels.append(
                    class_name
                )

        plt.figure(
            figsize=(10, 7)
        )

        plt.boxplot(
            box_data,
            tick_labels=box_labels,
            showmeans=True
        )

        plt.title(
            "V3 Confidence by Actual Class",
            fontsize=15
        )

        plt.xlabel(
            "Actual Class"
        )

        plt.ylabel(
            "Prediction Confidence (%)"
        )

        plt.xticks(
            rotation=30
        )

        save_figure(
            "v3_confidence_by_actual_class.png"
        )

    # ========================================================
    # 5. ERROR PATTERN
    # ========================================================

    section(
        "5. ERROR PATTERN VISUALIZATION"
    )

    error_pairs = (
        errors_df
        .groupby(
            [
                "visual_actual",
                "visual_prediction"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    if len(error_pairs) > 0:

        error_labels = []

        error_values = []

        for _, row in error_pairs.iterrows():

            label = (
                f"{row['visual_actual']} "
                f"→ "
                f"{row['visual_prediction']}"
            )

            error_labels.append(
                label
            )

            error_values.append(
                row["count"]
            )

        plt.figure(
            figsize=(11, 7)
        )

        bars = plt.bar(
            error_labels,
            error_values
        )

        plt.title(
            "V3 Domain-Disjoint Misclassification Patterns",
            fontsize=15
        )

        plt.xlabel(
            "Actual → Predicted"
        )

        plt.ylabel(
            "Number of Errors"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        for bar, value in zip(
            bars,
            error_values
        ):

            plt.text(
                bar.get_x()
                +
                bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom"
            )

        save_figure(
            "v3_error_patterns.png"
        )

    # ========================================================
    # 6. DOMAIN ACCURACY
    # ========================================================

    section(
        "6. DOMAIN ACCURACY"
    )

    domain_group_columns = [
        "visual_source",
        "visual_date",
        "visual_actual"
    ]

    domain_stats = (
        analysis
        .groupby(
            domain_group_columns
        )
        .agg(
            samples=(
                "visual_correct",
                "size"
            ),
            correct=(
                "visual_correct",
                "sum"
            ),
            mean_confidence=(
                "visual_confidence",
                "mean"
            )
        )
        .reset_index()
    )

    domain_stats["accuracy"] = (
        domain_stats["correct"]
        /
        domain_stats["samples"]
        *
        100
    )

    print(
        domain_stats.to_string(
            index=False
        )
    )

    domain_labels = []

    domain_accuracy = []

    for _, row in domain_stats.iterrows():

        label = (
            f"{row['visual_date']}\n"
            f"{row['visual_actual']}"
        )

        domain_labels.append(
            label
        )

        domain_accuracy.append(
            row["accuracy"]
        )

    plt.figure(
        figsize=(11, 7)
    )

    bars = plt.bar(
        domain_labels,
        domain_accuracy
    )

    plt.title(
        "V3 Domain-Disjoint Accuracy by Domain",
        fontsize=15
    )

    plt.xlabel(
        "Domain"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        domain_accuracy
    ):

        plt.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom"
        )

    save_figure(
        "v3_domain_accuracy.png"
    )

    domain_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "visual_domain_statistics.csv"
        ),
        index=False
    )

    # ========================================================
    # 7. DOMAIN CONFIDENCE
    # ========================================================

    section(
        "7. DOMAIN CONFIDENCE"
    )

    domain_confidence = (
        domain_stats[
            "mean_confidence"
        ]
        * 100
    )

    plt.figure(
        figsize=(11, 7)
    )

    bars = plt.bar(
        domain_labels,
        domain_confidence
    )

    plt.title(
        "V3 Prediction Confidence by Unseen Domain",
        fontsize=15
    )

    plt.xlabel(
        "Domain"
    )

    plt.ylabel(
        "Mean Confidence (%)"
    )

    plt.ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        domain_confidence
    ):

        if pd.notna(value):

            plt.text(
                bar.get_x()
                +
                bar.get_width() / 2,
                bar.get_height() + 2,
                f"{value:.2f}%",
                ha="center",
                va="bottom"
            )

    save_figure(
        "v3_domain_confidence.png"
    )

    # ========================================================
    # 8. CONFIDENCE VS CORRECTNESS
    # ========================================================

    section(
        "8. CONFIDENCE VS CORRECTNESS"
    )

    if len(confidence_valid) > 0:

        correct_values = confidence_valid[
            confidence_valid[
                "visual_correct"
            ]
        ]

        error_values = confidence_valid[
            ~confidence_valid[
                "visual_correct"
            ]
        ]

        plt.figure(
            figsize=(10, 7)
        )

        if len(correct_values) > 0:

            x_correct = np.ones(
                len(correct_values)
            )

            plt.scatter(
                x_correct,
                correct_values[
                    "visual_confidence"
                ] * 100,
                alpha=0.7,
                label="Correct"
            )

        if len(error_values) > 0:

            x_error = np.ones(
                len(error_values)
            ) * 2

            plt.scatter(
                x_error,
                error_values[
                    "visual_confidence"
                ] * 100,
                alpha=0.7,
                label="Error"
            )

        plt.title(
            "V3 Confidence vs Prediction Correctness",
            fontsize=15
        )

        plt.xlabel(
            "Prediction Status"
        )

        plt.ylabel(
            "Confidence (%)"
        )

        plt.xticks(
            [1, 2],
            ["Correct", "Error"]
        )

        plt.ylim(
            0,
            105
        )

        plt.legend()

        save_figure(
            "v3_confidence_vs_correctness.png"
        )

    # ========================================================
    # 9. ACTUAL VS PREDICTED PERCENTAGE
    # ========================================================

    section(
        "9. ACTUAL VS PREDICTED PERCENTAGE"
    )

    percentage_table = pd.crosstab(
        analysis[
            "visual_actual"
        ],
        analysis[
            "visual_prediction"
        ],
        normalize="index"
    ) * 100

    print(
        percentage_table.to_string()
    )

    classes = list(
        percentage_table.index
    )

    predicted_classes = list(
        percentage_table.columns
    )

    x = np.arange(
        len(classes)
    )

    bottom = np.zeros(
        len(classes)
    )

    plt.figure(
        figsize=(12, 7)
    )

    for prediction_class in predicted_classes:

        values = (
            percentage_table
            .get(
                prediction_class,
                pd.Series(
                    0,
                    index=classes
                )
            )
            .values
        )

        plt.bar(
            x,
            values,
            bottom=bottom,
            label=prediction_class
        )

        bottom += values

    plt.title(
        "Predicted Class Distribution Within Each Actual Class",
        fontsize=15
    )

    plt.xlabel(
        "Actual Class"
    )

    plt.ylabel(
        "Prediction Distribution (%)"
    )

    plt.xticks(
        x,
        classes,
        rotation=30,
        ha="right"
    )

    plt.ylim(
        0,
        100
    )

    plt.legend(
        title="Predicted"
    )

    save_figure(
        "v3_actual_vs_predicted_percentage.png"
    )

    percentage_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "visual_actual_vs_predicted_percentage.csv"
        )
    )

    # ========================================================
    # 10. HIGH-CONFIDENCE ERROR ANALYSIS
    # ========================================================

    section(
        "10. HIGH-CONFIDENCE ERROR ANALYSIS"
    )

    if len(confidence_valid) > 0:

        high_conf_errors = errors_df[
            errors_df[
                "visual_confidence"
            ] >= 0.90
        ].copy()

        print(
            "High-confidence errors "
            f"(>=90%): {len(high_conf_errors)}"
        )

        high_conf_pairs = (
            high_conf_errors
            .groupby(
                [
                    "visual_actual",
                    "visual_prediction"
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False
            )
        )

        print()

        print(
            high_conf_pairs.to_string(
                index=False
            )
        )

        high_conf_pairs.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "visual_high_confidence_errors.csv"
            ),
            index=False
        )

        if len(high_conf_pairs) > 0:

            labels = [
                (
                    f"{row['visual_actual']} "
                    f"→ "
                    f"{row['visual_prediction']}"
                )
                for _, row
                in high_conf_pairs.iterrows()
            ]

            values = (
                high_conf_pairs[
                    "count"
                ].values
            )

            plt.figure(
                figsize=(11, 7)
            )

            bars = plt.bar(
                labels,
                values
            )

            plt.title(
                "High-Confidence V3 Errors",
                fontsize=15
            )

            plt.xlabel(
                "Actual → Predicted"
            )

            plt.ylabel(
                "Number of Errors"
            )

            plt.xticks(
                rotation=30,
                ha="right"
            )

            for bar, value in zip(
                bars,
                values
            ):

                plt.text(
                    bar.get_x()
                    +
                    bar.get_width() / 2,
                    bar.get_height(),
                    str(value),
                    ha="center",
                    va="bottom"
                )

            save_figure(
                "v3_high_confidence_errors.png"
            )

    # ========================================================
    # 11. ERROR CONFIDENCE BY ERROR PATTERN
    # ========================================================

    section(
        "11. ERROR CONFIDENCE BY ERROR PATTERN"
    )

    if len(confidence_valid) > 0:

        error_confidence_data = (
            errors_df[
                errors_df[
                    "visual_confidence"
                ].notna()
            ]
            .copy()
        )

        if len(
            error_confidence_data
        ) > 0:

            error_confidence_data[
                "error_pattern"
            ] = (
                error_confidence_data[
                    "visual_actual"
                ]
                +
                " → " +
                error_confidence_data[
                    "visual_prediction"
                ]
            )

            patterns = (
                error_confidence_data[
                    "error_pattern"
                ]
                .unique()
                .tolist()
            )

            box_data = []

            labels = []

            for pattern in patterns:

                values = (
                    error_confidence_data[
                        error_confidence_data[
                            "error_pattern"
                        ] == pattern
                    ]["visual_confidence"]
                    * 100
                )

                if len(values) > 0:

                    box_data.append(
                        values.values
                    )

                    labels.append(
                        pattern
                    )

            if len(box_data) > 0:

                plt.figure(
                    figsize=(11, 7)
                )

                plt.boxplot(
                    box_data,
                    tick_labels=box_labels,
                    showmeans=True
                )

                plt.title(
                    "Confidence Distribution by Error Pattern",
                    fontsize=15
                )

                plt.xlabel(
                    "Error Pattern"
                )

                plt.ylabel(
                    "Confidence (%)"
                )

                plt.xticks(
                    rotation=30
                )

                plt.ylim(
                    0,
                    105
                )

                save_figure(
                    "v3_error_confidence_by_pattern.png"
                )

    # ========================================================
    # 12. DOMAIN ACCURACY VS CONFIDENCE
    # ========================================================

    section(
        "12. DOMAIN ACCURACY VS CONFIDENCE"
    )

    if len(domain_stats) > 0:

        x_accuracy = (
            domain_stats[
                "accuracy"
            ].values
        )

        y_confidence = (
            domain_stats[
                "mean_confidence"
            ].values
            *
            100
        )

        plt.figure(
            figsize=(9, 7)
        )

        plt.scatter(
            x_accuracy,
            y_confidence,
            s=100,
            alpha=0.8
        )

        for i, row in domain_stats.iterrows():

            label = (
                f"{row['visual_date']} "
                f"{row['visual_actual']}"
            )

            plt.annotate(
                label,
                (
                    row["accuracy"],
                    row["mean_confidence"] * 100
                ),
                xytext=(
                    7,
                    7
                ),
                textcoords="offset points"
            )

        plt.title(
            "V3 Domain Accuracy vs Confidence",
            fontsize=15
        )

        plt.xlabel(
            "Accuracy (%)"
        )

        plt.ylabel(
            "Mean Confidence (%)"
        )

        plt.xlim(
            0,
            100
        )

        plt.ylim(
            0,
            105
        )

        save_figure(
            "v3_domain_accuracy_vs_confidence.png"
        )

    # ========================================================
    # 13. FAILURE SUMMARY INFOGRAPHIC
    # ========================================================

    section(
        "13. FAILURE SUMMARY"
    )

    total_errors = len(
        errors_df
    )

    high_confidence_count = 0

    if len(confidence_valid) > 0:

        high_confidence_count = len(
            errors_df[
                errors_df[
                    "visual_confidence"
                ] >= 0.90
            ]
        )

    high_confidence_percentage = (
        high_confidence_count /
        total_errors *
        100
        if total_errors > 0
        else 0
    )

    top_error_pattern = (
        "N/A"
    )

    top_error_count = 0

    if len(error_pairs) > 0:

        first_error = (
            error_pairs.iloc[0]
        )

        top_error_pattern = (
            f"{first_error['visual_actual']}"
            f" → "
            f"{first_error['visual_prediction']}"
        )

        top_error_count = int(
            first_error["count"]
        )

    plt.figure(
        figsize=(12, 8)
    )

    plt.axis(
        "off"
    )

    summary_lines = [
        "CIRCULENS V3 DOMAIN-DISJOINT FAILURE",
        "",
        f"Total evaluated samples : {total}",
        f"Correct predictions      : {correct}",
        f"Errors                   : {errors}",
        f"Accuracy                 : {accuracy:.2f}%",
        "",
        f"High-confidence errors   : "
        f"{high_confidence_count}/{total_errors}",
        f"High-confidence ratio    : "
        f"{high_confidence_percentage:.2f}%",
        "",
        f"Dominant error pattern   : "
        f"{top_error_pattern}",
        f"Dominant error count     : "
        f"{top_error_count}",
        "",
        "Interpretation:",
        "V3 shows strong failure on unseen domains.",
        "The model remains highly confident while",
        "making systematic domain-disjoint errors."
    ]

    plt.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        family="monospace"
    )

    save_figure(
        "v3_domain_disjoint_failure_summary.png"
    )

    # ========================================================
    # SAVE VISUAL ANALYSIS DATA
    # ========================================================

    section(
        "SAVING VISUAL ANALYSIS DATA"
    )

    visual_columns = [
        "visual_actual",
        "visual_prediction",
        "visual_confidence",
        "visual_correct",
        "visual_source",
        "visual_date",
        "visual_target_domain"
    ]

    if path_col is not None:

        visual_columns.insert(
            0,
            path_col
        )

    visual_columns = [
        column
        for column in visual_columns
        if column in analysis.columns
    ]

    visual_analysis = analysis[
        visual_columns
    ].copy()

    visual_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v3_visual_analysis_data.csv"
        ),
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    section(
        "GENERATING VISUAL ANALYSIS REPORT"
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "v3_domain_disjoint_visual_report.txt"
    )

    report_lines = []

    report_lines.append(
        "CIRCULENS V3 DOMAIN-DISJOINT VISUAL ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        "SUMMARY"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        f"Total samples : {total}"
    )

    report_lines.append(
        f"Correct       : {correct}"
    )

    report_lines.append(
        f"Errors        : {errors}"
    )

    report_lines.append(
        f"Accuracy      : {accuracy:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        "CONFIDENCE"
    )

    report_lines.append(
        "-" * 70
    )

    if len(confidence_valid) > 0:

        report_lines.append(
            f"Mean confidence   : "
            f"{confidence_valid['visual_confidence'].mean():.4f}"
        )

        report_lines.append(
            f"Median confidence : "
            f"{confidence_valid['visual_confidence'].median():.4f}"
        )

        report_lines.append(
            f"Minimum confidence: "
            f"{confidence_valid['visual_confidence'].min():.4f}"
        )

        report_lines.append(
            f"Maximum confidence: "
            f"{confidence_valid['visual_confidence'].max():.4f}"
        )

        report_lines.append(
            f"Errors >=90%      : "
            f"{high_confidence_count}"
        )

    else:

        report_lines.append(
            "Confidence unavailable."
        )

    report_lines.append("")

    report_lines.append(
        "ERROR PATTERNS"
    )

    report_lines.append(
        "-" * 70
    )

    if len(error_pairs) > 0:

        report_lines.append(
            error_pairs.to_string(
                index=False
            )
        )

    else:

        report_lines.append(
            "No errors."
        )

    report_lines.append("")

    report_lines.append(
        "DOMAIN STATISTICS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        domain_stats.to_string(
            index=False
        )
    )

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append(
        "-" * 70
    )

    if accuracy == 0:

        report_lines.append(
            "V3 achieved 0% accuracy on the "
            "domain-disjoint evaluation set."
        )

    if (
        high_confidence_percentage
        >= 50
    ):

        report_lines.append(
            "Most errors are high-confidence errors, "
            "indicating severe out-of-domain "
            "overconfidence."
        )

    if len(error_pairs) > 0:

        report_lines.append(
            "The failure is systematic rather than "
            "random because one or more "
            "misclassification patterns dominate "
            "the evaluation set."
        )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT:"
    )

    report_lines.append(
        "These visualizations do not prove the exact "
        "root cause of the failure."
    )

    report_lines.append(
        "They provide evidence that should be "
        "combined with image-level inspection, "
        "dataset metadata inspection, preprocessing "
        "verification, and controlled retraining "
        "experiments."
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                report_lines
            )
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    section(
        "V3 DOMAIN-DISJOINT VISUAL ANALYSIS COMPLETED"
    )

    print(
        f"Samples : {total}"
    )

    print(
        f"Correct : {correct}"
    )

    print(
        f"Errors  : {errors}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print()

    print(
        "Generated visualizations:"
    )

    print(
        "  v3_domain_disjoint_confusion_matrix.png"
    )

    print(
        "  v3_actual_vs_predicted_distribution.png"
    )

    print(
        "  v3_confidence_distribution.png"
    )

    print(
        "  v3_confidence_by_actual_class.png"
    )

    print(
        "  v3_error_patterns.png"
    )

    print(
        "  v3_domain_accuracy.png"
    )

    print(
        "  v3_domain_confidence.png"
    )

    print(
        "  v3_confidence_vs_correctness.png"
    )

    print(
        "  v3_actual_vs_predicted_percentage.png"
    )

    print(
        "  v3_high_confidence_errors.png"
    )

    print(
        "  v3_error_confidence_by_pattern.png"
    )

    print(
        "  v3_domain_accuracy_vs_confidence.png"
    )

    print(
        "  v3_domain_disjoint_failure_summary.png"
    )

    print()

    print(
        "Data:"
    )

    print(
        "  v3_visual_analysis_data.csv"
    )

    print(
        "  visual_domain_statistics.csv"
    )

    print(
        "  visual_actual_vs_predicted_percentage.csv"
    )

    print(
        "  visual_high_confidence_errors.csv"
    )

    print()

    print(
        "Report:"
    )

    print(
        f"  {report_path}"
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