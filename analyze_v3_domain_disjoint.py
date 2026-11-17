import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = (
    "reports/v3_domain_disjoint/"
    "v3_domain_disjoint_predictions.csv"
)

OUTPUT_DIR = (
    "reports/v3_domain_disjoint_analysis"
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
# HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


def safe_mean(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return np.nan

    return series.mean()


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
        "V3 DOMAIN-DISJOINT FAILURE ANALYSIS"
    )

    print(
        f"Input:\n{INPUT_FILE}"
    )

    print()

    print(
        f"Output:\n{OUTPUT_DIR}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING PREDICTIONS"
    )

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Prediction file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows : {len(df)}"
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
            "label",
            "actual",
            "actual_label",
            "true_label"
        ]
    )

    prediction_col = find_column(
        df,
        [
            "prediction",
            "predicted",
            "predicted_label",
            "prediction_label"
        ]
    )

    confidence_col = find_column(
        df,
        [
            "confidence",
            "prediction_confidence"
        ]
    )

    source_col = find_column(
        df,
        [
            "source",
            "dataset",
            "source_name"
        ]
    )

    date_col = find_column(
        df,
        [
            "date",
            "capture_date"
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

    if actual_col is None:

        raise ValueError(
            "Actual-label column not found."
        )

    if prediction_col is None:

        raise ValueError(
            "Prediction column not found."
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
        f"Source column     : {source_col}"
    )

    print(
        f"Date column       : {date_col}"
    )

    print(
        f"Path column       : {path_col}"
    )

    # ========================================================
    # NORMALIZE LABEL REPRESENTATION
    # ========================================================

    analysis = df.copy()

    analysis["analysis_actual"] = (
        analysis[actual_col]
    )

    analysis["analysis_predicted"] = (
        analysis[prediction_col]
    )

    # --------------------------------------------------------
    # Actual labels
    # --------------------------------------------------------

    numeric_actual = pd.to_numeric(
        analysis["analysis_actual"],
        errors="coerce"
    )

    if numeric_actual.notna().all():

        analysis["analysis_actual"] = (
            numeric_actual.astype(int)
        )

    else:

        analysis["analysis_actual"] = (
            analysis["analysis_actual"]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Prediction labels
    # --------------------------------------------------------

    numeric_prediction = pd.to_numeric(
        analysis["analysis_predicted"],
        errors="coerce"
    )

    if numeric_prediction.notna().all():

        analysis["analysis_predicted"] = (
            numeric_prediction.astype(int)
        )

    else:

        analysis["analysis_predicted"] = (
            analysis["analysis_predicted"]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Confidence
    #
    # IMPORTANT:
    # Create analysis_confidence HERE before any section
    # that uses it.
    # This fixes the previous KeyError.
    # --------------------------------------------------------

    if confidence_col is not None:

        analysis["analysis_confidence"] = (
            pd.to_numeric(
                analysis[confidence_col],
                errors="coerce"
            )
        )

    else:

        analysis["analysis_confidence"] = np.nan

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if source_col is not None:

        analysis["analysis_source"] = (
            analysis[source_col]
            .astype(str)
        )

    else:

        analysis["analysis_source"] = (
            "unknown"
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if date_col is not None:

        analysis["analysis_date"] = (
            analysis[date_col]
            .astype(str)
        )

    else:

        analysis["analysis_date"] = (
            "unknown"
        )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    section(
        "BASIC SUMMARY"
    )

    total_samples = len(
        analysis
    )

    # Do NOT blindly trust the existing "correct" column.
    # Recalculate correctness from actual vs prediction.
    analysis["analysis_correct"] = (
        analysis["analysis_actual"]
        ==
        analysis["analysis_predicted"]
    )

    correct_count = int(
        analysis["analysis_correct"].sum()
    )

    error_count = (
        total_samples -
        correct_count
    )

    accuracy = (
        correct_count /
        total_samples *
        100
        if total_samples > 0
        else 0
    )

    print(
        f"Total samples : {total_samples}"
    )

    print(
        f"Correct       : {correct_count}"
    )

    print(
        f"Errors        : {error_count}"
    )

    print(
        f"Accuracy      : {accuracy:.2f}%"
    )

    # ========================================================
    # ACTUAL CLASS DISTRIBUTION
    # ========================================================

    section(
        "ACTUAL CLASS DISTRIBUTION"
    )

    actual_distribution = (
        analysis["analysis_actual"]
        .value_counts()
        .sort_index()
        .rename("count")
        .to_frame()
    )

    print(
        actual_distribution.to_string()
    )

    actual_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "actual_class_distribution.csv"
        )
    )

    # ========================================================
    # PREDICTED CLASS DISTRIBUTION
    # ========================================================

    section(
        "PREDICTED CLASS DISTRIBUTION"
    )

    predicted_distribution = (
        analysis["analysis_predicted"]
        .value_counts()
        .rename("count")
        .to_frame()
    )

    print(
        predicted_distribution.to_string()
    )

    predicted_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "predicted_class_distribution.csv"
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    section(
        "CONFUSION MATRIX"
    )

    confusion = pd.crosstab(
        analysis["analysis_actual"],
        analysis["analysis_predicted"]
    )

    confusion.index.name = (
        "analysis_actual"
    )

    confusion.columns.name = (
        "analysis_predicted"
    )

    print(
        confusion.to_string()
    )

    confusion.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "analysis_confusion_matrix.csv"
        )
    )

    # ========================================================
    # MISCLASSIFICATION PAIRS
    # ========================================================

    section(
        "MISCLASSIFICATION PAIRS"
    )

    errors = analysis[
        ~analysis["analysis_correct"]
    ].copy()

    misclassification_pairs = (
        errors
        .groupby(
            [
                "analysis_actual",
                "analysis_predicted"
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

    print(
        misclassification_pairs.to_string(
            index=False
        )
    )

    misclassification_pairs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "misclassification_pairs.csv"
        ),
        index=False
    )

    # ========================================================
    # ERROR RATE BY ACTUAL CLASS
    # ========================================================

    section(
        "ERROR RATE BY ACTUAL CLASS"
    )

    class_analysis = (
        analysis
        .groupby(
            "analysis_actual"
        )
        .agg(
            samples=(
                "analysis_actual",
                "size"
            ),
            correct=(
                "analysis_correct",
                "sum"
            )
        )
        .reset_index()
    )

    class_analysis["correct"] = (
        class_analysis["correct"]
        .astype(int)
    )

    class_analysis["errors"] = (
        class_analysis["samples"]
        -
        class_analysis["correct"]
    )

    class_analysis["accuracy"] = (
        class_analysis["correct"]
        /
        class_analysis["samples"]
        *
        100
    )

    class_analysis["error_rate"] = (
        class_analysis["errors"]
        /
        class_analysis["samples"]
        *
        100
    )

    print(
        class_analysis.to_string(
            index=False
        )
    )

    class_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "class_error_analysis.csv"
        ),
        index=False
    )

    # ========================================================
    # CONFIDENCE ANALYSIS
    # ========================================================

    section(
        "CONFIDENCE ANALYSIS"
    )

    confidence_valid = analysis[
        analysis["analysis_confidence"]
        .notna()
    ].copy()

    high_confidence_errors = (
        pd.DataFrame()
    )

    high_confidence_pairs = (
        pd.DataFrame(
            columns=[
                "analysis_actual",
                "analysis_predicted",
                "count"
            ]
        )
    )

    if len(confidence_valid) > 0:

        print(
            f"Samples with confidence: "
            f"{len(confidence_valid)}"
        )

        mean_confidence = (
            confidence_valid[
                "analysis_confidence"
            ].mean()
        )

        median_confidence = (
            confidence_valid[
                "analysis_confidence"
            ].median()
        )

        min_confidence = (
            confidence_valid[
                "analysis_confidence"
            ].min()
        )

        max_confidence = (
            confidence_valid[
                "analysis_confidence"
            ].max()
        )

        correct_confidence = (
            confidence_valid[
                confidence_valid[
                    "analysis_correct"
                ]
            ]["analysis_confidence"]
        )

        error_confidence = (
            confidence_valid[
                ~confidence_valid[
                    "analysis_correct"
                ]
            ]["analysis_confidence"]
        )

        print(
            f"Mean confidence    : "
            f"{mean_confidence:.4f}"
        )

        print(
            f"Median confidence  : "
            f"{median_confidence:.4f}"
        )

        print(
            f"Minimum confidence : "
            f"{min_confidence:.4f}"
        )

        print(
            f"Maximum confidence : "
            f"{max_confidence:.4f}"
        )

        if len(correct_confidence) > 0:

            print(
                f"Mean correct confidence : "
                f"{correct_confidence.mean():.4f}"
            )

        else:

            print(
                "Mean correct confidence : N/A"
            )

        if len(error_confidence) > 0:

            print(
                f"Mean error confidence   : "
                f"{error_confidence.mean():.4f}"
            )

        else:

            print(
                "Mean error confidence   : N/A"
            )

        # ----------------------------------------------------
        # CONFIDENCE BINS
        # ----------------------------------------------------

        def confidence_bin(value):

            percentage = value * 100

            if percentage <= 50:
                return "<=50%"

            elif percentage <= 70:
                return "50-70%"

            elif percentage <= 80:
                return "70-80%"

            elif percentage <= 90:
                return "80-90%"

            elif percentage <= 95:
                return "90-95%"

            elif percentage < 99:
                return "95-99%"

            else:
                return ">=99%"

        confidence_valid[
            "confidence_bin"
        ] = (
            confidence_valid[
                "analysis_confidence"
            ]
            .apply(confidence_bin)
        )

        bin_order = [
            "<=50%",
            "50-70%",
            "70-80%",
            "80-90%",
            "90-95%",
            "95-99%",
            ">=99%"
        ]

        confidence_distribution = (
            confidence_valid
            .groupby(
                "confidence_bin"
            )
            .agg(
                samples=(
                    "analysis_correct",
                    "size"
                ),
                correct=(
                    "analysis_correct",
                    "sum"
                )
            )
            .reindex(
                bin_order,
                fill_value=0
            )
            .reset_index()
        )

        confidence_distribution[
            "correct"
        ] = (
            confidence_distribution[
                "correct"
            ].astype(int)
        )

        confidence_distribution[
            "accuracy"
        ] = np.where(
            confidence_distribution[
                "samples"
            ] > 0,
            confidence_distribution[
                "correct"
            ]
            /
            confidence_distribution[
                "samples"
            ]
            *
            100,
            np.nan
        )

        print()

        print(
            "Confidence distribution:"
        )

        print(
            confidence_distribution.to_string(
                index=False
            )
        )

        confidence_distribution.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "confidence_distribution.csv"
            ),
            index=False
        )

    else:

        print(
            "No valid confidence values found."
        )

        confidence_distribution = pd.DataFrame()

    # ========================================================
    # DOMAIN ANALYSIS
    # ========================================================

    section(
        "DOMAIN ANALYSIS"
    )

    domain_group_columns = [
        "analysis_source",
        "analysis_date",
        "analysis_actual"
    ]

    domain_analysis = (
        analysis
        .groupby(
            domain_group_columns
        )
        .agg(
            samples=(
                "analysis_correct",
                "size"
            ),
            correct=(
                "analysis_correct",
                "sum"
            ),
            mean_confidence=(
                "analysis_confidence",
                "mean"
            )
        )
        .reset_index()
    )

    domain_analysis["correct"] = (
        domain_analysis["correct"]
        .astype(int)
    )

    domain_analysis["errors"] = (
        domain_analysis["samples"]
        -
        domain_analysis["correct"]
    )

    domain_analysis["accuracy"] = (
        domain_analysis["correct"]
        /
        domain_analysis["samples"]
        *
        100
    )

    domain_analysis["error_rate"] = (
        domain_analysis["errors"]
        /
        domain_analysis["samples"]
        *
        100
    )

    print(
        domain_analysis.to_string(
            index=False
        )
    )

    domain_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "domain_analysis.csv"
        ),
        index=False
    )

    # ========================================================
    # SOURCE ANALYSIS
    # ========================================================

    section(
        "SOURCE ANALYSIS"
    )

    source_analysis = (
        analysis
        .groupby(
            "analysis_source"
        )
        .agg(
            samples=(
                "analysis_correct",
                "size"
            ),
            correct=(
                "analysis_correct",
                "sum"
            ),
            mean_confidence=(
                "analysis_confidence",
                "mean"
            )
        )
        .reset_index()
    )

    source_analysis["correct"] = (
        source_analysis["correct"]
        .astype(int)
    )

    source_analysis["errors"] = (
        source_analysis["samples"]
        -
        source_analysis["correct"]
    )

    source_analysis["accuracy"] = (
        source_analysis["correct"]
        /
        source_analysis["samples"]
        *
        100
    )

    source_analysis["error_rate"] = (
        source_analysis["errors"]
        /
        source_analysis["samples"]
        *
        100
    )

    print(
        source_analysis.to_string(
            index=False
        )
    )

    source_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "source_analysis.csv"
        ),
        index=False
    )

    # ========================================================
    # DATE ANALYSIS
    # ========================================================

    section(
        "DATE ANALYSIS"
    )

    date_analysis = (
        analysis
        .groupby(
            "analysis_date"
        )
        .agg(
            samples=(
                "analysis_correct",
                "size"
            ),
            correct=(
                "analysis_correct",
                "sum"
            ),
            mean_confidence=(
                "analysis_confidence",
                "mean"
            )
        )
        .reset_index()
    )

    date_analysis["correct"] = (
        date_analysis["correct"]
        .astype(int)
    )

    date_analysis["errors"] = (
        date_analysis["samples"]
        -
        date_analysis["correct"]
    )

    date_analysis["accuracy"] = (
        date_analysis["correct"]
        /
        date_analysis["samples"]
        *
        100
    )

    date_analysis["error_rate"] = (
        date_analysis["errors"]
        /
        date_analysis["samples"]
        *
        100
    )

    print(
        date_analysis.to_string(
            index=False
        )
    )

    date_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "date_analysis.csv"
        ),
        index=False
    )

    # ========================================================
    # PREDICTION BIAS
    # ========================================================

    section(
        "PREDICTION BIAS"
    )

    prediction_bias = (
        analysis["analysis_predicted"]
        .value_counts()
        .rename(
            "predicted_count"
        )
        .reset_index()
    )

    prediction_bias.columns = [
        "predicted_class",
        "predicted_count"
    ]

    prediction_bias[
        "prediction_percentage"
    ] = (
        prediction_bias[
            "predicted_count"
        ]
        /
        total_samples
        *
        100
    )

    print(
        prediction_bias.to_string(
            index=False
        )
    )

    prediction_bias.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "prediction_bias.csv"
        ),
        index=False
    )

    # ========================================================
    # ACTUAL VS PREDICTED DISTRIBUTION
    # ========================================================

    section(
        "ACTUAL VS PREDICTED DISTRIBUTION"
    )

    actual_vs_predicted = (
        pd.crosstab(
            analysis["analysis_actual"],
            analysis["analysis_predicted"],
            normalize="index"
        )
        * 100
    )

    actual_vs_predicted = (
        actual_vs_predicted.round(4)
    )

    print(
        actual_vs_predicted.to_string()
    )

    actual_vs_predicted.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "actual_vs_predicted_percentage.csv"
        )
    )

    # ========================================================
    # HIGH-CONFIDENCE ERRORS
    # ========================================================

    section(
        "HIGH-CONFIDENCE ERRORS"
    )

    if len(confidence_valid) > 0:

        high_confidence_errors = (
            errors[
                errors["analysis_confidence"]
                >= 0.90
            ]
            .copy()
        )

        print(
            "Errors with confidence >= 90%: "
            f"{len(high_confidence_errors)}"
        )

        if len(high_confidence_errors) > 0:

            high_confidence_pairs = (
                high_confidence_errors
                .groupby(
                    [
                        "analysis_actual",
                        "analysis_predicted"
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

        else:

            high_confidence_pairs = (
                pd.DataFrame(
                    columns=[
                        "analysis_actual",
                        "analysis_predicted",
                        "count"
                    ]
                )
            )

        print()

        print(
            high_confidence_pairs.to_string(
                index=False
            )
        )

        high_confidence_pairs.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "high_confidence_errors.csv"
            ),
            index=False
        )

    else:

        print(
            "Confidence column not available."
        )

        high_confidence_errors = (
            pd.DataFrame()
        )

        high_confidence_pairs = (
            pd.DataFrame()
        )

        high_confidence_pairs.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "high_confidence_errors.csv"
            ),
            index=False
        )

    # ========================================================
    # SAMPLE-LEVEL ERROR TABLE
    # ========================================================

    section(
        "SAMPLE-LEVEL ERROR TABLE"
    )

    error_columns = []

    candidate_error_columns = [
        path_col,
        "filename",
        source_col,
        date_col,
        actual_col,
        prediction_col,
        "analysis_actual",
        "analysis_predicted",
        confidence_col,
        "analysis_confidence",
        "target_domain"
    ]

    for column in candidate_error_columns:

        if (
            column is not None
            and column in errors.columns
            and column not in error_columns
        ):

            error_columns.append(
                column
            )

    all_errors = errors[
        error_columns
    ].copy()

    print(
        f"Saved {len(all_errors)} error samples."
    )

    all_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_errors.csv"
        ),
        index=False
    )

    # ========================================================
    # TOP ERROR PATTERNS
    # ========================================================

    section(
        "TOP ERROR PATTERNS"
    )

    top_error_group_columns = [
        "analysis_actual",
        "analysis_predicted",
        "analysis_source",
        "analysis_date"
    ]

    top_error_patterns = (
        errors
        .groupby(
            top_error_group_columns
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

    print(
        top_error_patterns.to_string(
            index=False
        )
    )

    top_error_patterns.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_error_patterns.csv"
        ),
        index=False
    )

    # ========================================================
    # AUTOMATIC DIAGNOSTIC
    # ========================================================

    section(
        "AUTOMATIC DIAGNOSTIC"
    )

    diagnostics = []

    diagnostic_number = 1

    # --------------------------------------------------------
    # OVERALL ACCURACY
    # --------------------------------------------------------

    if accuracy == 0:

        message = (
            "CRITICAL: Model has zero correct "
            "predictions on the domain-disjoint "
            "evaluation set."
        )

    else:

        message = (
            f"INFO: Model achieved "
            f"{accuracy:.2f}% accuracy on "
            "the domain-disjoint evaluation set."
        )

    print(
        f"{diagnostic_number}. {message}"
    )

    diagnostics.append(
        message
    )

    diagnostic_number += 1

    # --------------------------------------------------------
    # HIGH CONFIDENCE OVERCONFIDENCE
    # --------------------------------------------------------

    if len(confidence_valid) > 0:

        high_conf_count = len(
            high_confidence_errors
        )

        if error_count > 0:

            high_conf_ratio = (
                high_conf_count /
                error_count *
                100
            )

        else:

            high_conf_ratio = 0

        if high_conf_ratio >= 50:

            message = (
                "CRITICAL: Most domain-disjoint "
                "errors are made with high confidence. "
                "This indicates strong out-of-domain "
                "overconfidence."
            )

        elif high_conf_count > 0:

            message = (
                "WARNING: Some domain-disjoint "
                "errors are made with high confidence."
            )

        else:

            message = (
                "INFO: No errors exceed the "
                "90% confidence threshold."
            )

        print(
            f"{diagnostic_number}. {message}"
        )

        diagnostics.append(
            message
        )

        diagnostic_number += 1

    # --------------------------------------------------------
    # DOMINANT FAILURE MODE
    # --------------------------------------------------------

    if len(misclassification_pairs) > 0:

        dominant = (
            misclassification_pairs.iloc[0]
        )

        dominant_count = int(
            dominant["count"]
        )

        dominant_percentage = (
            dominant_count /
            max(error_count, 1)
            *
            100
        )

        message = (
            "WARNING: Dominant failure mode is "
            f"{dominant['analysis_actual']} -> "
            f"{dominant['analysis_predicted']} "
            f"({dominant_percentage:.2f}% "
            "of all errors)."
        )

        print(
            f"{diagnostic_number}. {message}"
        )

        diagnostics.append(
            message
        )

        diagnostic_number += 1

    # --------------------------------------------------------
    # DOMAIN FAILURE
    # --------------------------------------------------------

    if len(domain_analysis) > 0:

        all_domains_failed = (
            domain_analysis["accuracy"]
            == 0
        ).all()

        if all_domains_failed:

            message = (
                "CRITICAL: Every evaluated domain "
                "has 0% accuracy. This strongly "
                "suggests the issue is related to "
                "domain generalization rather than "
                "one isolated domain."
            )

        else:

            message = (
                "INFO: Not every evaluated domain "
                "failed completely. Domain-specific "
                "factors may contribute to the failure."
            )

        print(
            f"{diagnostic_number}. {message}"
        )

        diagnostics.append(
            message
        )

        diagnostic_number += 1

    # --------------------------------------------------------
    # SOURCE SHIFT
    # --------------------------------------------------------

    unique_sources = (
        analysis[
            "analysis_source"
        ].nunique()
    )

    if unique_sources == 1:

        source_name = (
            analysis[
                "analysis_source"
            ].iloc[0]
        )

        message = (
            "INFO: All evaluated samples come "
            f"from a single source "
            f"({source_name}). The result is "
            "consistent with source/domain shift, "
            "but this evaluation alone cannot prove "
            "that source shift is the only cause."
        )

    else:

        message = (
            "INFO: Evaluation contains multiple "
            "sources, allowing comparison of "
            "source-specific generalization."
        )

    print(
        f"{diagnostic_number}. {message}"
    )

    diagnostics.append(
        message
    )

    diagnostic_number += 1

    # ========================================================
    # ENRICHED DATASET
    # ========================================================

    enriched_path = os.path.join(
        OUTPUT_DIR,
        "v3_domain_disjoint_enriched.csv"
    )

    analysis.to_csv(
        enriched_path,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    section(
        "GENERATING FINAL REPORT"
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "v3_domain_disjoint_analysis.txt"
    )

    report_lines = []

    report_lines.append(
        "V3 DOMAIN-DISJOINT FAILURE ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        f"Input: {INPUT_FILE}"
    )

    report_lines.append(
        f"Samples: {total_samples}"
    )

    report_lines.append(
        f"Correct: {correct_count}"
    )

    report_lines.append(
        f"Errors: {error_count}"
    )

    report_lines.append(
        f"Accuracy: {accuracy:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        "AUTOMATIC DIAGNOSTIC"
    )

    report_lines.append(
        "-" * 70
    )

    for number, diagnostic in enumerate(
        diagnostics,
        start=1
    ):

        report_lines.append(
            f"{number}. {diagnostic}"
        )

    report_lines.append("")

    report_lines.append(
        "MISCLASSIFICATION PAIRS"
    )

    report_lines.append(
        "-" * 70
    )

    if len(misclassification_pairs) > 0:

        report_lines.append(
            misclassification_pairs.to_string(
                index=False
            )
        )

    else:

        report_lines.append(
            "No misclassification pairs."
        )

    report_lines.append("")

    report_lines.append(
        "DOMAIN ANALYSIS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        domain_analysis.to_string(
            index=False
        )
    )

    report_lines.append("")

    report_lines.append(
        "PREDICTION BIAS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        prediction_bias.to_string(
            index=False
        )
    )

    report_lines.append("")

    report_lines.append(
        "HIGH-CONFIDENCE ERRORS"
    )

    report_lines.append(
        "-" * 70
    )

    if len(confidence_valid) > 0:

        report_lines.append(
            "Errors with confidence >= 90%: "
            f"{len(high_confidence_errors)}"
        )

        report_lines.append("")

        if len(high_confidence_pairs) > 0:

            report_lines.append(
                high_confidence_pairs.to_string(
                    index=False
                )
            )

        else:

            report_lines.append(
                "No high-confidence errors."
            )

    else:

        report_lines.append(
            "Confidence data unavailable."
        )

    report_lines.append("")

    report_lines.append(
        "OUTPUT DIRECTORY"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        OUTPUT_DIR
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report_lines)
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    section(
        "V3 DOMAIN-DISJOINT FAILURE ANALYSIS COMPLETED"
    )

    print(
        f"Samples : {total_samples}"
    )

    print(
        f"Correct : {correct_count}"
    )

    print(
        f"Errors  : {error_count}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print()

    print(
        "Important output files:"
    )

    print(
        "  v3_domain_disjoint_analysis.txt"
    )

    print(
        "  actual_class_distribution.csv"
    )

    print(
        "  predicted_class_distribution.csv"
    )

    print(
        "  analysis_confusion_matrix.csv"
    )

    print(
        "  misclassification_pairs.csv"
    )

    print(
        "  class_error_analysis.csv"
    )

    print(
        "  confidence_distribution.csv"
    )

    print(
        "  domain_analysis.csv"
    )

    print(
        "  source_analysis.csv"
    )

    print(
        "  date_analysis.csv"
    )

    print(
        "  prediction_bias.csv"
    )

    print(
        "  actual_vs_predicted_percentage.csv"
    )

    print(
        "  high_confidence_errors.csv"
    )

    print(
        "  all_errors.csv"
    )

    print(
        "  top_error_patterns.csv"
    )

    print(
        "  v3_domain_disjoint_enriched.csv"
    )

    print()

    print(
        "Report saved to:"
    )

    print(
        report_path
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