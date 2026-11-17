import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

PREDICTION_FILE = (
    "reports/v3_domain_disjoint/"
    "v3_domain_disjoint_predictions.csv"
)

OUTPUT_DIR = (
    "reports/v3_domain_disjoint_visual/"
    "failure_images"
)

RIPE_TO_OLD_DIR = os.path.join(
    OUTPUT_DIR,
    "ripe_to_old"
)

OLD_TO_DAMAGED_DIR = os.path.join(
    OUTPUT_DIR,
    "old_to_damaged"
)

CONTACT_SHEET_DIR = (
    "reports/v3_domain_disjoint_visual"
)


# ============================================================
# DISPLAY CONFIG
# ============================================================

THUMBNAIL_SIZE = (220, 180)

COLUMNS = 4

DPI = 120

BACKGROUND_COLOR = "white"

TEXT_COLOR = "black"

BORDER_WIDTH = 2

MAX_FILENAME_LENGTH = 28


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


def safe_text(value):

    if pd.isna(value):
        return "N/A"

    return str(value)


def format_confidence(value):

    try:

        value = float(value)

        if value <= 1:

            return f"{value * 100:.2f}%"

        return f"{value:.2f}%"

    except Exception:

        return "N/A"


def shorten_filename(filename):

    filename = safe_text(filename)

    if len(filename) <= MAX_FILENAME_LENGTH:
        return filename

    return (
        filename[:MAX_FILENAME_LENGTH - 3]
        + "..."
    )


def get_font(size=14):

    possible_fonts = [

        "C:/Windows/Fonts/arial.ttf",

        "C:/Windows/Fonts/segoeui.ttf",

        "C:/Windows/Fonts/calibri.ttf",

    ]

    for font_path in possible_fonts:

        if os.path.exists(font_path):

            try:

                return ImageFont.truetype(
                    font_path,
                    size
                )

            except Exception:
                pass

    return ImageFont.load_default()


def resolve_image_path(path):

    if pd.isna(path):

        return None

    path = str(path)

    # --------------------------------------------------------
    # Direct path
    # --------------------------------------------------------

    if os.path.exists(path):

        return path

    # --------------------------------------------------------
    # Normalize separators
    # --------------------------------------------------------

    normalized = path.replace(
        "\\",
        os.sep
    )

    if os.path.exists(normalized):

        return normalized

    # --------------------------------------------------------
    # Try relative to current project
    # --------------------------------------------------------

    project_relative = os.path.abspath(
        normalized
    )

    if os.path.exists(project_relative):

        return project_relative

    # --------------------------------------------------------
    # Try removing leading ./ 
    # --------------------------------------------------------

    if normalized.startswith(
        "." + os.sep
    ):

        stripped = normalized[2:]

        if os.path.exists(stripped):

            return stripped

    return None


def load_image(path):

    resolved = resolve_image_path(
        path
    )

    if resolved is None:

        return None, None

    try:

        image = Image.open(
            resolved
        ).convert("RGB")

        return image, resolved

    except Exception:

        return None, resolved


def create_image_card(
    image,
    metadata,
    card_width=360,
    card_height=310
):

    card = Image.new(
        "RGB",
        (
            card_width,
            card_height
        ),
        BACKGROUND_COLOR
    )

    draw = ImageDraw.Draw(
        card
    )

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    title_font = get_font(16)

    text_font = get_font(13)

    small_font = get_font(11)

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_area_width = card_width - 20

    image_area_height = 205

    if image is not None:

        display_image = ImageOps.contain(
            image,
            (
                image_area_width,
                image_area_height
            )
        )

        x = (
            card_width -
            display_image.width
        ) // 2

        y = 8

        card.paste(
            display_image,
            (
                x,
                y
            )
        )

        # Border

        draw.rectangle(
            [
                x - 1,
                y - 1,
                x + display_image.width,
                y + display_image.height
            ],
            outline=TEXT_COLOR,
            width=BORDER_WIDTH
        )

    else:

        draw.rectangle(
            [
                10,
                10,
                card_width - 10,
                image_area_height
            ],
            outline=TEXT_COLOR,
            width=BORDER_WIDTH
        )

        draw.text(
            (
                card_width // 2,
                image_area_height // 2
            ),
            "IMAGE NOT FOUND",
            fill=TEXT_COLOR,
            font=title_font,
            anchor="mm"
        )

    # --------------------------------------------------------
    # Metadata area
    # --------------------------------------------------------

    y = 218

    actual = metadata["actual"]

    prediction = metadata["prediction"]

    confidence = metadata["confidence"]

    source = metadata["source"]

    date = metadata["date"]

    filename = metadata["filename"]

    draw.text(
        (
            10,
            y
        ),
        f"Actual      : {actual}",
        fill=TEXT_COLOR,
        font=title_font
    )

    y += 22

    draw.text(
        (
            10,
            y
        ),
        f"Prediction  : {prediction}",
        fill=TEXT_COLOR,
        font=text_font
    )

    y += 20

    draw.text(
        (
            10,
            y
        ),
        f"Confidence  : {confidence}",
        fill=TEXT_COLOR,
        font=text_font
    )

    y += 19

    draw.text(
        (
            10,
            y
        ),
        f"Source      : {source}",
        fill=TEXT_COLOR,
        font=small_font
    )

    y += 17

    draw.text(
        (
            10,
            y
        ),
        f"Date        : {date}",
        fill=TEXT_COLOR,
        font=small_font
    )

    y += 17

    draw.text(
        (
            10,
            y
        ),
        f"File        : {shorten_filename(filename)}",
        fill=TEXT_COLOR,
        font=small_font
    )

    return card


def create_contact_sheet(
    dataframe,
    output_path,
    title
):

    if len(dataframe) == 0:

        print(
            f"No samples available for: {title}"
        )

        return

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    rows = math.ceil(
        len(dataframe) /
        COLUMNS
    )

    card_width = 360

    card_height = 310

    title_height = 60

    sheet_width = (
        COLUMNS *
        card_width
    )

    sheet_height = (
        title_height +
        rows *
        card_height
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        BACKGROUND_COLOR
    )

    draw = ImageDraw.Draw(
        sheet
    )

    title_font = get_font(22)

    subtitle_font = get_font(14)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    draw.text(
        (
            20,
            10
        ),
        title,
        fill=TEXT_COLOR,
        font=title_font
    )

    draw.text(
        (
            20,
            37
        ),
        f"Samples: {len(dataframe)}",
        fill=TEXT_COLOR,
        font=subtitle_font
    )

    # --------------------------------------------------------
    # Cards
    # --------------------------------------------------------

    for index, (_, row) in enumerate(
        dataframe.iterrows()
    ):

        row_index = (
            index //
            COLUMNS
        )

        column_index = (
            index %
            COLUMNS
        )

        x = (
            column_index *
            card_width
        )

        y = (
            title_height +
            row_index *
            card_height
        )

        image, resolved_path = load_image(
            row["path"]
        )

        metadata = {

            "actual": safe_text(
                row["actual"]
            ),

            "prediction": safe_text(
                row["prediction"]
            ),

            "confidence": format_confidence(
                row["confidence"]
            ),

            "source": safe_text(
                row["source"]
            ),

            "date": safe_text(
                row["date"]
            ),

            "filename": safe_text(
                row["filename"]
            )

        }

        card = create_image_card(
            image,
            metadata,
            card_width,
            card_height
        )

        sheet.paste(
            card,
            (
                x,
                y
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    sheet.save(
        output_path,
        quality=95
    )

    print(
        f"Saved contact sheet: {output_path}"
    )


def save_individual_images(
    dataframe,
    output_directory
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    saved = 0

    missing = 0

    for index, (_, row) in enumerate(
        dataframe.iterrows()
    ):

        image, resolved_path = load_image(
            row["path"]
        )

        if image is None:

            missing += 1

            continue

        filename = safe_text(
            row["filename"]
        )

        # ----------------------------------------------------
        # Sanitize filename
        # ----------------------------------------------------

        safe_filename = (
            filename
            .replace(
                "\\",
                "_"
            )
            .replace(
                "/",
                "_"
            )
            .replace(
                ":",
                "_"
            )
            .replace(
                "*",
                "_"
            )
            .replace(
                "?",
                "_"
            )
            .replace(
                '"',
                "_"
            )
            .replace(
                "<",
                "_"
            )
            .replace(
                ">",
                "_"
            )
            .replace(
                "|",
                "_"
            )
        )

        output_filename = (
            f"{index + 1:03d}_"
            f"{safe_filename}"
        )

        output_path = os.path.join(
            output_directory,
            output_filename
        )

        # ----------------------------------------------------
        # Save original image
        # ----------------------------------------------------

        image.save(
            output_path,
            quality=95
        )

        saved += 1

    return saved, missing


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    section(
        "V3 DOMAIN-DISJOINT IMAGE FAILURE INSPECTION"
    )

    print(
        f"Prediction file:\n{PREDICTION_FILE}"
    )

    print()

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        RIPE_TO_OLD_DIR,
        exist_ok=True
    )

    os.makedirs(
        OLD_TO_DAMAGED_DIR,
        exist_ok=True
    )

    os.makedirs(
        CONTACT_SHEET_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    section(
        "LOADING PREDICTIONS"
    )

    if not os.path.exists(
        PREDICTION_FILE
    ):

        raise FileNotFoundError(
            "Prediction file not found:\n"
            f"{PREDICTION_FILE}"
        )

    df = pd.read_csv(
        PREDICTION_FILE
    )

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
            "prediction_label",
            "label",
            "actual",
            "actual_label"
        ]
    )

    prediction_col = find_column(
        df,
        [
            "prediction",
            "predicted",
            "predicted_label"
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

    filename_col = find_column(
        df,
        [
            "filename",
            "file_name",
            "image_name"
        ]
    )

    correct_col = find_column(
        df,
        [
            "correct",
            "is_correct"
        ]
    )

    target_domain_col = find_column(
        df,
        [
            "target_domain"
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

    if path_col is None:

        raise ValueError(
            "Image path column not found."
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

    print(
        f"Filename column   : {filename_col}"
    )

    print(
        f"Correct column    : {correct_col}"
    )

    print(
        f"Target domain     : {target_domain_col}"
    )

    # ========================================================
    # PREPARE FAILURE DATA
    # ========================================================

    section(
        "PREPARING FAILURE DATA"
    )

    analysis = df.copy()

    analysis["actual"] = (
        analysis[actual_col]
    )

    analysis["prediction"] = (
        analysis[prediction_col]
    )

    if confidence_col is not None:

        analysis["confidence"] = (
            pd.to_numeric(
                analysis[
                    confidence_col
                ],
                errors="coerce"
            )
        )

    else:

        analysis["confidence"] = float(
            "nan"
        )

    if source_col is not None:

        analysis["source"] = (
            analysis[source_col]
        )

    else:

        analysis["source"] = "N/A"

    if date_col is not None:

        analysis["date"] = (
            analysis[date_col]
        )

    else:

        analysis["date"] = "N/A"

    if filename_col is not None:

        analysis["filename"] = (
            analysis[filename_col]
        )

    else:

        analysis["filename"] = (
            analysis[path_col]
            .astype(str)
            .apply(os.path.basename)
        )

    analysis["path"] = (
        analysis[path_col]
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    if correct_col is not None:

        correct_values = (
            analysis[correct_col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        analysis["is_error"] = ~(
            correct_values.isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        )

    else:

        # Important:
        # prediction may be numeric while actual
        # may be class names. Therefore first compare
        # string representations.

        analysis["is_error"] = (

            analysis["actual"]
            .astype(str)
            .str.strip()
            !=
            analysis["prediction"]
            .astype(str)
            .str.strip()

        )

    errors = analysis[
        analysis["is_error"]
    ].copy()

    print(
        f"Total samples : {len(analysis)}"
    )

    print(
        f"Error samples : {len(errors)}"
    )

    # ========================================================
    # NORMALIZE KNOWN CLASS NAMES
    # ========================================================

    section(
        "IDENTIFYING FAILURE PATTERNS"
    )

    # The current domain-disjoint evaluation has:
    #
    # actual label 3 = ripe
    # actual label 2 = old
    #
    # prediction:
    # old
    # damaged
    #
    # We use prediction_label if available because
    # it contains the human-readable class name.

    if "prediction_label" in errors.columns:

        errors["prediction_display"] = (
            errors["prediction_label"]
        )

    else:

        errors["prediction_display"] = (
            errors["prediction"]
        )

    # --------------------------------------------------------
    # Actual class display
    # --------------------------------------------------------

    if "class" in errors.columns:

        # In this dataset "class" is the same numeric
        # class representation used by the evaluation.

        errors["actual_display"] = (
            errors["class"]
        )

    else:

        errors["actual_display"] = (
            errors["actual"]
        )

    # --------------------------------------------------------
    # Map known numeric labels
    # --------------------------------------------------------

    CLASS_MAP = {

        0: "damaged",

        1: "dried",

        2: "old",

        3: "ripe",

        4: "unripe"

    }

    def display_actual(value):

        try:

            numeric = int(
                float(value)
            )

            if numeric in CLASS_MAP:

                return CLASS_MAP[
                    numeric
                ]

        except Exception:
            pass

        return safe_text(
            value
        )

    errors["actual_display"] = (
        errors["actual"]
        .apply(display_actual)
    )

    errors["prediction_display"] = (
        errors["prediction_display"]
        .apply(
            lambda value:
            safe_text(value)
        )
    )

    # ========================================================
    # PRINT FAILURE PATTERNS
    # ========================================================

    failure_patterns = (
        errors
        .groupby(
            [
                "actual_display",
                "prediction_display"
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
        failure_patterns.to_string(
            index=False
        )
    )

    # ========================================================
    # SEPARATE FAILURE GROUPS
    # ========================================================

    ripe_to_old = errors[
        (
            errors["actual_display"]
            .str.lower()
            == "ripe"
        )
        &
        (
            errors["prediction_display"]
            .str.lower()
            == "old"
        )
    ].copy()

    old_to_damaged = errors[
        (
            errors["actual_display"]
            .str.lower()
            == "old"
        )
        &
        (
            errors["prediction_display"]
            .str.lower()
            == "damaged"
        )
    ].copy()

    print()

    print(
        f"Ripe -> old      : "
        f"{len(ripe_to_old)}"
    )

    print(
        f"Old -> damaged   : "
        f"{len(old_to_damaged)}"
    )

    # ========================================================
    # SAVE FAILURE TABLE
    # ========================================================

    section(
        "SAVING FAILURE TABLES"
    )

    failure_table_columns = [

        "path",

        "filename",

        "source",

        "date",

        "actual",

        "actual_display",

        "prediction",

        "prediction_display",

        "confidence",

        "target_domain",

        "is_error"

    ]

    failure_table_columns = [

        column

        for column in failure_table_columns

        if column in errors.columns

    ]

    failure_table = errors[
        failure_table_columns
    ].copy()

    failure_table_path = os.path.join(
        OUTPUT_DIR,
        "all_failure_samples.csv"
    )

    failure_table.to_csv(
        failure_table_path,
        index=False
    )

    print(
        f"Saved: {failure_table_path}"
    )

    # ========================================================
    # SAVE INDIVIDUAL ORIGINAL IMAGES
    # ========================================================

    section(
        "SAVING INDIVIDUAL FAILURE IMAGES"
    )

    saved_ripe, missing_ripe = (
        save_individual_images(
            ripe_to_old,
            RIPE_TO_OLD_DIR
        )
    )

    print(
        f"Ripe -> old images saved : "
        f"{saved_ripe}"
    )

    print(
        f"Ripe -> old missing      : "
        f"{missing_ripe}"
    )

    saved_old, missing_old = (
        save_individual_images(
            old_to_damaged,
            OLD_TO_DAMAGED_DIR
        )
    )

    print(
        f"Old -> damaged images saved : "
        f"{saved_old}"
    )

    print(
        f"Old -> damaged missing      : "
        f"{missing_old}"
    )

    # ========================================================
    # CREATE RIPE -> OLD CONTACT SHEET
    # ========================================================

    section(
        "CREATING RIPE -> OLD CONTACT SHEET"
    )

    ripe_contact_sheet = os.path.join(
        CONTACT_SHEET_DIR,
        "ripe_to_old_contact_sheet.png"
    )

    create_contact_sheet(
        ripe_to_old,
        ripe_contact_sheet,
        "V3 DOMAIN-DISJOINT FAILURE: RIPE -> OLD"
    )

    # ========================================================
    # CREATE OLD -> DAMAGED CONTACT SHEET
    # ========================================================

    section(
        "CREATING OLD -> DAMAGED CONTACT SHEET"
    )

    old_contact_sheet = os.path.join(
        CONTACT_SHEET_DIR,
        "old_to_damaged_contact_sheet.png"
    )

    create_contact_sheet(
        old_to_damaged,
        old_contact_sheet,
        "V3 DOMAIN-DISJOINT FAILURE: OLD -> DAMAGED"
    )

    # ========================================================
    # CREATE ALL FAILURE CONTACT SHEET
    # ========================================================

    section(
        "CREATING ALL FAILURE CONTACT SHEET"
    )

    all_contact_sheet = os.path.join(
        CONTACT_SHEET_DIR,
        "all_domain_disjoint_failures.png"
    )

    create_contact_sheet(
        errors,
        all_contact_sheet,
        "V3 DOMAIN-DISJOINT: ALL FAILURE SAMPLES"
    )

    # ========================================================
    # SAVE FAILURE METADATA FOR MANUAL REVIEW
    # ========================================================

    section(
        "CREATING MANUAL REVIEW TABLE"
    )

    review_columns = [

        "filename",

        "path",

        "source",

        "date",

        "actual_display",

        "prediction_display",

        "confidence"

    ]

    review_columns = [

        column

        for column in review_columns

        if column in errors.columns

    ]

    manual_review = errors[
        review_columns
    ].copy()

    manual_review.insert(
        0,
        "review_id",
        range(
            1,
            len(manual_review) + 1
        )
    )

    manual_review[
        "human_visual_assessment"
    ] = ""

    manual_review[
        "suspected_cause"
    ] = ""

    manual_review[
        "notes"
    ] = ""

    manual_review_path = os.path.join(
        OUTPUT_DIR,
        "manual_visual_review.csv"
    )

    manual_review.to_csv(
        manual_review_path,
        index=False
    )

    print(
        f"Saved manual review table:"
    )

    print(
        manual_review_path
    )

    # ========================================================
    # FAILURE SUMMARY
    # ========================================================

    section(
        "FAILURE SUMMARY"
    )

    total_errors = len(
        errors
    )

    print(
        f"Total failure samples : "
        f"{total_errors}"
    )

    if total_errors > 0:

        print()

        for _, row in failure_patterns.iterrows():

            actual = row[
                "actual_display"
            ]

            prediction = row[
                "prediction_display"
            ]

            count = int(
                row["count"]
            )

            percentage = (
                count /
                total_errors *
                100
            )

            print(
                f"{actual} -> "
                f"{prediction}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

    # ========================================================
    # REPORT
    # ========================================================

    section(
        "GENERATING IMAGE INSPECTION REPORT"
    )

    report_path = os.path.join(
        CONTACT_SHEET_DIR,
        "v3_domain_disjoint_image_inspection_report.txt"
    )

    report_lines = []

    report_lines.append(
        "V3 DOMAIN-DISJOINT IMAGE FAILURE INSPECTION"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        f"Prediction file: {PREDICTION_FILE}"
    )

    report_lines.append(
        f"Total samples: {len(analysis)}"
    )

    report_lines.append(
        f"Failure samples: {len(errors)}"
    )

    report_lines.append("")

    report_lines.append(
        "FAILURE PATTERNS"
    )

    report_lines.append(
        "-" * 70
    )

    for _, row in failure_patterns.iterrows():

        actual = row[
            "actual_display"
        ]

        prediction = row[
            "prediction_display"
        ]

        count = int(
            row["count"]
        )

        percentage = (
            count /
            max(total_errors, 1)
            *
            100
        )

        report_lines.append(
            f"{actual} -> "
            f"{prediction}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    report_lines.append("")

    report_lines.append(
        "IMAGE FILES"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        f"Ripe -> old directory: "
        f"{RIPE_TO_OLD_DIR}"
    )

    report_lines.append(
        f"Old -> damaged directory: "
        f"{OLD_TO_DAMAGED_DIR}"
    )

    report_lines.append("")

    report_lines.append(
        "CONTACT SHEETS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        ripe_contact_sheet
    )

    report_lines.append(
        old_contact_sheet
    )

    report_lines.append(
        all_contact_sheet
    )

    report_lines.append("")

    report_lines.append(
        "MANUAL REVIEW"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        manual_review_path
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        "This analysis is intended for manual "
        "visual inspection of domain-disjoint "
        "failure samples."
    )

    report_lines.append(
        "Do not interpret the failure pattern "
        "as proof of a specific root cause "
        "without visual inspection."
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

    print(
        f"Report saved to:\n{report_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "V3 DOMAIN-DISJOINT IMAGE INSPECTION COMPLETED"
    )

    print(
        f"Total samples : "
        f"{len(analysis)}"
    )

    print(
        f"Failures      : "
        f"{len(errors)}"
    )

    print()

    print(
        "Failure groups:"
    )

    print(
        f"  ripe -> old       : "
        f"{len(ripe_to_old)}"
    )

    print(
        f"  old -> damaged    : "
        f"{len(old_to_damaged)}"
    )

    print()

    print(
        "Generated files:"
    )

    print(
        f"  {ripe_contact_sheet}"
    )

    print(
        f"  {old_contact_sheet}"
    )

    print(
        f"  {all_contact_sheet}"
    )

    print(
        f"  {failure_table_path}"
    )

    print(
        f"  {manual_review_path}"
    )

    print(
        f"  {report_path}"
    )

    print()

    print(
        "=" * 100
    )

if __name__ == "__main__":

    main()