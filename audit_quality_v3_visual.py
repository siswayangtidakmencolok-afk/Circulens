from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import math

MANIFEST = Path(
    "dataset/circulens_quality_clean_v3/quality_clean_v3_manifest.csv"
)

OUTPUT = Path(
    "reports/quality_clean_v3_visual_audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CIRCULENS QUALITY V3 VISUAL AUDIT")
print("=" * 90)


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING V3 MANIFEST")

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")
print(f"Columns: {list(df.columns)}")

# V3 manifest menggunakan kolom "split"
SPLIT_COL = "split"

if SPLIT_COL not in df.columns:
    raise ValueError(
        f"Column '{SPLIT_COL}' not found in manifest. "
        f"Available columns: {list(df.columns)}"
    )

print("\nSplit distribution:")
print(df[SPLIT_COL].value_counts())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSource distribution:")
print(df["source"].value_counts())


# ============================================================
# 2. VERIFY IMAGE PATHS
# ============================================================

print("\n2. VERIFYING IMAGE PATHS")

missing = []

for path in df["path"]:
    if not Path(path).exists():
        missing.append(path)

print(f"Missing images: {len(missing)}")

if missing:
    print("WARNING: Missing images detected.")

    print("\nFirst missing images:")
    for path in missing[:20]:
        print(f"  {path}")
else:
    print("PASS: All image paths exist.")


# ============================================================
# 3. SOURCE × CLASS × SPLIT
# ============================================================

print("\n3. SOURCE × CLASS × SPLIT")

matrix = (
    df.groupby(["source", "label", SPLIT_COL])
    .size()
    .reset_index(name="count")
)

print(matrix.to_string(index=False))

matrix.to_csv(
    OUTPUT / "source_class_split.csv",
    index=False
)


# ============================================================
# 4. COMPLETE SOURCE × CLASS × SPLIT MATRIX
# ============================================================

print("\n4. COMPLETE SOURCE × CLASS × SPLIT MATRIX")

sources = sorted(df["source"].unique())
classes = sorted(df["label"].unique())
splits = ["train", "val", "test"]

rows = []

for source in sources:
    for label in classes:
        for split in splits:

            count = len(
                df[
                    (df["source"] == source)
                    & (df["label"] == label)
                    & (df[SPLIT_COL] == split)
                ]
            )

            rows.append(
                {
                    "source": source,
                    "label": label,
                    "split": split,
                    "images": count,
                }
            )

complete_matrix = pd.DataFrame(rows)

print(complete_matrix.to_string(index=False))

complete_matrix.to_csv(
    OUTPUT / "complete_source_class_split_matrix.csv",
    index=False
)


# ============================================================
# 5. ZERO-SPLIT CHECK
# ============================================================

print("\n5. ZERO-SPLIT CHECK")

zero_cells = complete_matrix[
    complete_matrix["images"] == 0
]

if len(zero_cells) == 0:
    print(
        "PASS: Every source × class exists "
        "in train/val/test."
    )
else:
    print(
        f"WARNING: {len(zero_cells)} source × class × split "
        "combinations have zero images."
    )

    print(zero_cells.to_string(index=False))

zero_cells.to_csv(
    OUTPUT / "zero_split_combinations.csv",
    index=False
)


# ============================================================
# 6. CREATE SAMPLE GRID FUNCTION
# ============================================================

def create_grid(
    data,
    output_path,
    samples_per_group=20
):

    if len(data) == 0:
        print(f"SKIP: No images for {output_path}")
        return

    sample_count = min(
        len(data),
        samples_per_group
    )

    data = data.sample(
        sample_count,
        random_state=42
    )

    thumb_w = 220
    thumb_h = 180
    label_h = 60

    cols = 5
    rows = math.ceil(len(data) / cols)

    canvas_w = cols * thumb_w
    canvas_h = rows * (thumb_h + label_h)

    canvas = Image.new(
        "RGB",
        (canvas_w, canvas_h),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for i, (_, row) in enumerate(data.iterrows()):

        x = (i % cols) * thumb_w + 5
        y = (i // cols) * (thumb_h + label_h) + 5

        try:

            img_path = Path(row["path"])

            img = Image.open(img_path).convert("RGB")

            img = ImageOps.fit(
                img,
                (
                    thumb_w - 10,
                    thumb_h - 10
                )
            )

        except Exception as e:

            print(
                f"WARNING: Failed to load image "
                f"{row['path']}: {e}"
            )

            img = Image.new(
                "RGB",
                (
                    thumb_w - 10,
                    thumb_h - 10
                ),
                "gray"
            )

        canvas.paste(
            img,
            (x, y)
        )

        source_short = str(
            row["source"]
        )

        if len(source_short) > 24:
            source_short = source_short[:24]

        text = (
            f"label: {row['label']}\n"
            f"source: {source_short}\n"
            f"split: {row[SPLIT_COL]}"
        )

        draw.multiline_text(
            (x, y + thumb_h),
            text,
            fill="black",
            spacing=3
        )

    canvas.save(output_path)

    print(f"Saved: {output_path}")


# ============================================================
# 7. CLASS SAMPLE GRIDS
# ============================================================

print("\n6. CREATING CLASS SAMPLE GRIDS")

for label in sorted(
    df["label"].unique()
):

    subset = df[
        df["label"] == label
    ]

    output_path = (
        OUTPUT / f"class_{label}.png"
    )

    create_grid(
        subset,
        output_path,
        samples_per_group=25
    )


# ============================================================
# 8. SPLIT SAMPLE GRIDS
# ============================================================

print("\n7. CREATING SPLIT SAMPLE GRIDS")

for split in splits:

    subset = df[
        df[SPLIT_COL] == split
    ]

    output_path = (
        OUTPUT / f"split_{split}.png"
    )

    create_grid(
        subset,
        output_path,
        samples_per_group=25
    )


# ============================================================
# 9. SOURCE SAMPLE GRIDS
# ============================================================

print("\n8. CREATING SOURCE SAMPLE GRIDS")

for source in sorted(
    df["source"].unique()
):

    subset = df[
        df["source"] == source
    ]

    safe_name = (
        str(source)
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_path = (
        OUTPUT / f"source_{safe_name}.png"
    )

    create_grid(
        subset,
        output_path,
        samples_per_group=25
    )


# ============================================================
# 10. CLASS × SPLIT GRIDS
# ============================================================

print("\n9. CREATING CLASS × SPLIT SAMPLE GRIDS")

for label in sorted(
    df["label"].unique()
):

    for split in splits:

        subset = df[
            (df["label"] == label)
            & (df[SPLIT_COL] == split)
        ]

        if len(subset) == 0:
            continue

        safe_label = str(label).replace(
            " ",
            "_"
        )

        output_path = (
            OUTPUT
            / f"class_{safe_label}_{split}.png"
        )

        create_grid(
            subset,
            output_path,
            samples_per_group=15
        )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("V3 VISUAL AUDIT COMPLETED")
print("=" * 90)

print("\nTotal images:")
print(len(df))

print("\nSplit:")
print(df[SPLIT_COL].value_counts())

print("\nClasses:")
print(df["label"].value_counts())

print("\nSources:")
print(df["source"].value_counts())

print("\nZero split cells:")
print(len(zero_cells))

print("\nMissing images:")
print(len(missing))

print("\nOutput directory:")
print(OUTPUT)

print("\nGenerated PNG files:")

for file in sorted(
    OUTPUT.glob("*.png")
):

    print(f"  {file.name}")

print("\nGenerated CSV files:")

for file in sorted(
    OUTPUT.glob("*.csv")
):

    print(f"  {file.name}")

print("\nNEXT STEP:")
print("Open the generated PNG files and visually inspect:")

print("1. class_*.png")
print("2. split_*.png")
print("3. source_*.png")
print("4. class_*_train.png")
print("5. class_*_val.png")
print("6. class_*_test.png")

print("\nFocus on:")
print("- Apakah label terlihat benar?")
print("- Apakah ripe dan unripe terlihat berbeda?")
print("- Apakah old dan dried terlihat masuk akal?")
print("- Apakah damaged memang menunjukkan kerusakan?")
print("- Apakah ada gambar yang sangat berbeda dari gambar lain?")
print("- Apakah background/source terlalu mudah dikenali?")
print("- Apakah train/val/test terlihat berasal dari distribusi visual yang sama?")

print("\nDO NOT TRAIN THE MODEL YET.")