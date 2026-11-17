from pathlib import Path
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CIRCULENS CLEAN QUALITY VISUAL AUDIT
# ============================================================

MANIFEST = Path("dataset/circulens_quality_clean/quality_clean_manifest.csv")
OUTPUT = Path("reports/quality_clean_visual_audit")

IMAGE_SIZE = 180
GRID_COLS = 5
SAMPLES_PER_CLASS = 20
SEED = 42

OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

print("=" * 90)
print("CIRCULENS CLEAN QUALITY VISUAL AUDIT")
print("=" * 90)

# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING CLEAN MANIFEST")

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Support either new_split or split naming
if "new_split" in df.columns:
    SPLIT_COL = "new_split"
elif "split" in df.columns:
    SPLIT_COL = "split"
else:
    raise ValueError("No split column found.")

print("\nSplit distribution:")
print(df[SPLIT_COL].value_counts())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nClass × Split:")
print(
    pd.crosstab(
        df["label"],
        df[SPLIT_COL]
    )
)

# ============================================================
# 2. VERIFY PATHS
# ============================================================

print("\n2. VERIFYING IMAGE PATHS")

missing = []

for path in df["path"]:
    if not Path(path).exists():
        missing.append(path)

print(f"Missing images: {len(missing)}")

if missing:
    print("WARNING: Missing images detected.")
    for path in missing[:20]:
        print(path)
else:
    print("PASS: All image paths exist.")

# ============================================================
# 3. CLASS SAMPLE GRIDS
# ============================================================

print("\n3. CREATING CLASS SAMPLE GRIDS")

classes = sorted(df["label"].unique())

for label in classes:

    subset = df[df["label"] == label]

    sample_n = min(SAMPLES_PER_CLASS, len(subset))

    samples = subset.sample(
        n=sample_n,
        random_state=SEED
    )

    rows = (sample_n + GRID_COLS - 1) // GRID_COLS

    grid = Image.new(
        "RGB",
        (
            GRID_COLS * IMAGE_SIZE,
            rows * IMAGE_SIZE
        ),
        "white"
    )

    draw = ImageDraw.Draw(grid)

    for i, (_, row) in enumerate(samples.iterrows()):

        try:
            img = Image.open(row["path"]).convert("RGB")
            img.thumbnail((IMAGE_SIZE - 10, IMAGE_SIZE - 10))

            x = (i % GRID_COLS) * IMAGE_SIZE
            y = (i // GRID_COLS) * IMAGE_SIZE

            px = x + (IMAGE_SIZE - img.width) // 2
            py = y + (IMAGE_SIZE - img.height) // 2

            grid.paste(img, (px, py))

        except Exception as e:
            print(f"Failed: {row['path']} -> {e}")

    output_path = OUTPUT / f"class_{label}.png"

    grid.save(output_path)

    print(f"Saved: {output_path}")

# ============================================================
# 4. SPLIT SAMPLE GRIDS
# ============================================================

print("\n4. CREATING SPLIT SAMPLE GRIDS")

for split in sorted(df[SPLIT_COL].unique()):

    subset = df[df[SPLIT_COL] == split]

    sample_n = min(SAMPLES_PER_CLASS, len(subset))

    samples = subset.sample(
        n=sample_n,
        random_state=SEED
    )

    rows = (sample_n + GRID_COLS - 1) // GRID_COLS

    grid = Image.new(
        "RGB",
        (
            GRID_COLS * IMAGE_SIZE,
            rows * IMAGE_SIZE
        ),
        "white"
    )

    for i, (_, row) in enumerate(samples.iterrows()):

        try:
            img = Image.open(row["path"]).convert("RGB")
            img.thumbnail((IMAGE_SIZE - 10, IMAGE_SIZE - 10))

            x = (i % GRID_COLS) * IMAGE_SIZE
            y = (i // GRID_COLS) * IMAGE_SIZE

            px = x + (IMAGE_SIZE - img.width) // 2
            py = y + (IMAGE_SIZE - img.height) // 2

            grid.paste(img, (px, py))

        except Exception as e:
            print(f"Failed: {row['path']} -> {e}")

    output_path = OUTPUT / f"split_{split}.png"

    grid.save(output_path)

    print(f"Saved: {output_path}")

# ============================================================
# 5. SOURCE × SPLIT
# ============================================================

print("\n5. SOURCE × SPLIT")

if "source" in df.columns:

    source_split = pd.crosstab(
        df["source"],
        df[SPLIT_COL]
    )

    print(source_split)

    source_split.to_csv(
        OUTPUT / "source_split.csv"
    )

# ============================================================
# 6. SOURCE × CLASS × SPLIT
# ============================================================

print("\n6. SOURCE × CLASS × SPLIT")

if "source" in df.columns:

    source_class_split = (
        df.groupby(
            ["source", "label", SPLIT_COL]
        )
        .size()
        .reset_index(name="count")
    )

    print(source_class_split)

    source_class_split.to_csv(
        OUTPUT / "source_class_split.csv",
        index=False
    )

# ============================================================
# 7. SAVE SUMMARY
# ============================================================

summary = []

for label in classes:

    for split in sorted(df[SPLIT_COL].unique()):

        count = len(
            df[
                (df["label"] == label)
                &
                (df[SPLIT_COL] == split)
            ]
        )

        summary.append({
            "label": label,
            "split": split,
            "count": count
        })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT / "clean_visual_audit_summary.csv",
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 90)
print("CLEAN QUALITY VISUAL AUDIT COMPLETED")
print("=" * 90)

print("\nOutput directory:")
print(OUTPUT)

print("\nIMPORTANT FILES:")

for path in sorted(OUTPUT.iterdir()):
    print(f"  {path.name}")

print("\nNEXT STEP:")
print("Open the generated class_*.png files and split_*.png files.")
print("Check whether the classes remain visually distinguishable.")