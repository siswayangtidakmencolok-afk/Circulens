import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# CIRCULENS QUALITY VISUAL AUDIT
# ============================================================

MANIFEST = "reports/final_dataset/quality_manifest.csv"
OUTPUT_DIR = "reports/quality_visual_audit"

IMAGE_PER_CLASS = 12
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)

print("=" * 90)
print("CIRCULENS QUALITY VISUAL AUDIT")
print("=" * 90)

# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n1. LOADING MANIFEST")

df = pd.read_csv(MANIFEST)

print(f"Total images: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())

# ============================================================
# 2. VERIFY PATHS
# ============================================================

print("\n2. VERIFYING IMAGE PATHS")

missing = []

for path in df["path"]:
    if not os.path.exists(path):
        missing.append(path)

print(f"Missing images: {len(missing)}")

if missing:
    print("\nFirst missing paths:")
    for p in missing[:20]:
        print(p)

# ============================================================
# 3. SAMPLE IMAGES PER CLASS
# ============================================================

print("\n3. CREATING CLASS SAMPLE GRID")

classes = sorted(df["label"].unique())

for label in classes:

    subset = df[df["label"] == label]

    n = min(IMAGE_PER_CLASS, len(subset))

    samples = subset.sample(
        n=n,
        random_state=SEED
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    fig.suptitle(
        f"CircuLens Quality Audit — {label.upper()}",
        fontsize=16
    )

    for i, ax in enumerate(axes):

        ax.axis("off")

        if i >= len(samples):
            continue

        row = samples.iloc[i]

        try:

            image = Image.open(row["path"]).convert("RGB")

            ax.imshow(image)

            filename = os.path.basename(row["path"])

            ax.set_title(
                filename[:30],
                fontsize=8
            )

        except Exception as e:

            ax.text(
                0.5,
                0.5,
                f"ERROR\n{e}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"class_{label}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

# ============================================================
# 4. TRAIN / VAL / TEST VISUAL AUDIT
# ============================================================

print("\n4. CREATING SPLIT VISUALIZATION")

for split in ["train", "val", "test"]:

    split_df = df[df["split"] == split]

    print(
        f"{split}: {len(split_df)} images"
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    fig.suptitle(
        f"CircuLens Quality — {split.upper()}",
        fontsize=16
    )

    samples = split_df.sample(
        n=min(12, len(split_df)),
        random_state=SEED
    )

    for i, ax in enumerate(axes):

        ax.axis("off")

        if i >= len(samples):
            continue

        row = samples.iloc[i]

        try:

            image = Image.open(
                row["path"]
            ).convert("RGB")

            ax.imshow(image)

            ax.set_title(
                f"{row['label']}\n{os.path.basename(row['path'])[:25]}",
                fontsize=8
            )

        except Exception as e:

            ax.text(
                0.5,
                0.5,
                f"ERROR\n{e}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"split_{split}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

# ============================================================
# 5. CLASS × COMMODITY / SOURCE INSPECTION
# ============================================================

print("\n5. INSPECTING FILENAMES")

filename_keywords = [
    "chile",
    "pepper",
    "tomato",
    "bell",
    "new mexico"
]

for keyword in filename_keywords:

    mask = df["filename"].str.lower().str.contains(
        keyword,
        regex=False
    )

    count = mask.sum()

    if count > 0:

        print(
            f"{keyword:15s}: {count:5d}"
        )

# ============================================================
# 6. CLASS × FILENAME PATTERN
# ============================================================

print("\n6. CLASS × FILENAME PATTERN")

for label in classes:

    subset = df[df["label"] == label]

    print("\n" + "-" * 70)
    print(f"CLASS: {label}")
    print("-" * 70)

    for filename in subset["filename"].head(10):

        print(filename)

# ============================================================
# 7. SAVE AUDIT SUMMARY
# ============================================================

summary = []

for label in classes:

    subset = df[df["label"] == label]

    summary.append({
        "label": label,
        "count": len(subset),
        "train": len(
            subset[subset["split"] == "train"]
        ),
        "val": len(
            subset[subset["split"] == "val"]
        ),
        "test": len(
            subset[subset["split"] == "test"]
        )
    })

summary_df = pd.DataFrame(summary)

summary_path = os.path.join(
    OUTPUT_DIR,
    "quality_visual_audit_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved: {summary_path}"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 90)
print("QUALITY VISUAL AUDIT COMPLETED")
print("=" * 90)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nIMPORTANT FILES:")

for filename in sorted(os.listdir(OUTPUT_DIR)):

    print(
        f"  {filename}"
    )

print("\nNext step:")
print(
    "Open the generated class_*.png files and inspect "
    "whether the five quality classes are visually distinguishable."
)