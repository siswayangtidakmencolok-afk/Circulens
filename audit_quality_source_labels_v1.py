import os
import pandas as pd


MANIFEST = r"dataset\circulens_quality_clean_v4\quality_clean_v4_manifest_rebalanced.csv"


print("=" * 90)
print("CIRCULENS QUALITY — SOURCE × LABEL AUDIT")
print("=" * 90)

print(f"\nManifest:")
print(MANIFEST)

df = pd.read_csv(MANIFEST)

print(f"\nRows: {len(df)}")
print(f"Columns: {list(df.columns)}")


print("\n" + "=" * 90)
print("1. SOURCE DISTRIBUTION")
print("=" * 90)

print(df["source"].value_counts())


print("\n" + "=" * 90)
print("2. SOURCE × LABEL")
print("=" * 90)

source_label = pd.crosstab(
    df["source"],
    df["label"]
)

print(source_label)


print("\n" + "=" * 90)
print("3. SOURCE × LABEL (%)")
print("=" * 90)

source_label_pct = pd.crosstab(
    df["source"],
    df["label"],
    normalize="index"
) * 100

print(source_label_pct.round(2))


print("\n" + "=" * 90)
print("4. SPLIT × SOURCE × LABEL")
print("=" * 90)

split_source_label = pd.crosstab(
    [df["source"], df["split"]],
    df["label"]
)

print(split_source_label)


print("\n" + "=" * 90)
print("5. LABEL DISTRIBUTION BY SOURCE")
print("=" * 90)

for source in sorted(df["source"].unique()):

    subset = df[df["source"] == source]

    print(f"\nSOURCE: {source}")
    print(f"Samples: {len(subset)}")

    counts = subset["label"].value_counts()

    for label in sorted(df["label"].unique()):
        count = counts.get(label, 0)
        pct = count / len(subset) * 100

        print(f"  {label:10s}: {count:4d} ({pct:6.2f}%)")


print("\n" + "=" * 90)
print("6. SOURCE × LABEL × SPLIT")
print("=" * 90)

for source in sorted(df["source"].unique()):

    print(f"\nSOURCE: {source}")

    subset = df[df["source"] == source]

    table = pd.crosstab(
        subset["split"],
        subset["label"]
    )

    print(table)


print("\n" + "=" * 90)
print("7. SEQUENCE GROUP AUDIT")
print("=" * 90)

if "sequence_group" in df.columns:

    sequence_summary = (
        df.groupby(["source", "sequence_group"])
        .agg(
            samples=("filename", "count"),
            labels=("label", "nunique")
        )
        .reset_index()
    )

    print("\nSequence groups with multiple labels:")

    multi_label = sequence_summary[
        sequence_summary["labels"] > 1
    ]

    print(multi_label.head(50).to_string(index=False))

    print(
        f"\nTotal sequence groups : {len(sequence_summary)}"
    )

    print(
        f"Multi-label groups    : {len(multi_label)}"
    )


print("\n" + "=" * 90)
print("8. POTENTIAL LABEL CONFLICTS")
print("=" * 90)

if "sequence_group" in df.columns:

    conflicts = (
        df.groupby(["source", "sequence_group"])["label"]
        .nunique()
        .reset_index(name="label_count")
    )

    conflicts = conflicts[
        conflicts["label_count"] > 1
    ]

    print(
        f"Groups containing multiple labels: {len(conflicts)}"
    )

    if len(conflicts) > 0:
        print("\nExamples:")
        print(conflicts.head(30).to_string(index=False))


print("\n" + "=" * 90)
print("9. QUALITY CHECK")
print("=" * 90)

required_columns = [
    "source",
    "split",
    "label",
    "path",
    "filename"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("WARNING: Missing columns:")
    print(missing_columns)
else:
    print("PASS: Required columns exist.")


missing_paths = 0

for path in df["path"]:
    if not os.path.exists(path):
        missing_paths += 1

print(f"Missing image paths: {missing_paths}")


print("\n" + "=" * 90)
print("AUDIT COMPLETED")
print("=" * 90)