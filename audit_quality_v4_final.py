from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

MANIFEST_PATH = Path(
    "dataset/circulens_quality_clean_v4/quality_clean_v4_manifest.csv"
)

OUTPUT_ROOT = Path(
    "reports/quality_baseline_v4"
)

MODEL_DIR = OUTPUT_ROOT / "model"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

NUM_CLASSES = 5

CLASS_NAMES = [
    "damaged",
    "dried",
    "old",
    "ripe",
    "unripe",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("CIRCULENS QUALITY V4 BASELINE TRAINING")
print("=" * 90)

print()
print(f"Manifest   : {MANIFEST_PATH}")
print(f"Output     : {OUTPUT_ROOT}")
print(f"Image size : {IMAGE_SIZE}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Epochs     : {EPOCHS}")
print(f"Seed       : {SEED}")


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

print("\n" + "=" * 90)
print("1. LOADING MANIFEST")
print("=" * 90)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

df = pd.read_csv(MANIFEST_PATH)

print(f"Rows   : {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. VALIDATE MANIFEST
# ============================================================

print("\n" + "=" * 90)
print("2. VALIDATING MANIFEST")
print("=" * 90)

required_columns = [
    "task",
    "split",
    "label",
    "source",
    "sequence_group",
    "timestamp",
    "path",
    "filename",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("PASS: Required columns exist.")


# ============================================================
# 3. VALIDATE LABELS
# ============================================================

print("\n" + "=" * 90)
print("3. VALIDATING LABELS")
print("=" * 90)

unknown_labels = sorted(
    set(df["label"].unique())
    - set(CLASS_NAMES)
)

if unknown_labels:
    raise ValueError(
        f"Unknown labels detected: {unknown_labels}"
    )

print("Classes:")

for index, name in enumerate(CLASS_NAMES):
    print(f"{index}: {name}")


# ============================================================
# 4. VALIDATE IMAGE PATHS
# ============================================================

print("\n" + "=" * 90)
print("4. VALIDATING IMAGE PATHS")
print("=" * 90)

missing_images = []

for path in df["path"]:

    if not Path(path).exists():
        missing_images.append(path)

print(f"Total paths : {len(df)}")
print(f"Missing     : {len(missing_images)}")

if missing_images:

    print("\nExample missing paths:")

    for path in missing_images[:10]:
        print(path)

    raise FileNotFoundError(
        "Some image paths are missing."
    )

print("PASS: All image paths exist.")


# ============================================================
# 5. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("5. DATASET SUMMARY")
print("=" * 90)

print("\nSplit distribution:")

print(
    df["split"]
    .value_counts()
    .reindex(
        ["train", "val", "test"]
    )
)

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .reindex(CLASS_NAMES)
)

print("\nSplit × class:")

split_class = (
    df.groupby(
        ["split", "label"]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reindex(
        columns=CLASS_NAMES
    )
)

print(split_class)


# ============================================================
# 6. SPLIT DATA
# ============================================================

print("\n" + "=" * 90)
print("6. CREATING SPLITS")
print("=" * 90)

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()

print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")


# ============================================================
# 7. LABEL MAPPING
# ============================================================

label_to_index = {
    label: index
    for index, label in enumerate(CLASS_NAMES)
}

index_to_label = {
    index: label
    for label, index in label_to_index.items()
}

train_df["label_id"] = (
    train_df["label"]
    .map(label_to_index)
)

val_df["label_id"] = (
    val_df["label"]
    .map(label_to_index)
)

test_df["label_id"] = (
    test_df["label"]
    .map(label_to_index)
)


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 90)
print("7. COMPUTING CLASS WEIGHTS")
print("=" * 90)

classes_numeric = np.arange(NUM_CLASSES)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes_numeric,
    y=train_df["label_id"].values,
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}

class_weight_table = pd.DataFrame(
    {
        "class_id": classes_numeric,
        "label": CLASS_NAMES,
        "train_samples": [
            int(
                (train_df["label_id"] == index)
                .sum()
            )
            for index in classes_numeric
        ],
        "class_weight": class_weights_array,
    }
)

print(class_weight_table)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

class_weight_table.to_csv(
    OUTPUT_ROOT / "class_weights.csv",
    index=False,
)


# ============================================================
# 9. TF.DATA PIPELINE
# ============================================================

print("\n" + "=" * 90)
print("8. CREATING TF.DATA PIPELINES")
print("=" * 90)


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


def augment_image(image, label):

    image = tf.image.random_flip_left_right(
        image
    )

    image = tf.image.random_brightness(
        image,
        max_delta=0.10
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10
    )

    return image, label


def create_dataset(
    dataframe,
    training=False,
):

    paths = dataframe["path"].astype(str).values

    labels = dataframe["label_id"].astype(
        np.int32
    ).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

test_dataset = create_dataset(
    test_df,
    training=False
)

print(
    "PASS: Data pipelines created."
)


# ============================================================
# 10. BUILD MODEL
# ============================================================

print("\n" + "=" * 90)
print("9. BUILDING BASELINE MODEL")
print("=" * 90)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(
            0.05
        ),
        tf.keras.layers.RandomZoom(
            0.10
        ),
    ],
    name="augmentation",
)


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),
)

# IMPORTANT:
# Baseline = frozen ImageNet backbone.
base_model.trainable = False


inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

x = data_augmentation(
    inputs
)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classifier"
)(x)


model = tf.keras.Model(
    inputs,
    outputs
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ],
)


model.summary()


# ============================================================
# 11. CALLBACKS
# ============================================================

print("\n" + "=" * 90)
print("10. PREPARING CALLBACKS")
print("=" * 90)

best_model_path = (
    MODEL_DIR /
    "best_quality_baseline_v4.keras"
)

final_model_path = (
    MODEL_DIR /
    "quality_baseline_v4_final.keras"
)


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            best_model_path
        ),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ============================================================
# 12. TRAINING
# ============================================================

print("\n" + "=" * 90)
print("11. TRAINING")
print("=" * 90)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# 13. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    np.arange(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    OUTPUT_ROOT /
    "training_history.csv",
    index=False,
)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 90)
print("12. SAVING FINAL MODEL")
print("=" * 90)

model.save(
    final_model_path
)

print(
    f"Saved final model:\n"
    f"{final_model_path}"
)


# ============================================================
# 15. TEST EVALUATION
# ============================================================

print("\n" + "=" * 90)
print("13. TEST EVALUATION")
print("=" * 90)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(
    f"\nTest loss     : {test_loss:.6f}"
)

print(
    f"Test accuracy : {test_accuracy:.6f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 16. PREDICTIONS
# ============================================================

print("\n" + "=" * 90)
print("14. GENERATING TEST PREDICTIONS")
print("=" * 90)

probabilities = model.predict(
    test_dataset,
    verbose=1
)

predicted_ids = np.argmax(
    probabilities,
    axis=1
)

true_ids = test_df[
    "label_id"
].values

confidence = np.max(
    probabilities,
    axis=1
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 90)
print("15. CLASSIFICATION REPORT")
print("=" * 90)

report_dict = classification_report(
    true_ids,
    predicted_ids,
    labels=classes_numeric,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    true_ids,
    predicted_ids,
    labels=classes_numeric,
    target_names=CLASS_NAMES,
    zero_division=0,
)

print(report_text)

with open(
    OUTPUT_ROOT /
    "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report_text)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 90)
print("16. CONFUSION MATRIX")
print("=" * 90)

cm = confusion_matrix(
    true_ids,
    predicted_ids,
    labels=classes_numeric,
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

print(cm_df)

cm_df.to_csv(
    OUTPUT_ROOT /
    "confusion_matrix.csv"
)


# ============================================================
# 19. BALANCED METRICS
# ============================================================

print("\n" + "=" * 90)
print("17. BALANCED METRICS")
print("=" * 90)

balanced_accuracy = balanced_accuracy_score(
    true_ids,
    predicted_ids
)

macro_f1 = f1_score(
    true_ids,
    predicted_ids,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    true_ids,
    predicted_ids,
    average="weighted",
    zero_division=0
)

print(
    f"Balanced accuracy : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1       : "
    f"{weighted_f1:.4f}"
)


# ============================================================
# 20. TEST PREDICTION TABLE
# ============================================================

prediction_df = test_df[
    [
        "task",
        "split",
        "label",
        "source",
        "sequence_group",
        "timestamp",
        "path",
        "filename",
    ]
].copy()

prediction_df[
    "true_label"
] = prediction_df["label"]

prediction_df[
    "predicted_label"
] = [
    index_to_label[index]
    for index in predicted_ids
]

prediction_df[
    "confidence"
] = confidence

prediction_df[
    "correct"
] = (
    true_ids
    == predicted_ids
)

prediction_df.to_csv(
    OUTPUT_ROOT /
    "test_predictions.csv",
    index=False
)


# ============================================================
# 21. TEST ERRORS
# ============================================================

error_df = prediction_df[
    ~prediction_df["correct"]
].copy()

error_df.to_csv(
    OUTPUT_ROOT /
    "test_errors.csv",
    index=False
)


# ============================================================
# 22. HIGH-CONFIDENCE ERRORS
# ============================================================

high_confidence_errors = error_df[
    error_df["confidence"] >= 0.80
].copy()

high_confidence_errors.to_csv(
    OUTPUT_ROOT /
    "high_confidence_errors.csv",
    index=False
)


# ============================================================
# 23. PER-CLASS SUMMARY
# ============================================================

per_class_rows = []

for index, label in enumerate(
    CLASS_NAMES
):

    support = int(
        (true_ids == index).sum()
    )

    correct = int(
        (
            (true_ids == index)
            &
            (predicted_ids == index)
        ).sum()
    )

    predicted = int(
        (predicted_ids == index).sum()
    )

    recall = (
        correct / support
        if support > 0
        else 0
    )

    precision = (
        correct / predicted
        if predicted > 0
        else 0
    )

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0

    per_class_rows.append(
        {
            "class_id": index,
            "label": label,
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_df.to_csv(
    OUTPUT_ROOT /
    "per_class_summary.csv",
    index=False
)


# ============================================================
# 24. BASELINE SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("18. BASELINE SUMMARY")
print("=" * 90)

print(
    f"Classes              : "
    f"{NUM_CLASSES}"
)

print(
    f"Train samples        : "
    f"{len(train_df)}"
)

print(
    f"Validation samples   : "
    f"{len(val_df)}"
)

print(
    f"Test samples         : "
    f"{len(test_df)}"
)

print(
    f"Test accuracy        : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Balanced accuracy    : "
    f"{balanced_accuracy * 100:.2f}%"
)

print(
    f"Macro F1             : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1          : "
    f"{weighted_f1:.4f}"
)

print(
    f"Test errors          : "
    f"{len(error_df)}"
)

print(
    f"High-conf errors     : "
    f"{len(high_confidence_errors)}"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_ROOT
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 90)
print("QUALITY V4 BASELINE TRAINING COMPLETED")
print("=" * 90)

print(
    "\nIMPORTANT:"
)

print(
    "This is the V4 frozen-backbone baseline."
)

print(
    "Do NOT tune the model yet."
)

print(
    "Use the results to determine whether "
    "the next step should be fine-tuning, "
    "data balancing, or dataset investigation."
)

print("=" * 90)