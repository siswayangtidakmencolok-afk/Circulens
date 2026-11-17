from collections import Counter
from io import BytesIO

import torch
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image
from torchvision import models, transforms

from services.risk_engine import (
    calculate_at_risk_quantity,
    calculate_contextual_risk,
    calculate_quality_score,
)


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


# ======================================================
# MODEL CONFIGURATION
# ======================================================

MODEL_PATH = "../models/circulens_quality_efficientnet_b0_finetuned.pth"

CLASSES = [
    "damaged",
    "dried",
    "old",
    "ripe",
    "unripe",
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================
# LOAD MODEL
# ======================================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    len(CLASSES),
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )


model.to(device)
model.eval()


# ======================================================
# IMAGE TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ======================================================
# MODEL PREDICTION
# ======================================================

def predict_image(
    image: Image.Image,
):
    """
    Run EfficientNet classification
    for a single image.
    """

    image = image.convert(
        "RGB"
    )

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1,
        )

    quality = CLASSES[
        predicted.item()
    ]

    confidence_value = (
        float(
            confidence.item()
        )
        * 100
    )

    return (
        quality,
        confidence_value,
    )


# ======================================================
# BATCH CONDITION
# ======================================================

def get_batch_condition(
    distribution,
    total,
):
    """
    Determine the dominant batch condition.

    If one quality represents at least 70%
    of the samples, that quality becomes the
    batch condition.

    Otherwise the batch is classified as mixed.
    """

    if total <= 0:
        return "unknown"

    dominant_quality, dominant_count = (
        distribution.most_common(1)[0]
    )

    dominant_ratio = (
        dominant_count / total
    )

    if dominant_ratio >= 0.7:
        return dominant_quality

    return "mixed"


# ======================================================
# FINDINGS
# ======================================================

def generate_findings(
    distribution,
    total,
    condition,
):
    findings = []

    for quality in CLASSES:

        count = distribution.get(
            quality,
            0,
        )

        if count == 0:
            continue

        percentage = round(
            (
                count
                / total
            )
            * 100
        )

        findings.append({
            "type": quality,
            "message": (
                f"{count} of {total} samples "
                f"({percentage}%) were classified "
                f"as {quality}."
            ),
        })

    if condition == "mixed":

        findings.append({
            "type": "batch",
            "message": (
                "The batch contains mixed "
                "quality conditions and should "
                "be sorted before distribution."
            ),
        })

    return findings


# ======================================================
# RECOMMENDATIONS
# ======================================================

def generate_recommendations(
    distribution,
    total,
    risk_level,
    temperature=None,
    humidity="",
    harvest_age="",
    storage_location="",
):
    """
    Generate contextual action recommendations.

    Recommendations use:
    - visual quality distribution
    - contextual risk level
    - temperature
    - humidity
    - harvest age
    - storage location
    """

    if total <= 0:
        return []

    recommendations = []

    damaged_ratio = (
        distribution.get("damaged", 0) / total
    )

    dried_ratio = (
        distribution.get("dried", 0) / total
    )

    old_ratio = (
        distribution.get("old", 0) / total
    )

    ripe_ratio = (
        distribution.get("ripe", 0) / total
    )

    unripe_ratio = (
        distribution.get("unripe", 0) / total
    )

    humidity_normalized = (
        str(humidity)
        .lower()
        .strip()
    )

    harvest_normalized = (
        str(harvest_age)
        .lower()
        .strip()
    )

    storage_normalized = (
        str(storage_location)
        .lower()
        .strip()
    )

    # ==================================================
    # 1. DAMAGED PRODUCE
    # ==================================================

    if damaged_ratio > 0:

        if damaged_ratio >= 0.5:
            recommendations.append(
                "Separate heavily damaged produce immediately "
                "to prevent affected samples from remaining "
                "in the main batch."
            )

        else:
            recommendations.append(
                "Separate damaged produce from the main batch "
                "and prioritize it for processing or recovery."
            )

    # ==================================================
    # 2. DRIED / OLD PRODUCE
    # ==================================================

    if (
        dried_ratio > 0
        or old_ratio > 0
    ):

        recommendations.append(
            "Prioritize dried or aging produce for processing "
            "or preservation before further quality decline."
        )

    # ==================================================
    # 3. RIPE PRODUCE
    # ==================================================

    if ripe_ratio > 0:

        recommendations.append(
            "Prioritize suitable ripe produce for fresh "
            "distribution while quality is still acceptable."
        )

    # ==================================================
    # 4. UNRIPE PRODUCE
    # ==================================================

    if unripe_ratio > 0:

        recommendations.append(
            "Keep unripe produce under suitable conditions "
            "and monitor maturity before fresh distribution."
        )

    # ==================================================
    # 5. TEMPERATURE
    # ==================================================

    if temperature is not None:

        if temperature >= 30:

            recommendations.append(
                "Move the batch to a cooler and more controlled "
                "storage environment because the current "
                "temperature may accelerate deterioration."
            )

        elif temperature >= 27:

            recommendations.append(
                "Consider moving the batch to a cooler storage "
                "environment to reduce deterioration risk."
            )

        elif temperature <= 10:

            recommendations.append(
                "Verify that the low storage temperature is "
                "appropriate for the crop and does not cause "
                "cold-related quality damage."
            )

    # ==================================================
    # 6. HUMIDITY
    # ==================================================

    if humidity_normalized == "high":

        recommendations.append(
            "Improve ventilation or humidity control because "
            "high humidity may increase deterioration risk."
        )

    elif humidity_normalized == "low":

        recommendations.append(
            "Monitor moisture loss because low humidity may "
            "increase drying and quality decline."
        )

    # ==================================================
    # 7. HARVEST AGE
    # ==================================================

    if harvest_normalized in [
        "5-7",
        "7+",
    ]:

        recommendations.append(
            "Prioritize this batch for distribution, processing, "
            "or preservation because it has been stored for "
            "several days since harvest."
        )

    elif harvest_normalized == "3-4":

        recommendations.append(
            "Monitor the batch closely and prioritize suitable "
            "produce before further quality deterioration."
        )

    # ==================================================
    # 8. STORAGE LOCATION
    # ==================================================

    if storage_normalized in [
        "open",
        "transport",
        "market",
    ]:

        recommendations.append(
            "Move the batch to a more controlled storage "
            "environment when possible."
        )

    # ==================================================
    # 9. HIGH CONTEXTUAL RISK
    # ==================================================

    if risk_level == "high":

        recommendations.append(
            "Take action promptly on at-risk produce before "
            "further quality decline increases potential loss."
        )

    elif risk_level == "moderate":

        recommendations.append(
            "Monitor the batch regularly and prioritize "
            "at-risk produce for the most suitable next action."
        )

    # ==================================================
    # 10. FALLBACK
    # ==================================================

    if not recommendations:

        recommendations.append(
            "Maintain current storage conditions and continue "
            "monitoring the batch for quality changes."
        )

    # Remove duplicate recommendations
    unique_recommendations = []

    for recommendation in recommendations:

        if recommendation not in unique_recommendations:
            unique_recommendations.append(
                recommendation
            )

    return unique_recommendations[:5]


# ======================================================
# CIRCULAR OPTIONS
# ======================================================

def generate_circular_options(
    distribution,
):
    options = []

    # Fresh distribution
    if distribution.get(
        "ripe",
        0,
    ) > 0:

        options.append(
            "Sell Fresh"
        )

    # Processing
    if (
        distribution.get(
            "damaged",
            0,
        ) > 0

        or distribution.get(
            "old",
            0,
        ) > 0
    ):

        options.append(
            "Process"
        )

    # Preservation
    if (
        distribution.get(
            "dried",
            0,
        ) > 0

        or distribution.get(
            "old",
            0,
        ) > 0
    ):

        options.append(
            "Preserve"
        )

    # Recovery
    if distribution.get(
        "damaged",
        0,
    ) > 0:

        options.append(
            "Recover"
        )

    return options


# ======================================================
# ANALYZE BATCH
# ======================================================

@router.post(
    "/analyze"
)
async def analyze_batch(
    crop_type: str = Form(...),
    quantity: float = Form(...),
    harvest_age: str = Form(...),
    storage_location: str = Form(""),
    temperature: float | None = Form(None),
    humidity: str = Form(""),
    notes: str = Form(""),
    images: list[UploadFile] = File(...),
):

    predictions = []

    # ==================================================
    # IMAGE PREDICTION
    # ==================================================

    for image_file in images:

        image_bytes = (
            await image_file.read()
        )

        try:

            image = Image.open(
                BytesIO(
                    image_bytes
                )
            )

            quality, confidence = (
                predict_image(
                    image
                )
            )

            predictions.append({
                "filename": (
                    image_file.filename
                ),
                "quality": quality,
                "confidence": round(
                    confidence,
                    2,
                ),
            })

        except Exception as error:

            print(
                "Failed to process image:",
                image_file.filename,
                error,
            )

    # ==================================================
    # NO VALID PREDICTIONS
    # ==================================================

    if not predictions:

        return {
            "success": False,
            "message": (
                "No valid images were provided."
            ),
        }

    # ==================================================
    # DISTRIBUTION
    # ==================================================

    distribution = Counter(
        prediction["quality"]
        for prediction
        in predictions
    )

    total = len(
        predictions
    )

    # ==================================================
    # BATCH CONDITION
    # ==================================================

    batch_condition = (
        get_batch_condition(
            distribution,
            total,
        )
    )

    # ==================================================
    # QUALITY SCORE
    # ==================================================

    quality_score = (
        calculate_quality_score(
            distribution=distribution,
            total=total,
        )
    )

    # ==================================================
    # CONTEXTUAL RISK
    # ==================================================

    risk_result = (
        calculate_contextual_risk(
            distribution=distribution,
            total=total,
            temperature=temperature,
            humidity=humidity,
            harvest_age=harvest_age,
            storage_location=storage_location,
        )
    )

    risk_level = (
        risk_result[
            "risk_level"
        ]
    )

    risk_score = (
        risk_result[
            "risk_score"
        ]
    )

    risk_factors = (
        risk_result[
            "risk_factors"
        ]
    )

    # ==================================================
    # FINDINGS
    # ==================================================

    findings = (
        generate_findings(
            distribution,
            total,
            batch_condition,
        )
    )

    # ==================================================
    # RECOMMENDATIONS
    # ==================================================

    recommendations = generate_recommendations(
    distribution,
    total,
    risk_result["risk_level"],
    temperature=temperature,
    humidity=humidity,
    harvest_age=harvest_age,
    storage_location=storage_location,
)

    # ==================================================
    # CIRCULAR OPTIONS
    # ==================================================

    circular_options = (
        generate_circular_options(
            distribution,
        )
    )

    # ==================================================
    # AVERAGE CONFIDENCE
    # ==================================================

    average_confidence = round(
        sum(
            prediction[
                "confidence"
            ]
            for prediction
            in predictions
        )
        / total,
        2,
    )

    # ==================================================
    # DISTRIBUTION RESULT
    # ==================================================

    distribution_result = {
        quality: distribution.get(
            quality,
            0,
        )
        for quality in CLASSES
    }

    distribution_percentage = {
        quality: round(
            (
                distribution.get(
                    quality,
                    0,
                )
                / total
            )
            * 100,
            2,
        )
        for quality in CLASSES
    }

    # ==================================================
    # ESTIMATED AT-RISK QUANTITY
    # ==================================================

    estimated_at_risk = (
        calculate_at_risk_quantity(
            quantity=quantity,
            distribution=distribution,
            total=total,
            risk_level=risk_level,
        )
    )

    # ==================================================
    # FINAL RESPONSE
    # ==================================================

    return {

        "success": True,

        "message": (
            "Batch analyzed successfully"
        ),

        "batch": {

            "crop_type": crop_type,

            "quantity": quantity,

            "harvest_age": harvest_age,

            "storage_location": (
                storage_location
            ),

            "temperature": temperature,

            "humidity": humidity,

            "notes": notes,

            "image_count": total,
        },

        "analysis": {

            "quality": batch_condition,

            "quality_score": quality_score,

            "confidence": average_confidence,

            "risk_level": risk_level,

            "risk_score": risk_score,

            "risk_factors": risk_factors,

            "estimated_at_risk_kg": (
                estimated_at_risk
            ),
        },

        "distribution": (
            distribution_result
        ),

        "distribution_percentage": (
            distribution_percentage
        ),

        "findings": findings,

        "recommendations": (
            recommendations
        ),

        "circular_options": (
            circular_options
        ),

        "predictions": predictions,
    }