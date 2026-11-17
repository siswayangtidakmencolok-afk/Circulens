from pathlib import Path

import torch
import torch.nn as nn
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image
from torchvision import transforms, models


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)

MODEL_PATH = Path(
    "../models/circulens_quality_efficientnet_b0_finetuned.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

classes = checkpoint["classes"]

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


def predict_image(image):
    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            1,
        )

    predicted_class = classes[prediction.item()]
    confidence_value = confidence.item() * 100

    return predicted_class, confidence_value


@router.post("/analyze")
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

    for image_file in images:
        image_data = await image_file.read()

        try:
            from io import BytesIO

            image = Image.open(
                BytesIO(image_data)
            )

            predicted_class, confidence = predict_image(
                image
            )

            predictions.append({
                "filename": image_file.filename,
                "quality": predicted_class,
                "confidence": round(
                    confidence,
                    2,
                ),
            })

        except Exception:
            predictions.append({
                "filename": image_file.filename,
                "quality": "unknown",
                "confidence": 0,
            })

    valid_predictions = [
        prediction
        for prediction in predictions
        if prediction["quality"] != "unknown"
    ]

    if valid_predictions:
        average_confidence = sum(
            prediction["confidence"]
            for prediction in valid_predictions
        ) / len(valid_predictions)

        quality_counts = {}

        for prediction in valid_predictions:
            quality = prediction["quality"]

            quality_counts[quality] = (
                quality_counts.get(
                    quality,
                    0,
                ) + 1
            )

        dominant_quality = max(
            quality_counts,
            key=quality_counts.get,
        )
    else:
        average_confidence = 0
        dominant_quality = "unknown"

    if dominant_quality in ["damaged", "dried"]:
        risk_level = "high"
    elif dominant_quality == "old":
        risk_level = "moderate"
    elif dominant_quality == "ripe":
        risk_level = "low"
    elif dominant_quality == "unripe":
        risk_level = "low"
    else:
        risk_level = "unknown"

    return {
        "success": True,
        "message": "Batch analyzed successfully",

        "batch": {
            "crop_type": crop_type,
            "quantity": quantity,
            "harvest_age": harvest_age,
            "storage_location": storage_location,
            "temperature": temperature,
            "humidity": humidity,
            "notes": notes,
            "image_count": len(images),
        },

        "analysis": {
            "quality": dominant_quality,
            "confidence": round(
                average_confidence,
                2,
            ),
            "risk_level": risk_level,
        },

        "predictions": predictions,
    }