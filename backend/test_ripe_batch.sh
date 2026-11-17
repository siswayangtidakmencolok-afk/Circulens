#!/bin/bash

for f in "../dataset/chili/dataset 1/Test/Bell Pepper_Ripe"/*.jpg; do

    echo "=========================================="
    echo "Testing: $(basename "$f")"
    echo "=========================================="

    curl -s -X POST http://127.0.0.1:8000/api/analyze \
    -F "crop_type=chili" \
    -F "quantity=10" \
    -F "harvest_age=0-2" \
    -F "storage_location=warehouse" \
    -F "temperature=20" \
    -F "humidity=low" \
    -F "notes=batch validation" \
    -F "images=@$f"

    echo
    echo

done
