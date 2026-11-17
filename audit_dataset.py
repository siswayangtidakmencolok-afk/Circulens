from pathlib import Path
from PIL import Image

ROOT = Path("dataset/circulens_quality")

total = 0
bad = []

for file in ROOT.rglob("*"):
    if not file.is_file():
        continue

    if file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    total += 1

    try:
        with Image.open(file) as image:
            image.verify()
    except Exception as e:
        bad.append((file, str(e)))

print(f"Total images: {total}")
print(f"Corrupt images: {len(bad)}")

if bad:
    print("\nFiles with problems:")

    for file, error in bad:
        print(f"- {file}")
        print(f"  {error}")
else:
    print("All images are valid.")