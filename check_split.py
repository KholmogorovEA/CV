from pathlib import Path

train_images = set(p.name for p in Path("dataset/train/images").glob("*.jpg"))
valid_iamges = set(p.name for p in Path("dataset/valid/images").glob("*.jpg"))

intersections = train_images & valid_iamges

if intersections:
    print(f"утчка {len(intersections)}")