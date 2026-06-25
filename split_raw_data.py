from pathlib import Path
import shutil, random
from collections import defaultdict

img = list(Path("all_video").glob("*.jpg"))

groups = defaultdict(list)

for p in img:
    video_id = p.stem.rsplit("_frame", 1)[0]
    groups[video_id].append(p)

video_ids = list(groups.keys())
random.seed(42)
random.shuffle(video_ids)
n_val = int(len(video_ids) * 0.3)
val_videos = set(video_ids[:n_val])

for vid, path in groups.items():
    split = "valid" if vid in video_ids else "train"
    for img in path:
        shutil.copy(img, f"dataset/{split}/images/{img.name}")


