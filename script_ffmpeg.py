
import subprocess
from pathlib import Path

raw_video = Path("raw_video")
all_images = Path("all_images")
all_images.mkdir(exist_ok=True)


for video in raw_video.glob("*.mp4"):
    video_name = video.stem
    output = str(all_images / f"{video_name}_frame%04d.jpg")
    
    subprocess.run([
        "ffmpeg",
        "-i",
        str(video),
        "-vf",
        "fps=0.1",
        "-q:v",
        "2",
        output,
    ], check=True)



