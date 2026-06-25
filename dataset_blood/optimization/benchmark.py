import time, numpy as np, torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark(model_path, imgsz=640, runs=200):
    model = YOLO(model_path)
    model.to(device)

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.int8)

    for _ in range(10):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(dummy, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)

    ms = np.array(times) * 1000

    print(f"{model_path}: p50={np.percentile(ms, 50):.1f}ms"
          f"p95={np.percentile(ms, 95):.1f}ms FPS={1000/ms.mean():.0f}")
    

    
