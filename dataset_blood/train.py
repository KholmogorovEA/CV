from ultralytics import YOLO
import torch
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt")
model.to(device)


if torch.cuda.is_available():
    print(f"GPU - {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA не доступене")


print(f"тип yolo - {model.model.__class__.__name__}")


def train_model():
    model.train(
    data="data.yaml",
    epochs=70,
    imgsz=640,
    batch=32,
    patience=15,
    exist_ok=True,
    pretrained=True,
    workers=2,
    device=0,
    optimizer="auto",
    cos_lr=True,
    seed=42,
    project="runs",
    name="detect_blood",
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    )
    

if __name__ == '__main__':
    print("start train")
    train_model()
    print("finish train")