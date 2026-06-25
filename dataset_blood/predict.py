from ultralytics import YOLO
import torch
from collections import Counter
import cv2 
from pathlib import Path
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

trh = {'Platelets': 0.444, 'RBC': 0.304, 'WBC': 0.392}
out = Path("out")
out.mkdir(exist_ok=True)
passed, failed = Counter(), Counter()

model = YOLO("C:\\Users\\bykho\\OneDrive\\Desktop\\yolo_trenirovka\\dataset_blood\\runs\\detect\\runs\\detect_blood\\weights\\best.pt")
model.to(device)

for res in model.predict("test/images/", stream=True, conf=0.25):
    keep = []
    
    for box in res.boxes:
        name = model.names[int(box.cls)]
        ok = float(box.conf) >= trh[name]
        keep.append(ok)
        (passed if ok else failed)[name] +=1

    res.boxes = res.boxes[np.array(keep, dtype=bool)]
    cv2.imwrite(str(out / Path(res.path).name), res.plot())

for name in trh:
    print(f"{name} прошло {passed[name]}  , отсеяно {failed[name]}")


"""
Platelets прошло 41  , отсеяно 4
RBC прошло 609  , отсеяно 67
WBC прошло 37  , отсеяно 0
"""