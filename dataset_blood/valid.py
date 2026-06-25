
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("C:\\Users\\bykho\\OneDrive\\Desktop\\yolo_trenirovka\\dataset_blood\\runs\\detect\\runs\\detect_blood\\weights\\best.pt")
model.to(device)

def get_metrics():
    metrics = model.val(data="data.yaml")

    print("После обучения общий порог conf 0.339")
    print("Присутствует дисбаланс, по этой причине посчитаем conf для каждого класа")

    conf, f1 = metrics.box.curves_results[1][:2]
    conf_per_class = {model.names[c]: round(float(conf[f1[i].argmax()]), 3) for i, c in enumerate(metrics.box.ap_class_index)}
    print(f"{conf_per_class}")

    print("metrics all")
    print("----------------------------")
    print("mAP@0.5: ", metrics.box.map50)
    print("mAP@0.75: ", metrics.box.map75)
    print("mAP@0.5-0.95 : ", metrics.box.map)
    print("Precision : ", metrics.box.mp)
    print("Recall : ", metrics.box.mr)

    print("metrics per class: ")
    print("----------------------------")

    for cls_id, ap in enumerate(metrics.box.maps):
        print(f"{model.names[cls_id]} : {ap:.3f}")

    
if __name__ == '__main__':
    get_metrics()