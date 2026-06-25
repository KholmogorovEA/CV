from importlib import simple
from ultralytics import YOLO
from benchmark import benchmark


def optimize_for_prodaction(weights, data, imgsz=640):
    model = YOLO(weights)

    model.export(format="onnx", imgsz=imgsz, dynamic=True, nms=True, simplify=True, opset=18)
    model.export(format="engine", imgsz=imgsz, half=True, nms=True)

    base_metric = YOLO(weights).val(data=data).box.map
    trt_metric = YOLO(weights.replace(".pt", ".engine")).val(data=data).box.map

    print(f"map: .pt{base_metric:.4f}  TRT-FP16={trt_metric:.4f}  потеря={base_metric - trt_metric:.4f}")

    benchmark(weights)
    benchmark(weights.replace(".pt", ".engine"))


if __name__ == '__main__':
    data = "C:\\Users\\bykho\\OneDrive\\Desktop\\yolo_trenirovka\\dataset_blood\\data.yaml"
    weights = "C:\\Users\\bykho\\OneDrive\\Desktop\\yolo_trenirovka\\dataset_blood\\runs\\detect\\runs\\detect_blood\\weights\\best.pt"
    optimize_for_prodaction(weights, data)


"""
.pt: 10.9 ms на картинку и TRT-FP16: 3.7 ms. Это почти 3× ускорение на чистом инференсе 
По всему пайплайну (препроцесс + инференс + постпроцесс): 16.8 ms и 8.4 ms, то есть почти 2× end-to-end 
цена за скорост - map: .pt0.6412  TRT-FP16=0.6210  потеря=0.0201 (потеря 0.020, примерно 3% относительно)
"""