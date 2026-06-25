import cv2                            # чтение картинок, resize, NMS
import numpy as np                   # работа с массивами/математика
import tritonclient.http as http     # HTTP-клиент Triton (порт 8000)

IMG = 640                                          # размер входа сети (квадрат 640x640)
NAMES = {0: "Platelets", 1: "RBC", 2: "WBC"}       # id класса -> человекочитаемое имя
CONF = {0: 0.444, 1: 0.304, 2: 0.392}              # per-class пороги уверенности (из predict.py)
IOU = 0.45                                          # порог IoU для NMS (насколько боксы пересекаются)

client = http.InferenceServerClient("localhost:8000")  # одно подключение к Triton на весь модуль


def letterbox(img):
    """Вписать картинку в 640x640 без искажений пропорций + серый паддинг по краям."""
    h, w = img.shape[:2]                            # исходные высота и ширина
    r = min(IMG / h, IMG / w)                       # коэффициент масштаба по меньшей стороне
    nh, nw = round(h * r), round(w * r)             # новые размеры после масштабирования
    pad_h, pad_w = (IMG - nh) // 2, (IMG - nw) // 2 # отступы, чтобы поставить картинку по центру
    resized = cv2.resize(img, (nw, nh))             # масштабируем с сохранением пропорций
    canvas = np.full((IMG, IMG, 3), 114, np.uint8)  # серый холст 640x640 (114 — как в YOLO)
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized   # кладём картинку в центр холста
    return canvas, r, pad_w, pad_h                  # r и паддинги нужны для обратного пересчёта боксов


def preprocess(img):
    """Одна BGR-картинка -> тензор [3,640,640] float32 + (r, pw, ph). БЕЗ оси батча."""
    canvas, r, pw, ph = letterbox(img)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)        # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, np.float32) / 255.0
    return x, (r, pw, ph)                            # отдаём «обратный адрес» этой картинки


def postprocess_one(dets, meta):
    """[300,6] -> список боксов в координатах оригинала."""
    r, pw, ph = meta
    dets[:, [0, 2]] = (dets[:, [0, 2]] - pw) / r        # X -> оригинал
    dets[:, [1, 3]] = (dets[:, [1, 3]] - ph) / r        # Y -> оригинал

    result = []
    for x1, y1, x2, y2, conf, c in dets:
        c = int(c)
        if conf >= CONF.get(c, 1.0):                    # per-class порог (паддинг с conf=0 отсеется)
            result.append({"cls": NAMES[c], "conf": round(float(conf), 3),
                           "xyxy": [int(x1), int(y1), int(x2), int(y2)]})
    return result


def detect_batch(imgs, model="detect_blood_trt"):
    """Список картинок -> список результатов (по одному на картинку)."""
    xs, metas = [], []
    for img in imgs:
        x, meta = preprocess(img)
        xs.append(x); metas.append(meta)             # копим тензоры и их адреса
    
    batch = np.stack(xs, 0)                          # [N, 3, 640, 640]

    inp = http.InferInput("images", batch.shape, "FP32")
    inp.set_data_from_numpy(batch)
    out = client.infer(model, [inp],
                       outputs=[http.InferRequestedOutput("output0")])
    out = out.as_numpy("output0")                    # [N, 300, 6]

    # КАЖДОЙ картинке — её строки выхода и её адрес
    return [postprocess_one(out[i], metas[i]) for i in range(len(imgs))]


def detect(img, model="detect_blood"):
    """Удобство для одной картинки."""
    return detect_batch([img], model)[0]


def draw(img, boxes):
    """Нарисовать боксы и подписи на КОПИИ картинки (для наглядной демонстрации)."""
    vis = img.copy()                                # не портим оригинал
    for b in boxes:                                 # по каждому найденному объекту
        x1, y1, x2, y2 = b["xyxy"]                  # углы бокса
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)        # зелёный прямоугольник
        label = f'{b["cls"]} {b["conf"]:.2f}'       # текст: класс + уверенность
        cv2.putText(vis, label, (x1, y1 - 5),       # подпись над боксом
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis                                      # картинка с разметкой




