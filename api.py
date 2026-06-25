"""FastAPI-ручка: картинка -> bbox'ы (JSON / картинка), видео -> асинхронная обработка с трекингом."""
import asyncio
import subprocess                                    # вызов ffmpeg для транскода в H.264
from concurrent.futures import ThreadPoolExecutor
import os, uuid, tempfile                            # пути, id задач, временная папка
import cv2                                           # декодирование/кодирование картинки
import numpy as np                                   # буфер байтов -> массив
from fastapi import FastAPI, UploadFile, Query, File, BackgroundTasks, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse
import io               # чтобы вернуть картинку (бинарный ответ)
import aiofiles #type: ignore
from triton.inference import detect, detect_batch, draw  # детект (одна/батч) + отрисовка

app = FastAPI(title="детекция клеток крови")          # создаём приложение

NAME2ID = {"Platelets": 0, "RBC": 1, "WBC": 2}        # имя класса -> id (для supervision)
jobs = {}                                            # in-memory реестр видео-задач: job_id -> статус


@app.get("/health")                                  # GET /health — проверка, что сервис жив
def health():
    return {"status": "ok"}                          # простой ответ для healthcheck/k8s-проб


@app.post("/detect")                                 # POST /detect — основная ручка детекции
async def detect_endpoint(
    file: UploadFile,                                # загруженный файл картинки (multipart/form-data)
    model: str = Query("detect_blood"),              # какая модель: 'detect_blood' (onnx) | 'detect_blood_trt'
):
    raw = await file.read()                          # читаем байты загруженного файла
    arr = np.frombuffer(raw, np.uint8)               # байты -> 1D-массив uint8
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)        # декодируем в BGR-картинку (как cv2.imread)
    boxes = detect(img, model)                       # препроцесс + инференс Triton + постпроцесс
    return {"model": model, "count": len(boxes), "detections": boxes}  # отдаём боксы клиенту




@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...), model: str = Query("detect_blood_trt")):
    data = await file.read()
    
    def work():
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        boxes = detect_batch([img])[0]
        vis = draw(img, boxes)                               # рисуем рамки
        ok, buf = cv2.imencode(".jpg", vis)
        return buf.tobytes()

    jpg = await asyncio.to_thread(work)

    return StreamingResponse(io.BytesIO(jpg), media_type="image/jpeg")



@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
    async with aiofiles.open(path, "wb") as f:
        await f.write(await file.read())
    jobs[job_id] = {"status": "processing"}
    asyncio.create_task(asyncio.to_thread(process_video, job_id, path))
    return {"job_id": job_id, "status": "processing"}

@app.get("/jobs/{job_id}")                               # клиент опрашивает статус
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    return jobs[job_id]


@app.get("/jobs/{job_id}/video")                         # отдать готовое размеченное видео файлом
def get_video(job_id: str):
    job = jobs.get(job_id)                               # ищем задачу
    if not job or job.get("status") != "done":          # нет задачи или ещё не готова
        raise HTTPException(404, "видео не готово")
    return FileResponse(job["result"], media_type="video/mp4")  # отдаём H.264-mp4 (играется в браузере)


def process_video(job_id: str, path: str):
    """Фоновая обработка всего видео: детекция + трекинг + запись размеченного файла."""
    import supervision as sv #type: ignore
    tracker = sv.ByteTrack()                             # трекер — В API (хранит ID между кадрами)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20                 # fps берём из исходника (запасной — 20)
    out_path = path.replace(".mp4", "_out.mp4")
    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect_batch([frame])[0]              # детекция кадра → Triton (список {cls,conf,xyxy})
            # адаптер наш формат -> sv.Detections (вручную, метода from_inference_like нет)
            if dets:
                tracked = sv.Detections(
                    xyxy=np.array([d["xyxy"] for d in dets], dtype=float),
                    confidence=np.array([d["conf"] for d in dets], dtype=float),
                    class_id=np.array([NAME2ID[d["cls"]] for d in dets], dtype=int))
            else:
                tracked = sv.Detections.empty()          # кадр без объектов
            tracked = tracker.update_with_detections(tracked)   # трекинг → присваивает ID
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                if tid is None:                          # неподтверждённый трек — пропустить
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)

        if writer:
            writer.release()                             # закрыть mp4v-файл ДО транскода (сбросить буфер на диск)

        # перекодировать mp4v -> H.264 (играется в браузере и обычных плеерах)
        h264_path = path.replace(".mp4", "_h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path,             # вход — наш mp4v-файл (-y: перезаписать без вопросов)
             "-c:v", "libx264", "-pix_fmt", "yuv420p",   # видеокодек H.264 + совместимый пиксельный формат
             "-movflags", "+faststart",                  # метаданные в начало -> браузер начинает играть сразу
             h264_path],
            check=True, capture_output=True)             # check=True -> исключение, если ffmpeg упал

        jobs[job_id] = {"status": "done", "result": h264_path}   # отдаём путь к H.264-файлу
    except Exception as e:
        jobs[job_id] = {"status": "error", "detail": str(e)}
    finally:
        cap.release()
        if writer: writer.release()                      # повторный release безопасен (no-op)