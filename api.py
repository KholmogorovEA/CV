
import asyncio
import subprocess                                 
from concurrent.futures import ThreadPoolExecutor
import os, uuid, tempfile                            
import cv2                                         
import numpy as np                               
from fastapi import FastAPI, UploadFile, Query, File, BackgroundTasks, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse
import io               
import aiofiles #type: ignore
from triton.inference import detect, detect_batch, draw  

app = FastAPI(title="детекция клеток крови")        

NAME2ID = {"Platelets": 0, "RBC": 1, "WBC": 2}       
jobs = {}                                           


@app.get("/health")                                  
def health():
    return {"status": "ok"}                        


@app.post("/detect")                                
async def detect_endpoint(
    file: UploadFile,                               
    model: str = Query("detect_blood"),             
):
    raw = await file.read()                         
    arr = np.frombuffer(raw, np.uint8)              
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)        
    boxes = detect(img, model)                       
    return {"model": model, "count": len(boxes), "detections": boxes}  # отдаём боксы клиенту




@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...), model: str = Query("detect_blood_trt")):
    data = await file.read()
    
    def work():
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        boxes = detect_batch([img])[0]
        vis = draw(img, boxes)                              
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

@app.get("/jobs/{job_id}")                              
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    return jobs[job_id]


@app.get("/jobs/{job_id}/video")                   
def get_video(job_id: str):
    job = jobs.get(job_id)                            
    if not job or job.get("status") != "done":          
        raise HTTPException(404, "видео не готово")
    return FileResponse(job["result"], media_type="video/mp4") 


def process_video(job_id: str, path: str):
    import supervision as sv #type: ignore
    tracker = sv.ByteTrack()                           
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20              
    out_path = path.replace(".mp4", "_out.mp4")
    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect_batch([frame])[0]             
            # адаптер наш формат -> sv.Detections 
            if dets:
                tracked = sv.Detections(
                    xyxy=np.array([d["xyxy"] for d in dets], dtype=float),
                    confidence=np.array([d["conf"] for d in dets], dtype=float),
                    class_id=np.array([NAME2ID[d["cls"]] for d in dets], dtype=int))
            else:
                tracked = sv.Detections.empty()         
            tracked = tracker.update_with_detections(tracked)   
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                if tid is None:                         
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
            writer.release()                           

        h264_path = path.replace(".mp4", "_h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path,          
             "-c:v", "libx264", "-pix_fmt", "yuv420p",   
             "-movflags", "+faststart",                 
             h264_path],
            check=True, capture_output=True)             

        jobs[job_id] = {"status": "done", "result": h264_path}  
    except Exception as e:
        jobs[job_id] = {"status": "error", "detail": str(e)}
    finally:
        cap.release()
        if writer: writer.release()                     
