import cv2                            
import numpy as np                  
import tritonclient.http as http     

IMG = 640                                        
NAMES = {0: "Platelets", 1: "RBC", 2: "WBC"}       
CONF = {0: 0.444, 1: 0.304, 2: 0.392}             
IOU = 0.45                                         

client = http.InferenceServerClient("localhost:8000") 

def letterbox(img):
    h, w = img.shape[:2]                           
    r = min(IMG / h, IMG / w)                    
    nh, nw = round(h * r), round(w * r)             
    pad_h, pad_w = (IMG - nh) // 2, (IMG - nw) // 2 
    resized = cv2.resize(img, (nw, nh))            
    canvas = np.full((IMG, IMG, 3), 114, np.uint8)  
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized   
    return canvas, r, pad_w, pad_h                

def preprocess(img):
    canvas, r, pw, ph = letterbox(img)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)       
    x = np.ascontiguousarray(x, np.float32) / 255.0
    return x, (r, pw, ph)                          

def postprocess_one(dets, meta):
    r, pw, ph = meta
    dets[:, [0, 2]] = (dets[:, [0, 2]] - pw) / r        
    dets[:, [1, 3]] = (dets[:, [1, 3]] - ph) / r      

    result = []
    for x1, y1, x2, y2, conf, c in dets:
        c = int(c)
        if conf >= CONF.get(c, 1.0):                   
            result.append({"cls": NAMES[c], "conf": round(float(conf), 3),
                           "xyxy": [int(x1), int(y1), int(x2), int(y2)]})
    return result


def detect_batch(imgs, model="detect_blood_trt"):
    xs, metas = [], []
    for img in imgs:
        x, meta = preprocess(img)
        xs.append(x); metas.append(meta)           
    
    batch = np.stack(xs, 0)                          

    inp = http.InferInput("images", batch.shape, "FP32")
    inp.set_data_from_numpy(batch)
    out = client.infer(model, [inp],
                       outputs=[http.InferRequestedOutput("output0")])
    out = out.as_numpy("output0")             

    return [postprocess_one(out[i], metas[i]) for i in range(len(imgs))]


def detect(img, model="detect_blood"):
    return detect_batch([img], model)[0]


def draw(img, boxes):
    vis = img.copy()                             
    for b in boxes:                            
        x1, y1, x2, y2 = b["xyxy"]             
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)       
        label = f'{b["cls"]} {b["conf"]:.2f}'      
        cv2.putText(vis, label, (x1, y1 - 5),     
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis                                   




