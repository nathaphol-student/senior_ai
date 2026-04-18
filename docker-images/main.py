from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io
import os
import cv2
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "best-senior-ai.pt")
CONF_THRES = float(os.getenv("CONF_THRES", 0.12))
IOU_THRES = float(os.getenv("IOU_THRES", 0.5))
MAX_POINTS = int(os.getenv("MAX_POINTS", 7))
DEVICE = os.getenv("DEVICE", "cpu")

model = YOLO(MODEL_PATH)

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/model")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return "-1,-1"

        results = model.predict(
            source=img,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_POINTS,
            save=False,
            verbose=False,
            device=DEVICE,
        )

        result = results[0]

        # No any keypoint found
        if result.keypoints is None or len(result.keypoints.xyn) == 0:
            return "-1,-1"

        kpts = result.keypoints.xyn.cpu().numpy()
        cnfs = result.keypoints.conf.cpu().numpy()

        det_confs = cnfs[:, 0] if cnfs.ndim == 2 else cnfs
        best = int(np.argmax(det_confs))

        return f'{float(kpts[best][0][0])},{float(kpts[best][0][1])}'

    except Exception:
        return "-1,-1"


@app.post("/model_visualize")
async def predict_visualize(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return "-1,-1"

        results = model.predict(
            source=img,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_POINTS,
            save=False,
            verbose=False,
            device=DEVICE,
        )

        result = results[0]
        h, w = img.shape[:2]

        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result.keypoints is not None and len(result.keypoints.xyn) > 0:
            kpts = result.keypoints.xyn.cpu().numpy()
            cnfs = result.keypoints.conf.cpu().numpy()
            det_confs = cnfs[:, 0] if cnfs.ndim == 2 else cnfs
            best = int(np.argmax(det_confs))

            for i, (kpt_set, conf_set) in enumerate(zip(kpts, det_confs)):
                px, py = int(kpt_set[0][0] * w), int(kpt_set[0][1] * h)
                color = (0, 0, 255) if i == best else (255, 200, 0)
                cv2.circle(img, (px, py), 6, color, -1)
                nx, ny = float(kpt_set[0][0]), float(kpt_set[0][1])
                label = f"({nx:.5f},{ny:.5f}) {float(conf_set):.2f}"
                cv2.putText(img, label, (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        _, encoded = cv2.imencode(".png", img)
        return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")

    except Exception:
        return "-1,-1"
