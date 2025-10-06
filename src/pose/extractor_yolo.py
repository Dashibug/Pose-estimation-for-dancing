import numpy as np, cv2
from ultralytics import YOLO

class YOLOPoseExtractor:
    """
    Извлекает (T,17,3) = [x,y,conf] из видео.
    """
    def __init__(self, model_name="yolov8n-pose.pt", device=None, conf=0.25):
        self.model = YOLO(model_name)
        if device is not None:
            self.model.to(device)
        self.conf = conf

    def extract(self, video_path, resize_long_side=720):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        frames = []
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = 1.0
        if max(W,H) > resize_long_side:
            scale = resize_long_side / max(W,H)

        kpts_all = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            if scale != 1.0:
                frame = cv2.resize(frame, (int(W*scale), int(H*scale)))
            res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
            if len(res.keypoints) == 0:
                # пустой кадр
                kpts_all.append(np.zeros((17,3), dtype=np.float32))
                continue
            # возьмём персону с макс. conf
            i = int(np.argmax(res.boxes.conf.cpu().numpy()))
            k = res.keypoints[i].data[0].cpu().numpy()  # (17,3)
            # нормализуем координаты в [0,1]
            h, w = frame.shape[:2]
            k[:,0] /= w
            k[:,1] /= h
            kpts_all.append(k.astype(np.float32))
        cap.release()
        return np.stack(kpts_all)  # (T,17,3)
