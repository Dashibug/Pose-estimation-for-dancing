import cv2
import numpy as np

# совпадает с тем, что мы используем в графе
EDGES = [
    (5,6), (5,11), (6,12), (11,12),
    (5,7), (7,9), (6,8), (8,10),
    (11,13), (13,15), (12,14), (14,16),
    (0,5), (0,6)  # нос-плечи
]

def draw_skeleton(frame, kpts_xyc, conf_thr=0.25, color_j=(0,255,0), color_e=(255,0,0)):
    """
    frame: BGR-кадр (H,W,3)
    kpts_xyc: (17,3) в нормализ координатах [x,y,conf] относительно кадра
              (если у тебя уже пиксели — просто не умножай на w/h ниже)
    conf_thr: порог уверенности точки
    """
    h, w = frame.shape[:2]
    pts = []
    for i in range(17):
        x, y, c = kpts_xyc[i]
        if c >= conf_thr:
            cx, cy = int(x * w), int(y * h)
            pts.append((cx, cy))
            cv2.circle(frame, (cx, cy), 4, color_j, -1)
        else:
            pts.append(None)

    for i, j in EDGES:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(frame, pts[i], pts[j], color_e, 2)

    return frame
