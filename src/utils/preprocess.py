import numpy as np

COCO_PAIRS = [
    (5,6), (5,11), (6,12), (11,12),
    (5,7), (7,9), (6,8), (8,10),
    (11,13), (13,15), (12,14), (14,16)
]

def normalize_skeleton(seq_xyc, min_conf=0.2):
    """
    seq_xyc: (T, 17, 3) в пикселях или норм. [x,y,conf]
    Возврат: (T, 17, 2) нормировано: центр = mid-hip, масштаб = torso
    """
    T = seq_xyc.shape[0]
    out = np.zeros((T, 17, 2), dtype=np.float32)

    for t in range(T):
        kp = seq_xyc[t]  # (17,3)
        conf = kp[:,2]
        m = conf >= min_conf

        # center — средняя точка бёдер
        if m[11] and m[12]:
            center = (kp[11,:2] + kp[12,:2]) / 2.0
        else:
            vis = kp[m,:2]
            center = vis.mean(axis=0) if len(vis)>0 else np.array([0.0,0.0])

        xy = kp[:,:2] - center

        # масштаб — расстояние плечи - бёдра (по вертикали/евклидовое)
        torso = None
        if m[5] and m[6] and m[11] and m[12]:
            sh = (kp[5,:2] + kp[6,:2])/2.0
            hp = (kp[11,:2] + kp[12,:2])/2.0
            torso = np.linalg.norm(sh - hp)
        if torso is None or torso < 1e-6:
            # fallback: медиана расстояний вдоль связей
            dists = []
            for i,j in COCO_PAIRS:
                if m[i] and m[j]:
                    dists.append(np.linalg.norm(kp[i,:2]-kp[j,:2]))
            torso = np.median(dists) if dists else 1.0

        out[t] = xy / max(torso, 1e-6)

    return out.astype(np.float32)

def make_windows(seq_xy_norm, T=100, hop=25):
    """
    seq_xy_norm: (T0, 17, 2)
    -> список окон (T, 17, 2)
    """
    T0 = seq_xy_norm.shape[0]
    wins = []
    for s in range(0, max(1, T0 - T + 1), hop):
        e = s + T
        if e <= T0:
            wins.append(seq_xy_norm[s:e])
    return wins
