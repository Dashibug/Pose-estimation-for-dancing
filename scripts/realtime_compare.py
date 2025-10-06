import cv2, yaml, torch, numpy as np, os
from collections import deque
from ultralytics import YOLO

from src.utils.preprocess import normalize_skeleton
from src.models.encoder_ctrgcn import PoseEncoder
from src.models.triplet_head import TripletHead


# пары суставов COCO-17
COCO_PAIRS = [
    (5,6), (5,11), (6,12), (11,12),
    (5,7), (7,9), (6,8), (8,10),
    (11,13), (13,15), (12,14), (14,16),
    (0,5), (0,6)
]

def cosine(a,b):
    return (a@b)/(np.linalg.norm(a)+1e-6)/(np.linalg.norm(b)+1e-6)

def render_skeleton_canvas(kpts_norm, size=512, conf_min=0.2, title=""):
    """
    Рисует скелет на белом холсте. kpts_norm: (17,3) с x,y в [0,1].
    """
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255

    if kpts_norm is None:
        cv2.putText(canvas, "No data", (size//2-70, size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120,120,120), 2)
    else:
        pts = (kpts_norm[:, :2] * size).astype(np.int32)
        conf = kpts_norm[:, 2]

        # кости
        for i, j in COCO_PAIRS:
            if conf[i] >= conf_min and conf[j] >= conf_min:
                cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), (0,0,0), 2, cv2.LINE_AA)
        # суставы
        for i, (x, y) in enumerate(pts):
            if conf[i] >= conf_min:
                cv2.circle(canvas, (int(x), int(y)), 5, (0,0,0), -1)

    # рамка и заголовок
    cv2.rectangle(canvas, (0,0), (size-1, size-1), (200,200,200), 2)
    if title:
        cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,60), 2)
    return canvas

def slice_windows_seq(seq_xyc, T, hop):
    """Режем (T0,17,3) на окна (T,17,3) с шагом hop."""
    T0 = seq_xyc.shape[0]
    wins = []
    for s in range(0, max(1, T0 - T + 1), hop):
        e = s + T
        if e <= T0:
            wins.append(seq_xyc[s:e])
    return wins

def main():
    CFG = yaml.safe_load(open("configs/default.yaml"))
    T        = int(CFG["window"]["frames"])
    IN_CH    = int(CFG["model"]["in_channels"])
    GCH      = CFG["model"]["gcn_channels"]
    KT       = int(CFG["model"]["kernel_t"])
    DROPOUT  = float(CFG["model"]["dropout"])
    EMBED    = int(CFG["model"]["embed_dim"])
    HEADDIM  = int(CFG["model"]["head_dim"])
    MINCONF  = float(CFG["window"]["min_conf"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # модели
    yolo = YOLO("yolov8n-pose.pt")
    if device == "cuda":
        yolo.to(device)

    enc  = PoseEncoder(in_ch=IN_CH, gcn_channels=GCH, k_t=KT, dropout=DROPOUT, embed_dim=EMBED).to(device)
    head = TripletHead(in_dim=EMBED, out_dim=HEADDIM, p=DROPOUT).to(device)
    enc.load_state_dict(torch.load("data/models/encoder.pt", map_location=device))
    head.load_state_dict(torch.load("data/models/triplet_head.pt", map_location=device))
    enc.eval(); head.eval()

    # реф-библиотека
    lib = np.load("data/ref/library.npz", allow_pickle=True)
    ref_embeds = [np.asarray(e) for e in lib["embeds"]]  # [(K,D), ...]
    ref_meta   = list(lib["meta"])                       # ["video1.npz", ...]

    # кэш окон исходных kpts для каждого реф-видео
    ref_windows_cache = {}  # name -> list of (T,17,3)

    def get_ref_midframe(name, idx):
        """
        Возвращает центральный кадр окна idx для эталона 'name' (17,3) в [0,1].
        Окна режутся с hop=T (как при build_ref_library.py).
        """
        if name not in ref_windows_cache:
            path1 = os.path.join("data", "sessions", name)
            path = path1 if os.path.isfile(path1) else (name if os.path.isfile(name) else None)
            if path is None:
                return None
            npz = np.load(path, allow_pickle=True)
            seq = np.asarray(npz["kpts"], dtype=np.float32)  # (N,17,3) в [0,1]
            ref_windows_cache[name] = slice_windows_seq(seq, T=T, hop=T)
        wins = ref_windows_cache[name]
        if not wins:
            return None
        idx = max(0, min(idx, len(wins)-1))
        win = wins[idx]
        return win[len(win)//2]  # центральный кадр (17,3)

    cap = cv2.VideoCapture(0)
    buf = deque(maxlen=T)

    user_canvas = render_skeleton_canvas(None, title="User")
    ref_canvas  = render_skeleton_canvas(None, title="Reference")
    size = user_canvas.shape[0]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        res = yolo.predict(frame, conf=0.25, verbose=False)[0]

        # --- получаем текущие kpts пользователя (в [0,1]) ---
        if len(res.keypoints) > 0:
            i = int(np.argmax(res.boxes.conf.detach().cpu().numpy()))
            k = res.keypoints[i].data[0].detach().cpu().numpy().astype(np.float32)  # (17,3)
            k[:,0] /= max(1, w); k[:,1] /= max(1, h)
            buf.append(k)
            # отрисуем текущий кадр пользователя
            user_canvas = render_skeleton_canvas(k, size=size, conf_min=MINCONF, title="User")
        else:
            buf.append(np.zeros((17,3), np.float32))
            user_canvas = render_skeleton_canvas(None, size=size, title="User • No person")

        # --- когда буфер полон, считаем эмбеддинг и находим лучший реф-окно ---
        score_txt = "collecting..."
        if len(buf) == T:
            seq = np.stack(buf)  # (T,17,3)
            norm = normalize_skeleton(seq, min_conf=MINCONF)  # (T,17,2)
            X = torch.from_numpy(norm.transpose(2,0,1)).unsqueeze(0).to(device)  # (1,2,T,17)

            with torch.no_grad():
                z = head(enc(X)).cpu().numpy()[0]  # (D,)

            best_sim, best_name, best_idx = -1.0, "N/A", -1
            for name, Z in zip(ref_meta, ref_embeds):
                sims = (Z @ z) / (np.linalg.norm(Z,axis=1)+1e-6)/(np.linalg.norm(z)+1e-6)
                idx  = int(np.argmax(sims))
                cur  = float(sims[idx])
                if cur > best_sim:
                    best_sim, best_name, best_idx = cur, name, idx

            score_txt = f"{best_name}: {best_sim:.2f} (win#{best_idx})"
            ref_mid = get_ref_midframe(best_name, best_idx)
            ref_canvas = render_skeleton_canvas(
                ref_mid, size=size, conf_min=MINCONF, title=f"Reference • {score_txt}"
            )

        # --- единое окно: слева user, справа reference ---
        combo = cv2.hconcat([user_canvas, ref_canvas])
        cv2.imshow("DancePose - Skeletons", combo)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
