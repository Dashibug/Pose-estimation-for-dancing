import argparse, cv2, numpy as np
from src.utils.viz import draw_skeleton

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    K = data["kpts"]              # (T,17,3) норм. [x,y,conf]
    H, W = 720, 1280              # холст для просмотра
    delay = max(1, int(1000/args.fps))

    for t in range(len(K)):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_skeleton(frame, K[t], conf_thr=0.25)
        cv2.putText(frame, f"{t+1}/{len(K)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("NPZ preview", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
