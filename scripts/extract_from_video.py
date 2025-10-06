# видео → .npz (последовательность поз)
import argparse, os, numpy as np
from src.pose.extractor_yolo import YOLOPoseExtractor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="data/sessions")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    extractor = YOLOPoseExtractor("yolov8n-pose.pt")
    kpts = extractor.extract(args.video)  # (T,17,3)

    base = os.path.splitext(os.path.basename(args.video))[0]
    outp = os.path.join(args.outdir, f"{base}.npz")
    np.savez_compressed(outp, kpts=kpts)
    print(f"saved {outp}   shape:{kpts.shape}")

if __name__ == "__main__":
    main()
