# реф-эмбеддинги из .npz
import os, glob, yaml, torch, numpy as np
from tqdm import tqdm

from src.utils.dataset import WindowsDataset
from src.models.encoder_ctrgcn import PoseEncoder
from src.models.triplet_head import TripletHead

def main():
    CFG = yaml.safe_load(open("configs/default.yaml"))
    T        = int(CFG["window"]["frames"])
    HOP      = int(CFG["window"]["hop"])
    MINCONF  = float(CFG["window"]["min_conf"])
    IN_CH    = int(CFG["model"]["in_channels"])
    GCH      = CFG["model"]["gcn_channels"]
    KT       = int(CFG["model"]["kernel_t"])
    DROPOUT  = float(CFG["model"]["dropout"])
    EMBED    = int(CFG["model"]["embed_dim"])
    HEADDIM  = int(CFG["model"]["head_dim"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc  = PoseEncoder(in_ch=IN_CH, gcn_channels=GCH, k_t=KT, dropout=DROPOUT, embed_dim=EMBED).to(device)
    head = TripletHead(in_dim=EMBED, out_dim=HEADDIM, p=DROPOUT).to(device)
    enc.load_state_dict(torch.load("data/models/encoder.pt", map_location=device))
    head.load_state_dict(torch.load("data/models/triplet_head.pt", map_location=device))
    enc.eval(); head.eval()

    os.makedirs("data/ref", exist_ok=True)
    paths = sorted(glob.glob("data/sessions/*.npz"))
    ref_embeds = []
    ref_meta   = []

    with torch.no_grad():
        for p in tqdm(paths, desc="Ref build"):
            ds = WindowsDataset(root=os.path.dirname(p), T=T, hop=T, min_conf=MINCONF)  #
            # фильтруем по одному файлу
            keep_idx = [i for i,(x,_) in enumerate(ds.items)]
            X = torch.stack([ds.items[i] for i in range(len(ds.items))]).to(device) if len(ds)>0 else None
            if X is None or len(X)==0:
                continue
            z = head(enc(X)).cpu().numpy()  # (K,D)
            ref_embeds.append(z)
            ref_meta.append(os.path.basename(p))

    np.savez_compressed("data/ref/library.npz", embeds=ref_embeds, meta=ref_meta)
    print(f"saved data/ref/library.npz videos:{len(ref_meta)}")

if __name__ == "__main__":
    main()
