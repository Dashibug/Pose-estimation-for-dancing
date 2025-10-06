# обучение энкодера + головы с TripletLoss
import os, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset import WindowsDataset
from src.models.encoder_ctrgcn import PoseEncoder
from src.models.triplet_head import TripletHead
from src.losses.triplet import batch_hard_triplet

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
    BS       = int(CFG["train"]["batch_size"])
    EPOCHS   = int(CFG["train"]["epochs"])
    LR       = float(CFG["train"]["lr"])
    MARGIN   = float(CFG["train"]["margin"])
    NW       = int(CFG["train"]["num_workers"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = WindowsDataset(root="data/sessions", T=T, hop=HOP, min_conf=MINCONF)
    if len(ds) < 8:
        print(f"[WARN] Very few windows ({len(ds)}). Add more videos!")
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=NW, pin_memory=True, drop_last=True)

    enc  = PoseEncoder(in_ch=IN_CH, gcn_channels=GCH, k_t=KT, dropout=DROPOUT, embed_dim=EMBED).to(device)
    head = TripletHead(in_dim=EMBED, out_dim=HEADDIM, p=DROPOUT).to(device)

    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    os.makedirs("data/models", exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        enc.train(); head.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}")
        losses, posm, negm = [], [], []
        for x, y in pbar:
            x = x.to(device)   # (B,2,T,17)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                z = enc(x)
                z = head(z)
                loss, pmean, nmean = batch_hard_triplet(z, y, margin=MARGIN)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item()); posm.append(pmean); negm.append(nmean)
            pbar.set_postfix(loss=np.mean(losses), pos=np.mean(posm), neg=np.mean(negm))

        torch.save(enc.state_dict(),  "data/models/encoder.pt")
        torch.save(head.state_dict(), "data/models/triplet_head.pt")
        print("💾 saved data/models/encoder.pt & triplet_head.pt")

if __name__ == "__main__":
    main()
