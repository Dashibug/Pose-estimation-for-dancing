import os, glob, numpy as np, torch
from torch.utils.data import Dataset
from src.utils.preprocess import normalize_skeleton, make_windows

class WindowsDataset(Dataset):
    """
    Загружает все .npz из data/sessions/
    Возвращает окна (C=2,T,V=17) + метку video_id (для майнинга).
    """
    def __init__(self, root="data/sessions", T=100, hop=25, min_conf=0.2):
        self.items = []
        self.labels = []
        paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        for vid, p in enumerate(paths):
            npz = np.load(p, allow_pickle=True)
            seq = npz["kpts"]  # (N,17,3) x,y,conf
            seq_norm = normalize_skeleton(seq, min_conf=min_conf)  # (N,17,2)
            wins = make_windows(seq_norm, T=T, hop=hop)
            for w in wins:
                # to tensor NCHW (C,T,V) => (2,T,17)
                x = torch.from_numpy(w.transpose(2,0,1))  # (2,T,17)
                self.items.append(x.float())
                self.labels.append(vid)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.labels[i]
