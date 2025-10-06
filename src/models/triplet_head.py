import torch, torch.nn as nn
import torch.nn.functional as F

class TripletHead(nn.Module):
    """
    Небольшая MLP-голова поверх эмбеддинга (опционально).
    """
    def __init__(self, in_dim=256, out_dim=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, z):
        z = self.net(z)
        return F.normalize(z, dim=-1)
