import torch, torch.nn as nn
from src.models.coco_graph import coco_adjacency

class GraphConv(nn.Module):
    """
    X: (N,C,T,V) → A*X → 1x1conv по каналам
    Edge importance: learnable scale для A.
    """
    def __init__(self, in_ch, out_ch, V=17):
        super().__init__()
        self.A = coco_adjacency().unsqueeze(0)  # (1,V,V)
        self.edge_importance = nn.Parameter(torch.ones(1, V, V))
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (N,C,T,V)
        N,C,T,V = x.shape
        A = (self.A.to(x.device) * self.edge_importance).clamp(min=0)
        # матр. умножение по V: (N,C,T,V) @ (V,V) -> (N,C,T,V)
        x = torch.einsum('nctv,vw->nctw', x, A[0])
        x = self.proj(x)
        return x

class TemporalConv(nn.Module):
    def __init__(self, ch, k=9, p=0.1):
        super().__init__()
        pad = (k - 1)//2
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(k,1), padding=(pad,0), groups=ch),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.BatchNorm2d(ch),
            nn.Dropout(p),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_t=9, p=0.1):
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch)
        self.tcn = TemporalConv(out_ch, k=k_t, p=p)
        self.down = nn.Identity()
        if in_ch != out_ch:
            self.down = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.down(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = x + res
        return self.act(x)

class PoseEncoder(nn.Module):
    """
    Вход: (N, C=2, T, V=17) → эмбеддинг D, L2-нормирован
    """
    def __init__(self, in_ch=2, gcn_channels=(64,128,256), k_t=9, dropout=0.1, embed_dim=256):
        super().__init__()
        chs = [in_ch] + list(gcn_channels)
        blocks = []
        for i in range(len(chs)-1):
            blocks.append(GCNBlock(chs[i], chs[i+1], k_t, dropout))
        self.backbone = nn.Sequential(*blocks)
        self.pool_t = nn.AdaptiveAvgPool2d((1, None))  # усреднить по времени
        self.pool_v = nn.AdaptiveAvgPool2d((1, 1))     # потом по суставам
        self.fc = nn.Linear(chs[-1], embed_dim)
        self.l2 = nn.functional.normalize

    def forward(self, x):
        # x: (N,2,T,17)
        x = self.backbone(x)
        x = self.pool_t(x)   # (N,C,1,V)
        x = self.pool_v(x)   # (N,C,1,1)
        x = x.squeeze(-1).squeeze(-1)  # (N,C)
        x = self.fc(x)       # (N,D)
        return self.l2(x, dim=-1)
