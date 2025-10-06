# граф COCO-17
import torch

# Рёбра COCO-17
_EDGES = [
    (5,6), (5,11), (6,12), (11,12),
    (5,7), (7,9), (6,8), (8,10),
    (11,13), (13,15), (12,14), (14,16),
    (0,5), (0,6)  # от носа к плечам (опц.)
]

def coco_adjacency(norm=True, device="cpu"):
    V = 17
    A = torch.zeros((V, V), dtype=torch.float32, device=device)
    for i,j in _EDGES:
        A[i,j] = 1.0
        A[j,i] = 1.0
    for i in range(V):
        A[i,i] = 1.0
    if norm:
        D = torch.sum(A, dim=1, keepdim=True)  # степени
        A = A / torch.clamp(D, min=1.0)        # нормализация
    return A  # (17,17)
