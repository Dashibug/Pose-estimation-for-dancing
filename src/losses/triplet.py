import torch, torch.nn.functional as F

def pairwise_dist(z):
    # z: (B,D) unit-norm → расстояние = 1 - cosine
    sim = z @ z.t()                 # (B,B)
    dist = (1.0 - sim).clamp_(min=0)
    return dist

def batch_hard_triplet(z, labels, margin=0.2):
    """
    z: (B,D), labels: (B,)
    Для каждого якоря: hardest positive, hardest negative в батче.
    """
    with torch.no_grad():
        same   = labels.unsqueeze(1).eq(labels.unsqueeze(0))  # (B,B)
        diff   = ~same
        eye    = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        same[eye] = False

    D = pairwise_dist(z)  # (B,B)

    pos = (D + (~same)*1e6).min(dim=1).values  # min dist среди позитивов
    neg = (D + (~diff)*1e6).max(dim=1).values  # max dist среди негативов (hardest)

    loss = (pos - neg + margin).clamp(min=0).mean()
    return loss, pos.mean().item(), neg.mean().item()
