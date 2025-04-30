import torch, random, numpy as np

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def nce_loss(center_vec, target_idx, embed, K=20):
    neg_idx = torch.randint(0, embed.weight.size(0), (len(center_vec), K), device=center_vec.device)
    pos = (center_vec * embed(target_idx)).sum(-1)
    neg = (embed(neg_idx) * center_vec.unsqueeze(1)).sum(-1)
    return (-torch.log(torch.sigmoid(pos)).mean() - torch.log(torch.sigmoid(-neg)).mean())